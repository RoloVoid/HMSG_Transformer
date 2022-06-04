# personal test for Hierarchical Multi-Scale Gaussian Transformer
# dzy 2022.4.27

'''Ding, Qianggang, Sifan Wu, Hao Sun, Jiadong Guo and Jian Guo. 
《Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction》. 
Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, 
4640–46. Yokohama, Japan: International Joint Conferences on Artificial Intelligence Organization, 2020. 
https://doi.org/10.24963/ijcai.2020/640.
'''

import torch
import torch.nn as nn
import torch.functional as F
import basicattn, config

# d_model = config.d_model
device = config.device
# n_heads = config.n_heads
# n_layers = config.n_layers
# batch_size = config.batch_size
# seq_len = config.seq_len

# Hmsg_Transformer Model.
# It is an encoder_only network.
# Instead of decoder layer, it uses a temporal attention layer to get the final product

'''
raw_input = [seq_len, batch_size, d_model]
'''
class PreLayer(nn.Module):
    def __init__(self,d_model):
        super(PreLayer,self).__init__()
        self.prehandler = nn.Sequential(
            basicattn.PositionalEncoding(d_model),
            nn.Linear(d_model,d_model,bias=False),
            nn.Tanh()
        )

    def forward(self,raw_input):
        return self.prehandler(raw_input)

"""
enc_inputs: [batch_size, seq_len, d_model]
enc_self_attn_mask: [batch_size, seq_len, seq_len]
"""
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_q,
        d_k,
        d_v,
        n_heads,
        d_model,
        d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = basicattn.MultiHeadAttention(d_q,d_k,d_v,n_heads,d_model)
        self.pos_ffn = basicattn.PoswiseFeedForwardNet(d_ff,d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) 
        # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs, attn

'''
Encoder: [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model] 
'''
class Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_q,
        d_k,
        d_v,
        n_heads,
        d_ff,
        n_layers
        ):
        super(Encoder,self).__init__()
        self.prelayer = PreLayer()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_q,d_k,d_v,n_heads,d_model,d_ff) for _ in range (n_layers)])

    def forward(self,raw_input):
        enc_out = self.prelayer(raw_input).transpose(0,1)
        enc_self_attn_mask = basicattn.get_attn_pad_mask(enc_out,enc_out)
        for l in self.encoder_layers:
            enc_out, _ = l(enc_out,enc_self_attn_mask)
        return enc_out

'''
TemporalAttnLayer: [batch_size, seq_len, d_model] -> [batch_size, d_model]
'''
class TemporalAttnLayer(nn.Module):    
    def __init__(
        self,
        H,
        d_model,
        batch_size,
        seq_len
        ):
        """
        H: The length of hidden size of lstm
        """
        super(TemporalAttnLayer, self).__init__()
        self.M = d_model
        self.H = H
        self.T = seq_len
        self.batch_size = batch_size

        self.encoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.H)

        self.w_tilda = nn.Linear(self.M + 1, 1)
        
        #equation 12 matrices
        self.W_d = nn.Linear(2*self.H, self.M)
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias = False)
    
    def forward(self, enc_inputs,y):
        # initializing hidden states
        d_tm1 = torch.zeros((enc_inputs.size(0), self.H))
        s_prime_tm1 = torch.zeros((enc_inputs.size(0), self.H))
        c_t = torch.zeros((enc_inputs.size(0), self.H))
        tgt = torch.zeros(self.batch_size, self.M)
        beta_i_t = torch.ones
        for t in range(self.T):
            # concatenate hidden states -> [batch_size, 2H]
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)
            # [batch_size, 2H] -> [batch_size, d_model] -> [batch_size, 1, d_model] -> [batch_size, seq_len, d_model]
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, enc_inputs.shape[1], 1)
            # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
            y1 = self.U_d(enc_inputs)
            # [batch_size, seq_len, d_model]
            z1 = torch.tanh(x1 + y1)
            # [batch_size, seq_len, 1]
            l_i_t = self.v_d(z1)
            
            beta_i_t = F.softmax(l_i_t, dim=1) 
            # [batch_size, seq_len, 1] * [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model] -> [batch_size, d_model]
            c_t = torch.sum(beta_i_t * enc_inputs, dim=1)
            tgt = tgt + c_t # get target series
            
            # [batch_size, d_model+1]
            y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1)
            # [batch_size, 1]
            y_tilda_t = self.w_tilda(y_c_concat)
            
            # next hidden states
            d_tm1, s_prime_tm1 = self.encoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))
        
        return tgt

# [batch_size,d_model] -> [batch_size,1] -> [batch_size]
class AggregationLayer(nn.Module):
    def __init__(
        self,
        d_model
        ):
        super(AggregationLayer,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,1,bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,tgt):
        return self.net(tgt).squeeze(1)

# [batch_size, seq_len, d_model] -> [batch_size]
class TemporalAggregation(nn.Module):
    def __init__(
        self,
        H,
        d_model,
        batch_size,
        seq_len
        ):
        super(TemporalAggregation,self).__init__()
        self.H = H
        self.TALayer = TemporalAttnLayer(H,d_model,batch_size,seq_len)
        self.ALayer = AggregationLayer(d_model)
    def forward(self,enc_out):
        return self.ALayer(self.TALayer(self.H)(enc_out))

    
# [seq_len, batch_size, d_model] -> [batch_size]
class HMSG_Transformer(nn.Module):
    def __init__(
        self,
        H,
        d_q,
        d_k,
        d_v,
        n_heads,
        n_layers,
        d_ff,
        batch_size,
        device,
        d_model,
        seq_len
        ):
        super(HMSG_Transformer,self).__init__()
        self.encoder = Encoder(self,d_model,d_q,d_k,d_v,n_heads,d_ff,n_layers)
        self.temporalaggr = TemporalAggregation(H,d_model,batch_size,seq_len)

    def forward(self,raw_input):
        return self.temporalaggr(self.encoder(raw_input))