# personal test for Hierarchical Multi-Scale Gaussian Transformer
# dzy 2022.4.27

'''
Ding, Qianggang, Sifan Wu, Hao Sun, Jiadong Guo and Jian Guo. 
《Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction》. 
Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, 
4640–46. Yokohama, Japan: International Joint Conferences on Artificial Intelligence Organization, 2020. 
https://doi.org/10.24963/ijcai.2020/640.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basicattn import PositionalEncoding, \
        MultiHeadAttention, \
        PoswiseFeedForwardNet, \
        get_attn_subsequence_mask, \
        get_hierarchical_mask

# Hmsg_Transformer Model.
# It is an encoder_only network.
# Instead of decoder layer, it uses a temporal attention layer to get the final product

sigma_hs = [5,10,20,40]

'''
raw_input = [batch_size, seq_len, f_size]
'''
class PreLayer(nn.Module):
    def __init__(self,seq_len,f_size,device):
        super(PreLayer,self).__init__()
        self.pos = PositionalEncoding(seq_len,device)
        self.linear = nn.Linear(f_size,f_size,bias=False)

    def forward(self,raw_input):
        return torch.tanh(self.linear(self.pos(raw_input)))

"""
enc_inputs: [batch_size, seq_len, f_size]
enc_self_attn_mask: [batch_size, seq_len, seq_len]
"""
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_q,
        d_k,
        d_v,
        n_heads,
        f_size,
        d_ff,
        device
        ):
        super(EncoderLayer, self).__init__()
        self.device = device
        self.enc_self_attn = MultiHeadAttention(d_q,d_k,d_v,n_heads,f_size,device)
        self.pos_ffn = PoswiseFeedForwardNet(d_ff,f_size,device)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_inputs = enc_inputs.to(self.device)
        enc_self_attn_mask = enc_self_attn_mask.to(self.device)
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask).to(self.device) 
        # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, seq_len, f_size]
        return enc_outputs

'''
Encoder: [batch_size, seq_len, f_size] -> [batch_size, seq_len, f_size] 
'''
class Encoder(nn.Module):
    def __init__(
        self,
        seq_len,
        d_q,
        d_k,
        d_v,
        n_heads,
        d_ff,
        n_layers,
        f_size,
        device
        ):
        super(Encoder,self).__init__()
        self.prelayer = PreLayer(seq_len,f_size,device)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_q,d_k,d_v,n_heads,f_size,d_ff,device) for _ in range (n_layers)])
        self.w_vhs = []
        self.device = device

    def forward(self,raw_input):
        # [batch_size, seq_len, f_size]
        enc_out = self.prelayer(raw_input).to(self.device)
        seq_attn_mask = get_attn_subsequence_mask(enc_out).to(self.device)
        enc_self_attn_mask = seq_attn_mask

        # extract weights of w_v for Orthogonal Regularization
        # weight: [d_v * n_heads, f_size]
        counter = 0
        for l in self.encoder_layers:
            # Add hierarchical mask as trading gap spliter
            h_mask = get_hierarchical_mask(enc_out,sigma_hs[counter]).to(self.device)
            h_enc_self_attn_mask = (h_mask*enc_self_attn_mask).to(self.device)
            enc_out = l(enc_out,h_enc_self_attn_mask).to(self.device)
            self.w_vhs.append(l.enc_self_attn.W_V.weight)
        return enc_out

# [batch_size,seq_len,f_size]
class TemporalAttnWithLstm(nn.Module):
    def __init__(
        self,
        H,
        E,
        f_size,
        device
        ):
        super(TemporalAttnWithLstm,self).__init__()
        self.H = H
        self.E = E
        self.device = device
        self.LSTM = nn.LSTMCell(f_size,H)
        self.W_a = nn.Linear(H,E)
        self.U_a_T = nn.Linear(E,1,bias=False)

    def forward(self,enc_out):
        # [batch_size,seq_len,f_size] -> [seq_len, batch_size, f_size]
        enc_out = enc_out.transpose(0,1)
        batch_size = enc_out.size(1)
        hx = torch.zeros(batch_size,self.H)
        cx = torch.zeros(batch_size,self.H)

        output = []
        for i in range (enc_out.size(0)):
            hx,cx = self.LSTM(enc_out[i],(hx,cx))
            output.append(hx)

        # [seq_len*batch_size, self.H]
        output = torch.cat(output,dim=0)

        # [seq_len*batch_size, 1]->[batch_size,seq_len]
        a_s_tilda = self.U_a_T(torch.tanh(self.W_a(output))).squeeze(1).reshape(batch_size,-1)
        a_s = F.softmax(a_s_tilda,dim=1)

        # [seq_len, batch_size, f_size] -> [f_size, batch_size, seq_len]
        enc_out = enc_out.permute(2,1,0)

        # [batch_size, f_size]
        return torch.sum(enc_out*a_s,dim=2).transpose(0,1)

# [batch_size,f_size] -> [batch_size,1] -> [batch_size]
class AggregationLayer(nn.Module):
    def __init__(
        self,
        f_size
        ):
        super(AggregationLayer,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(f_size,1,bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,tgt):
        return self.net(tgt.transpose(0,1)).squeeze(1)

# [batch_size, seq_len, f_size] -> [batch_size]
class TemporalAggregation(nn.Module):
    def __init__(
        self,
        H,
        E,
        f_size,
        seq_len
        ):
        super(TemporalAggregation,self).__init__()
        # self.TALayer = TemporalAttnLayer(H,f_size,batch_size,seq_len)
        self.TALayer = TemporalAttnWithLstm(H,E,f_size,seq_len)
        self.ALayer = AggregationLayer(f_size)
    def forward(self,enc_out):
        return self.ALayer(self.TALayer(enc_out))
    
# [seq_len, batch_size, f_size] -> [batch_size]
class HMSGTransformer(nn.Module):
    def __init__(
        self,
        H,
        E,
        d_q,
        d_k,
        d_v,
        n_heads,
        n_layers,
        d_ff,
        f_size,
        seq_len,
        device
        ):
        super(HMSGTransformer,self).__init__()
        self.encoder = Encoder(seq_len,d_q,d_k,d_v,n_heads,d_ff,n_layers,f_size,device)
        self.temporalaggr = TemporalAggregation(H,E,f_size,seq_len)

    def forward(self,raw_input):
        temp = self.encoder(raw_input)
        return self.temporalaggr(temp)