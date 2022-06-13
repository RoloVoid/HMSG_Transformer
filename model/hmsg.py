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
import model.basicattn as basicattn

# Hmsg_Transformer Model.
# It is an encoder_only network.
# Instead of decoder layer, it uses a temporal attention layer to get the final product

'''
raw_input = [seq_len, batch_size, d_model]
'''
class PreLayer(nn.Module):
    def __init__(self,seq_len,d_model,device):
        super(PreLayer,self).__init__()
        self.prehandler = nn.Sequential(
            basicattn.PositionalEncoding(seq_len,device),
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
        d_ff,
        device
        ):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = basicattn.MultiHeadAttention(d_q,d_k,d_v,n_heads,d_model,device)
        self.pos_ffn = basicattn.PoswiseFeedForwardNet(d_ff,d_model,device)

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
        seq_len,
        d_q,
        d_k,
        d_v,
        n_heads,
        d_ff,
        n_layers,
        d_model,
        device
        ):
        super(Encoder,self).__init__()
        self.prelayer = PreLayer(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_q,d_k,d_v,n_heads,d_model,d_ff,device) for _ in range (n_layers)])

    def forward(self,raw_input):
        enc_out = self.prelayer(raw_input).transpose(0,1)
        enc_self_attn_mask = basicattn.get_attn_pad_mask(enc_out,enc_out)

        # concat weights of w_v for Orthogonal Regularization
        # weight: [d_v * n_heads, d_model]
        w_vhs = []
        for l in self.encoder_layers:
            enc_out = l(enc_out,enc_self_attn_mask)
            w_vhs.append(l.enc_self_attn.W_V.weight)
        self.w_vhs = w_vhs
        return enc_out


class TemporalAttention(nn.Module):
    def __init__(
        self,
        d_model,
        seq_len,
        ):
        super(TemporalAttention,self).__init__()

        self.seq_len = seq_len
        self.u_t = nn.Linear(1,1,bias=False)
        self.W_e = nn.Linear(d_model,1)

    def forward(self,enc_out):
        batch_size = enc_out.size(0)
        tgtweight = torch.zeros(self.seq_len,batch_size,1)
        # [batch_size,seq_len,d_model] -> [seq_len,batch_size,d_model]
        enc_out = enc_out.transpose(0,1)
        for m in range (self.seq_len):
            alpha_s_t = self.u_t(self.W_e(enc_out))
            tgtweight[m]=alpha_s_t
        # [seq_len,batch_size,1] -> [seq_len,batch_size] -> [batch_size,seq_len]
        tgtweight = F.softmax(tgtweight).squeeze(2).transpose(0,1)
        # [seq_len,batch_size,d_model] -> [d_model,batch_size,seq_len]
        enc_out = enc_out.transpose(0,2)
        # [d_model,batch_size,seq_len]*[batch_size,seq_len] -> [d_model,batch_size,seq_len] -> [d_model, batch_size]
        return torch.sum((enc_out*tgtweight),dim=2)


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
        return self.net(tgt.transpose(0,1)).squeeze(1)

# [batch_size, seq_len, d_model] -> [batch_size]
class TemporalAggregation(nn.Module):
    def __init__(
        self,
        d_model,
        seq_len
        ):
        super(TemporalAggregation,self).__init__()
        # self.TALayer = TemporalAttnLayer(H,d_model,batch_size,seq_len)
        self.TALayer = TemporalAttention(d_model,seq_len)
        self.ALayer = AggregationLayer(d_model)
    def forward(self,enc_out):
        return self.ALayer(self.TALayer(enc_out))

    
# [seq_len, batch_size, d_model] -> [batch_size]
class HMSGTransformer(nn.Module):
    def __init__(
        self,
        H,
        d_q,
        d_k,
        d_v,
        n_heads,
        n_layers,
        d_ff,
        d_model,
        seq_len
        ):
        super(HMSGTransformer,self).__init__()
        self.encoder = Encoder(d_model,d_q,d_k,d_v,n_heads,d_ff,n_layers,d_model)
        self.temporalaggr = TemporalAggregation(H,d_model,seq_len)

    def forward(self,raw_input):
        return self.temporalaggr(self.encoder(raw_input))