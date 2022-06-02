# personal test for Hierarchical Multi-Scale Gaussian Transformer
# dzy 2022.4.27

'''Ding, Qianggang, Sifan Wu, Hao Sun, Jiadong Guo and Jian Guo. 
《Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction》. 
Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, 
4640–46. Yokohama, Japan: International Joint Conferences on Artificial Intelligence Organization, 2020. 
https://doi.org/10.24963/ijcai.2020/640.
'''

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.functional as F
from . import basicattn, config

d_model = config.d_model
device = config.device
n_heads = config.n_heads
n_layers = config.n_layers
batch_size = config.batch_size
seq_len = config.seq_len

# Hmsg_Transformer Model.
# It is an encoder_only network.
# Instead of decoder layer, it uses a temporal attention layer to get the final product


'''
raw_input = [seq_len, batch_size, d_model]
'''
class PreLayer(nn.Module):
    def __init__(self):
        super(PreLayer,self).__init__()
        self.prehandler = nn.Sequential(
            basicattn.PositionalEncoding(),
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
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = basicattn.MultiHeadAttention()
        self.pos_ffn = basicattn.PoswiseFeedForwardNet()

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
    def __init__(self):
        super(Encoder,self).__init__()
        self.prelayer = PreLayer()
        self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range (n_layers)])

    def forward(self,raw_input):
        enc_out = self.prelayer(raw_input).transpose(0,1)
        enc_self_attn_mask = basicattn.get_attn_pad_mask(enc_out,enc_out)
        for l in self.encoder_layers:
            enc_out, _ = l(enc_out,enc_self_attn_mask)
        return enc_out

'''
TemporalAttnLayer: [batch_size, seq_len, d_model] -> [batch_size, d_model,seq_len] -> [batch_size,d_model]
'''
class TemporalAttnLayer(nn.Module):
    
    __constants__ = ["P", "T"]
    
    def __init__(self, M, P, T, stateful=False):
        """
        :param: M: int
            number of encoder LSTM units
        :param: P:
            number of deocder LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.M = M
        self.P = P
        self.T = T
        self.stateful = stateful
        
        self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.P)
        
        #equation 12 matrices
        self.W_d = nn.Linear(2*self.P, self.M)
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias = False)
        
        #equation 15 matrix
        self.w_tilda = nn.Linear(self.M + 1, 1)
        
        #equation 22 matrices
        self.W_y = nn.Linear(self.P + self.M, self.P)
        self.v_y = nn.Linear(self.P, 1)
    
    def forward(self, encoded_inputs, y):
        
        #initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).cuda()
        s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).cuda()
        c_t = torch.zeros((encoded_inputs.size(0), self.P)).cuda()
        for t in range(self.T):
            #concatenate hidden states
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)
            #print(d_s_prime_concat)
            #temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1)
            y1 = self.U_d(encoded_inputs)
            z1 = torch.tanh(x1 + y1)
            l_i_t = self.v_d(z1)
            
            #normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1)
            
            #create context vector (equation_14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1)
            
            #concatenate c_t and y_t
            y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1)
            #create y_tilda
            y_tilda_t = self.w_tilda(y_c_concat)
            
            #calculate next hidden states (equation 16)
            d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))
        
        #concatenate context vector at step T and hidden state at step T
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)

        #calculate output
        y_Tp1 = self.v_y(self.W_y(d_c_concat))
        return y_Tp1

# class TemporalAttnLayer(nn.Module):
#     def __init__(self):
#         super(TemporalAttnLayer,self).__init__()

#         # paramters for equations
#         self.W_d = nn.Linear(seq_len, seq_len)
#         self.U_d = nn.Linear(seq_len, seq_len, bias=False)
#         self.v_d = nn.Linear(seq_len, 1, bias = False)

#     def forward(self,enc_out):
#         enc_out = enc_out.transpose(1,2)
#         r1 = self.W_d(enc_out)
#         r2 = torch.tanh(self.U_d(enc_out)+r1)
#         r3 = self.v_d(r2).squeeze(-1)

class AggregationLayer(nn.Module):
    def __init__(self):
        super(AggregationLayer,self).__init__()


class HMSG_Transformer(nn.Module):
    def __init__(self,
        encoder,
        temporal_aggr_layer,
        ):
        super(HMSG_Transformer,self).__init__()
        self.encoder = encoder
        self.temporal_aggr_layer = temporal_aggr_layer

    def forward(self):
        print("f")