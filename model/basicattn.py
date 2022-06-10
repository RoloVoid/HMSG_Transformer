'''
Basic transformer functions, including normal attn and gaussian prior
'''

import math
import cmath
import torch
import numpy as np
from torch import nn

# d_model = config.d_model
# device = config.device
# d_ff = config.d_ff
# n_heads = config.n_heads
# d_k,d_v = config.d_k,config.d_v
# sigma_hs = [5,10,20,40]

# for posfeedforward part of encoder
class PoswiseFeedForwardNet(nn.Module):
    def __init__(
        self,
        d_ff,
        d_model,
        device="cpu"
        ):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.device = device
        self.fc = nn.Sequential(
            # seq_len*d_ff
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            # seq_len*d_ff
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual)  # [batch_size, seq_len, d_model]

# d_model is the number features
# basic postional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, seq_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, seq_len, 2).float() * (-math.log(10000.0) / seq_len))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# pad mask function
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  

# seq mask function
def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) 
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

# Gaussian Prior Bias
# sigma_h is a hyperparameter to enchance data locality
# sigma_h is unique in each head
def GenerateGaussianPrior(n_heads,len_q,len_k,*sigma_hs) -> torch.Tensor:
    GaussianMask = target.zeros(n_heads,len_q,len_k)
    dim0,dim1 = target.shape
    for i in n_heads:
        target = torch.zeros(len_q,len_k)
        sigma_h = sigma_hs[i]
        for i in range (dim0):
            for j in range (dim1):
                if i >= j: target[i][j] = cmath.exp(-math.pow(j-i,2)/(2*pow(sigma_h,2))) 
        GaussianMask[i] = target
    return GaussianMask

# Basic ScaledDotProduct for transformer
class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        d_k
        ):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) 
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores) 
        context = torch.matmul(attn, V)  
        return context, attn

# ScaledDotProduct With Gaussian Prior
class GPSDPAttention(nn.Module):
    def __init__(
        self,
        d_k
        ):
        super(GPSDPAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores: batch_size,n_heads,len_q,len_k
        batch_size,n_heads,len_q,len_k = scores.shape
        # add gaussian prior
        gm = GenerateGaussianPrior(n_heads,len_q,len_k)
        for i in range(batch_size):
            scores[i] +=  gm
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores) 
        context = torch.matmul(attn, V)  
        return context, attn

# basic multihead_attention with attn func as parameter
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_k,
        d_q,
        d_v,
        n_heads,
        d_model,
        ):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.attn = GPSDPAttention
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_v = d_v

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # with gaussian prior
        context, attn = self.attn()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)

        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual), attn