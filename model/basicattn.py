'''
Basic transformer functions
This part contains traditional transformer encoder part and some methods from paper
'''

import math
import torch
import numpy as np
from torch import nn

sigma_hs = [5,10,20,40]

# for posfeedforward part of encoder
class PoswiseFeedForwardNet(nn.Module):
    def __init__(
        self,
        d_ff,
        f_size,
        device
        ):
        super(PoswiseFeedForwardNet, self).__init__()
        self.f_size = f_size
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(f_size, d_ff, bias=False,device=device),
            nn.ReLU(),
            nn.Linear(d_ff, f_size, bias=False,device=device)
        )

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, f_size]
        residual = inputs.to(self.device)
        inputs = inputs.to(self.device)
        output = self.fc(inputs)
        return nn.LayerNorm(self.f_size).to(self.device)(output + residual)  # [batch_size, seq_len, f_size]

# f_size is the number features
# basic postional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, f_size, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # [max_len,f_size]
        check = (f_size % 2 != 0)
        if check: f_size+=1
        pe = torch.zeros(max_len, f_size)
        # [max_len] -> [max_len,1]
        position = torch.arange(0, max_len, dtype=torch.float,device=device).unsqueeze(1)
        # [ceil(f_size/2)]
        div_term = torch.exp(torch.arange(0, f_size, 2,device=device).float() * (-math.log(10000.0) / f_size))

        # [max_len,seq_len]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if check: pe = pe[:,:-1]

        # [1,max_len,f_size] -> [max_len,1,f_size]
        pe = pe.unsqueeze(0).transpose(0,1)

        # means this part will not be updated when training
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, f_size]
        x = x.transpose(0,1)
        pos = self.pe[:x.size(0),:]
        
        x = x + pos
        return x.transpose(0,1)

# pad mask function
# for this structure, pad is not neccessary because all the sequences have a same length

# seq mask function
# in this model, tgt_len = seq_len
def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) 
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

# hierachical mask function for Trading Gap Filter
# fre is different when using different dataset 
def get_hierarchical_mask(seq,fre):
    len = seq.size(1)
    attn_shape = [seq.size(0),len,len]
    target= np.fromfunction(lambda i,j:abs(i//fre-j//fre),attn_shape[1:])
    target[target != 0]=-np.Inf
    target = torch.from_numpy(target).unsqueeze(0).repeat(seq.size(0),1,1)
    return target
    
# Gaussian Prior Bias
# sigma_h is a hyperparameter to enchance data locality
# sigma_h is unique in each head
def GenerateGaussianPrior(batch_size,n_heads,len_q,len_k,sigma_hs) -> torch.Tensor:
    GaussianMask = torch.zeros(n_heads,len_q,len_k)   
    for i in range (n_heads):
        target = torch.zeros(len_q,len_k)
        dim0,dim1 = target.shape 
        sigma_h = sigma_hs[i]
        for k in range (dim0):
            for j in range (dim1):
                if k >= j: target[i][j] = math.exp(-math.pow(j-k,2)/(2*pow(sigma_h,2))) 
        GaussianMask[i] = target
    return GaussianMask.unsqueeze(0).repeat(batch_size,1,1,1)

# Basic ScaledDotProduct for traditional transformer
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask):
        d_k = K.size(3)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        # scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores) 
        context = torch.matmul(attn, V) 
        return context

# ScaledDotProduct With Gaussian Prior
class GPSDPAttention(nn.Module):
    def __init__(self,device):
        super(GPSDPAttention, self).__init__()
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        d_k = K.size(3)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores: [batch_size,n_heads,len_q,len_k]
        batch_size,n_heads,len_q,len_k = scores.shape

        # add gaussian prior
        gm = GenerateGaussianPrior(batch_size,n_heads,len_q,len_k,sigma_hs)
        gm = gm.to(self.device)
        scores+=gm
        # scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores) 
        context = torch.matmul(attn, V)  
        return context

# basic multihead_attention with attn func as parameter
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_k,
        d_q,
        d_v,
        n_heads,
        f_size,
        device
        ):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(f_size, d_k * n_heads, bias=False, device=device)
        self.W_K = nn.Linear(f_size, d_k * n_heads, bias=False, device=device)
        self.W_V = nn.Linear(f_size, d_v * n_heads, bias=False, device=device)
        self.fc = nn.Linear(n_heads * d_v, f_size, bias=False, device=device)
        self.attn = GPSDPAttention(device=device) # with gaussian prior
        self.f_size = f_size
        self.n_heads = n_heads
        self.d_v = d_v
        self.d_k = d_k
        self.device = device

    # we assume d_q=d_k
    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, f_size]
        input_K: [batch_size, len_k, f_size]
        input_V: [batch_size, len_v(=len_k), f_size]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        
        residual, batch_size = input_Q.to(self.device), input_Q.size(0)
        input_Q,input_K,input_V = input_Q.to(self.device),input_K.to(self.device),input_V.to(self.device)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) 
        context = self.attn(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        
        output = self.fc(context)  # [batch_size, len_q, f_size]

        return nn.LayerNorm(self.f_size).to(self.device)(output + residual)