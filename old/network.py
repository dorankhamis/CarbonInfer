import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import copy
from typing import Optional, Any, Union, Callable
from torch import Tensor

from soil_moisture.components.mlp import MLP, DenseBlock
from soil_moisture.components.causal_conv1d import CausalBlock, CausalConv1d

class MyTransformer(nn.Module):    
    def __init__(self, encoder, src_embed, decoder):
        """
            - src: (B, S, F) = (batch, seq_len, features)
            - src_mask: (S, S): causal masking            
            - src_key_padding_mask: (B, S): padding masking, also missing data masking??
        """
        super(MyTransformer, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.decoder = decoder
        
    def forward(self, src, src_mask=None):
        "Take in and process masked source sequence."
        return self.decode(self.encode(src, src_mask))
    
    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory):
        return self.decoder(memory)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])   
    
class Encoder(nn.Module):
    "Generic core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    "Generic core encoder is a stack of N layers"
    def __init__(self, seq_len, d_model, dropout):
        super(Decoder, self).__init__()
        self.global_pool = nn.AvgPool1d(seq_len)
        self.dense1 = nn.Linear(d_model, d_model//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(d_model//2, 1)
            
    def forward(self, x):
        x = self.global_pool(x.transpose(-2,-1)) # pool over features
        x = self.dense1(x.squeeze(2)) # dense over sequence
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
        
class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2        

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
        
        
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
class Embeddings(nn.Module):
    "Rewrite this for our input data: i.e. won't use nn.Embedding"
    def __init__(self, d_model, features, dropout,
                 conv_dil_list, conv_kern_list, embed_ds):
        super(Embeddings, self).__init__()        
        self.init_embed = nn.Conv1d(features, embed_ds[0], 1)
        self.conv_blocks = nn.ModuleList([])
        for i in range(len(conv_dil_list)):            
            self.conv_blocks.append(
                CausalBlock(embed_ds[i], embed_ds[i+1], conv_kern_list[i], 
                            dilation=conv_dil_list[i], dropout=dropout)
            )
        self.dyn_embed = nn.Conv1d(embed_ds[-2], embed_ds[-1], 1)
        self.final_embed = nn.Conv1d(features+embed_ds[-1], d_model, 1)
        self.d_model = d_model

    def forward(self, x):
        # first encode the short-term dynamics using causal convolutions
        x1 = self.init_embed(x)
        for cconv in self.conv_blocks:
            x1 = cconv(x1)
        x1 = self.dyn_embed(x1)
        # join to raw data and do final point-wise embedding
        x1 = torch.cat((x, x1), dim=1) # channel-wise
        x1 = self.final_embed(x1)
        # before returning, scale by sqrt(d) and transpose to (B,S,C) = (batch, seq_len, channels)
        return x1.transpose(-2,-1) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
        
def make_model(features, seq_len, embed_ds, conv_dil_list, conv_kern_list,
               N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = MyTransformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),        
        nn.Sequential(Embeddings(d_model, features, dropout,
                                 conv_dil_list, conv_kern_list, embed_ds),
                      c(position)),
        Decoder(seq_len, d_model, dropout)
    )      
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
