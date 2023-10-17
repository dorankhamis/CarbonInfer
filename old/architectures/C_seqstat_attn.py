import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import pkbar
from torch import Tensor
from torch.autograd import Variable

'''
from .components.mlp import MLP, DenseBlock, PositionwiseFeedForward
from .components.causal_conv1d import CausalBlock, CausalConv1d
from .components.attention import MultiHeadedAttention
from .components.layer_norm import LayerNorm, SublayerConnection
from .components.positional_encoding import PositionalEncoding
from .components.nn_utils import subsequent_mask, clones
'''

from soil_moisture.architectures.components.mlp import MLP, DenseBlock, PositionwiseFeedForward
from soil_moisture.architectures.components.causal_conv1d import CausalBlock, CausalConv1d
from soil_moisture.architectures.components.attention import MultiHeadedAttention
from soil_moisture.architectures.components.layer_norm import LayerNorm, SublayerConnection
from soil_moisture.architectures.components.positional_encoding import PositionalEncoding
from soil_moisture.architectures.components.nn_utils import subsequent_mask, clones

class AttentionBlock(nn.Module):    
    def __init__(self, layer, N):
        super(AttentionBlock, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, xs, selfattn_mask=None, crossattn_mask=None):        
        for layer in self.layers:
            xs = layer(x, xs, selfattn_mask, crossattn_mask)
        return self.norm(xs)

class EncoderLayer(nn.Module):    
    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size
 
    def forward(self, xd, xs, selfattn_mask=None, crossattn_mask=None):
        xd = self.sublayer[0](xd, lambda xd: self.self_attn(xd, xd, xd, selfattn_mask))
        xs = self.sublayer[1](xs, lambda xs: self.cross_attn(xs, xd, xd, crossattn_mask))
        return self.sublayer[2](xs, self.feed_forward)

class SeqStatAttn(nn.Module):
    def __init__(self, features_d, embed_ds, features_s, dropout,
                 Natt_h, Natt_l, d_ff):
        super(SeqStatAttn, self).__init__()
        
        # conv blocks and embedding
        self.initial_embed = nn.Conv1d(features_d, embed_ds[0], kernel_size=1)        
        self.context_embed = CausalBlock(embed_ds[0], embed_ds[1], 5, 1, dropout=dropout)
        self.pe = PositionalEncoding(embed_ds[1], dropout)
        
        # static embedding
        self.s_ff = PositionwiseFeedForward(features_s, d_ff, dropout)
        self.ls = nn.Linear(features_s, embed_ds[1])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # attention        
        c = copy.deepcopy
        attn = MultiHeadedAttention(Natt_h, embed_ds[1])
        ff = PositionwiseFeedForward(embed_ds[1], d_ff, dropout)
        self.attblck = AttentionBlock(EncoderLayer(embed_ds[1], c(attn), c(attn), c(ff), dropout), Natt_l)
        
        # prediction
        self.predict = nn.Linear(embed_ds[1], 1)
        
    def forward(self, x, selfattn_mask=None, crossattn_mask=None):
        # x: [dynamic_input, static_input]
        x1 = self.initial_embed(x[0])
        x1 = self.context_embed(x1)
        
        # add positional encoding
        x1 = self.pe(x1.transpose(-2,-1))
        
        # embed static info
        s1 = self.s_ff(x[1])
        s1 = self.ls(s1)
        s1 = self.gelu(s1)
        s1 = self.dropout(s1)        
        
        # attention block: self-attn across time series, cross-attn to static info
        xs = self.attblck(x1, s1.unsqueeze(1), selfattn_mask, crossattn_mask)
        
        # predict
        return self.predict(xs).squeeze(-1)


'''
bb = get_seqstat_batch(train_dg, 6, nan_val=-99, noise_soc=0.05, device=None)
selfattn_mask = bb.selfattn_mask
crossattn_mask = bb.crossattn_mask
x = bb.src
features_d = x[0].shape[1]
features_s = x[1].shape[-1]

embed_ds = [64, 128]
dropout = 0.1
Natt_h = 4
Natt_l = 2
d_ff = 256
                 
# conv blocks and embedding
initial_embed = nn.Conv1d(features_d, embed_ds[0], kernel_size=1)
context_embed = CausalBlock(embed_ds[0], embed_ds[1], 5, 1, dropout=dropout)
pe = PositionalEncoding(embed_ds[1], dropout)

# static embedding
s_ff = PositionwiseFeedForward(features_s, d_ff, dropout)
ls = nn.Linear(features_s, embed_ds[1])
gelu = nn.GELU()

# attention        
c = copy.deepcopy
attn = MultiHeadedAttention(Natt_h, embed_ds[1])
ff = PositionwiseFeedForward(embed_ds[1], d_ff, dropout)
attblck = AttentionBlock(EncoderLayer(embed_ds[1], c(attn), c(attn), c(ff), dropout), Natt_l)

# prediction
predict = nn.Linear(embed_ds[1], 1)

## run
x1 = initial_embed(x[0])
x1 = context_embed(x1)

# add positional encoding
x1 = pe(x1.transpose(-2,-1))

# embed static info
s1 = s_ff(x[1])
s1 = ls(s1)
s1 = gelu(s1)

# attention block: self-attn across time series, cross-attn to static info
xs = attblck(x1, s1.unsqueeze(1), selfattn_mask, crossattn_mask)

ypred = predict(xs)


attn(s1,x1,x1,selfattn_mask)
attn(s1,x1,x1,crossattn_mask)
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
    
# We assume d_v always equals d_k
d_k = d_model // h
h = h
linears = clones(nn.Linear(d_model, d_model), 4)
attn = None        


if mask is not None:
    # Same mask applied to all h heads.
    mask = mask.unsqueeze(1)
nbatches = query.size(0)

# 1) Do all the linear projections in batch from d_model => h x d_k 
query, key, value = \
    [l(x).view(nbatches, -1, h, d_k).transpose(1, 2)
     for l, x in zip(linears, (query, key, value))]

# 2) Apply attention on all the projected vectors in batch. 
x, attn = attention(query, key, value, mask=mask, 
                         dropout=self.dropout)

# 3) "Concat" using a view and apply a final linear. 
x = x.transpose(1, 2).contiguous() \
     .view(nbatches, -1, h * d_k)
return linears[-1](x)

'''
