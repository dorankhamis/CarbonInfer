import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch import Tensor
'''
from .components.mlp import MLP, DenseBlock, PositionwiseFeedForward
from .components.causal_conv1d import CausalBlock, CausalConv1d
from .components.attention import MultiHeadedAttention
from .components.layer_norm import LayerNorm, SublayerConnection
from .components.positional_encoding import PositionalEncoding
from .components.encoder_decoder import EncoderDecoder
from .components.nn_utils import subsequent_mask, clones
'''
from soil_moisture.architectures.components.mlp import MLP, DenseBlock, PositionwiseFeedForward
from soil_moisture.architectures.components.causal_conv1d import CausalBlock, CausalConv1d
from soil_moisture.architectures.components.attention import MultiHeadedAttention
from soil_moisture.architectures.components.layer_norm import LayerNorm, SublayerConnection
from soil_moisture.architectures.components.positional_encoding import PositionalEncoding
from soil_moisture.architectures.components.encoder_decoder import EncoderDecoder
from soil_moisture.architectures.components.nn_utils import subsequent_mask, clones

class ModelGenerator(nn.Module):
    def __init__(self, model, model_size, target_size, dropout=0.1):
        super(ModelGenerator, self).__init__()
        self.model = model        
        self.investigator = PositionwiseFeedForward(model_size, 8*target_size, dropout)
        self.adjudicator = nn.Linear(model_size, target_size)
        
    def forward(self, x):        
        y = self.investigator(self.run_model(x))
        return F.log_softmax(self.adjudicator(y), dim=-1)
    
    def run_model(self, x):
        return self.model(x)
        

class ModelGeneratorRegressor(nn.Module):
    def __init__(self, model, model_size, dropout=0.1):
        super(ModelGenerator, self).__init__()
        self.model = model        
        self.investigator = PositionwiseFeedForward(model_size, 2*model_size, dropout)
        self.adjudicator = nn.Linear(model_size, 1)
        
    def forward(self, x):        
        y = self.investigator(self.run_model(x))
        return self.adjudicator(y)
    
    def run_model(self, x):
        return self.model(x)


class DynamicStaticJoiner(nn.Module):    
    def __init__(self, dynamic_model, static_model):
        super(DynamicStaticJoiner, self).__init__()
        self.dynamic_model = dynamic_model
        self.static_model = static_model        

    def forward(self, x):
        d_out = self.dynamic_model(x[0])
        s_out = self.static_model(x[1])
        return torch.cat([d_out, s_out], dim=-1)


class Encoder(nn.Module):    
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):        
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SeqDecoder(nn.Module):    
    def __init__(self, d_model, out_size, seq_len, dropout, pool_kernel=30):
        super(Decoder, self).__init__()
        # prediction for each element based on attention
        # across local + long range causal history
        self.dense1 = nn.Linear(d_model, out_size)
        # squeeze and then pool across the sequence        
        self.pool = nn.MaxPool1d(pool_kernel)
        self.dense2 = nn.Linear(seq_len//pool_kernel, 1)
        self.dropout = nn.Dropout(dropout)        
            
    def forward(self, x, src_mask=None, tgt=None, tgt_mask=None):
        x = self.dense1(x)
        x = self.dropout(x)        
        x = self.pool(x.transpose(-1,-2))
        x = self.dense2(x)
        return x.squeeze(-1)


class ChunkDecoder(nn.Module):    
    def __init__(self, d_model, num_chunks, out_size, dropout):
        super(ChunkDecoder, self).__init__()        
        #self.pool = nn.AvgPool1d(num_chunks) # across chunks
        self.pool = nn.MaxPool1d(num_chunks) # across chunks
        self.decode = nn.Linear((d_model), out_size)
            
    def forward(self, x, src_mask=None, tgt=None, tgt_mask=None):
        x = self.pool(x.transpose(-2,-1))        
        return self.decode(x.squeeze(-1))

        
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

   
class Embeddings(nn.Module):    
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
        self.dyn_embed = nn.Conv1d(embed_ds[-1], d_model, 1)        
        self.d_model = d_model

    def forward(self, x):
        # first encode the short-term dynamics using causal convolutions
        x1 = self.init_embed(x)
        for cconv in self.conv_blocks:
            x1 = cconv(x1)
        x1 = self.dyn_embed(x1)
        # transpose to (B,S,C) = (batch, seq_len, channels)
        return x1.transpose(-2,-1)


class DynamicDataDense(nn.Module):
    def __init__(self, d_model, features, d_ff, seq_len, out_size,
                 conv_dil_list, conv_kern_list, embed_ds, dropout):
        super(DynamicDataConv, self).__init__()
        self.causal_embed = Embeddings(d_model, features, dropout, conv_dil_list, 
                                       conv_kern_list, embed_ds)
        self.pointwise_ff = PositionwiseFeedForward(d_model, d_ff, dropout, 'relu')
        self.pointwise_pred = nn.Linear(d_model, 1)
        self.seq_pred = nn.Linear(seq_len, out_size)
        
    def forward(self, x):
        x = self.causal_embed(x)
        x = self.pointwise_ff(x)
        x = self.pointwise_pred(x).squeeze(-1)
        return self.seq_pred(x)
        

class ChunkEmbedder(nn.Module):
    def __init__(self, d_model, features, d_ff, dropout, chunk_kernel):
        super(ChunkEmbedder, self).__init__()        
        self.chunk_conv = nn.Conv1d(features, d_model, chunk_kernel)
        self.pointwise_ff = PositionwiseFeedForward(d_model, d_ff, dropout, 'relu')
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)        
        
        # static embedding
        self.s_ff = PositionwiseFeedForward(features_s, d_ff, dropout)
        self.ls = nn.Linear(features_s, embed_s)
        
    def forward(self, x):
        x = torch.stack([self.chunk_conv(xi).squeeze(-1) for xi in x], dim=0)
        x = self.pointwise_ff(x)
        return x


class ChunkStaticEmbedder(nn.Module):
    def __init__(self, d_model, d_in, s_in, d_ff, dropout, chunk_kernel):
        super(ChunkStaticEmbedder, self).__init__()        
        self.chunk_conv = nn.Conv1d(features, d_model, chunk_kernel)
        self.pointwise_ff = PositionwiseFeedForward(d_model, d_ff, dropout, 'relu')                
        
    def forward(self, x):
        # embed dynamic chunks
        x1 = torch.stack([self.chunk_conv(xi).squeeze(-1) for xi in x[0]], dim=0)
        x1 = self.pointwise_ff(x1)
        
        # embed static info
        s1 = self.s_ff(x[1])
        s1 = self.ls(s1)
        s1 = self.gelu(s1)
        s1 = self.dropout(s1)
        
        # append to time series on channel dim        
        s1 = (s1.unsqueeze(1)
                .expand(s1.shape[0], x1.shape[1], s1.shape[-1]))
        x1 = torch.cat((x1, s1), dim=-1) 
        
        return x

        
class StaticDataMLP(nn.Module):
    def __init__(self, s_feats, d_ff, chans_out, out_size, dropout=0.1):
        super(StaticDataMLP, self).__init__()
        self.ff = PositionwiseFeedForward(s_feats, d_ff, dropout)
        self.mlp = MLP(s_feats, chans_out, dropout=dropout)        
        self.dense = nn.Linear(chans_out[-1], out_size)
        
    def forward(self, x):
        x = self.ff(x)
        x = self.mlp(x)
        return self.dense(x)


def make_model(d_features, s_features, target_size, seq_len, 
               embed_ds, conv_dil_list, conv_kern_list,
               N=4, d_model=256, d_ff=1024, h=4, dropout=0.1,
               d_ff_s=512, Nls=4):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    dynamic_model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),        
        nn.Sequential(Embeddings(d_model, d_features, dropout,
                                 conv_dil_list, conv_kern_list, embed_ds),
                      c(position)),
        Decoder(d_model, target_size, seq_len, dropout)
    )
    static_model = StaticDataParser(s_features, target_size, d_ff_s, Nls, dropout)
    model = ModelGenerator(DynamicStaticJoiner(dynamic_model, static_model),
                           2*target_size, target_size, dropout)    
        
    # Initialize parameters with Glorot / fan_avg.    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
    
def make_static_model(s_features, target_size, d_ff=256, chans_list=[], dropout=0.1):        
    model = ModelGenerator(StaticDataMLP(s_features, d_ff, chans_list, target_size, dropout),
                           target_size, target_size, dropout)
    # Initialize parameters with Glorot / fan_avg.    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def make_chunked_model(d_in, d_out, target_size, num_chunks, chunk_size,
                       N=2, d_model=256, d_ff=512, h=4, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = ModelGenerator(
        EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            ChunkEmbedder(d_model, d_in, d_ff, dropout, chunk_size),
            ChunkDecoder(d_model, num_chunks, d_out, dropout)
        ),
        d_out, target_size, dropout
    )
    # Initialize parameters with Glorot / fan_avg.    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
    
def make_chunked_and_static_model(d_in, d_out, target_size, num_chunks, chunk_size,
                                  s_features, s_out, s_ff=256, chans_list=[],
                                  N=2, d_model=256, d_ff=512, h=4, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    dynamic_model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        ChunkEmbedder(d_model, d_in, d_ff, dropout, chunk_size),
        ChunkDecoder(d_model, num_chunks, d_out, dropout)
    )
    static_model = StaticDataMLP(s_features, s_ff, chans_list, s_out, dropout)
    model = ModelGenerator(DynamicStaticJoiner(dynamic_model, static_model),
                           d_out+s_out, target_size, dropout)    
    # Initialize parameters with Glorot / fan_avg.    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

if __name__=='__main__':
    '''
    x, y = get_batch(train_dg, 8)
    x1 = model.encode(x)
    # prediction for each element based on attention
    # across local + long range causal history
    dense1 = nn.Linear(d_model, target_size)
    # squeeze and then pool across the sequence    
    max_pool = nn.MaxPool1d(30)
    dense2 = nn.Linear(12, 1)
    self.dropout = nn.Dropout(dropout)
    
    # static model
    layer = PositionwiseFeedForward(xs.shape[1], 512, 0.1, 'gelu')    
    '''
    
    '''        
    d_in = len(train_dg.dynamic_features)
    chunk_kernel = train_dg.maxsize
    chans_out = 256
    d_ff = 512
    chunk_conv = nn.Conv1d(d_in, chans_out, chunk_kernel)
    pointwise_ff = PositionwiseFeedForward(chans_out, d_ff, dropout, 'relu')
    self.pointwise_pred = nn.Linear(d_model, 1)
    self.seq_pred = nn.Linear(seq_len, out_size)
        
    x = torch.stack([chunk_conv(xi).squeeze(-1) for xi in xbatch], dim=1)
    x = pointwise_ff(x)
    x = self.chunk_conv(x)
    x = self.pointwise_ff(x)
    x = self.pointwise_pred(x).squeeze(-1)
    '''
    
