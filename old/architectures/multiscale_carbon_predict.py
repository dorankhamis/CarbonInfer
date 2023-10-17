import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch import Tensor
from torch.autograd import Variable

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
from soil_moisture.architectures.components.nn_utils import subsequent_mask, clones, alltrue_mask


class AttnEncoder(nn.Module):    
    def __init__(self, layer, N):
        super(AttnEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class AttnLayer(nn.Module):    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(AttnLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
      

class ChunkEmbedder(nn.Module):
    def __init__(self, d_in, d_model, d_out, N_attn, num_chunks, self_attn, feed_forward, dropout):
        super(ChunkEmbedder, self).__init__()
        # feature map creation
        self.c_local = clones(nn.Conv1d(d_in, d_model//2, 1), 2)
        self.c_causal = CausalConv1d(d_in, d_model//2, 5) 
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()        
        self.pe = PositionalEncoding(d_model//2, dropout)
        self.attn = AttnEncoder(AttnLayer(d_model//2, self_attn, feed_forward, dropout), N_attn)        
        # convolutional encoding, assuming seq_len = 84 = 7 * 12 (7days, 12 twohourly points)
        self.resblock1 = ResConvBlock(3 * d_model//2, d_model//4, 5, padding='valid')
        self.resblock2 = ResConvBlock(d_model//4, d_model//2, 5, padding='valid')
        self.resblock3 = ResConvBlock(d_model//2, d_model, 5, padding='valid')
        self.cc4 = nn.Conv1d(d_model, d_model, 7, stride=1, padding='valid')
        self.pool = nn.MaxPool1d(num_chunks) # across chunks
        self.decode = nn.Linear(d_model, d_out)        
                        
    def forward(self, x):
        ## x is a list of length batchsize with elements of (N, C, L)
        ## == (num_chunks, channels, seq_len)
        # could add a handful of timepoints at beginning for the causal
        # convs at the start of the sequence and then chop those off?
        x1 = torch.stack([self.c_local[0](xi) for xi in x], dim=0)
        x2 = torch.stack([self.c_causal(xi) for xi in x], dim=0)
        x3 = torch.stack([
            self.attn(
                self.pe(
                    self.c_local[1](xi).transpose(-2,-1)
                ), mask=None) for xi in x], dim=0)
        x_embed = torch.cat([x1,x2,x3.transpose(-2,-1)], dim=-2)
        x_embed = self.dropout(x_embed)
        x_embed = self.gelu(x_embed)
        x_embed = torch.stack(
            [self.cc4(
                self.resblock3(
                    self.resblock2(
                        self.resblock1(x_embed[i,:,:,:]))))
             for i in range(x_embed.shape[0])], dim=0)
        x_embed = self.pool(x_embed.squeeze(-1).transpose(-2,-1))        
        return self.decode(x_embed.squeeze(-1))


class ResConvBlock(nn.Module):
    def __init__(self, d_in, filters, kernel, dropout=0.1, padding='valid', padding_mode='zeros'):
        super(ResConvBlock, self).__init__()
        self.c1 = nn.Conv1d(d_in, filters, kernel, stride=2, padding=padding, padding_mode=padding_mode)
        self.c2 = nn.Conv1d(filters, filters, kernel, stride=1, padding='same', padding_mode=padding_mode)
        self.cres = nn.Conv1d(d_in, filters, kernel, stride=2, padding=padding, padding_mode=padding_mode)
        self.gelu = nn.GELU()
        self.gn1 = nn.GroupNorm(max(1, d_in//8), d_in)
        self.gn2 = nn.GroupNorm(max(1, filters//8), filters)
        self.gn3 = nn.GroupNorm(max(1, filters//8), filters)
        self.do  = nn.Dropout(dropout)
    
    def forward(self, x):
        x1 = self.gn1(x)
        x1 = self.gelu(x1)
        x1 = self.do(x1)
        x1 = self.c1(x1)        
        x1 = self.gn2(x1)
        x1 = self.gelu(x1)
        x1 = self.do(x1)
        x1 = self.c2(x1)
        res = self.cres(x)
        res = self.gn3(res)
        res = self.do(res)
        return torch.add(x1, res)
        

class YearCycleEmbedder(nn.Module):
    def __init__(self, d_in=1, chans=[32,64,128,256,96], d_out=128, dropout=0.1):
        super(YearCycleEmbedder, self).__init__()
        self.c_init = nn.Conv1d(d_in, chans[0], 5, stride=1, padding=2, padding_mode='circular')
        self.resblock1 = ResConvBlock(chans[0], chans[0], 3, dropout=dropout, padding=1, padding_mode='circular')
        self.resblock2 = ResConvBlock(chans[0], chans[1], 3, dropout=dropout, padding=1, padding_mode='circular')
        self.resblock3 = ResConvBlock(chans[1], chans[2], 3, dropout=dropout, padding=1, padding_mode='circular')
        self.c_fin = nn.Conv1d(chans[2], chans[3], 7)
        self.lin_c = nn.Linear(chans[3], chans[4])
        self.pe = PositionalEncoding(chans[0], 0.1, max_len=100)
        self.attn = MultiHeadedAttention(4, chans[0])
        self.lin_out = nn.Linear(chans[0]+chans[4], d_out)
    
    def forward(self, x):
        xe = self.c_init(x)
        # circular convolutional embedding branch
        xc = self.resblock1(xe)
        xc = self.resblock2(xc)
        xc = self.resblock3(xc)
        xc = self.c_fin(xc)
        xc = self.lin_c(xc.transpose(-2,-1))
        
        # attentional embedding branch
        xe = self.pe(xe.transpose(-2,-1))
        # add dummy entry
        xe = torch.cat([xe, Variable(torch.ones((xe.size()[0],1,xe.size()[2]))).to(xe.device)], dim=1)
        # mask dummy entry from attention
        mask = Variable(torch.zeros((xe.size()[0],xe.size()[1]))).to(xe.device)
        mask[:,-1] = 1
        mask = (mask != 1).unsqueeze(-2)
        mask = mask & Variable(alltrue_mask(xe.size()[1]).type_as(mask.data)).to(xe.device)
        xattn = self.attn(xe, xe, xe, mask=mask)
        # attention mask is then self.attn.attn, with size (batch,heads,seqlen,channels)
        xattn_out = xattn.transpose(-2,-1)[:,:,-1] # take dummy response as "global" view
        xj = torch.cat([xc.squeeze(), xattn_out], dim=-1)
        return self.lin_out(xj)
        

class YearConvEmbedder(nn.Module):
    def __init__(self, d_in=1, chans=[16,32,64,128,256,512], d_out=128, dropout=0.1):
        super(YearConvEmbedder, self).__init__()
        self.c_init = nn.Conv1d(d_in, chans[0], 5, stride=1, padding=2, padding_mode='circular')
        resconv_layers = []
        num_levels = len(chans)
        for i in range(num_levels-1):
            resconv_layers += [ResConvBlock(chans[i], chans[i+1], 3, dropout=dropout, padding=1, padding_mode='circular')]
        self.conv_blocks = nn.Sequential(*resconv_layers)
        self.flat = nn.Flatten()        
        self.lin_c = nn.Linear(chans[-1]*2, d_out*2)        
        self.lin_out = nn.Linear(d_out*2, d_out)
    
    def forward(self, x):
        xe = self.c_init(x)
        # circular convolutional embedding branch
        xc = self.conv_blocks(xe)
        xc = self.flat(xc)        
        xc = self.lin_c(xc)
        return self.lin_out(xc)
        

class MultiScaleCPredict(nn.Module):
    def __init__(self, d_chunks, d_year, d_model, d_out, num_chunks, h, N, dropout):
        super(MultiScaleCPredict, self).__init__()        
        self.yce = YearCycleEmbedder(d_year, [32,64,128,d_model,96], d_out, dropout=dropout)
        chunk_attn = MultiHeadedAttention(h, d_model//2)
        chunk_ff = PositionwiseFeedForward(d_model//2, d_model, dropout)
        self.chu = ChunkEmbedder(d_chunks, d_model, d_out, N, num_chunks,
                                 chunk_attn, chunk_ff, dropout)
        self.pred_mu = nn.Linear(2*d_out, 1)
        self.pred_sigma = nn.Linear(2*d_out, 1)
        self.bound_mu = nn.Sigmoid()
        self.init_p()

    def forward(self, x):
        year_embed = self.yce(x[0])
        chunk_embed = self.chu(x[1])
        joined = torch.concat([year_embed, chunk_embed], dim=-1)
        mu = self.pred_mu(joined)
        logsigma = self.pred_sigma(joined)
        #return torch.cat((mu, logsigma), dim=-1)
        return torch.cat((self.bound_mu(mu), logsigma), dim=-1)

    def init_p(self):
        # Initialize parameters with Glorot / fan_avg.    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class LargeScaleCPredict(nn.Module):
    def __init__(self, d_year, d_model, d_out, dropout):
        super(LargeScaleCPredict, self).__init__()        
        self.yce = YearCycleEmbedder(d_year, [32,64,128,d_model,96], d_out, dropout=dropout)
        self.pred_mu = nn.Linear(d_out, 1)
        self.pred_sigma = nn.Linear(d_out, 1)
        #self.bound_mu = nn.Sigmoid()
        self.init_p()

    def forward(self, x):
        year_embed = self.yce(x[0])        
        mu = self.pred_mu(year_embed)
        logsigma = self.pred_sigma(year_embed)
        return torch.cat((mu, logsigma), dim=-1)
        #return torch.cat((self.bound_mu(mu), logsigma), dim=-1)

    def init_p(self):
        # Initialize parameters with Glorot / fan_avg.    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
class LargeScaleCPredictConv(nn.Module):
    def __init__(self, d_year, d_model, d_out, dropout):
        super(LargeScaleCPredictConv, self).__init__()        
        self.yce = YearConvEmbedder(d_year, [16,32,64,128,256,d_model], d_out, dropout=dropout)
        self.pred_mu = nn.Linear(d_out, 1)
        self.pred_sigma = nn.Linear(d_out, 1)        
        self.init_p()

    def forward(self, x):
        year_embed = self.yce(x[0])        
        mu = self.pred_mu(year_embed)
        logsigma = self.pred_sigma(year_embed)
        return torch.cat((mu, logsigma), dim=-1)        

    def init_p(self):
        # Initialize parameters with Glorot / fan_avg.    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class SmallScaleCPredict(nn.Module):
    def __init__(self, d_chunks, d_model, d_out, num_chunks, h, N, dropout):
        super(SmallScaleCPredict, self).__init__()        
        chunk_attn = MultiHeadedAttention(h, d_model//2)
        chunk_ff = PositionwiseFeedForward(d_model//2, d_model, dropout)
        self.chu = ChunkEmbedder(d_chunks, d_model, d_out, N, num_chunks,
                                 chunk_attn, chunk_ff, dropout)
        self.pred_mu = nn.Linear(d_out, 1)
        self.pred_sigma = nn.Linear(d_out, 1)
        #self.bound_mu = nn.Sigmoid()
        self.init_p()

    def forward(self, x):        
        chunk_embed = self.chu(x[1])        
        mu = self.pred_mu(chunk_embed)
        logsigma = self.pred_sigma(chunk_embed)
        return torch.cat((mu, logsigma), dim=-1)
        #return torch.cat((self.bound_mu(mu), logsigma), dim=-1)

    def init_p(self):
        # Initialize parameters with Glorot / fan_avg.    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
