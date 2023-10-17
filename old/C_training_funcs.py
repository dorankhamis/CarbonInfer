import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable
import pkbar

from soil_moisture.training_funcs import (save_checkpoint, make_train_step,
                                          make_val_step, prepare_run)
from soil_moisture.architectures.components.nn_utils import subsequent_mask

def grab_time_chunks(datagen, SID, num_chunks, window_size, noise_soc=0.05):
    chunk_list = []
    datagen.maxsize = window_size
    datagen.minsize = window_size
    datagen.static_features = ['SOIL_ORGANIC_CARBON']
    for i in range(num_chunks):
        dt_window, _ = datagen.grab_necessary_contiguous_chunk_from_site(SID)    
        chunk, soc, _ = datagen.process_inputs(dt_window, SID)
        chunk_list.append(chunk)
    return torch.stack(chunk_list), soc*(1+np.random.normal(0,noise_soc))  
    
def create_chunk_batches(datagen, batchsize, num_chunks, chunk_size, noise_soc=0.1, nan_val=-99):
    xbatch = []
    xsbatch = []
    ybatch = []
    site_probs = datagen.static_data['SOIL_ORGANIC_CARBON']/datagen.static_data['SOIL_ORGANIC_CARBON'].sum()
    for b in range(batchsize):
        gotgood = False
        while gotgood is False:            
            SID = np.random.choice(datagen.static_data.index, 1, p=site_probs)[0]
            # dynamic inputs
            chunks, soc = grab_time_chunks(datagen, SID, num_chunks, chunk_size, noise_soc)
            if chunks.shape[-1]==0: gotgood = False
            else: gotgood = True
            # static inputs
            stat = torch.from_numpy(
                np.asarray(
                    datagen.static_data.drop('SOIL_ORGANIC_CARBON', axis=1).loc[SID],
                    dtype=np.float32
                )
            )
            stat[torch.isnan(stat)] = nan_val
        xbatch.append(chunks)
        xsbatch.append(stat)
        ybatch.append(soc)
    return xbatch, torch.stack(xsbatch), torch.stack(ybatch)
    
def create_allsite_chunks(datagen, num_chunks, chunk_size, noise_soc=0.0, nan_val=-99):
    xbatch = []
    xsbatch = []
    ybatch = []
    sites_out = []
    for SID in datagen.sites:        
        chunks, soc = grab_time_chunks(datagen, SID, num_chunks, chunk_size, noise_soc)
        if chunks.shape[-1]==0: continue
        stat = torch.from_numpy(
            np.asarray(
                datagen.static_data.drop('SOIL_ORGANIC_CARBON', axis=1).loc[SID],
                dtype=np.float32
            )
        )
        stat[torch.isnan(stat)] = nan_val
        xbatch.append(chunks)        
        xsbatch.append(stat)
        ybatch.append(soc)
        sites_out.append(SID)
    return xbatch, torch.stack(xsbatch), torch.stack(ybatch), sites_out

def weight_vector(v, w):
    return np.sum(w[w>0] * v[w>0]) / np.sum(w)

def generate_vis_samples(ypred, c_grid):
    try:
        probs = torch.exp(ypred).detach().cpu().numpy()
    except:
        probs = np.exp(ypred)
    N_samp = 1000
    vis_out = pd.DataFrame()
    for bb in range(ypred.shape[0]):
        samp_list = []
        for qq in range(probs.shape[-1]-1):
            bin_edges = (c_grid[qq], c_grid[qq+1])            
            samp_list = (samp_list + 
                list(np.random.uniform(
                    bin_edges[0], bin_edges[1],
                    int(np.round(N_samp*probs[bb,qq])))
                )
            )
        vis_out = pd.concat([vis_out, pd.DataFrame({'c_samples':samp_list, 'site_num':bb})], axis=0)
    return vis_out
    
def calculate_pred_means(ypred, c_grid):
    try:
        probs = torch.exp(ypred).detach().cpu().numpy()
    except:
        probs = np.exp(ypred)
    yp_means = []
    site_num = []    
    for b in range(ypred.shape[0]):
        yp_means.append(weight_vector(c_grid, probs[b,:]))
        site_num.append(b)
    means_out = pd.DataFrame({'ypred_mean':yp_means, 'site_num':site_num})
    return means_out

def linear_map(x, a, b, c, d):
    """ map x defined in [a,b] to [c,d] """
    return (d-c) * (x-a) / (b-a) + c
        
def pseudo_onehot_encoding(x, target_size):
    """ gap-filled pseudo one-hot encoding """
    # x is [0,0.35) SOC
    # map to [0,target_size-1] index space    
    int_long = torch.LongTensor()
    hi = 0.35
    lo = 0.
    inds = linear_map(torch.clamp(x, min=lo, max=hi), lo, hi, 0, (target_size-1))
    disc_bins = torch.zeros(x.shape[0], x.shape[1], target_size)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            flr = torch.floor(inds[i,j])
            cel = torch.ceil(inds[i,j])
            if flr==cel:
                disc_bins[i, j, inds[i,j].type_as(int_long)] = 1.
            else:
                upper = 1 - (cel - inds[i,j])
                disc_bins[i, j, cel.type_as(int_long)] = upper
                disc_bins[i, j, flr.type_as(int_long)] = 1 - upper
    return disc_bins    

def get_batch(datagen, batch_size, model_type, nan_val=-99):    
    site_probs = datagen.static_data['SOIL_ORGANIC_CARBON']/datagen.static_data['SOIL_ORGANIC_CARBON'].sum()
    if model_type=='dynamic' or model_type=='both':
        xbatch = []
        xsbatch = []
        ybatch = []
        for bs in range(batch_size):        
            gotgood = False
            while not gotgood:
                SID = np.random.choice(datagen.static_data.index, 1, p=site_probs)[0]            
                all_dt = datagen.grab_chunk(SID, 0) # make sure we have only one chunk!
                try:
                    dt_idx = int(np.random.choice(range(all_dt.shape[0] - datagen.maxsize), 1))
                    dt_window = all_dt[dt_idx:(dt_idx + datagen.maxsize)]
                    gotgood = True
                except ValueError:
                    continue            
            # dynamic inputs
            sub_dyn_input, SOC, _ = datagen.process_inputs(dt_window, SID)
            sub_dyn_input[torch.isnan(sub_dyn_input)] = nan_val
            xbatch.append(sub_dyn_input)
            if model_type=='both':
                # static inputs
                stat = torch.from_numpy(
                    np.asarray(
                        datagen.static_data.drop('SOIL_ORGANIC_CARBON', axis=1).loc[SID],
                        dtype=np.float32
                    )
                )
                stat[torch.isnan(stat)] = nan_val
                xsbatch.append(stat)
            else: xsbatch.append(torch.tensor([]))
            # target
            ybatch.append(SOC)
        xbatch = torch.stack(xbatch)
        xsbatch = torch.stack(xsbatch)
        ybatch = torch.stack(ybatch)
    elif model_type=='static':
        batch_data = datagen.static_data.sample(batch_size, weights=site_probs)
        ybatch = torch.from_numpy(np.asarray(batch_data[['SOIL_ORGANIC_CARBON']], dtype=np.float32))
        xsbatch = torch.from_numpy(np.asarray(batch_data.drop('SOIL_ORGANIC_CARBON', axis=1), dtype=np.float32))
        xsbatch[torch.isnan(xsbatch)] = nan_val
        xbatch = torch.tensor([])   
    return xbatch, xsbatch, ybatch #pseudo_onehot_encoding(ybatch, target_size).squeeze()

def get_seqstat_batch(datagen, batch_size, nan_val=-99, noise_soc=0.05, device=None):
    site_probs = datagen.static_data['SOIL_ORGANIC_CARBON']/datagen.static_data['SOIL_ORGANIC_CARBON'].sum()    
    batch = []
    for bs in range(batch_size):        
        gotgood = False
        while not gotgood:
            SID = np.random.choice(datagen.static_data.index, 1, p=site_probs)[0]                                            
            dt_window, chunk_id = datagen.grab_necessary_contiguous_chunk_from_site(SID)
            if chunk_id != -1: gotgood = True
        # dynamic inputs
        dyn, SOC, _ = datagen.process_inputs(dt_window, SID)
        dyn[torch.isnan(dyn)] = nan_val        
        # static inputs
        stat = torch.from_numpy(
            np.asarray(
                datagen.static_data.drop('SOIL_ORGANIC_CARBON', axis=1).loc[SID],
                dtype=np.float32
            )
        )
        stat[torch.isnan(stat)] = nan_val        
        batch.append(([dyn, stat], SOC*(1+np.random.normal(0, noise_soc))))
    xbatch, ybatch = pad_batch(batch, pad=nan_val)
    return Batch(list(xbatch), ybatch, pad=nan_val, device=device)

def create_allsite_seqstat_batch(datagen, noise_soc=0.0, nan_val=-99, device=None):
    batch = []
    sites_out = []
    for SID in datagen.sites:        
        dt_window, chunk_id = datagen.grab_necessary_contiguous_chunk_from_site(SID)
        if chunk_id == -1: continue
        # dynamic inputs
        dyn, SOC, _ = datagen.process_inputs(dt_window, SID)
        dyn[torch.isnan(dyn)] = nan_val        
        # static inputs
        stat = torch.from_numpy(
            np.asarray(
                datagen.static_data.drop('SOIL_ORGANIC_CARBON', axis=1).loc[SID],
                dtype=np.float32
            )
        )
        stat[torch.isnan(stat)] = nan_val        
        sites_out.append(SID)
        batch.append(([dyn, stat], SOC))
    xbatch, ybatch = pad_batch(batch, pad=nan_val)
    return Batch(list(xbatch), ybatch, pad=nan_val, device=device), sites_out

def batch_onesite_seqstat_variants(datagen, SID, N_dist=100, nan_val=-99, device=None):
    batch = []
    for i in range(N_dist):
        dt_window, chunk_id = datagen.grab_necessary_contiguous_chunk_from_site(SID)
        if chunk_id == -1: continue
        # dynamic inputs
        dyn, SOC, _ = datagen.process_inputs(dt_window, SID)
        dyn[torch.isnan(dyn)] = nan_val        
        # static inputs
        stat = torch.from_numpy(
            np.asarray(
                datagen.static_data.drop('SOIL_ORGANIC_CARBON', axis=1).loc[SID],
                dtype=np.float32
            )
        )
        stat[torch.isnan(stat)] = nan_val        
        batch.append(([dyn, stat], SOC))
    if len(batch)==0: return Batch([torch.tensor([]), torch.tensor([])], torch.tensor([]), pad=nan_val, device=device)
    xbatch, ybatch = pad_batch(batch, pad=nan_val)
    return Batch(list(xbatch), ybatch, pad=nan_val, device=device)

def pad_batch(batch, pad=-99):    
    # "batch" is a tuple (data, label)
    # Sort the batch in the descending order    
    sorted_batch = sorted(batch, key=lambda x: x[0][0].shape[-1], reverse=True)
    sequences = [x[0][0].T for x in sorted_batch] # permute for padding
    statics = torch.stack([x[0][1] for x in sorted_batch])    
    # pad each sequence
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad)    
    sequences_padded = [x.T for x in sequences_padded] # unpermute
    # sort labels
    labels = [x[1] for x in sorted_batch]    
    return (torch.stack(sequences_padded), statics), torch.stack(list(labels))

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, labels, pad=-99, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        x_dyn = src[0] # (B, C, S) == batch, channels, length
        x_stat = src[1]        
        self.src = src      
        self.selfattn_mask = self.make_pad_mask(x_dyn[:,0,:], pad) # (B, S, S)
        self.crossattn_mask = self.make_crossattn_mask(x_dyn[:,0,:], pad) # (B, 1, S)
        self.labels = labels.to(device)   
        self.src = [item.to(device) for item in self.src]                        
        self.selfattn_mask = self.selfattn_mask.to(device)
        self.crossattn_mask = self.crossattn_mask.to(device)

    @staticmethod
    def make_pad_mask(var, pad):
        "Create a mask to hide padding"
        var_mask = (var != pad).unsqueeze(-2)
        tri_mask = Variable(subsequent_mask(var.size(-1)).type_as(var_mask.data))
        return var_mask & tri_mask.fill_(True)   
        
    @staticmethod
    def make_crossattn_mask(var, pad):
        "Create a mask to hide padding"
        return (var != pad).unsqueeze(-2)
          
        
def batch_one_site_variants(datagen, SID, N_dist=100, nan_val=-99):
    all_dt = datagen.grab_chunk(SID, 0)
    all_subinds = np.array(range(all_dt.shape[0] - datagen.maxsize), dtype=np.int32)
    dt_idxs = np.random.choice(all_subinds, size=min(N_dist,len(all_subinds)), replace=False)
    xbatch = []
    ybatch = []
    for dt_idx in dt_idxs:
        dt_window = all_dt[dt_idx:(dt_idx + datagen.maxsize)]
        sub_dyn_input, SOC, _ = datagen.process_inputs(dt_window, SID)
        sub_dyn_input[torch.isnan(sub_dyn_input)] = nan_val
        xbatch.append(sub_dyn_input)
        ybatch.append(SOC)
    if len(xbatch)==0: return torch.tensor([]), torch.tensor([])
    xbatch = torch.stack(xbatch)
    ybatch = torch.stack(ybatch)
    return xbatch, ybatch

class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')        
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, ypred, ytrue):
        assert ypred.size(-1) == self.size # num slots
        nbatch = ypred.shape[0]
        true_dist = ytrue.data.clone()
        true_dist = true_dist + (self.smoothing / (self.size - 2))        
        true_dist = true_dist / true_dist.sum(dim=-1, keepdim=True)
        self.true_dist = true_dist
        return self.criterion(ypred, Variable(true_dist, requires_grad=False))

def fit(model, optimizer, loss_fn, train_dg, val_dg, batchsize, target_size,
        train_len=50, val_len=15, max_epochs=5, nan_val=-99,
        num_chunks=26, chunk_size=14, soc_train_noise=0.05, outdir='./logs/', 
        checkpoint=None, device=None, model_type='both'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_loss = 0
    is_best = False
    train_step = make_train_step(model, optimizer, loss_fn)
    val_step = make_val_step(model, loss_fn)    
    losses, val_losses, curr_epoch, best_loss = prepare_run(checkpoint)    
    for epoch in range(curr_epoch, max_epochs):
        kbar = pkbar.Kbar(target=train_len, epoch=epoch, num_epochs=max_epochs,
                          width=12, always_stateful=False)
        model.train()
        for i in range(train_len):
            if model_type=='chunked' or model_type=='chunked+static':
                xb, xsb, yb = create_chunk_batches(train_dg, batchsize, num_chunks, 
                                                   chunk_size, noise_soc=soc_train_noise,
                                                   nan_val=nan_val)
                xb = [item.to(device) for item in xb]
                xsb = xsb.to(device)
            else:
                xb, xsb, yb = get_batch(train_dg, batchsize, model_type, nan_val=nan_val)
                xb = xb.to(device)
                xsb = xsb.to(device)
            yb = pseudo_onehot_encoding(yb, target_size).squeeze()                
            yb = yb.to(device)
            if model_type=='both' or model_type=='chunked+static':
                    loss = train_step([xb, xsb], yb)
            elif model_type=='static': loss = train_step(xsb, yb)
            elif model_type=='dynamic': loss = train_step(xb, yb)
            elif model_type=='chunked': loss = train_step(xb, yb)            
            epoch_loss += loss
            kbar.update(i, values=[("loss", loss)])
        losses.append(epoch_loss / max(float(train_len),1.))
        epoch_loss = 0
        with torch.no_grad():
            for i in range(val_len):
                if model_type=='chunked' or model_type=='chunked+static':
                    xb, xsb, yb = create_chunk_batches(train_dg, batchsize, num_chunks, 
                                                       chunk_size, noise_soc=soc_train_noise,
                                                       nan_val=nan_val)
                    xb = [item.to(device) for item in xb]
                    xsb = xsb.to(device)
                else:
                    xb, xsb, yb = get_batch(val_dg, batchsize, model_type, nan_val=nan_val)
                    xb = xb.to(device)
                    xsb = xsb.to(device)
                yb = pseudo_onehot_encoding(yb, target_size).squeeze()                
                yb = yb.to(device)
                model.eval()                
                if model_type=='both' or model_type=='chunked+static':
                        loss = val_step([xb, xsb], yb)
                elif model_type=='static': loss = val_step(xsb, yb)
                elif model_type=='dynamic': loss = val_step(xb, yb)
                elif model_type=='chunked': loss = val_step(xb, yb)
                epoch_loss += loss
            val_losses.append(epoch_loss / max(float(val_len), 1.))
            kbar.add(1, values=[("val_loss", val_losses[-1])])
            is_best = bool(val_losses[-1] < best_loss)
            best_loss = min(val_losses[-1], best_loss)
            checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'torch_random_state': torch.random.get_rng_state(),
                        'numpy_random_state': np.random.get_state(),
                        'losses': losses,
                        'val_losses': val_losses
                        }
            save_checkpoint(checkpoint, is_best, outdir)
        print("Done epoch %d" % epoch)
    return model, losses, val_losses

def make_masked_train_step(model, optimizer, loss_fn):
    def train_step(x, selfattn_mask, crossattn_mask, ytrue):
        model.train()
        ypred = model(x, selfattn_mask, crossattn_mask)
        loss = loss_fn(ypred, ytrue)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

def make_masked_val_step(model, loss_fn):
    def val_step(x, selfattn_mask, crossattn_mask, ytrue):
        model.eval()
        ypred = model(x, selfattn_mask, crossattn_mask)
        loss = loss_fn(ypred, ytrue)
        return loss.item()
    return val_step  

def fit_seqstat_attn(model, optimizer, loss_fn, train_dg, val_dg, batchsize,
                     train_len=50, val_len=15, max_epochs=5, nan_val=-99,
                     soc_train_noise=0.05, outdir='./logs/', 
                     checkpoint=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_loss = 0
    is_best = False
    train_step = make_masked_train_step(model, optimizer, loss_fn)
    val_step = make_masked_val_step(model, loss_fn)    
    losses, val_losses, curr_epoch, best_loss = prepare_run(checkpoint)    
    for epoch in range(curr_epoch, max_epochs):
        kbar = pkbar.Kbar(target=train_len, epoch=epoch, num_epochs=max_epochs,
                          width=12, always_stateful=False)
        model.train()
        for i in range(train_len):            
            batch = get_seqstat_batch(train_dg, batchsize, nan_val=nan_val,
                                            noise_soc=soc_train_noise)            
            loss = train_step(batch.src, batch.selfattn_mask, batch.crossattn_mask, batch.labels)
            epoch_loss += loss
            kbar.update(i, values=[("loss", loss)])
        losses.append(epoch_loss / max(float(train_len),1.))
        epoch_loss = 0
        with torch.no_grad():
            model.eval()
            for i in range(val_len):                    
                batch = get_seqstat_batch(val_dg, batchsize, nan_val=nan_val, noise_soc=0)                
                loss = val_step(batch.src, batch.selfattn_mask, batch.crossattn_mask, batch.labels)
                epoch_loss += loss
            val_losses.append(epoch_loss / max(float(val_len), 1.))
            kbar.add(1, values=[("val_loss", val_losses[-1])])
            is_best = bool(val_losses[-1] < best_loss)
            best_loss = min(val_losses[-1], best_loss)
            checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'torch_random_state': torch.random.get_rng_state(),
                        'numpy_random_state': np.random.get_state(),
                        'losses': losses,
                        'val_losses': val_losses
                        }
            save_checkpoint(checkpoint, is_best, outdir)
        print("Done epoch %d" % epoch)
    return model, losses, val_losses
