import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
import pkbar

from soil_moisture.training_funcs import load_checkpoint, save_checkpoint

def get_batch(datagen, batch_size, nan_val=-99):
    xbatch = []
    ybatch = []
    site_probs = datagen.static_data['SOIL_ORGANIC_CARBON']/datagen.static_data['SOIL_ORGANIC_CARBON'].sum()
    for bs in range(batch_size):        
        SID = np.random.choice(datagen.static_data.index, 1, p=site_probs)[0]
        all_dt = datagen.grab_chunk(SID, 0)
        dt_idx = np.random.choice(range(all_dt.shape[0] - datagen.maxsize), 1)
        dt_window = all_dt[dt_idx:(dt_idx + datagen.maxsize)]
        sub_dyn_input, SOC, _ = datagen.process_inputs(dt_window, SID)
        sub_dyn_input[torch.isnan(sub_dyn_input)] = nan_val
        xbatch.append(sub_dyn_input)
        ybatch.append(SOC)
    xbatch = torch.stack(xbatch)
    ybatch = torch.stack(ybatch)
    return xbatch, ybatch

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

def fit(model, optimizer, loss_fn, train_dg, val_dg, batchsize,
        train_len=50, val_len=15, max_epochs=5, nan_val=-99,
        outdir='./logs/', checkpoint=None, device=None):        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_loss = 0
    is_best = False
    train_step = make_train_step(model, optimizer, loss_fn)
    val_step = make_val_step(model, loss_fn)    
    if checkpoint is None:
        curr_epoch = 0
        best_loss = np.inf
        losses = []
        val_losses = []
    else:
        curr_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        torch.random.set_rng_state(checkpoint['torch_random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        try:
            losses = checkpoint['losses']
            val_losses = checkpoint['val_losses']
        except:
            losses = []
            val_losses = []    
    
    for epoch in range(curr_epoch, max_epochs):
        kbar = pkbar.Kbar(target=train_len, epoch=epoch, num_epochs=max_epochs,
                          width=12, always_stateful=False)
        model.train()
        for i in range(train_len):
            xb, yb = get_batch(train_dg, batchsize, nan_val=nan_val)
            xb = xb.to(device)
            yb = yb.to(device)
            loss = train_step(xb, yb)
            epoch_loss += loss
            kbar.update(i, values=[("loss", loss)])
        losses.append(epoch_loss / max(float(train_len),1.))
        epoch_loss = 0
        with torch.no_grad():
            for i in range(val_len):
                xb, yb = get_batch(val_dg, batchsize, nan_val=nan_val)
                xb = xb.to(device)
                yb = yb.to(device)
                model.eval()
                loss = val_step(xb, yb)                            
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
