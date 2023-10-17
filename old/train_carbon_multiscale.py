import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pkbar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset

from soil_moisture.training_funcs import (make_train_step, make_val_step,
                                          save_checkpoint, prepare_run, load_checkpoint)
from soil_moisture.utils import indexify_datetime, zeropad_strint
from soil_moisture.data_classes.cosmos_data import CosmosMetaData
from soil_moisture.prepare_general import (import_params, grab_data, normalise_data,
                                           drop_add_contig_sincos, generators_and_loaders,
                                           assign_contiguous_groups) 
from soil_moisture.data_loaders import (read_one_cosmos_site, proj_dir,
                                        provide_weekly_dynamic_cosmos_data,
                                        provide_twohourly_dynamic_cosmos_data,
                                        provide_WorldClim_at_cosmos_sites)
from soil_moisture.architectures.multiscale_carbon_predict import (MultiScaleCPredict,
    LargeScaleCPredict, SmallScaleCPredict, LargeScaleCPredictConv)

sns.set_theme()
plt.rcParams['figure.figsize'] = [14,8]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelname = f'carbon_largescale'
#modelname = f'carbon_smallscale'
#modelname = f'carbon_multiscale'
model_outdir = f'{proj_dir}/logs/{modelname}/'
Path(model_outdir).mkdir(parents=True, exist_ok=True)
eval_outdir = f'{proj_dir}/eval/{modelname}/'
Path(eval_outdir).mkdir(parents=True, exist_ok=True)
TRAIN = False
EVAL = True
load_prev_chkpnt = True

metadata = CosmosMetaData()

class ChunkGenerator(Dataset):
    def __init__(self, weekly_data, twohourly_data, static_data,
                 smallscale_features, largescale_features, n_chunks, chunk_length):
        self.weekly_data = weekly_data
        self.twohourly_data = twohourly_data
        self.static_data = static_data
        self.sites = weekly_data.keys()
        self.possible_targets = ['TDT1_VWC','TDT2_VWC']
        self.smallscale_features = smallscale_features
        self.largescale_features = largescale_features
        self.n_chunks = n_chunks
        self.chunk_length = chunk_length
        self.soc = 'SOIL_ORGANIC_CARBON'
        self.rawsoc = 'raw_soc'
        self.targets = [self.soc, self.soc+'_SD']
        self.site_probs = (1+8*static_data[self.rawsoc]) / (1+8*static_data[self.rawsoc]).sum()
        self.gen_smallscale = True        
        
    def __len__(self):
        return len(self.sites)

    def __getitem__(self, idx):
        x, y, dt = self.return_elem_random_site()
        return x, y, dt
        
    def return_elem_random_site(self):
        # find weekly and subdaily dt windows
        gotgood = False
        while not gotgood:
            SID, TARG, dt_window = self.grab_random_contiguous_chunk(self.weekly_data, 52)            
            if self.gen_smallscale is True:
                chunk_dt_list = []
                for i in range(self.n_chunks):
                    chunk_dt, chunk_id = self.grab_constrained_contiguous_chunk(
                        self.twohourly_data[SID], TARG, dt_window, self.chunk_length)
                    if chunk_id == -1: continue # catch a window with no small scale continuity
                    chunk_dt_list.append(chunk_dt)
                if len(chunk_dt_list)==self.n_chunks:
                    gotgood = True
                else:
                    gotgood = False
                    return [], 0, 0
            else: gotgood = True
        # turn dt windows into data tensors
        year_ref = torch.from_numpy(np.asarray(
            np.sin(dt_window.day_of_year/365. * 2*np.pi)).astype(np.float32)).unsqueeze(0)
        year_scale = torch.from_numpy(np.asarray(
            self.weekly_data[SID].loc[dt_window][self.largescale_features+[TARG]], dtype=np.float32)).T
        year_scale = torch.cat([year_scale, year_ref], dim=0)
        if self.gen_smallscale is True:
            chunk_dat_list = []
            for i in range(self.n_chunks):
                chunk_dat_list.append(
                    torch.from_numpy(np.asarray(
                        self.twohourly_data[SID]
                            .loc[chunk_dt_list[i]][self.smallscale_features+[TARG]],
                    dtype=np.float32)).T
                )
            small_scale = torch.stack(chunk_dat_list)        
        else: small_scale = torch.tensor([])
        # get target: carbon
        carbon = torch.tensor([self.static_data.loc[SID][self.targets]], dtype=torch.float32)
        return [year_scale, small_scale], carbon.flatten(), dt_window

    def return_elem_for_site(self, SID):
        # find weekly and subdaily dt windows
        gotgood = False
        while not gotgood:        
            TARG, dt_window = self.grab_site_contiguous_chunk(self.weekly_data, 52, SID)
            if len(dt_window)==0: return [], 0, 0
            if self.gen_smallscale is True:
                chunk_dt_list = []
                for i in range(self.n_chunks):
                    chunk_dt, chunk_id = self.grab_constrained_contiguous_chunk(
                        self.twohourly_data[SID], TARG, dt_window, self.chunk_length)
                    if chunk_id == -1: continue # catch a window with no small scale continuity
                    chunk_dt_list.append(chunk_dt)
                if len(chunk_dt_list)==self.n_chunks:
                    gotgood = True
                else:
                    gotgood = False
                    return [], 0, 0
            else: gotgood = True
        # turn dt windows into data tensors
        year_ref = torch.from_numpy(np.asarray(
            np.sin(dt_window.day_of_year/365. * 2*np.pi)).astype(np.float32)).unsqueeze(0)
        year_scale = torch.from_numpy(np.asarray(
            self.weekly_data[SID].loc[dt_window][self.largescale_features+[TARG]], dtype=np.float32)).T
        year_scale = torch.cat([year_scale, year_ref], dim=0)        
        if self.gen_smallscale is True:
            chunk_dat_list = []
            for i in range(self.n_chunks):
                chunk_dat_list.append(
                    torch.from_numpy(np.asarray(
                        self.twohourly_data[SID]
                            .loc[chunk_dt_list[i]][self.smallscale_features+[TARG]],
                    dtype=np.float32)).T
                )
            small_scale = torch.stack(chunk_dat_list)
        else: small_scale = torch.tensor([])
        # get target: carbon
        carbon = torch.tensor([self.static_data.loc[SID][self.targets]], dtype=torch.float32)
        return [year_scale, small_scale], carbon.flatten(), dt_window
        
    def grab_random_contiguous_chunk(self, DATA, length):
        gotgood = False
        SID = 'FAIL'
        dt_window = pd.Index([])        
        while not gotgood:
            TARG = np.random.choice(self.possible_targets, 1)[0]
            SID = np.random.choice(self.static_data.index, 1, p=self.site_probs)[0]
            contig_ids = self.view_contiguous_chunks(DATA[SID], TARG)
            np.random.shuffle(contig_ids)
            for chunk_id in contig_ids:
                dt_window = self.grab_chunk(DATA[SID], TARG, chunk_id)
                seqlen = dt_window.shape[0]
                if seqlen<length:
                    continue
                if seqlen>length:
                    # grab random subset of size maxsize
                    dt_idx = np.random.randint(seqlen - length)
                    dt_window = dt_window[dt_idx:(dt_idx + length)]
                gotgood = True
                break            
        return SID, TARG, dt_window
    
    def grab_site_contiguous_chunk(self, DATA, length, SID):
        gotgood = False        
        dt_window = pd.Index([])
        shuff_targs = self.possible_targets.copy()
        np.random.shuffle(shuff_targs)
        for TARG in shuff_targs:            
            contig_ids = self.view_contiguous_chunks(DATA[SID], TARG)
            np.random.shuffle(contig_ids)
            for chunk_id in contig_ids:
                dt_window = self.grab_chunk(DATA[SID], TARG, chunk_id)
                seqlen = dt_window.shape[0]
                if seqlen<length:
                    continue
                if seqlen>length:
                    # grab random subset of size maxsize
                    dt_idx = np.random.randint(seqlen - length)
                    dt_window = dt_window[dt_idx:(dt_idx + length)]
                gotgood = True
                return TARG, dt_window
        return 'FAIL', []

    def view_contiguous_chunks(self, DATA_SID, TARG):
        contig_ids = (np.unique(DATA_SID[[TARG+'_contig']])
            [~np.isnan(np.unique(DATA_SID[[TARG+'_contig']]))])
        return contig_ids
        
    def grab_chunk(self, DATA_SID, TARG, chunk_id):
        dt_window = DATA_SID[DATA_SID[TARG+'_contig']==chunk_id].index
        return dt_window

    def grab_constrained_contiguous_chunk(self, DATA_SID, TARG, mask_dt_window, length):
        gotgood = False            
        dt_window = pd.Index([])            
        subset_mask = np.bitwise_and(
            DATA_SID.index > (mask_dt_window[0] - np.timedelta64(7, 'D')), 
            DATA_SID.index <= mask_dt_window[-1])
        c_df = DATA_SID[subset_mask]
        c_df = assign_contiguous_groups(c_df, self.smallscale_features+[TARG])
        c_df = c_df.rename(columns={'contig_chunk':TARG+'_contig'})
        # define threshold for rain event to start window
        # could actually change to checking for no rain before start, as
        # a "fresh event".
        thresh = c_df.PRECIP[c_df.PRECIP>0].quantile(0.15)
        contig_ids = self.view_contiguous_chunks(c_df, TARG)            
        np.random.shuffle(contig_ids)
        for chunk_id in contig_ids:
            dt_window = self.grab_chunk(c_df, TARG, chunk_id)
            seqlen = dt_window.shape[0]
            if seqlen<length:
                continue
            if seqlen>length:
                # grab random subset, conditioning on starting with a "heavy" rainfall event
                try:
                    start_dt = np.random.choice(c_df.loc[dt_window[:-length]][
                        c_df.loc[dt_window[:-length]].PRECIP>thresh].index)
                    dt_idx = np.where(dt_window == start_dt)[0][0]
                except:                
                    dt_idx = np.random.randint(seqlen - length)
                dt_window = dt_window[dt_idx:(dt_idx + length)]
            gotgood = True
            break    
        if gotgood is False:
            return [], -1
        else:
            return dt_window, chunk_id

def get_batch(dg, batchsize, device=None, soc_noise=0):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    xbatcha = []
    xbatchb = []
    ybatch = []
    for i in range(batchsize):
        x, y, _ = dg[0]
        y[0] = y[0] * (1+np.random.normal(0, soc_noise))
        xbatcha.append(x[0])
        xbatchb.append(x[1])
        ybatch.append(y)    
    xbatch = [torch.stack(xbatcha).to(device), [xb.to(device) for xb in xbatchb]]
    ybatch = torch.stack(ybatch).to(device)
    return xbatch, ybatch

def gaussian_KL(ypred, ytrue):
    EPS = 1e-12
    mu_2 = ypred[:,0]
    sigma_2 = torch.exp(ypred[:,1])
    mu_1 = ytrue[:,0]
    sigma_1 = ytrue[:,1]
    t1 = torch.log(sigma_2/sigma_1 + EPS)
    t2 = (sigma_1*sigma_1) / (sigma_2*sigma_2)
    t3 = (mu_1 - mu_2)*(mu_1 - mu_2) / (sigma_2*sigma_2) - 1
    gKL = t1 + 0.5 * (t2 + t3)
    return torch.mean(gKL, dim=(0,), keepdim=False)
    
def build_NLL(sample_truth=False):
    EPS = 1e-12   
    sqrt_two_pi = np.sqrt(np.pi * 2.)
    
    def NLL(ypred, ytrue):
        mu_2 = ypred[:,0]
        sigma_2 = torch.exp(ypred[:,1]) # found in log space
        mu_1 = ytrue[:,0]
        sigma_1 = ytrue[:,1]
        if sample_truth is True:
            x = torch.normal(mu_1, sigma_1)
            #negs = True
            #while negs is True:
            #    x = torch.normal(mu_1, sigma_1)
            #    negs = bool(torch.any(x<0))
        else:
            x = mu_1
        expterm = -0.5 * torch.div((x - mu_2)*(x - mu_2), sigma_2*sigma_2 + EPS)
        prefactor = -torch.log(sqrt_two_pi * sigma_2)
        return -torch.mean(expterm + prefactor)
                
    return NLL

def fit(model, optimizer, loss_fn, train_dg, val_dg, batchsize, 
        train_len=50, val_len=15, max_epochs=5, outdir='./logs/', 
        checkpoint=None, device=None, soc_noise=0):
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
            xb, yb = get_batch(train_dg, batchsize, device=device, soc_noise=soc_noise)
            loss = train_step(xb, yb)            
            epoch_loss += loss
            kbar.update(i, values=[("loss", loss)])
        losses.append(epoch_loss / max(float(train_len),1.))
        epoch_loss = 0
        with torch.no_grad():
            for i in range(val_len):                
                xb, yb = get_batch(val_dg, batchsize, device=device, soc_noise=0)
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


##############################################
## loading sub-daily data to see if short timescale 
## patterns in SM are informative
subhourly_features = ['TDT1_VWC', 'TDT2_VWC', 'PRECIP',
                      'STP_TSOIL10', 'SWIN', 'RH', 'TA', 'PA']
# 'UX', 'UY' wind speed is not measured at some sites
weekly_features = subhourly_features.copy()[:3]
weekly_data = provide_weekly_dynamic_cosmos_data(metadata, weekly_features, forcenew=False)
twohourly_data = provide_twohourly_dynamic_cosmos_data(metadata, subhourly_features, forcenew=False)

'''
# calculate wind speed from Ux, Uy
for SID in twohourly_data.keys():
    twohourly_data[SID]['WS'] = np.sqrt(
        np.square(twohourly_data[SID].UX) + np.square(twohourly_data[SID].UY)
    )
    twohourly_data[SID] = twohourly_data[SID].drop(['UX','UY'], axis=1)
subhourly_features = np.setdiff1d(subhourly_features, ['UX', 'UY'])
subhourly_features = list(subhourly_features) + ['WS']
'''


## IDEA: 
## take the 52 week data and reshuffle so that it always starts with
## January, say. Then we remove the positional problem, help learning?
## also can then treat as a feature vector and feed into something like 
## a random forest? Can probably cast as a classification problem, and
## generate more "samples" for each carbon case by taking all possible
## 52 week periods (for tdt1 and tdt2)
## Also: Think of calculating statistics for individual rainfall/drying events
## so clean up the pulling of two-hourly data windows to make sure we capture 
## "start" of rain
'''
fig, ax = plt.subplots(2,2, sharey=True)
weekly_data['BUNNY'].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[0,0])
soc = np.around(float(metadata.site[metadata.site['SITE_ID']=='BUNNY'].SOIL_ORGANIC_CARBON), 3)
ax[0,0].set_title(f'BUNNY {soc}')
weekly_data['ALIC1'].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[0,1])
soc = np.around(float(metadata.site[metadata.site['SITE_ID']=='ALIC1'].SOIL_ORGANIC_CARBON), 3)
ax[0,1].set_title(f'ALIC1 {soc}')
weekly_data['STIPS'].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[1,0])
soc = np.around(float(metadata.site[metadata.site['SITE_ID']=='STIPS'].SOIL_ORGANIC_CARBON), 3)
ax[1,0].set_title(f'STIPS {soc}')
weekly_data['TADHM'].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[1,1])
soc = np.around(float(metadata.site[metadata.site['SITE_ID']=='TADHM'].SOIL_ORGANIC_CARBON), 3)
ax[1,1].set_title(f'TADHM {soc}')
ax[0,0].set_xlabel('')
ax[0,1].set_xlabel('')
ax[1,0].set_xlabel('')
ax[1,1].set_xlabel('')
ax[0,0].set_ylabel('VWC')
ax[1,0].set_ylabel('VWC')
plt.show()

fig, ax = plt.subplots(2,2)
twohourly_data['BUNNY'].iloc[:5000].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[0,0])
soc = np.around(float(metadata.site[metadata.site['SITE_ID']=='BUNNY'].SOIL_ORGANIC_CARBON), 3)
ax[0,0].set_title(f'BUNNY {soc}')
twohourly_data['ALIC1'].iloc[:5000].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[0,1])
soc = np.around(float(metadata.site[metadata.site['SITE_ID']=='ALIC1'].SOIL_ORGANIC_CARBON), 3)
ax[0,1].set_title(f'ALIC1 {soc}')
twohourly_data['STIPS'].iloc[:5000].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[1,0])
soc = np.around(float(metadata.site[metadata.site['SITE_ID']=='STIPS'].SOIL_ORGANIC_CARBON), 3)
ax[1,0].set_title(f'STIPS {soc}')
twohourly_data['TADHM'].iloc[:5000].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[1,1])
soc = np.around(float(metadata.site[metadata.site['SITE_ID']=='TADHM'].SOIL_ORGANIC_CARBON), 3)
ax[1,1].set_title(f'TADHM {soc}')
ax[0,0].set_xlabel('')
ax[0,1].set_xlabel('')
ax[1,0].set_xlabel('')
ax[1,1].set_xlabel('')
ax[0,0].set_ylabel('VWC')
ax[1,0].set_ylabel('VWC')
plt.show()

'''

## normalise data
norm_by_const_f = weekly_features + ['SWIN']
for sbf in np.setdiff1d(subhourly_features, norm_by_const_f):
    x = []
    for SID in twohourly_data.keys():
        x += list(twohourly_data[SID][sbf])
    x = np.array(x)
    mu = np.mean(x[~np.isnan(x)])
    sd = np.std(x[~np.isnan(x)])
    for SID in twohourly_data.keys():        
        twohourly_data[SID][sbf] = (twohourly_data[SID][sbf] - mu) / sd
    
tdt_feats = ['TDT1_VWC', 'TDT2_VWC']
for SID in weekly_data.keys():    
    weekly_data[SID][tdt_feats] = weekly_data[SID][tdt_feats]/100.
    twohourly_data[SID][tdt_feats] = twohourly_data[SID][tdt_feats]/100.
    weekly_data[SID][['PRECIP']] = weekly_data[SID][['PRECIP']]/25.
    twohourly_data[SID][['PRECIP']] = twohourly_data[SID][['PRECIP']]/25.
    twohourly_data[SID][['SWIN']] = twohourly_data[SID][['SWIN']]/400.
            
static_data = metadata.site.set_index('SITE_ID')[['SOIL_ORGANIC_CARBON', 'SOIL_ORGANIC_CARBON_SD']]
# find mean relative error on SOC measurement to fix NaNs in error col
meanrelerr = (static_data['SOIL_ORGANIC_CARBON'] / static_data['SOIL_ORGANIC_CARBON_SD']).mean()
missing_sd_sites = static_data.index[static_data['SOIL_ORGANIC_CARBON_SD'].isna()]
for sid in missing_sd_sites:
    approx_sd = static_data.loc[sid]['SOIL_ORGANIC_CARBON'] / meanrelerr
    static_data.loc[sid,'SOIL_ORGANIC_CARBON_SD'] = approx_sd
soc_mean = 0.06
soc_sd = 0.066
static_data['raw_soc'] = static_data.SOIL_ORGANIC_CARBON
static_data.loc[:,'SOIL_ORGANIC_CARBON'] = (static_data.SOIL_ORGANIC_CARBON - soc_mean) / soc_sd
static_data.loc[:,'SOIL_ORGANIC_CARBON_SD'] = static_data.SOIL_ORGANIC_CARBON_SD / soc_sd

## add labels to contiguous weekly chunks
# where the chunk being consistent in the weekly granularity means we 
# should be able to then grab weeks within that at the higher resolution
# for the finer dynamic chunks (though means might ignore NaNs in weekly smoothing)
variables = ['TDT1_VWC', 'TDT2_VWC']
consts = list(np.setdiff1d(weekly_features, variables))
for SID in weekly_data.keys():
    for vv in variables:
        temp_df = assign_contiguous_groups(weekly_data[SID], consts+[vv])
        weekly_data[SID] = weekly_data[SID].merge(
            temp_df.rename(columns={'contig_chunk':vv+'_contig'}),
            on=['DATE_TIME']+list(weekly_data[SID].columns), how='inner')

# split data
sorted_sites = metadata.site.sort_values('SOIL_ORGANIC_CARBON')['SITE_ID']
train_sites = ['TADHM', 'RDMER', 'GLENS', 'SOURH']
val_sites = list(np.unique(list(np.setdiff1d(sorted_sites, train_sites)[::3]) + ['HARWD', 'STIPS']))
test_sites = list(np.setdiff1d(sorted_sites, val_sites+train_sites)[::5]) + ['GLENW']
train_sites = list(np.setdiff1d(sorted_sites, val_sites+test_sites))
    
twohour_feats = list(np.setdiff1d(subhourly_features, tdt_feats))
year_feats = []
n_chunks = 20
chunk_length = 7 * 12 # ~week of two-hourly data

if TRAIN is True:
    train_dat = dict(weekly_data={}, twohourly_data={})
    val_dat = dict(weekly_data={}, twohourly_data={})
    test_dat = dict(weekly_data={}, twohourly_data={})
    for sid in train_sites:
        train_dat['weekly_data'][sid] = weekly_data[sid]
        train_dat['twohourly_data'][sid] = twohourly_data[sid]
    for sid in val_sites:
        val_dat['weekly_data'][sid] = weekly_data[sid]
        val_dat['twohourly_data'][sid] = twohourly_data[sid]
    for sid in test_sites:
        test_dat['weekly_data'][sid] = weekly_data[sid]
        test_dat['twohourly_data'][sid] = twohourly_data[sid]
    
    train_dg = ChunkGenerator(train_dat['weekly_data'], train_dat['twohourly_data'],
                              static_data.loc[train_sites], twohour_feats, year_feats,
                              n_chunks, chunk_length)
    val_dg = ChunkGenerator(val_dat['weekly_data'], val_dat['twohourly_data'],
                            static_data.loc[val_sites], twohour_feats, year_feats,
                            n_chunks, chunk_length)   
    test_dg = ChunkGenerator(test_dat['weekly_data'], test_dat['twohourly_data'],
                            static_data.loc[test_sites], twohour_feats, year_feats,
                            n_chunks, chunk_length)
else:
    all_dg = ChunkGenerator(weekly_data, twohourly_data,
                            static_data, twohour_feats, year_feats,
                            n_chunks, chunk_length)

## make model, optimizer and loss function
d_chunks = len(twohour_feats) + 1
d_year = len(year_feats) + 2 # add sin wave
d_model = 512
d_out = 128
num_chunks = n_chunks
h_att = 4
N_att = 2
dropout = 0.1

#model = MultiScaleCPredict(d_chunks, d_year, d_model, d_out,
#                           num_chunks, h_att, N_att, dropout)
#model = SmallScaleCPredict(d_chunks, d_model, d_out,
#                           num_chunks, h_att, N_att, dropout)
#model = LargeScaleCPredict(d_year, d_model, d_out, dropout)
model = LargeScaleCPredictConv(d_year, d_model, d_out, dropout)
try:
    train_dg.gen_smallscale = False
    val_dg.gen_smallscale = False
    test_dg.gen_smallscale = False
except:
    all_dg.gen_smallscale = False

optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = build_NLL(sample_truth=True) #regularized_loss #gaussian_KL

model.to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(model)
print(f'Number of trainable parameters = {params}')

if load_prev_chkpnt:
    loadmodel = model_outdir + 'best_model.pth'    
    try:
        model, optimizer, checkpoint = load_checkpoint(loadmodel, model, optimizer, device)
    except:
        print('Failed loading checkpoint')
        checkpoint = None
else: 
  checkpoint = None
  loadmodel = None
  
if loadmodel != model_outdir + 'best_model.pth' and not checkpoint is None:
    checkpoint['best_loss'] = np.inf
    checkpoint['epoch'] = 0

if TRAIN is True:
    train_len = 200
    val_len = 100
    batchsize = 25
    soc_noise = 0.
    max_epochs = 500
    model, losses, val_losses = fit(model, optimizer, loss_fn, train_dg, val_dg, batchsize, 
                                    train_len=train_len, val_len=val_len, max_epochs=max_epochs,
                                    outdir=model_outdir, checkpoint=checkpoint, device=device,
                                    soc_noise=soc_noise)

'''
## investigate data
SID = list(train_dg.sites)[-9]
fig, ax = plt.subplots()
for i in range(5):
    x, y, dt = train_dg.return_elem_for_site(SID)
    res = pd.DataFrame({'x':x[0].numpy().flatten(), 't':dt}).set_index('t')
    res.plot(y='x', ax=ax)
fig.suptitle(f'{SID} {np.round(float(y[0]),3)}')
plt.show()
'''

## evaluate
###########################
def create_site_batch(dg, batchsize, SID, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    xbatcha = []
    xbatchb = []
    ybatch = []
    for i in range(batchsize):
        x, y, _ = dg.return_elem_for_site(SID)
        if len(x)==0: return [], 0
        xbatcha.append(x[0])
        xbatchb.append(x[1])
        ybatch.append(y)    
    xbatch = [torch.stack(xbatcha).to(device), [xb.to(device) for xb in xbatchb]]
    ybatch = torch.stack(ybatch).to(device)
    return xbatch, ybatch
   
if EVAL is True:   
    ## run for repeats of each site
    batchsize = 25
    res = dict(mu_pred=[], sigma_pred=[],
               mu_true=[], sigma_true=[],
               SITE_ID=[], data_type=[])
    for SID in static_data.index:
        xbatch, ytrue = create_site_batch(all_dg, batchsize, SID, device=device)        
        if len(xbatch)==0: continue
        try:
            ypred = model(xbatch)
            loss = loss_fn(ypred, ytrue)
        except: continue
        if SID in train_sites: data_type = 'train'
        elif SID in val_sites: data_type = 'validate'
        elif SID in test_sites: data_type = 'test'        
        mu = ypred[:,0].detach().cpu().numpy()        
        sigma = torch.exp(ypred[:,1].detach()).cpu().numpy()
        pop_mu = np.mean(mu)
        pop_sig = np.sqrt(np.sum(sigma * sigma)) / len(sigma)
        res['mu_pred'].append(pop_mu)
        res['sigma_pred'].append(pop_sig)
        res['mu_true'].append(float(ytrue[0,0].cpu().numpy()))
        res['sigma_true'].append(float(ytrue[0,1].cpu().numpy()))
        res['SITE_ID'].append(SID)
        res['data_type'].append(data_type)
    res = pd.DataFrame(res)
    res.to_csv(eval_outdir + 'allsite_results.csv', index=False)
    
    
    '''
    ## explore results
    res = pd.read_csv(eval_outdir + 'allsite_results.csv')
    
    fig, ax = plt.subplots(figsize=(18,10))
    pldat = res
    xlocs = pldat[['SITE_ID']].copy()
    dodge = 0.1
    xlocs['raw'] = list(range(xlocs.shape[0]))
    xlocs['pred'] = np.array(range(xlocs.shape[0])) - dodge
    xlocs['true'] = np.array(range(xlocs.shape[0])) + dodge
    pldat = pldat.merge(xlocs)
    cpal = {'train':'g', 'validate':'y', 'test':'r', 'true':'k'}
    for dat_type in ['train', 'validate', 'test', 'true']:        
        if dat_type=='true':
            markers, caps, bars = ax.errorbar(pldat['true'], pldat['mu_true'],
                                              yerr=pldat['sigma_true'], fmt='.', ms=0,
                                              ecolor=cpal[dat_type], capsize=1, capthick=1)
        else:
            subdat = pldat[pldat['data_type']==dat_type]
            markers, caps, bars = ax.errorbar(subdat['pred'], subdat['mu_pred'],
                                              yerr=subdat['sigma_pred'], fmt='.', ms=0,
                                              ecolor=cpal[dat_type], capsize=1, capthick=1)
        [bar.set_alpha(0.55) for bar in bars]
        [cap.set_alpha(0.55) for cap in caps]
    sns.scatterplot(x='pred', y='mu_pred', data=pldat,
                    hue='data_type', palette=cpal, ax=ax, s=15)
    sns.scatterplot(x='true', y='mu_true', data=pldat,
                    hue=np.repeat(['true'], pldat.shape[0]),
                    palette=['k'], ax=ax, s=15)    
    plt.xticks(pldat.raw, pldat.SITE_ID, rotation=90)
    plt.draw()
    plt.show()
    
    from soil_moisture.eval_funcs import score
    score(res.mu_pred, res.mu_true)
    score(res[res['data_type']=='train'].mu_pred, res[res['data_type']=='train'].mu_true)
    score(res[res['data_type']=='validate'].mu_pred, res[res['data_type']=='validate'].mu_true)
    score(res[res['data_type']=='test'].mu_pred, res[res['data_type']=='test'].mu_true)
    '''        
