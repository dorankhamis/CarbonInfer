import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.optim import Adam
import torch
import pkbar

from soil_moisture.utils import indexify_datetime
from soil_moisture.data_classes.cosmos_data import CosmosMetaData
from soil_moisture.training_funcs import load_checkpoint, save_checkpoint
from soil_moisture.prepare_general import (import_params, grab_data, normalise_data,
                                           drop_add_contig_sincos, generators_and_loaders)
from network import make_model
from fit_funcs import get_batch, fit, batch_one_site_variants, mse_loss_exp

sns.set_theme()
plt.rcParams['figure.figsize'] = [14,8]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metadata = CosmosMetaData()
modelname = 'carbon_predict'
model_outdir = f'./logs/{modelname}/'
Path(model_outdir).mkdir(parents=True, exist_ok=True)

## import params
trainmodel = False
load_prev_chkpnt = False
evalmodel = True
new_set = False
fp, sp = import_params(model_outdir, new_set)

## prepare data
extra_features = ['SOIL_ORGANIC_CARBON']
dynamic_targets = ['COSMOS_VWC', 'D86_75M']
drop_d_feats = ['STP_TSOIL20', 'STP_TSOIL50', 'G1', 'G2']
drop_d_feats += list(np.setdiff1d(fp.tdt_soilmoisturenames, ['TDT1_VWC', 'TDT2_VWC']))
drop_d_feats += fp.tdt_soiltempnames
av_groups = [dict(name='topsoil_temp', cols=['STP_TSOIL2', 'STP_TSOIL5', 'STP_TSOIL10']),
             dict(name='TDT_VWC', cols=['TDT1_VWC', 'TDT2_VWC'])]             
drop_s_feats = ['LATITUDE', 'LONGITUDE']

(dynamic_data, dynamic_features, dynamic_targets,
    static_data, static_features, static_targets) = grab_data(
        metadata, fp, sp, extra_features, dynamic_targets, forcenew=False
)
dynamic_data, static_data = normalise_data(metadata, fp, sp, dynamic_data, static_data)

contig_cosmos = False
contig_tdt = False
train_carbon_sites = True
dynamic_data, dynamic_features, static_data, static_features = drop_add_contig_sincos(
    metadata, fp, sp, 
    dynamic_data, dynamic_features, dynamic_targets,
    static_data, static_features, static_targets,
    drop_d_feats, av_groups, drop_s_feats,
    contig_cosmos=contig_cosmos, contig_tdt=contig_tdt 
)

## splits here must be different as we want train, val and test sets
## that each can have various 365 day windows (i.e. temporal fractions
## not going to work). So just split sites by stratified sampling on SOC
train_dat = dict(dyn={})
val_dat = dict(dyn={})
test_dat = dict(dyn={})
sorted_sites = static_data.sort_values('SOIL_ORGANIC_CARBON').index
val_sites = list(sorted_sites[-2::-3])
test_sites = list(np.setdiff1d(sorted_sites[-4::-6], val_sites))
train_sites = list(np.setdiff1d(sorted_sites, val_sites+test_sites))
for sid in train_sites: train_dat['dyn'][sid] = dynamic_data[sid]
for sid in val_sites: val_dat['dyn'][sid] = dynamic_data[sid]
for sid in test_sites: test_dat['dyn'][sid] = dynamic_data[sid]
train_dat['stat'] = static_data.loc[train_sites]
val_dat['stat'] = static_data.loc[val_sites]
test_dat['stat'] = static_data.loc[test_sites]

## create datasets and dataloaders
collate_fn = None
sp.maxsize = 365
sp.minsize = 365
sp.historysize = 0
sp.pre_shift = 0
sp.post_shift = 0
sp.separate_history = False
sp.dynamics_with_history = False
fp.dynamic_features = dynamic_features
fp.static_features = static_features
fp.dynamic_targets = dynamic_targets
fp.static_targets = static_targets
fp.columns_to_dynamise = np.setdiff1d(static_data.columns, ['SOIL_ORGANIC_CARBON'])
sp.use_cosmos = False
sp.use_tdt = False

(train_dg, val_dg, test_dg,
    train_dl, val_dl, test_dl) = generators_and_loaders(
        metadata, fp, sp, train_dat, val_dat, test_dat,
        collate_fn=collate_fn, use_cosmos=sp.use_cosmos, use_tdt=sp.use_tdt
)

# save params to logs folder
with open(model_outdir+'session_parms.pkl', 'wb') as fo:
    pickle.dump(sp, fo)
with open(model_outdir+'feature_parms.pkl', 'wb') as fo:
    pickle.dump(fp, fo)

## grab a sample like this:
#SID, dt_window = train_dg.grab_random_contiguous_chunk()
#sub_dyn_input, SOC, _ = train_dg.process_inputs(dt_window, SID)
# where SOC is the carbon as it is the only static feature left 
# after the rest are stuck on the dynamic time series

## define model, optimizer and loss function
batchsize = 16
d_in = len(train_dg.dynamic_features) + len(train_dg.columns_to_dynamise)
d_model = 256
num_enc_layers = 4
num_heads = 4
d_feedforward = 512
dropout = 0.1
NAN_MASK = -99.
seq_len = train_dg.maxsize
embed_ds = [64, 128, 256, 16] # dynamics embedding
conv_dil_list = [1, 3] # dynamics embedding
conv_kern_list = [5, 5] # dynamics embedding
model = make_model(d_in, seq_len, embed_ds, conv_dil_list, conv_kern_list,
                   N=num_enc_layers, d_model=d_model,
                   d_ff=d_feedforward, h=num_heads, dropout=dropout)
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = mse_loss_exp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

if trainmodel is True:
    model.train()
    model, losses, val_losses = fit(model, optimizer, loss_fn, train_dg, val_dg,
                                    batchsize, train_len=250, val_len=100, max_epochs=25,
                                    nan_val=NAN_MASK, outdir=model_outdir, device=device)

## evaluate
###########################
if evalmodel is True:
    model, optimizer, checkpoint = load_checkpoint(model_outdir + 'best_model.pth', model, optimizer, device)
    model.eval()
    eval_outdir = f'./eval/{modelname}/'
    Path(eval_outdir).mkdir(parents=True, exist_ok=True)

    '''
    xbatch1, ytrue1 = get_batch(train_dg, 50, nan_val=-99)
    xbatch2, ytrue2 = get_batch(val_dg, 50, nan_val=-99)
    xbatch3, ytrue3 = get_batch(test_dg, 50, nan_val=-99)
    ypred1 = model(xbatch1)
    ypred2 = model(xbatch2)
    ypred3 = model(xbatch3)

    fig, ax = plt.subplots()
    sns.scatterplot(x=ytrue1.squeeze().numpy(), y=ypred1.detach().squeeze().numpy(), label='train')
    sns.scatterplot(x=ytrue2.squeeze().numpy(), y=ypred2.detach().squeeze().numpy(), label='val')
    sns.scatterplot(x=ytrue3.squeeze().numpy(), y=ypred3.detach().squeeze().numpy(), label='test')
    plt.show()
    '''

    ## want to get a distribution of predictions by moving the 365 day window 
    ## over data from a single site and calculate stats from predictions.
    output = pd.DataFrame()
    for SID in train_sites:
        xb, yb = batch_one_site_variants(train_dg, SID, N_dist=50, nan_val=NAN_MASK)
        if xb.shape[0]==0: continue        
        yp = model(xb.to(device))
        yp = yp.detach().squeeze().cpu().numpy()
        output = pd.concat([output, pd.DataFrame({'y_pred':yp, 'SITE_ID':SID,
                                                  'data_type':'train'})], axis=0)
    for SID in val_sites:
        xb, yb = batch_one_site_variants(val_dg, SID, N_dist=50, nan_val=NAN_MASK)
        if xb.shape[0]==0: continue
        yp = model(xb.to(device))
        yp = yp.detach().squeeze().cpu().numpy()
        output = pd.concat([output, pd.DataFrame({'y_pred':yp, 'SITE_ID':SID,
                                                  'data_type':'val'})], axis=0)
    for SID in test_sites:
        xb, yb = batch_one_site_variants(test_dg, SID, N_dist=50, nan_val=NAN_MASK)
        if xb.shape[0]==0: continue
        yp = model(xb.to(device))
        yp = yp.detach().squeeze().cpu().numpy()
        output = pd.concat([output, pd.DataFrame({'y_pred':yp, 'SITE_ID':SID,
                                                  'data_type':'test'})], axis=0)
    
    output.to_csv(eval_outdir + '/eval_output_distributions.csv', index=False)

    '''
    output = pd.read_csv(eval_outdir + '/eval_output_distributions.csv')
    output['y_pred'] = np.exp(output['y_pred'])
    
    def q_l95(x): return(np.quantile(x, 0.025))
    def q_u95(x): return(np.quantile(x, 0.975))
    def q_m50(x): return(np.quantile(x, 0.50))
    summary_stats = output.groupby(['SITE_ID','data_type']).agg([q_m50, q_l95, q_u95])
    summary_stats.columns = summary_stats.columns.get_level_values(1)
    summary_stats = summary_stats.reset_index()
    summary_stats = summary_stats.merge(static_data.reset_index()[['SITE_ID', 'SOIL_ORGANIC_CARBON']], on='SITE_ID')
    output = output.merge(static_data.reset_index()[['SITE_ID', 'SOIL_ORGANIC_CARBON']], on='SITE_ID')

    g = sns.FacetGrid(data=output, hue='data_type', col='SITE_ID', col_wrap=5)
    g.map(sns.histplot, 'y_pred', palette="muted", element='step').add_legend()
    plt.show()

    fig, ax = plt.subplots()
    sns.boxplot(x='SITE_ID', y='y_pred', hue='data_type', data=output, ax=ax)
    sns.scatterplot(x='SITE_ID', y='SOIL_ORGANIC_CARBON', 
                    data=static_data.loc[output['SITE_ID'].unique()].reset_index(), ax=ax, palette=['r'])
    plt.show()

    fig, ax = plt.subplots()
    markers, caps, bars = ax.errorbar(summary_stats['SOIL_ORGANIC_CARBON'],
                                      summary_stats['q_m50'],
                                      yerr=np.asarray(summary_stats[['q_l95', 'q_u95']]).T,
                                      fmt='.', ms=0, ecolor='black', capsize=1, capthick=1)
    [bar.set_alpha(0.55) for bar in bars]
    [cap.set_alpha(0.55) for cap in caps]
    sns.scatterplot(x='SOIL_ORGANIC_CARBON', y='q_m50', data=summary_stats,
                    hue='data_type', ax=ax)
    xx = np.mean(ax.get_xlim()) 
    ax.axline((xx,xx), slope=1, linestyle='--')
    plt.show()
    '''    
