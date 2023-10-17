import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.autograd import Variable
import pkbar

from soil_moisture.utils import indexify_datetime, zeropad_strint
from soil_moisture.data_loaders import proj_dir, provide_WorldClim_at_cosmos_sites
from soil_moisture.data_classes.cosmos_data import CosmosMetaData
from soil_moisture.training_funcs import load_checkpoint, mse_loss_exp
from soil_moisture.C_training_funcs import *
from soil_moisture.prepare_general import (import_params, grab_data, normalise_data,
                                           drop_add_contig_sincos, generators_and_loaders)
from soil_moisture.architectures.C_seqstat_attn import SeqStatAttn
from soil_moisture.explore_carbon_predictors import (accumulated_temperature,
                                                     average_signals_annual_monthly)

sns.set_theme()
plt.rcParams['figure.figsize'] = [14,8]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metadata = CosmosMetaData()
modelname = f'carbon_seqstat_attn'
model_outdir = f'{proj_dir}/logs/{modelname}/'
Path(model_outdir).mkdir(parents=True, exist_ok=True)

## import params
TRAIN = False
EVAL = True
load_prev_chkpnt = False
fp, sp = import_params(model_outdir, True)

## prepare data
extra_features = ['SOIL_ORGANIC_CARBON']
dynamic_targets = ['COSMOS_VWC', 'D86_75M']
drop_d_feats = ['STP_TSOIL20', 'STP_TSOIL50', 'G1', 'G2', 'SWOUT']
drop_d_feats += list(np.setdiff1d(fp.tdt_soilmoisturenames, ['TDT1_VWC', 'TDT2_VWC']))
drop_d_feats += fp.tdt_soiltempnames + ['STP_TSOIL10'] # 'SWI_010', 
av_groups = [dict(name='topsoil_temp', cols=['STP_TSOIL2', 'STP_TSOIL5']), # 'STP_TSOIL10'
             dict(name='TDT_VWC', cols=['TDT1_VWC', 'TDT2_VWC'])]
             #dict(name='SWI', cols=['SWI_002', 'SWI_005'])] # , 'SWI_010'
drop_s_feats = ['LATITUDE', 'LONGITUDE'] # don't keep lat and long as predictors

(dynamic_data, dynamic_features, dynamic_targets,
    static_data, static_features, static_targets) = grab_data(
        metadata, fp, sp, extra_features, dynamic_targets,
        forcenew=False, satellite_prods=[]
)

train_carbon_sites = True
#sp.add_yearly_periodicity = True might want this to help with year-window order problem?
dynamic_data, dynamic_features, static_data, static_features = drop_add_contig_sincos(
    metadata, fp, sp, 
    dynamic_data, dynamic_features, dynamic_targets,
    static_data, static_features, static_targets,
    drop_d_feats, av_groups, drop_s_feats,
    contig_cosmos=False, contig_tdt=False 
)

## annual accumulated temperature above 10 degrees (calculate before normalising)    
acc10_output = accumulated_temperature(dynamic_data, baseline_temp=10)
static_data = static_data.merge(acc10_output, on='SITE_ID')        

## normalise
dynamic_data, static_data = normalise_data(metadata, fp, sp, dynamic_data, static_data)

## gather and normalise World Clim historical data
worldclim_dat = provide_WorldClim_at_cosmos_sites(metadata)
worldclim_dat = (worldclim_dat.set_index('SITE_ID')
    .drop(['LATITUDE', 'LONGITUDE'], axis=1))    
months = np.array([zeropad_strint(a) for a in list(range(1,13))], dtype=object)

rad_norm = 18000
worldclim_dat['prec_'+months] = worldclim_dat['prec_'+months] / sp.precip_norm
worldclim_dat['srad_'+months] = worldclim_dat['srad_'+months] / rad_norm
worldclim_dat['tavg_'+months] = (worldclim_dat['tavg_'+months]-sp.tsoil_mu) / sp.tsoil_sd
worldclim_dat['tmin_'+months] = (worldclim_dat['tmin_'+months]-sp.tsoil_mu) / sp.tsoil_sd
worldclim_dat['tmax_'+months] = (worldclim_dat['tmax_'+months]-sp.tsoil_mu) / sp.tsoil_sd

## average temperature and precipitation total and by month
output = average_signals_annual_monthly(dynamic_data, take_sum = [],#['PRECIP'],
                                        take_mean = ['TDT_VWC', 'topsoil_temp']) # 'SWIN' # 'SWI'
static_data = static_data.merge(output, on='SITE_ID').merge(worldclim_dat, on='SITE_ID')

## splits here must be different as we want train, val and test sets
## that each can have various 365 day windows (i.e. temporal fractions
## not going to work). So just split sites by stratified sampling on SOC
train_dat = dict(dyn={})
val_dat = dict(dyn={})
test_dat = dict(dyn={})
sorted_sites = static_data.sort_values('SOIL_ORGANIC_CARBON').index

'''
SITE_ID  elev      slope                   SOC
MOORH    2.457582  2.005890                0.076
HENFS    1.628905  3.412375                0.077
SOURH    1.576985  2.191104                0.086
PLYNL    3.621365  1.993496                0.098
STIPS    2.125132  0.374795                0.104
GLENW         NaN       NaN                0.153
GLENS    2.327573  2.159747                0.203
RDMER   -1.477617 -1.523094                0.238
HARWD    1.455168 -0.619774                0.304
TADHM   -1.436227 -1.492035                0.314
'''

train_sites = ['TADHM', 'RDMER', 'GLENS', 'SOURH']
val_sites = list(np.unique(list(np.setdiff1d(sorted_sites, train_sites)[::3]) + ['HARWD', 'STIPS']))
test_sites = list(np.setdiff1d(sorted_sites, val_sites+train_sites)[::5]) + ['GLENW']
train_sites = list(np.setdiff1d(sorted_sites, val_sites+test_sites))

for sid in train_sites: train_dat['dyn'][sid] = dynamic_data[sid]
for sid in val_sites: val_dat['dyn'][sid] = dynamic_data[sid]
for sid in test_sites: test_dat['dyn'][sid] = dynamic_data[sid]

train_dat['stat'] = static_data.loc[train_sites]
val_dat['stat'] = static_data.loc[val_sites]
test_dat['stat'] = static_data.loc[test_sites]

## create data generators
collate_fn = None
sp.maxsize = 365
sp.minsize = 90
sp.historysize = 0
sp.pre_shift = 0
sp.post_shift = 0
sp.separate_history = False
sp.dynamics_with_history = False
fp.dynamic_features = dynamic_features
fp.static_features = ['SOIL_ORGANIC_CARBON']
fp.dynamic_targets = dynamic_targets
fp.static_targets = static_targets
fp.columns_to_dynamise = [] #np.setdiff1d(static_data.columns, ['SOIL_ORGANIC_CARBON'])
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

## define model, optimizer and loss function
batchsize = 6
d_in = len(train_dg.dynamic_features) + len(train_dg.columns_to_dynamise)
s_in = len(np.setdiff1d(static_data.columns, 'SOIL_ORGANIC_CARBON'))
dropout = 0.15
embed_ds = [128, 256]
Natt_h = 4
Natt_l = 4
d_ff = 512
NAN_MASK = -99

model = SeqStatAttn(d_in, embed_ds, s_in, dropout, Natt_h, Natt_l, d_ff)
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = mse_loss_exp

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
    model.train()    
    model, losses, val_losses = fit_seqstat_attn(
        model, optimizer, loss_fn, train_dg, val_dg,
        batchsize, train_len=250, val_len=150, max_epochs=50, 
        nan_val=NAN_MASK,  soc_train_noise=0.05, outdir=model_outdir, 
        checkpoint=checkpoint, device=device
    )
    
## evaluate
###########################
if EVAL is True:
    model, optimizer, checkpoint = load_checkpoint(model_outdir + 'best_model.pth', model, optimizer, device)
    model.eval()
    eval_outdir = f'{proj_dir}/eval/{modelname}/'
    Path(eval_outdir).mkdir(parents=True, exist_ok=True)

    batch1, sites1 = create_allsite_seqstat_batch(train_dg, noise_soc=0.0, nan_val=-99, device=device)
    batch2, sites2 = create_allsite_seqstat_batch(val_dg, noise_soc=0.0, nan_val=-99, device=device)
    batch3, sites3 = create_allsite_seqstat_batch(test_dg, noise_soc=0.0, nan_val=-99, device=device)

    ypred1 = torch.exp(model(batch1.src, batch1.selfattn_mask, batch1.crossattn_mask))
    ypred2 = torch.exp(model(batch2.src, batch2.selfattn_mask, batch2.crossattn_mask))
    ypred3 = torch.exp(model(batch3.src, batch3.selfattn_mask, batch3.crossattn_mask))

    res1 = pd.DataFrame({'ypred':ypred1.detach().squeeze().cpu(),
                         'SITE_ID':sites1,
                         'ytrue':batch1.labels.squeeze().cpu()})
    res2 = pd.DataFrame({'ypred':ypred2.detach().squeeze().cpu(),
                         'SITE_ID':sites2,
                         'ytrue':batch2.labels.squeeze().cpu()})
    res3 = pd.DataFrame({'ypred':ypred3.detach().squeeze().cpu(),
                         'SITE_ID':sites3,
                         'ytrue':batch3.labels.squeeze().cpu()})


    fig, ax = plt.subplots()
    sns.scatterplot(x='ytrue', y='ypred', label='train', data=res1)
    sns.scatterplot(x='ytrue', y='ypred', label='val', data=res2)
    sns.scatterplot(x='ytrue', y='ypred', label='test', data=res3)
    xx = np.mean(ax.get_xlim()) 
    ax.axline((xx,xx), slope=1, linestyle='--')
    plt.show()
    
    from soil_moisture.eval_funcs import mse, mae, score
    score(res1['ypred'], res1['ytrue'])
    score(res2['ypred'], res2['ytrue'])
    score(res3['ypred'], res3['ytrue'])    
    score(pd.concat([y_res2['ypred_mean'], y_res3['ypred_mean']], axis=0),
          pd.concat([y_res2['ytrue']     , y_res3['ytrue']], axis=0))

    ## compare to random forest model of static data
    from sklearn.ensemble import RandomForestRegressor
    site_feat_vecs = static_data.fillna(-99)
    targ = ['SOIL_ORGANIC_CARBON']
    feats = list(np.setdiff1d(site_feat_vecs.columns, targ))
    Xy = [site_feat_vecs.loc[train_dg.sites][feats+targ],
          site_feat_vecs.loc[list(val_dg.sites) + list(test_dg.sites)][feats+targ]]
    rfmodel = RandomForestRegressor(min_samples_leaf=2, random_state=22,
                                    max_features=max(1, int(len(feats)*0.75)),
                                    criterion='squared_error')
    rfmodel.fit(Xy[0][feats], Xy[0][targ])
    ypred_train = rfmodel.predict(Xy[0][feats])
    ypred_test = rfmodel.predict(Xy[1][feats])
    r2score_train = rfmodel.score(Xy[0][feats], Xy[0][targ])
    r2score_test = rfmodel.score(Xy[1][feats], Xy[1][targ])
    res = pd.concat([Xy[0][targ].assign(soc_pred = ypred_train).assign(data_type='train'),
                     Xy[1][targ].assign(soc_pred = ypred_test).assign(data_type='test')], axis=0)

    fig, ax = plt.subplots(figsize=(6,6))
    cpal = ['g', 'r']
    pl1 = res[res['data_type']=='train']
    pl2 = res[res['data_type']=='test']
    lab1 = 'Train: R^2 = '+str(np.around(r2score_train, 3))
    lab2 = 'Test: R^2 = '+str(np.around(r2score_test, 3))
    pldat = pd.concat([pl1.assign(label=lab1), pl2.assign(label=lab2)])
    sns.scatterplot(x='SOIL_ORGANIC_CARBON', y='soc_pred', data=pl1, ax=ax,
                    hue='data_type', palette=cpal[0:1], alpha=0.9, s=20)
    sns.scatterplot(x='SOIL_ORGANIC_CARBON', y='soc_pred', data=pl2, ax=ax,
                    hue='data_type', palette=cpal[1:2], alpha=0.9, s=20)
    xx = np.mean(ax.get_xlim())
    ax.axline((xx,xx), slope=1, linestyle='--')
    new_labels = [lab1, lab2]
    for t, l in zip(ax.legend_.texts, new_labels):
        t.set_text(l)
    plt.show()

    feat_importance = pd.DataFrame(rfmodel.feature_importances_[np.newaxis,...], columns = feats)
    feat_importance.melt().sort_values('value', ascending=False).set_index('variable').plot(kind='bar')
    plt.show()

    RUN_DIST_EVAL = False
    if RUN_DIST_EVAL is True:
        ## want to get a distribution of predictions by moving the 365 day window 
        ## over data from a single site and calculate stats from predictions.
        output = pd.DataFrame()
        N_dist = 100
        for SID in train_sites:
            batch = batch_onesite_seqstat_variants(train_dg, SID, N_dist=N_dist, nan_val=NAN_MASK, device=device)
            if batch.src[0].shape[0]==0: continue
            yp = model(batch.src, batch.selfattn_mask, batch.crossattn_mask)
            yp = np.exp(yp.detach().squeeze().cpu().numpy())
            output = pd.concat([output, pd.DataFrame({'y_pred':yp, 'SITE_ID':SID,
                                                      'data_type':'train'})], axis=0)
        for SID in val_sites:
            batch = batch_onesite_seqstat_variants(val_dg, SID, N_dist=N_dist, nan_val=NAN_MASK, device=device)
            if batch.src[0].shape[0]==0: continue
            yp = model(batch.src, batch.selfattn_mask, batch.crossattn_mask)
            yp = np.exp(yp.detach().squeeze().cpu().numpy())
            output = pd.concat([output, pd.DataFrame({'y_pred':yp, 'SITE_ID':SID,
                                                      'data_type':'val'})], axis=0)
        for SID in test_sites:
            batch = batch_onesite_seqstat_variants(test_dg, SID, N_dist=N_dist, nan_val=NAN_MASK, device=device)
            if batch.src[0].shape[0]==0: continue
            yp = model(batch.src, batch.selfattn_mask, batch.crossattn_mask)
            yp = np.exp(yp.detach().squeeze().cpu().numpy())
            output = pd.concat([output, pd.DataFrame({'y_pred':yp, 'SITE_ID':SID,
                                                      'data_type':'test'})], axis=0)    
        output.to_csv(eval_outdir + '/eval_output_distributions.csv', index=False)

    output = pd.read_csv(eval_outdir + '/eval_output_distributions.csv')    
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

    from soil_moisture.eval_funcs import mse, mae, score
    r2train = score(summary_stats[summary_stats['data_type']=='train']['q_m50'],
                    summary_stats[summary_stats['data_type']=='train']['SOIL_ORGANIC_CARBON'])
          
    r2val = score(summary_stats[summary_stats['data_type']=='val']['q_m50'],
                  summary_stats[summary_stats['data_type']=='val']['SOIL_ORGANIC_CARBON'])
          
    r2test = score(summary_stats[summary_stats['data_type']=='test']['q_m50'],
                   summary_stats[summary_stats['data_type']=='test']['SOIL_ORGANIC_CARBON'])
    r2valtest = score(summary_stats[summary_stats['data_type']!='train']['q_m50'],
                      summary_stats[summary_stats['data_type']!='train']['SOIL_ORGANIC_CARBON'])    
    
    summary_stats = summary_stats.set_index('SITE_ID')

    summary_stats.loc[val_dg.sites, 'data_type'] = 'test'
    
    
    fig, ax = plt.subplots(figsize=(5,5))
    cpal = ['g', 'r']
    pl1 = summary_stats[summary_stats['data_type']=='train'].rename({'q_m50':'soc_pred'}, axis=1)
    pl2 = summary_stats[summary_stats['data_type']=='test'].rename({'q_m50':'soc_pred'}, axis=1)
    lab1 = 'Train: R^2 = '+str(np.around(r2train, 3))
    lab2 = 'Test: R^2 = '+str(np.around(r2valtest, 3))   
    pldat = pd.concat([pl1.assign(label=lab1), pl2.assign(label=lab2)])
    sns.scatterplot(x='SOIL_ORGANIC_CARBON', y='soc_pred', data=pl1, ax=ax,
                    hue='data_type', palette=cpal[0:1], alpha=0.9, s=25)
    sns.scatterplot(x='SOIL_ORGANIC_CARBON', y='soc_pred', data=pl2, ax=ax,
                    hue='data_type', palette=cpal[1:2], alpha=0.9, s=25)
    xx = np.mean(ax.get_xlim())
    ax.axline((xx,xx), slope=1, linestyle='--')
    new_labels = [lab1, lab2]
    for t, l in zip(ax.legend_.texts, new_labels):
        t.set_text(l)
    plt.show()
    
    fig, ax = plt.subplots()    
    sns.scatterplot(x='SOIL_ORGANIC_CARBON', y='q_m50', data=summary_stats,
                    hue='data_type', ax=ax)
    xx = np.mean(ax.get_xlim()) 
    ax.axline((xx,xx), slope=1, linestyle='--')
    new_labels = [lab2, lab1]
    for t, l in zip(ax.legend_.texts, new_labels):
        t.set_text(l)
    plt.show()

    
    fig, ax = plt.subplots()
    markers, caps, bars = ax.errorbar(summary_stats['SOIL_ORGANIC_CARBON'],
                                      summary_stats['q_m50'],
                                      yerr=np.asarray(summary_stats[['q_l95', 'q_u95']]).T,
                                      fmt='.', ms=0, ecolor='black', capsize=1, capthick=1, alpha=0.7)
    [bar.set_alpha(0.55) for bar in bars]
    [cap.set_alpha(0.55) for cap in caps]
    sns.scatterplot(x='SOIL_ORGANIC_CARBON', y='q_m50', data=summary_stats,
                    hue='data_type', ax=ax)
    xx = np.mean(ax.get_xlim()) 
    ax.axline((xx,xx), slope=1, linestyle='--')
    plt.show()

    
