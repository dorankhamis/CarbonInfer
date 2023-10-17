import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pkbar
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn.metrics as metrics
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

sns.set_theme()
plt.rcParams['figure.figsize'] = [14,8]
metadata = CosmosMetaData()

class ChunkGenerator(Dataset):
    def __init__(self, weekly_data, static_data, largescale_features):
        self.weekly_data = weekly_data
        self.static_data = static_data
        self.sites = list(weekly_data.keys())
        self.possible_targets = ['TDT1_VWC','TDT2_VWC']
        self.largescale_features = largescale_features
        self.soc = 'SOIL_ORGANIC_CARBON'        
        self.targets = [self.soc, self.soc+'_SD']
        self.site_probs = (1+8*static_data[self.soc]) / (1+8*static_data[self.soc]).sum()   
        
    def __len__(self):
        return len(self.sites)

    def return_elem_for_site(self, SID):
        # find weekly and subdaily dt windows
        gotgood = False
        while not gotgood:        
            TARG, dt_window = self.grab_site_contiguous_chunk(self.weekly_data, 52, SID)
            if len(dt_window)==0: return 0, 0, 0            
            gotgood = True
        # turn dt windows into data tensors
        temp = self.weekly_data[SID].loc[dt_window][self.largescale_features+[TARG]]
        temp['doy'] = temp.index.day_of_year
        temp = temp.sort_values('doy')
        year_scale = np.asarray(temp[self.largescale_features+[TARG]], dtype=np.float32).T        
        carbon = self.static_data.loc[SID][self.targets]
        return year_scale, carbon, dt_window

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

##############################################
## load weekly data for 52 element feature vec
subhourly_features = ['TDT1_VWC', 'TDT2_VWC', 'PRECIP', 'SWIN', 
                      'STP_TSOIL10', 'STP_TSOIL5', 'STP_TSOIL2',
                      'RH', 'TA', 'PA']
# add features like days about 10 degrees or raw LST, cumulative rain, cumulative SWIN
weekly_features = subhourly_features.copy()[:-3]
weekly_data = provide_weekly_dynamic_cosmos_data(metadata, weekly_features, forcenew=False)
twohourly_data = provide_twohourly_dynamic_cosmos_data(metadata, subhourly_features, forcenew=False)

for SID in weekly_data.keys():
    weekly_data[SID]['SOIL_TEMP'] = weekly_data[SID][
        ['STP_TSOIL10', 'STP_TSOIL5', 'STP_TSOIL2']].mean(axis=1)
    weekly_data[SID] = weekly_data[SID].drop(['STP_TSOIL10', 'STP_TSOIL5', 'STP_TSOIL2'], axis=1)
    
extract fraction of year above 10 degrees?    
    
## IDEA: 
##      + take the 52 week data and reshuffle so that it always starts with
## January, say. Then we remove the positional problem, help learning?
## also can then treat as a feature vector and feed into something like 
## a random forest? Can probably cast as a classification problem, and
## generate more "samples" for each carbon case by taking all possible
## 52 week periods (for tdt1 and tdt2)
##      + Also: Think of calculating statistics for individual rainfall/drying events
## so clean up the pulling of two-hourly data windows to make sure we capture 
## "start" of rain
##      + Add the land surface temperature weekly averaged data to the
## feature vector?
##      + renormalize the 52 week vectors by ~99% and cap at 1 in order to 
##          explicitly capture periods of saturation?

## add labels to contiguous weekly chunks
from soil_moisture.utils import interpolate_df
variables = ['TDT1_VWC', 'TDT2_VWC']
consts = list(np.setdiff1d(weekly_features, variables))
for SID in weekly_data.keys():
    weekly_data[SID] = interpolate_df(weekly_data[SID], gap_size=1)
    for vv in variables:
        temp_df = assign_contiguous_groups(weekly_data[SID], consts+[vv])
        weekly_data[SID] = weekly_data[SID].merge(
            temp_df.rename(columns={'contig_chunk':vv+'_contig'}),
            on=['DATE_TIME']+list(weekly_data[SID].columns), how='inner')

# find mean relative error on SOC measurement to fix NaNs in error col
static_data = metadata.site.set_index('SITE_ID')[['SOIL_ORGANIC_CARBON', 'SOIL_ORGANIC_CARBON_SD']]
meanrelerr = (static_data['SOIL_ORGANIC_CARBON'] / static_data['SOIL_ORGANIC_CARBON_SD']).mean()
missing_sd_sites = static_data.index[static_data['SOIL_ORGANIC_CARBON_SD'].isna()]
for sid in missing_sd_sites:
    approx_sd = static_data.loc[sid]['SOIL_ORGANIC_CARBON'] / meanrelerr
    static_data.loc[sid,'SOIL_ORGANIC_CARBON_SD'] = approx_sd
    
# create all possible contiguous 52-week feature vectors
dg = ChunkGenerator(weekly_data, static_data, [])
length = 52
samples = []
carbon_msd = []
site_samples = []
for SID in dg.sites:
    for TARG in dg.possible_targets:
        contig_ids = dg.view_contiguous_chunks(dg.weekly_data[SID], TARG)
        for chunk_id in contig_ids:
            dt_window_main = dg.grab_chunk(dg.weekly_data[SID], TARG, chunk_id)
            seqlen = dt_window_main.shape[0]
            if seqlen<length:
                continue
            elif seqlen>length:
                for dt_idx in range(seqlen-length):
                    dt_window = dt_window_main[dt_idx:(dt_idx + length)]
                    temp = dg.weekly_data[SID].loc[dt_window]
                    temp['doy'] = dt_window.day_of_year
                    temp = temp.sort_values('doy')
                    year_scale = np.asarray(temp[dg.largescale_features+[TARG]],
                                            dtype=np.float32).flatten()     
                    carbon = np.asarray(dg.static_data.loc[SID][dg.targets], dtype=np.float32)
                    # normalise VWC by 97.5th percentile
                    year_scale = np.clip(year_scale / np.quantile(year_scale, 0.975), 0, 1)
                    samples.append(year_scale)
                    carbon_msd.append(carbon)
                    site_samples.append(SID)
            else:
                temp = dg.weekly_data[SID].loc[dt_window_main]
                temp['doy'] = dt_window_main.day_of_year
                temp = temp.sort_values('doy')
                year_scale = np.asarray(temp[dg.largescale_features+[TARG]],
                                        dtype=np.float32).flatten()     
                carbon = np.asarray(dg.static_data.loc[SID][dg.targets], dtype=np.float32)                                    
                # normalise VWC by 97.5th percentile
                year_scale = np.clip(year_scale / np.quantile(year_scale, 0.975), 0, 1)
                samples.append(year_scale)
                carbon_msd.append(carbon)
                site_samples.append(SID)

samples = np.stack(samples)
carbon_msd = np.stack(carbon_msd)
site_samples = np.array(site_samples)
Path(proj_dir+'/carbon_prediction/data/').mkdir(parents=True, exist_ok=True)
np.save(proj_dir+'/carbon_prediction/data/year_feature_vectors_normed.npy', samples)
np.save(proj_dir+'/carbon_prediction/data/year_sample_carbon_mean_std.npy', carbon_msd)
np.save(proj_dir+'/carbon_prediction/data/year_sample_sitenames.npy', site_samples)


## load feature vecs
samples =  np.load(proj_dir+'/carbon_prediction/data/year_feature_vectors_normed.npy')
carbon_msd = np.load(proj_dir+'/carbon_prediction/data/year_sample_carbon_mean_std.npy')
site_samples = np.load(proj_dir+'/carbon_prediction/data/year_sample_sitenames.npy')

## data splits 
# test/train splits with unseen sites
sorted_sites = metadata.site.sort_values('SOIL_ORGANIC_CARBON')['SITE_ID']
train_sites = ['TADHM', 'RDMER', 'GLENS', 'SOURH']
test_sites = list(np.unique(list(np.setdiff1d(sorted_sites, train_sites)[::3]) + ['HARWD', 'STIPS']))
train_sites = list(np.setdiff1d(sorted_sites, test_sites))
train_inds = np.hstack([np.where(site_samples==s)[0] for s in train_sites])
test_inds = np.hstack([np.where(site_samples==s)[0] for s in test_sites])
for i in range(10):
    np.random.shuffle(train_inds)
    np.random.shuffle(test_inds)
# or randomly between all sites:
'''
inds = np.array(range(samples.shape[0]), dtype=np.int32)
for i in range(10):
    np.random.shuffle(inds)
    samples = samples[inds,:]
    carbon_msd = carbon_msd[inds,:]
    site_samples = site_samples[inds]
train_inds = np.random.choice(inds, int(inds.shape[0]*0.5), replace=False)
test_inds = np.setdiff1d(inds, train_inds)
'''
xtrain = samples[train_inds,:]
# either sample from the measurement noise or just take the mean
samp_meas_noise = True
if samp_meas_noise is True:
    ytrain = np.random.normal(carbon_msd[train_inds,0], carbon_msd[train_inds,1])
    where_neg = np.where(ytrain<0)[0]
    while where_neg.shape[0]>0:
        ytrain[where_neg] = np.random.normal(carbon_msd[train_inds,0][where_neg],
                                             carbon_msd[train_inds,1][where_neg])
        where_neg = np.where(ytrain<0)[0]
else:
    ytrain = carbon_msd[train_inds,0]
ytrain = ytrain.astype(np.float32)
sites_train = site_samples[train_inds]

xtest = samples[test_inds,:]
if samp_meas_noise is True:
    ytest = np.random.normal(carbon_msd[test_inds,0], carbon_msd[test_inds,1])
    where_neg = np.where(ytest<0)[0]
    while where_neg.shape[0]>0:
        ytest[where_neg] = np.random.normal(carbon_msd[test_inds,0][where_neg],
                                             carbon_msd[test_inds,1][where_neg])
        where_neg = np.where(ytest<0)[0]
else:
    ytest = carbon_msd[test_inds,0]
ytest = ytest.astype(np.float32)
sites_test = site_samples[test_inds]

## create random forest model
min_samples_leaf = 15
n_estimators = 150
max_depth = 15
model = RandomForestRegressor(
    random_state=22,
    max_features=max(1,int(xtrain.shape[1]*0.3)),
    min_samples_leaf=min_samples_leaf,
    n_estimators=n_estimators,
    max_depth=max_depth
)

## fit just 52 week VWC data
# train
model.fit(xtrain, ytrain)
ypred_train = model.predict(xtrain)
print(metrics.r2_score(ytrain, ypred_train))
# test
ypred_test = model.predict(xtest)
print(metrics.r2_score(ytest, ypred_test))
# importance
fimportance = pd.DataFrame({'score':model.feature_importances_,
                            'feature':list(range(length))})
fimportance = fimportance.sort_values('score', ascending=False)
# plot scatter
fig, ax = plt.subplots(1,2,figsize=(18,10))
tr_res = pd.DataFrame({'ytrue':ytrain, 'ypred':ypred_train, 'SITE_ID':sites_train})
sns.scatterplot(x='ytrue', y='ypred', hue='SITE_ID', data=tr_res, ax=ax[0], s=15)
te_res = pd.DataFrame({'ytrue':ytest, 'ypred':ypred_test, 'SITE_ID':sites_test})
sns.scatterplot(x='ytrue', y='ypred', hue='SITE_ID', data=te_res, ax=ax[1], s=15)
for a in ax:
    xx = np.mean(a.get_xlim()) 
    a.axline((xx,xx), slope=1)
plt.show()

## add historical climate data to feature vector!
worldclim_dat = provide_WorldClim_at_cosmos_sites(metadata)
worldclim_dat = (worldclim_dat.set_index('SITE_ID')
    .drop(['LATITUDE', 'LONGITUDE'], axis=1))

# add noised worldclim data (for "repeats")
wc_noise = 0.001
xtrain_new = np.hstack([xtrain,
    worldclim_dat.loc[sites_train] + np.random.normal(0, worldclim_dat.loc[sites_train].abs()*wc_noise)
])
missing_inds = np.unique(np.where(np.isnan(xtrain_new))[0])
xtrain_new = np.delete(xtrain_new, missing_inds, axis=0)
ytrain_new = np.delete(ytrain, missing_inds, axis=0)
sites_train_new = np.delete(sites_train, missing_inds, axis=0)

xtest_new = np.hstack([xtest,
    worldclim_dat.loc[sites_test] +  + np.random.normal(0, worldclim_dat.loc[sites_test].abs()*wc_noise)
])
missing_inds = np.unique(np.where(np.isnan(xtest_new))[0])
xtest_new = np.delete(xtest_new, missing_inds, axis=0)
ytest_new = np.delete(ytest, missing_inds, axis=0)
sites_test_new = np.delete(sites_test, missing_inds, axis=0)

## train
model.fit(xtrain_new, ytrain_new)
ypred_train_new = model.predict(xtrain_new)
print(metrics.r2_score(ytrain_new, ypred_train_new))

## test
ypred_test_new = model.predict(xtest_new)
print(metrics.r2_score(ytest_new, ypred_test_new))

fimportance = pd.DataFrame({'score':model.feature_importances_,
                            'feature':list(range(length))+list(worldclim_dat.columns)})
fimportance = fimportance.sort_values('score', ascending=False)

fig, ax = plt.subplots(1,2,figsize=(18,10))
tr_res = pd.DataFrame({'ytrue':ytrain_new, 'ypred':ypred_train_new, 'SITE_ID':sites_train_new})
sns.scatterplot(x='ytrue', y='ypred', hue='SITE_ID', data=tr_res, ax=ax[0], s=15)
te_res = pd.DataFrame({'ytrue':ytest_new, 'ypred':ypred_test_new, 'SITE_ID':sites_test_new})
sns.scatterplot(x='ytrue', y='ypred', hue='SITE_ID', data=te_res, ax=ax[1], s=15)
for a in ax:
    xx = np.mean(a.get_xlim()) 
    a.axline((xx,xx), slope=1)
plt.show()

## just worldclim data...
xtrain_wc = xtrain_new[:,52:]
xtest_wc = xtest_new[:,52:]
model.fit(xtrain_wc, ytrain_new)
ypred_train_wc = model.predict(xtrain_wc)
print(metrics.r2_score(ytrain_new, ypred_train_wc))
ypred_test_wc = model.predict(xtest_wc)
print(metrics.r2_score(ytest_new, ypred_test_wc))

fimportance = pd.DataFrame({'score':model.feature_importances_,
                            'feature':list(worldclim_dat.columns)})
fimportance = fimportance.sort_values('score', ascending=False)

fig, ax = plt.subplots(1,2,figsize=(18,10))
tr_res = pd.DataFrame({'ytrue':ytrain_new, 'ypred':ypred_train_wc, 'SITE_ID':sites_train_new})
sns.scatterplot(x='ytrue', y='ypred', hue='SITE_ID', data=tr_res, ax=ax[0], s=15)
te_res = pd.DataFrame({'ytrue':ytest_new, 'ypred':ypred_test_wc, 'SITE_ID':sites_test_new})
sns.scatterplot(x='ytrue', y='ypred', hue='SITE_ID', data=te_res, ax=ax[1], s=15)
for a in ax:
    xx = np.mean(a.get_xlim()) 
    a.axline((xx,xx), slope=1)
plt.show()

## so: LST weekly to featyre vector
## and perhaps fitting drying curves for feature vector too?
