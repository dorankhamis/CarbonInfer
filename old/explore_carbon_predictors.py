import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.optim import Adam
import torch
import torch.nn as nn
import pkbar

from soil_moisture.utils import indexify_datetime, zeropad_strint
from soil_moisture.data_loaders import proj_dir, provide_WorldClim_at_cosmos_sites
from soil_moisture.data_classes.cosmos_data import CosmosMetaData
from soil_moisture.prepare_general import (import_params, grab_data, normalise_data,
                                           drop_add_contig_sincos, generators_and_loaders) 

def accumulated_temperature(dynamic_data, baseline_temp=10):    
    acc10_output = pd.DataFrame()
    for SID in list(dynamic_data.keys()):
        acc10 = (dynamic_data[SID][['topsoil_temp']]
            .assign(acc10 = lambda x: x.topsoil_temp - baseline_temp,
                    year = lambda x: x.index.year)
            .rolling(365, min_periods=360)
            .sum()
            .dropna()
            .mean())
        acc10['SITE_ID'] = SID
        acc10_output = pd.concat([acc10_output,
            pd.DataFrame(acc10).T[['SITE_ID', 'acc10']]], axis=0)
    acc10_output['acc10'] = acc10_output['acc10']/abs(acc10_output['acc10']).max()
    acc10_output = acc10_output.set_index('SITE_ID', drop=True)
    return acc10_output
    
def average_signals_annual_monthly(dynamic_data, take_sum = ['PRECIP'],
                                   take_mean = ['SWI', 'SWIN', 'TDT_VWC', 'topsoil_temp']):
    output = pd.DataFrame()
    for SID in list(dynamic_data.keys()):        
        site_avs = dynamic_data[SID][take_mean].rolling(365, min_periods=360).mean().mean()
        site_avs = pd.DataFrame(site_avs).T.assign(SITE_ID=SID).set_index('SITE_ID', drop=True)
        site_avs.columns = site_avs.columns + '_mean'
        
        site_sms = dynamic_data[SID][take_sum].rolling(365, min_periods=360).sum().mean()
        site_sms = pd.DataFrame(site_sms).T.assign(SITE_ID=SID).set_index('SITE_ID', drop=True)
        site_sms.columns = site_sms.columns + '_mean_annual_total'

        month_avs = (dynamic_data[SID].assign(month = lambda x: x.index.month)
            .groupby('month')
            .agg('mean')[take_mean]
            .reset_index())
        month_avs = month_avs.melt(id_vars='month')
        month_avs['VarName'] = month_avs['variable'] + '_month' + month_avs['month'].astype(str)
        month_avs = (month_avs.drop(['month','variable'], axis=1)
                              .assign(SITE_ID = SID)
                              .pivot(index='SITE_ID', columns='VarName', values='value'))
                              
        month_sms = (dynamic_data[SID].assign(month = lambda x: x.index.month, year = lambda x: x.index.year)
            .groupby(['month','year'])
            .agg('sum')[take_sum]
            .groupby('month')
            .agg('mean')
            .reset_index())
        month_sms = month_sms.melt(id_vars='month')
        month_sms['VarName'] = month_sms['variable'] + '_month' + month_sms['month'].astype(str)
        month_sms = (month_sms.drop(['month','variable'], axis=1)
                              .assign(SITE_ID = SID)
                              .pivot(index='SITE_ID', columns='VarName', values='value'))
        output = pd.concat([output, 
                            pd.concat([site_avs, site_sms, month_avs, month_sms], axis=1)], axis=0)
    return output
    
if __name__=='__main__':

    sns.set_theme()
    plt.rcParams['figure.figsize'] = [14,8]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = CosmosMetaData()
    fp, sp = import_params(new_set=True)

    ## prepare data
    extra_features = ['SOIL_ORGANIC_CARBON']
    dynamic_targets = ['COSMOS_VWC', 'D86_75M']
    drop_d_feats = ['STP_TSOIL20', 'STP_TSOIL50', 'G1', 'G2', 'SWOUT']
    drop_d_feats += list(np.setdiff1d(fp.tdt_soilmoisturenames, ['TDT1_VWC', 'TDT2_VWC']))
    drop_d_feats += fp.tdt_soiltempnames + ['SWI_010', 'STP_TSOIL10']
    av_groups = [dict(name='topsoil_temp', cols=['STP_TSOIL2', 'STP_TSOIL5']), # 'STP_TSOIL10'
                 dict(name='TDT_VWC', cols=['TDT1_VWC', 'TDT2_VWC']),
                 dict(name='SWI', cols=['SWI_002', 'SWI_005'])] # , 'SWI_010'
    drop_s_feats = [] #['LATITUDE', 'LONGITUDE'] # keep lat and long as predictors??

    (dynamic_data, dynamic_features, dynamic_targets,
        static_data, static_features, static_targets) = grab_data(
            metadata, fp, sp, extra_features, dynamic_targets,
            forcenew=False, satellite_prods=['SWI']
    )

    # dyn
    for SID in metadata.site['SITE_ID']:
        dynamic_data[SID] = dynamic_data[SID].drop(drop_d_feats, axis=1)
        for avgr in av_groups:            
            dynamic_data[SID][avgr['name']] = dynamic_data[SID][avgr['cols']].mean(axis=1)
            dynamic_data[SID] = dynamic_data[SID].drop(avgr['cols'], axis=1)
    extradrop = []
    for avgr in av_groups:
        extradrop += avgr['cols']
        dynamic_features.append(avgr['name'])
    dynamic_features = list(np.setdiff1d(dynamic_features, drop_d_feats+extradrop))

    # stat
    static_data = static_data.drop(drop_s_feats, axis=1)
    static_features = list(np.setdiff1d(static_features, drop_s_feats))

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
                                            take_mean = ['SWI', 'TDT_VWC', 'topsoil_temp']) # 'SWIN'    
    static_data = static_data.merge(output, on='SITE_ID').merge(worldclim_dat, on='SITE_ID')

    '''
    ## LST anomalies
    temp_anomalies = pd.DataFrame()
    usevar = 'topsoil_temp' # STP_TSOIL2'
    for SID in list(dynamic_data.keys()):
        month_avs = (dynamic_data[SID].assign(month = lambda x: x.index.month)
            .groupby('month')
            .agg('median')[[usevar]]
            .reset_index()
            .rename({usevar:'month_av'}, axis='columns'))
        by_year = (dynamic_data[SID].assign(month = lambda x: x.index.month,
                                            year = lambda x: x.index.year)
            .resample('M')
            .agg('mean')[[usevar,'month','year']])
        T_anomalies = (by_year.merge(month_avs, on='month')
            .assign(T_anomaly = lambda x: x[usevar] - x.month_av)
            .drop([usevar,'month_av'], axis=1))
        # take abs max anomaly for each month?
        absmax_anom = T_anomalies.drop('year',axis=1).groupby('month').agg(lambda x: x.abs().max())
        T_anomalies.assign()

        T_anomalies[['year']] = T_anomalies[['year']].astype(np.int32)
        T_anomalies[['month']] = T_anomalies[['month']].astype(np.int32)
        temp_anomalies = pd.concat([temp_anomalies, 
                                    T_anomalies.assign(SITE_ID = SID)], axis=0)
                                    
    temp_anomalies = (temp_anomalies.assign(day=1)
        .assign(DATE = lambda x: pd.to_datetime(x[['year','month','day']]))
        .set_index('SITE_ID', drop=True)
        .merge(static_data[['SOIL_ORGANIC_CARBON']], on='SITE_ID')
        .reset_index())
            
    fig,ax=plt.subplots()
    sns.lineplot(x='DATE', y='T_anomaly', hue='SOIL_ORGANIC_CARBON', style='SITE_ID', data=temp_anomalies, ax=ax)
    ax.legend(ax.lines[:7], ax.legend_.texts[:7])
    plt.show()
    '''
    
    '''
    ## plot all features against soil organic carbon
    pl_feats = np.setdiff1d(static_data.columns, 'SOIL_ORGANIC_CARBON')
    nr = int(np.sqrt(len(pl_feats))-1)
    nc = int(np.ceil(len(pl_feats)/nr))
    fig, axes = plt.subplots(nr, nc, sharey=True)
    for nf, feat in enumerate(pl_feats):
        sns.scatterplot(x=feat, y='SOIL_ORGANIC_CARBON', data=static_data, ax=axes[nf//nc,nf%nc], s=15)
    '''
        
    ## can we use these static features in a simple random forest regression to find the carbon?
    from sklearn.ensemble import RandomForestRegressor
    site_feat_vecs = static_data.dropna()
    sorted_sites = site_feat_vecs.sort_values('SOIL_ORGANIC_CARBON').index
    test_sites = list(sorted_sites[-2::-2])
    train_sites = list(np.setdiff1d(sorted_sites, test_sites))
    targ = ['SOIL_ORGANIC_CARBON']
    feats = list(np.setdiff1d(site_feat_vecs.columns, targ))
    Xy = [site_feat_vecs.dropna().loc[train_sites][feats+targ],
          site_feat_vecs.dropna().loc[test_sites][feats+targ]]
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

    fig, ax = plt.subplots(figsize=(5,5))
    cpal = ['g', 'r']
    pl1 = res[res['data_type']=='train']
    pl2 = res[res['data_type']=='test']
    lab1 = 'Train: R^2 = '+str(np.around(r2score_train, 3))
    lab2 = 'Test: R^2 = '+str(np.around(r2score_test, 3))
    pldat = pd.concat([pl1.assign(label=lab1), pl2.assign(label=lab2)])
    sns.scatterplot(x='SOIL_ORGANIC_CARBON', y='soc_pred', data=pl1, ax=ax,
                    hue='data_type', palette=cpal[0:1], alpha=0.9, s=30)
    sns.scatterplot(x='SOIL_ORGANIC_CARBON', y='soc_pred', data=pl2, ax=ax,
                    hue='data_type', palette=cpal[1:2], alpha=0.9, s=30)
    xx = np.mean(ax.get_xlim())
    ax.axline((xx,xx), slope=1, linestyle='--')
    new_labels = [lab1, lab2]
    for t, l in zip(ax.legend_.texts, new_labels):
        t.set_text(l)
    plt.show()

    feat_importance = pd.DataFrame(rfmodel.feature_importances_[np.newaxis,...], columns = feats)    
    (feat_importance
        .melt()
        .sort_values('value', ascending=False)
        .set_index('variable')
        .iloc[:35]
        .plot(kind='bar'))
    plt.xticks(rotation=75)
    plt.show()
    
    
    '''
    ## or can we calculate power spectra of rainfall, soil moisture, SWI, and 
    ## combine all these different types of data?
    from soil_moisture.utils import calculate_power_spectrum, resample
    seconds_per_day = 3600. * 24
    max_period = 365.2425 # max temporal period in days 
    min_period = 3.0 # min temporal period in days 
    window_size = 10 # percent of frequency length
    freq_grid = np.linspace(0.02747387143166888, 0.3086300693193843, 256)
    outdf = pd.DataFrame()
    for SID in list(dynamic_data.keys()):      
        subdat = (dynamic_data[SID]
            .assign(days = lambda x: (x.index - np.min(dynamic_data[SID].index)).total_seconds() / seconds_per_day))
        ps_precip = calculate_power_spectrum(subdat, 'days', 'PRECIP', min_period, max_period, smooth=True, w_pc=window_size)
        ps_tdt = calculate_power_spectrum(subdat, 'days', 'TDT_VWC', min_period, max_period, smooth=True, w_pc=window_size)
        ps_swi = calculate_power_spectrum(subdat, 'days', 'SWI', min_period, max_period, smooth=True, w_pc=window_size)
        ps_precip = resample(ps_precip[2], ps_precip[3], freq_grid)
        ps_tdt = resample(ps_tdt[2], ps_tdt[3], freq_grid)
        ps_swi = resample(ps_swi[2], ps_swi[3], freq_grid)
        
    fig, ax = plt.subplots()
    sns.lineplot(x = 1. / freq_grid, y = ps_precip, ax=ax)
    sns.lineplot(x = 1. / freq_grid, y = ps_tdt, ax=ax)
    sns.lineplot(x = 1. / freq_grid, y = ps_swi, ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
    '''



    ##############################################
    ## loading sub-daily data to see if short timescale 
    ## patterns in SM are informative
    from soil_moisture.data_loaders import (read_one_cosmos_site,
                                            provide_weekly_dynamic_cosmos_data,
                                            provide_twohourly_dynamic_cosmos_data)
    metadata = CosmosMetaData()
    subhourly_features = ['TDT1_VWC', 'TDT2_VWC', 'PRECIP']
    weekly_data = provide_weekly_dynamic_cosmos_data(metadata, subhourly_features)
    twohourly_data = provide_twohourly_dynamic_cosmos_data(metadata, subhourly_features)
    
    fig, ax = plt.subplots(2,3, sharey=True)
    weekly_data['BUNNY'].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[0,0])
    weekly_data['ALIC1'].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[0,1])
    weekly_data['STIPS'].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[1,0])
    weekly_data['TADHM'].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[1,1])
    weekly_data['SHEEP'].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[0,2])
    weekly_data['RDMER'].plot(y=['TDT1_VWC','TDT2_VWC'], ax=ax[1,2])
    plt.show()
             
    ## normalise data
    for SID in weekly_data.keys():
        weekly_data[SID][subhourly_features] = weekly_data[SID][subhourly_features]/100
        twohourly_data[SID][subhourly_features] = twohourly_data[SID][subhourly_features]/100
                
    ## add labels to contiguous weekly chunks
    from soil_moisture.prepare_general import assign_contiguous_groups    
    import torch
    from torch.utils.data import Dataset
    # where the chunk being consistent in the weekly granularity means we 
    # should be able to then grab weeks within that at the higher resolution
    # for the finer dynamic chunks (though means might ignore NaNs in weekly smoothing)
    consts = ['PRECIP']
    variables = ['TDT1_VWC', 'TDT2_VWC']
    for SID in weekly_data.keys():
        for vv in variables:
            temp_df = assign_contiguous_groups(weekly_data[SID], consts+[vv])
            weekly_data[SID] = weekly_data[SID].merge(
                temp_df.rename(columns={'contig_chunk':vv+'_contig'}),
                on=['DATE_TIME']+list(weekly_data[SID].columns), how='inner'
            )

    # toy data loader for 52 week chunks (for wrapped convolutions)
    class ChunkGenerator(Dataset):
        def __init__(self, weekly_data, twohourly_data, static_data,
                     zoom_in_features, zoom_out_features, n_chunks, chunk_length):
            self.weekly_data = weekly_data
            self.twohourly_data = twohourly_data
            self.static_data = static_data
            self.sites = weekly_data.keys()
            self.possible_targets = ['TDT1_VWC','TDT2_VWC']
            self.zoom_in_features = zoom_in_features
            self.zoom_out_features = zoom_out_features
            self.n_chunks = n_chunks
            self.chunk_length = chunk_length
            self.targets = ['SOIL_ORGANIC_CARBON', 'SOIL_ORGANIC_CARBON_SD']
        
        def __len__(self):
            return len(self.sites)

        def __getitem__(self, idx):
            # find weekly and subdaily dt windows
            gotgood = False
            while not gotgood:
                SID, TARG, dt_window = self.grab_random_contiguous_chunk(self.weekly_data, 52)
                chunk_dt_list = []
                for i in range(self.n_chunks):
                    chunk_dt, chunk_id = self.grab_constrained_contiguous_chunk(
                        self.twohourly_data[SID], TARG, dt_window, self.chunk_length)
                    if chunk_id == -1: continue # catch a window with no small scale continuity
                    chunk_dt_list.append(chunk_dt)
                gotgood = True
            # turn dt windows into data tensors
            year_scale = torch.from_numpy(np.asarray(
                self.weekly_data[SID].loc[dt_window][self.zoom_out_features+[TARG]], dtype=np.float32)).T
            chunk_dat_list = []
            for i in range(self.n_chunks):
                chunk_dat_list.append(
                    torch.from_numpy(np.asarray(
                        self.twohourly_data[SID]
                            .loc[chunk_dt_list[i]][self.zoom_in_features+[TARG]],
                    dtype=np.float32)).T
                )
            small_scale = torch.stack(chunk_dat_list)
            # get target: carbon
            carbon = torch.tensor([self.static_data.loc[SID][self.targets]], dtype=torch.float32)
            return [year_scale, small_scale], carbon.flatten()
            
        def grab_random_contiguous_chunk(self, DATA, length):
            gotgood = False
            SID = 'FAIL'
            dt_window = pd.Index([])
            while not gotgood:
                TARG = np.random.choice(self.possible_targets, 1)[0]
                SID = self.static_data.index[np.random.randint(self.static_data.shape[0])]
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
            c_df = assign_contiguous_groups(c_df, self.zoom_in_features+[TARG])
            c_df = c_df.rename(columns={'contig_chunk':TARG+'_contig'})
            contig_ids = self.view_contiguous_chunks(c_df, TARG)            
            np.random.shuffle(contig_ids)
            for chunk_id in contig_ids:
                dt_window = self.grab_chunk(c_df, TARG, chunk_id)
                seqlen = dt_window.shape[0]
                if seqlen<length:
                    continue
                if seqlen>length:
                    # grab random subset of size maxsize
                    dt_idx = np.random.randint(seqlen - length)
                    dt_window = dt_window[dt_idx:(dt_idx + length)]
                gotgood = True
                break    
            if gotgood is False:
                return [], -1
            else:
                return dt_window, chunk_id

    def get_batch(dg, batchsize, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        xbatcha = []
        xbatchb = []
        ybatch = []
        for i in range(batchsize):
            x, y = dg[0]
            xbatcha.append(x[0])
            xbatchb.append(x[1])
            ybatch.append(y)
        
        xbatch = [torch.stack(xbatcha).to(device), [xb.to(device) for xb in xbatchb]]
        ybatch = torch.stack(ybatch).to(device)
        return xbatch, ybatch
   
    # split data
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
    sorted_sites = metadata.site.sort_values('SOIL_ORGANIC_CARBON')['SITE_ID']
    train_sites = ['TADHM', 'RDMER', 'GLENS', 'SOURH']
    val_sites = list(np.unique(list(np.setdiff1d(sorted_sites, train_sites)[::3]) + ['HARWD', 'STIPS']))
    test_sites = list(np.setdiff1d(sorted_sites, val_sites+train_sites)[::5]) + ['GLENW']
    train_sites = list(np.setdiff1d(sorted_sites, val_sites+test_sites))

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
        
    static_data = metadata.site.set_index('SITE_ID')[['SOIL_ORGANIC_CARBON', 'SOIL_ORGANIC_CARBON_SD']]
    # find mean relative error on SOC measurement to fix NaNs in error col
    meanrelerr = (static_data['SOIL_ORGANIC_CARBON'] / static_data['SOIL_ORGANIC_CARBON_SD']).mean()
    missing_sd_sites = static_data.index[static_data['SOIL_ORGANIC_CARBON_SD'].isna()]
    for sid in missing_sd_sites:
        approx_sd = static_data.loc[sid]['SOIL_ORGANIC_CARBON'] / meanrelerr
        static_data.loc[sid,'SOIL_ORGANIC_CARBON_SD'] = approx_sd
    twohour_feats = ['PRECIP']
    year_feats = []
    n_chunks = 20
    chunk_length = 7 * 12 # ~week of two-hourly data
    
    train_dg = ChunkGenerator(train_dat['weekly_data'], train_dat['twohourly_data'],
                              static_data.loc[train_sites], twohour_feats, year_feats,
                              n_chunks, chunk_length)
    val_dg = ChunkGenerator(val_dat['weekly_data'], val_dat['twohourly_data'],
                            static_data.loc[val_sites], twohour_feats, year_feats,
                            n_chunks, chunk_length)   
    test_dg = ChunkGenerator(test_dat['weekly_data'], test_dat['twohourly_data'],
                            static_data.loc[test_sites], twohour_feats, year_feats,
                            n_chunks, chunk_length)   
    
    def gaussian_KL(ypred, ytrue):
        EPS = 1e-12
        mu_2 = ypred[:,0]
        sigma_2 = torch.exp(ypred[:,1])
        mu_1 = ytrue[:,0]
        sigma_1 = ytrue[:,1]
        t1 = torch.log(sigma_2/sigma_1 + EPS)
        t2 = sigma_1 * sigma_1 + (mu_1 - mu_2)*(mu_1 - mu_2) / (2*sigma_2 * sigma_2) - 0.5
        return torch.mean(t1+t2, dim=(0,), keepdim=False)
    
    ## make model, optimizer and loss function
    from soil_moisture.architectures.multiscale_carbon_predict import MultiScaleCPredict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    d_chunks = len(twohour_feats) + 1
    d_year = len(year_feats) + 1
    d_model = 128
    d_out = 128
    num_chunks = n_chunks
    h_att = 4
    N_att = 2
    dropout = 0.1
    
    model = MultiScaleCPredict(d_chunks, d_year, d_model, d_out,
                               num_chunks, h_att, N_att, dropout)
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = gaussian_KL

    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(model)
    print(f'Number of trainable parameters = {params}')

    from soil_moisture.training_funcs import (make_train_step, make_val_step,
                                              save_checkpoint, prepare_run)

    def fit(model, optimizer, loss_fn, train_dg, val_dg, batchsize, 
            train_len=50, val_len=15, max_epochs=5, outdir='./logs/', 
            checkpoint=None, device=None):
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
                xb, yb = get_batch(train_dg, batchsize, device=device)
                loss = train_step(xb, yb)            
                epoch_loss += loss
                kbar.update(i, values=[("loss", loss)])
            losses.append(epoch_loss / max(float(train_len),1.))
            epoch_loss = 0
            with torch.no_grad():
                for i in range(val_len):                
                    xb, yb = get_batch(val_dg, batchsize, device=device)
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
        
        
    modelname = f'carbon_multiscale'
    model_outdir = f'{proj_dir}/logs/{modelname}/'
    train_len = 25
    val_len = 12
    batchsize = 8
    checkpoint = None
    model, losses, val_losses = fit(model, optimizer, loss_fn, train_dg, val_dg, batchsize, 
                                    train_len=train_len, val_len=val_len, max_epochs=5,
                                    outdir=model_outdir, checkpoint=checkpoint, device=device)     
        
        
        
        
        
        
        
        
    
    

