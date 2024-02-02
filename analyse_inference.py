import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

import experiment_setup as es

run_meta = pd.read_csv("run_metadata.csv")

run_id = 'single_site_single_param'
run_meta = run_meta.loc[run_meta.run_id==run_id]

pft_order = ['BDT', 'NET', 'C3G', 'DSh', 'C3Cr']
pfts_run = run_meta.pft_num.unique()

param_order = ['tlow_io', 'tupp_io', 'gpp_st_io']
params_run = run_meta.opt_param.unique()

for pftnum in pfts_run:    
    pft_name = pft_order[pftnum]
    pft_meta = run_meta.loc[run_meta.pft_num==pftnum]
    for param_name in params_run:
        param_pft_meta = pft_meta.loc[pft_meta.opt_param==param_name]
        uniq_sites = param_pft_meta.site_id.unique()
        for sid in uniq_sites:
            print('')
            print(f'pft num: {pftnum}')
            print(f'pft name: {pft_name}')
            print(f'param: {param_name}')
            print(f'site id: {sid}')
            
            site_meta = param_pft_meta.loc[param_pft_meta.site_id==sid]
            chains = pd.DataFrame()
            for chain_id in site_meta.chain_id:
                chain_dat = pd.read_csv(f'./inference/{run_id}/{chain_id}_pft_{pftnum}_site_{sid}_param_{param_name}.csv')
                chains = pd.concat([chains, chain_dat.assign(chain = chain_id)], axis=0)
                
                # chain acceptance ratio
                print(f'{chain_id} acceptance ratio:')
                print((chain_dat.diff().iloc[1:].assign(accept = lambda x: x.tlow_io!=0).accept.sum() / chain_dat.shape[0]))

            #sns.lineplot(x='iter', y='Lpost', hue='chain', data=chains)
            #plt.title(f'Log likelihood, {pft_name}, {sid}')
            #plt.show()
            
            # burn in
            burn = 50
            chains = chains.loc[chains.iter > burn]
            
            # thin
            thin_by = 2
            chains = chains.loc[chains.iter % thin_by == 0]
            
            fig, ax = plt.subplots()
            sns.lineplot(x='iter', y=param_name, hue='chain', data=chains, ax=ax)
            ax.set_title(f'{param_name}, {pft_name}, {sid}')
            plt.show()

            fig, ax = plt.subplots()
            sns.histplot(x=param_name, data=chains, ax=ax)
            ax.set_title(f'{param_name}, {pft_name}, {sid}')
            plt.show()
            
            
# outdir = f'./output/{chain_id}/{sid}/'
# pred = xr.load_dataset(outdir + f'/jrun_{sid}.Daily_gb.nc')
# data = pd.read_csv(es.site_info.loc[sid].observation_data_path)

# pred = (pred[['gpp_gb', 'resp_p_gb', 'resp_s_gb']].to_dataframe()
    # .assign(nee_gb = lambda x: x.resp_p_gb + x.resp_s_gb - x.gpp_gb) # check signs
    # .reset_index()
    # .drop(['latitude', 'longitude', 'y', 'x'], axis=1)
    # .assign(DATE_TIME = lambda x: pd.to_datetime(x.time, utc=True))
# )

# pred = (data.assign(DATE_TIME = lambda x: pd.to_datetime(x.DATE_TIME, utc=True))
    # .merge(pred, on='DATE_TIME', how='left')
    # .set_index('DATE_TIME')
    # .drop(['time', 'gpp_gb', 'resp_p_gb', 'resp_s_gb'], axis=1)
    # .dropna()
# )

# pred = pred / es.C_kg_p_mmco2
# pred = (pred[['NEE', 'nee_gb']].resample(resample_str).mean().dropna()
    # .merge(pred[['NEE_sd']].resample(resample_str).apply(resample_sd),
           # on='DATE_TIME', how='left') # NEE_sd must resample as sqrt(sum(sd^2) / n)
# )
