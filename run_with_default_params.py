import numpy as np
import pandas as pd
import xarray as xr
import os
import pickle
import string
import random
import math
import copy
import datetime
import time
import multiprocessing
import argparse

from pathlib import Path
from scipy.stats import gamma, norm

import experiment_setup as es
from jules import Jules
from utils import resample_sd, gaussian_loglikelihood
from inference import *
from single_site_runs import run_chains_parallel

if __name__=="__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("site_num", help = "number of site to run")        
        args = parser.parse_args()        
        site_num = int(args.site_num)
        
    except:
        print('Not running as script, using default params')              
        ## define run vars
        site_num = 2
    
    nchains = 1        
    ncpu = nchains
    
    RUN_ID = 'default_params'
    inference_dir = os.getcwd() + f'/inference/{RUN_ID}/'
    Path(inference_dir).mkdir(exist_ok=True, parents=True)
    
    runout_dir = es.output_directory + f'/{RUN_ID}'
    Path(runout_dir).mkdir(exist_ok=True, parents=True)
    runmeta_out = runout_dir + '/run_metadata.csv'
    
    ## create a Jules object for each of N chains
    j_objs = [Jules(run_type='normal', prepare_nmls=False) for i in range(nchains)]

    ## subset site info to the single site site_num
    site_info = es.site_info.iloc[[site_num]].copy()
    landcover_use = site_info.landcover_simp.values[0]

    ## choose PFT type to run with from the land cover string
    pft_order = ['BDT', 'NET', 'C3G', 'DSh', 'C3Cr']

    if landcover_use in ['Broadleaf tree']:
        pft_use = 'BDT'
    if landcover_use in ['Needleleaf tree']:
        pft_use = 'NET'
    if landcover_use in ['Grassland', 'Fen', 'Bog']:
        pft_use = 'C3G'
    if landcover_use in ['Shrubland']:
        pft_use = 'DSh'
    if landcover_use in ['Agriculture']:
        pft_use = 'C3Cr'
    
    PFT_RUN = pft_order.index(pft_use)

    ## not optimizing anything here
    opt_params    = []
    positive_only = []
    
    # run strings for site and opt params
    site_id_string = ';'.join(list(site_info.index))
    opt_param_string = ';'.join(opt_params)
    
    ## define randomised folder to work in for each chain
    ## (to save edited namelists temporary jules output)
    chain_ids = [f'{site_id_string}' for i in range(nchains)]
    
    ## output run metadata to identify chain output folders
    (pd.DataFrame({'chain_id':chain_ids,
                   'pft_num':PFT_RUN,
                   'pft_id':pft_use,
                   'run_date':pd.to_datetime(datetime.datetime.today()),
                   'site_id':site_id_string,
                   'opt_param':opt_param_string,
                   'run_id':RUN_ID
                   })
        .to_csv(runmeta_out, index=False, mode='a',
                header=not os.path.exists(runmeta_out))
    )
    
    for i, j in enumerate(j_objs):
        j.site_nml_path = es.nml_directory + f'/{RUN_ID}/{chain_ids[i]}/'
        j.output_path = es.output_directory + f'/{RUN_ID}/{chain_ids[i]}/'
        j.site_list = list(site_info.index)
    
    ## prepare namelists now we have altered the working folders
    ## default params are already loaded here
    [j.prepare_site_nmls() for j in j_objs]
    
    ## run jules with initial parameter set for all chains
    run_chains_parallel(j_objs, ncpu=ncpu)

    ## calculate loss against evaluation data
    cur_preds = []    
    for i, j_obj in enumerate(j_objs):
        loglikelihood, preds = calc_run_loglik(j_obj, resample_str='1D')
        cur_preds.append(preds)

    # output plots and site preds
    [plot_site_preds(j_obj, cur_preds[i], itr=0) for i, j_obj in enumerate(j_objs)]
