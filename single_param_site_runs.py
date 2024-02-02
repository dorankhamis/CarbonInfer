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
        parser.add_argument("nchains", help = "number of chains to run")
        parser.add_argument("niters", help = "number of iterations to run")
        parser.add_argument("ncpu", help = "number of cpu cores to call on")
        parser.add_argument("site_num", help = "number of site to run")
        parser.add_argument("param_num", help = "number of param to optimize")
        
        args = parser.parse_args()
        
        nchains = int(args.nchains)
        niters = int(args.niters)
        ncpu = int(args.ncpu)
        site_num = int(args.site_num)
        par_num = int(args.param_num)
    except:
        print('Not running as script, using default params')              
        ## define run vars
        nchains = 2
        niters = 2
        ncpu = 2 # ==nchains for quickest running
        site_num = 0
        par_num = 0
    
    RUN_ID = 'single_site_single_param'
    inference_dir = os.getcwd() + f'/inference/{RUN_ID}/'
    Path(inference_dir).mkdir(exist_ok=True, parents=True)
    
    ## create a Jules object for each of N chains
    j_objs = [Jules(run_type='normal', prepare_nmls=False) for i in range(nchains)]

    ## subset site info to the single site site_num
    es.site_info = es.site_info.iloc[[site_num]]
    landcover_use = es.site_info.landcover_simp.values[0]

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

    ## subset the opt parameter dicts to the selected PFT and define prior dists
    all_params        = ['tlow_io', 'tupp_io']#, 'gpp_st_io'] # 'tleaf_of_io'
    all_positive_only = [False    , False]#    , True       ]
    # select single param to optimize
    opt_params = [all_params[par_num]]
    positive_only = [all_positive_only[par_num]]
    
    prior_dists = create_prior_dists(opt_params, positive_only, PFT_RUN)
    # then can call:
    #   - prior_dists[k].logpdf(x) for x a proposed parameter value
    #   - prior_dists[k].rvs() for a sample from the prior

    ## define randomised folder to work in for each chain
    ## (to save edited namelists temporary jules output)
    chain_ids = [f'chain_{id_generator()}' for i in range(nchains)]
    for i, j in enumerate(j_objs):
        j.site_nml_path = es.nml_directory + f'/{RUN_ID}/{chain_ids[i]}/'
        j.output_path = es.output_directory + f'/{RUN_ID}/{chain_ids[i]}/'
        j.site_list = list(es.site_info.index)
    
    ## output run metadata to identify chain output folders
    site_id_string = ';'.join(list(es.site_info.index))
    opt_param_string = ';'.join(opt_params)
    runmeta_out = es.output_directory + f'/{RUN_ID}/run_metadata.csv'
    (pd.DataFrame({'chain_id':chain_ids,
                   'pft_num':PFT_RUN,
                   'pft_id':pft_use,
                   'run_date':pd.to_datetime(datetime.datetime.today()),
                   'site_id':site_id_string,
                   'opt_param':opt_param_string,
                   'run_id':RUN_ID})
        .to_csv(runmeta_out, index=False, mode='a',
                header=not os.path.exists(runmeta_out))
    )

    ## prepare namelists now we have altered the working folders
    [j.prepare_site_nmls() for j in j_objs]

    ## sample from priors as initial conditions
    cur_params = sample_priors(j_objs, prior_dists, opt_params, PFT_RUN)

    ## update namelists
    [j.update_nmls(cur_params[i]) for i, j in enumerate(j_objs)]

    ## run jules with initial parameter set for all chains
    run_chains_parallel(j_objs, ncpu=ncpu)

    ## calculate loss against evaluation data
    cur_preds = []
    cur_L = []
    for i, j_obj in enumerate(j_objs):
        logposterior, loglikelihood, logprior, preds = calc_run_posterior(
            j_obj, prior_dists, cur_params[i], PFT_RUN, opt_params, resample_str='1D'
        )
        cur_L.append((logposterior, loglikelihood, logprior))
        cur_preds.append(preds)
    # cur_L = [calc_run_posterior(j_obj, prior_dists, cur_params[i], PFT_RUN,
                                # opt_params, resample_str='1D') for i, j_obj in enumerate(j_objs)]

    # output plots and site preds
    [plot_site_preds(j_obj, cur_preds[i], itr=0) for i, j_obj in enumerate(j_objs)]

    ## create chain dataframe for updates
    chain_dfs = [pd.DataFrame(
        {'iter':0, 'Lpost':cur_L[i][0], 'Llik':cur_L[i][1], 'Lprior':cur_L[i][2]} | 
        {k:cur_params[i][k][PFT_RUN] for k in opt_params}, index=[0]
    ) for i in range(nchains)]

    ## start MCMC loop:    
    for cur_iter in range(1, niters+1):
        print(f'Starting iteration {cur_iter}')
        ## create new parameters set using Metropolis-Hastings proposal        
        new_params = propose_params(
            chain_dfs,
            cur_params,
            j_objs,
            prior_dists,
            opt_params,
            PFT_RUN,
            stepsize_as_frac_prior_width=0.1,
            update_type='de',
            gamma=0.4
        )
         
        ## update run namelists with proposed params
        [j.update_nmls(new_params[i]) for i, j in enumerate(j_objs)]

        ## run jules with new parameter set for all chains
        run_chains_parallel(j_objs, ncpu=ncpu)
        
        ## calculate loss against evaluation data
        new_preds = []
        new_L = []
        for i, j_obj in enumerate(j_objs):
            logposterior, loglikelihood, logprior, preds = calc_run_posterior(
                j_obj, prior_dists, new_params[i], PFT_RUN, opt_params, resample_str='1D'
            )
            new_L.append((logposterior, loglikelihood, logprior))
            new_preds.append(preds)
        # new_L = [calc_run_posterior(j_obj, prior_dists, new_params[i], PFT_RUN,
                                    # opt_params, resample_str='1D') for i, j_obj in enumerate(j_objs)]
        
        ## accept or reject new parameter set and update chains
        chain_dfs, cur_params, cur_L, accept_mask = update_chain(
            chain_dfs,
            cur_params,
            new_params,
            opt_params,
            cur_L,
            new_L,
            cur_iter,
            PFT_RUN
        )
        
        for i in range(len(accept_mask)):
            if accept_mask[i]==1:
                # output plots and site preds
                cur_preds[i] = new_preds[i]
                plot_site_preds(j_objs[i], cur_preds[i], itr=cur_iter)
        
        if cur_iter % 5 == 0:
            ## save chain
            for i, j in enumerate(j_objs):
                chain_dfs[i].to_csv(inference_dir + f'/{chain_ids[i]}_pft_{PFT_RUN}_site_{site_id_string}_param_{opt_param_string}.csv', index=False)
                
    ## finally, save chain
    for i, j in enumerate(j_objs):
        chain_dfs[i].to_csv(inference_dir + f'/{chain_ids[i]}_pft_{PFT_RUN}_site_{site_id_string}_param_{opt_param_string}.csv', index=False)
        
    # aceptance probability
    # dat = pd.read_csv("./inference/chain_TZA6O2_pft_2.csv")
    # (dat.diff().iloc[1:].assign(accept = lambda x: x.tlow_io!=0).sum() / dat.shape[0]).accept
