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

def run_chains_parallel(j_objs, ncpu=1):
    # chunk site list into ncpu length chunks
    chunk_size = ncpu
    n_chunks = len(j_objs) // chunk_size
    leftover = len(j_objs) - chunk_size * n_chunks
    chunk_sizes = np.repeat(chunk_size, n_chunks)
    if leftover>0:
        chunk_sizes = np.hstack([chunk_sizes, leftover])
    
    csum = np.hstack([0, np.cumsum(chunk_sizes)])
    chain_nums = np.arange(len(j_objs))
    
    for chunk_num in range(len(chunk_sizes)):
        start_time = time.perf_counter()
        processes = []
        
        # takes subset of chains to work with, length <= ncpu
        chain_slice = chain_nums[csum[chunk_num]:csum[chunk_num+1]]

        # Creates chunk_sizes[chunk_num] processes then starts them
        for i in range(chunk_sizes[chunk_num]):
            print(f'Running chain {chain_slice[i]} as task {i}')
            p = multiprocessing.Process(target = j_objs[chain_slice[i]].run_all_sites)
            p.start()
            processes.append(p)
        
        # Joins all the processes 
        for p in processes:
            p.join()
     
        finish_time = time.perf_counter()
        print(f"Finished chain slice in {(finish_time-start_time)/60.} minutes")

if __name__=="__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("nchains", help = "number of chains to run")
        parser.add_argument("niters", help = "number of iterations to run")
        parser.add_argument("ncpu", help = "number of cpu cores to call on")
        parser.add_argument("site_num", help = "number of site to run")
        parser.add_argument("continue_chains", help = "1: True, 0: Start new chains")
        
        args = parser.parse_args()
        
        nchains = int(args.nchains)
        niters = int(args.niters)
        ncpu = int(args.ncpu)
        site_num = int(args.site_num)
        continue_chains = bool(int(args.continue_chains))
    except:
        print('Not running as script, using default params')              
        ## define run vars
        nchains = 8
        niters = 2
        ncpu = nchains
        site_num = 2
        continue_chains = True
    
    RUN_ID = 'single_site'
    inference_dir = os.getcwd() + f'/inference/{RUN_ID}/'
    Path(inference_dir).mkdir(exist_ok=True, parents=True)
    
    runmeta_out = es.output_directory + f'/{RUN_ID}/run_metadata.csv'
    
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

    ## subset the opt parameter dicts to the selected PFT and define prior dists
    opt_params    = ['tlow_io', 'tupp_io']#, 'gpp_st_io'] # 'tleaf_of_io'
    positive_only = [False    , False] #    , True       ]
    prior_dists = create_prior_dists(opt_params, positive_only, PFT_RUN)
    
    # run strings for site and opt params
    site_id_string = ';'.join(list(site_info.index))
    opt_param_string = ';'.join(opt_params)

    # then can call:
    #   - prior_dists[k].logpdf(x) for x a proposed parameter value
    #   - prior_dists[k].rvs() for a sample from the prior

    if continue_chains is True:
        run_meta = pd.read_csv(runmeta_out)
        this_run = run_meta.loc[
            (run_meta.pft_num==PFT_RUN) & 
            (run_meta.site_id==site_id_string) &
            (run_meta.opt_param==opt_param_string) &
            (run_meta.run_id==RUN_ID)
        ]
        if this_run.shape[0] > 0:
            nchains = this_run.shape[0]
            ncpu = nchains
            chain_ids = list(this_run.chain_id)
        else:
            continue_chains = False
            print("Can't find chains to continue, starting afresh.")
    
    if continue_chains is False:
        ## define randomised folder to work in for each chain
        ## (to save edited namelists temporary jules output)
        chain_ids = [f'chain_{id_generator()}' for i in range(nchains)]
        
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
    [j.prepare_site_nmls() for j in j_objs]

    if continue_chains is False:
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
        
        init_iter = 1        
    else:
        ## grab latest iteration info from saved chains
        chain_dfs = []
        cur_L = []
        cur_params = []
        cur_preds = []
        for jj, ch in enumerate(chain_ids):
            # load the chain
            curr_inf = pd.read_csv(inference_dir + f'/{ch}_pft_{PFT_RUN}_site_{site_id_string}.csv')
            chain_dfs.append(curr_inf.copy())
            # subset to last iteration
            last_iter_dat = curr_inf.iloc[-1]
            # append likelihoods
            cur_L.append((last_iter_dat.Lpost, last_iter_dat.Llik, last_iter_dat.Lprior))
            # and parameter values
            dflt_params = {k:es.default_params[k].copy() for k in opt_params}
            for k in opt_params:
                dflt_params[k][PFT_RUN] = last_iter_dat[k]
            cur_params.append(dflt_params.copy())
            del(dflt_params)
            # preds
            last_changed_iter = np.where((curr_inf.diff()==0)['Lpost']==False)[0][-1]
            cur_preds.append(
                pd.read_parquet(j_objs[jj].output_path + f'/output_iter_{last_changed_iter}.parquet')
            )            
            del(curr_inf)
            # define our next iter to run
            init_iter = int(last_iter_dat.iter + 1)

    ## start MCMC loop:    
    for cur_iter in range(init_iter, niters+1):
        print(f'Starting iteration {cur_iter}')
        ## create new parameters set using Metropolis-Hastings proposal        
        new_params = propose_params(
            chain_dfs,
            cur_params,
            j_objs,
            prior_dists,
            opt_params,
            PFT_RUN,
            stepsize_as_frac_prior_width=0.15
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
        
        ## save chain
        for i, j in enumerate(j_objs):
            chain_dfs[i].to_csv(inference_dir + f'/{chain_ids[i]}_pft_{PFT_RUN}_site_{site_id_string}.csv', index=False)
                
    ## finally, save chain
    for i, j in enumerate(j_objs):
        chain_dfs[i].to_csv(inference_dir + f'/{chain_ids[i]}_pft_{PFT_RUN}_site_{site_id_string}.csv', index=False)
        
    # aceptance probability
    # dat = pd.read_csv("./inference/chain_TZA6O2_pft_2.csv")
    # (dat.diff().iloc[1:].assign(accept = lambda x: x.tlow_io!=0).sum() / dat.shape[0]).accept
