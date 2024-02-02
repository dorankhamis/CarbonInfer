import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
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
from utils import resample_sd, gaussian_loglikelihood, score

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def gamma_from_mu_sigma(mu, sigma):
    k = mu*mu / (sigma*sigma) # shape
    theta = (sigma*sigma) / mu # scale
    return gamma(k, scale = theta)

def plot_site_preds(j_obj, preds, itr=0):
    for sid in preds.site_id.unique():
        pred = preds.loc[preds.site_id==sid] 
        
        ## save a small plot of the time series and a scatter for each iteration?
        savename = f'{j_obj.site_env_vars[sid]["$OUTDIR"]}/timeseries_iter_{itr}.png'
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(pred.index, pred['NEE'], '-')
        ax.fill_between(
            x=pred.index,
            y1=pred['NEE'] - pred['NEE_sd'],
            y2=pred['NEE'] + pred['NEE_sd'],
            alpha=0.4
        )
        ax.plot(pred.index, pred['nee_gb'], '-')
        plt.title(sid)
        plt.savefig(savename, bbox_inches='tight')
        #plt.show()
        plt.close()
        
        savename = f'{j_obj.site_env_vars[sid]["$OUTDIR"]}/scatter_iter_{itr}.png'
        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot(pred['NEE'], pred['nee_gb'], 'o', markersize=2.5)
        ax.errorbar(pred['NEE'], pred['nee_gb'], yerr=None, xerr=pred['NEE_sd'],
                 fmt='none', ecolor=None, elinewidth=None, capsize=None, barsabove=False,
                 lolims=False, uplims=False, xlolims=False, xuplims=False,
                 errorevery=1, capthick=None, alpha=0.4)
        xx = np.mean(ax.get_xlim())
        ax.axline((xx,xx), slope=1, linestyle='--', color='k')
        
        thisscore = score(pred['nee_gb'], pred['NEE'])
        newlab = f'R^2 = {np.around(thisscore, 3)}'
        xx = np.quantile(ax.get_xlim(), 0.05)
        yy = np.quantile(ax.get_ylim(), 0.925)
        ax.text(xx, yy, newlab, fontsize = 11)
        plt.title(sid)
        plt.savefig(savename, bbox_inches='tight')
        #plt.show()
        plt.close()
    
    ## and output the bit of the timeseries that overlaps with the observed data
    ## for each iter parameters, for easy access later?
    savename = j_obj.output_path + f'/output_iter_{itr}.parquet'
    preds.to_parquet(savename)

def calculate_site_loglik(j_obj, sid, resample_str='1D'):
    pred = xr.load_dataset(f'{j_obj.site_env_vars[sid]["$OUTDIR"]}/{j_obj.site_env_vars[sid]["$RUNID"]}.Daily_gb.nc')
    data = pd.read_csv(es.site_info.loc[sid].observation_data_path)
    
    pred = (pred[['gpp_gb', 'resp_p_gb', 'resp_s_gb']].to_dataframe()
        .assign(nee_gb = lambda x: x.resp_p_gb + x.resp_s_gb - x.gpp_gb) # check signs
        .reset_index()
        .drop(['latitude', 'longitude', 'y', 'x'], axis=1)
        .assign(DATE_TIME = lambda x: pd.to_datetime(x.time, utc=True))
    )
    
    pred = (data.assign(DATE_TIME = lambda x: pd.to_datetime(x.DATE_TIME, utc=True))
        .merge(pred, on='DATE_TIME', how='left')
        .set_index('DATE_TIME')
        .drop(['time', 'gpp_gb', 'resp_p_gb', 'resp_s_gb'], axis=1)
        .dropna()
    )
    
    pred = pred / es.C_kg_p_mmco2
    pred = (pred[['NEE', 'nee_gb']].resample(resample_str).mean().dropna()
        .merge(pred[['NEE_sd']].resample(resample_str).apply(resample_sd),
               on='DATE_TIME', how='left') # NEE_sd must resample as sqrt(sum(sd^2) / n)
    )

    return gaussian_loglikelihood(pred.nee_gb, pred.NEE, pred.NEE_sd).sum(), pred

def calc_run_loglik(j_obj, resample_str='1D'):
    loglik = 0
    preds = pd.DataFrame()
    for sid in j_obj.site_nmls.keys():
        ll, pred = calculate_site_loglik(j_obj, sid, resample_str=resample_str)
        loglik += ll
        preds = pd.concat([preds, pred.assign(site_id = sid)], axis=0)
    return loglik, preds

def calc_run_posterior(j_obj, prior_dists, params, PFT_RUN, opt_params, resample_str='1D'):
    loglikelihood, preds = calc_run_loglik(j_obj, resample_str=resample_str)
    logprior = np.sum([prior_dists[k].logpdf(params[k][PFT_RUN]) for k in opt_params])
    logposterior = loglikelihood + logprior
    return logposterior, loglikelihood, logprior, preds
            
def create_prior_dists(opt_params, positive_only, PFT_RUN):
    prior_params = {}
    prior_dists = {}
    for i, k in enumerate(opt_params):
        prior_params[k] = es.default_params[k][PFT_RUN], es.default_stderr[k][PFT_RUN]
        if positive_only[i]:
            prior_dists[k] = gamma_from_mu_sigma(prior_params[k][0], prior_params[k][1])
        else:
            prior_dists[k] = norm(loc=prior_params[k][0], scale=prior_params[k][1])
    return prior_dists

def sample_priors(j_objs, prior_dists, opt_params, PFT_RUN):
    xx = es.opt_namelist_struct[0]
    yy = es.opt_namelist_struct[1]
    nchains = len(j_objs)
    params = [{k:j_objs[i].base_nml_dic[xx][yy][k] for k in opt_params} for i in range(nchains)]
    for i in range(nchains):
        for ii, k in enumerate(opt_params):
            params[i][k][PFT_RUN] = prior_dists[k].rvs()
    return params

def propose_params(chain_dfs, cur_params, j_objs, prior_dists,
                   opt_params, PFT_RUN, stepsize_as_frac_prior_width=0.1,
                   update_type='de', gamma=0.4):
    '''
    update_type = ['randomwalk', 'de', 'cov']
    randomwalk: sample from a diagonal multivariate gaussian
    de: differential evolution, x_p = x_i + gamma (x_R1 − x_R2) + e
        where x_p is the proposed params, x_i is the current params,
        x_R1 and x_R2 are current params sampled from the other chains 
        without replacement, gamma is a constant of order 0.4 -- 1,
        and e ~ N(0, b)^d for b small
    cov: approximate covariance matrix from parameter history within chain,
        then sample from a multivariate normal using this covariance matrix.
    '''
    xx = es.opt_namelist_struct[0]
    yy = es.opt_namelist_struct[1]
    nchains = len(j_objs)
    params = copy.deepcopy(cur_params)
    for i in range(nchains):
        if update_type=='cov':
            # use covariance matrix between parameters
            # to sample multivariate-aware proposal
            if len(opt_params)>1:
                chain_cov = chain_dfs[i][opt_params].cov()
            else:
                chain_cov = np.array([0])
            if not (np.all(chain_cov.isna()) or np.all(chain_cov == 0)):
                proposal =  np.random.multivariate_normal(
                    chain_dfs[i].iloc[-1:][opt_params].values.flatten(),
                    chain_cov.values
                )
                for ii, k in enumerate(opt_params):
                    params[i][k][PFT_RUN] = proposal[ii]
            else:
                for k in opt_params:
                    params[i][k][PFT_RUN] = gaussian_proposal(
                        cur_params[i][k][PFT_RUN],
                        math.sqrt(prior_dists[k].var()) * stepsize_as_frac_prior_width
                    )
        elif update_type=='de' and nchains>4:
            # run DE-MC
            r_1 = i
            r_2 = i
            while r_1==i:
                r_1 = np.random.randint(nchains)
            while r_2==i or r_2==r_1:
                r_2 = np.random.randint(nchains)
            for k in opt_params:
                params[i][k][PFT_RUN] = (cur_params[i][k][PFT_RUN] + 
                    gamma * (cur_params[r_1][k][PFT_RUN] - cur_params[r_2][k][PFT_RUN]) + 
                    gaussian_proposal(0, math.sqrt(prior_dists[k].var()) * 0.01)
                )
        else: # random walk
            for k in opt_params:
                params[i][k][PFT_RUN] = gaussian_proposal(
                    cur_params[i][k][PFT_RUN],
                    math.sqrt(prior_dists[k].var()) * stepsize_as_frac_prior_width
                )
    return params

def gaussian_proposal(mu, sigma):
    return mu + np.random.normal(scale=sigma)

def update_chain(chain_dfs, cur_params, new_params, opt_params,
                 cur_L, new_L, cur_iter, PFT_RUN):
    accept_mask = []
    for i in range(len(chain_dfs)):
        # do accept/reject
        logalpha = new_L[i][0] - cur_L[i][0]
        u01 = np.random.uniform()
        if np.log(u01) <= logalpha:
            # accept proposal
            cur_params[i] = new_params[i].copy()
            cur_L[i] = new_L[i]
            accept_mask.append(1)
        else:
            # reject proposal
            accept_mask.append(0)
            
    # update chain
    update_df = [pd.concat([chain_dfs[i], pd.DataFrame(
        {'iter':cur_iter, 'Lpost':cur_L[i][0], 'Llik':cur_L[i][1], 'Lprior':cur_L[i][2]} | 
        {k:cur_params[i][k][PFT_RUN] for k in opt_params}, index=[cur_iter]
    )], axis=0) for i in range(len(chain_dfs))]
    
    return update_df, cur_params, cur_L, accept_mask

def run_sites_parallel(j_obj, ncpu=1):
    # chunk site list into ncpu length chunks
    chunk_size = ncpu
    n_chunks = len(j_obj.site_list) // chunk_size
    leftover = len(j_obj.site_list) - chunk_size * n_chunks
    chunk_sizes = np.repeat(chunk_size, n_chunks)
    if leftover>0:
        chunk_sizes = np.hstack([chunk_sizes, leftover])
    
    csum = np.hstack([0, np.cumsum(chunk_sizes)])
    
    for chunk_num in range(len(chunk_sizes)):
        start_time = time.perf_counter()
        processes = []
        
        # takes subset of site list to work with, length <= ncpu
        sid_slice = j_obj.site_list[csum[chunk_num]:csum[chunk_num+1]]

        # Creates chunk_sizes[chunk_num] processes then starts them
        for i in range(chunk_sizes[chunk_num]):
            print(f'Running {sid_slice[i]} as task {i}')
            p = multiprocessing.Process(target = j_obj.run_jules, args=(sid_slice[i],))
            p.start()
            processes.append(p)
        
        # Joins all the processes 
        for p in processes:
            p.join()
     
        finish_time = time.perf_counter()
        print(f"Finished site slice in {(finish_time-start_time)/60.} minutes")

if __name__=="__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("nchains", help = "number of chains to run")
        parser.add_argument("niters", help = "number of iterations to run")
        parser.add_argument("ncpu", help = "number of cpu cores to call on")
        parser.add_argument("pft", help = "index of PFT to run, in order ['BDT', 'NET', 'C3G', 'DSh', 'C3Cr']")
        args = parser.parse_args()
        
        nchains = int(args.nchains)
        niters = int(args.niters)
        ncpu = int(args.ncpu)
        PFT_RUN = int(args.pft)
    except:
        print('Not running as script, using default params')              
        ## define run vars
        nchains = 1
        niters = 2
        ncpu = 1
        PFT_RUN = 3
    
    inference_dir = os.getcwd() + '/inference/'
    Path(inference_dir).mkdir(exist_ok=True, parents=True)
    
    ## create a Jules object for each of N chains
    j_objs = [Jules(run_type='normal', prepare_nmls=False) for i in range(nchains)]

    ## choose PFT type to run with and cut down the site list
    pft_order = ['BDT', 'NET', 'C3G', 'DSh', 'C3Cr']
    pft_use = pft_order[PFT_RUN]
    if pft_use=='BDT':
        landcover_use = ['Broadleaf tree']
    if pft_use=='NET':
        landcover_use = ['Needleleaf tree']
    if pft_use=='C3G':
        # potentially split bog into separate PFT using grass-like params
        landcover_use = ['Grassland', 'Fen', 'Bog']
    if pft_use=='DSh':
        landcover_use = ['Shrubland']
    if pft_use=='C3Cr':
        landcover_use = ['Agriculture']

    es.site_info = es.site_info[es.site_info['landcover_simp'].isin(landcover_use)]

    ## subset the opt parameter dicts to the selected PFT and define prior dists
    opt_params    = ['tlow_io', 'tupp_io', 'tleaf_of_io', 'gpp_st_io']
    positive_only = [False    , False    , True         , True       ]
    prior_dists = create_prior_dists(opt_params, positive_only, PFT_RUN)

    # then can call:
    #   - prior_dists[k].logpdf(x) for x a proposed parameter value
    #   - prior_dists[k].rvs() for a sample from the prior

    if False:
        ## FOR TESTING
        ## cut down to single site
        es.site_info = es.site_info.loc[['Conwy'],:]

    ## define randomised folder to work in for each chain
    ## (to save edited namelists temporary jules output)
    chain_ids = [f'chain_{id_generator()}' for i in range(nchains)]
    for i, j in enumerate(j_objs):
        j.site_nml_path = es.nml_directory + f'/{chain_ids[i]}/'
        j.output_path = es.output_directory + f'/{chain_ids[i]}/'
        j.site_list = list(es.site_info.index)
    
    ## output run metadata to identify chain output folders
    (pd.DataFrame({'chain_id':chain_ids,
                   'pft_num':PFT_RUN,
                   'pft_id':pft_use,
                   'run_date':pd.to_datetime(datetime.datetime.today()),
                   'site_id':';'.join(list(es.site_info.index))})
        .to_csv('./run_metadata.csv', index=False, mode='a',
                header=not os.path.exists('./run_metadata.csv'))
    )

    ## prepare namelists now we have altered the working folders
    [j.prepare_site_nmls() for j in j_objs]

    ## sample from priors as initial conditions
    cur_params = sample_priors(j_objs, prior_dists, opt_params, PFT_RUN)

    ## update namelists
    [j.update_nmls(cur_params[i]) for i, j in enumerate(j_objs)]

    ## run jules with initial parameter set at all sites
    #[j.run_all_sites(print_proc=True) for j in j_objs]
    [run_sites_parallel(j, ncpu=ncpu) for j in j_objs]

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
            stepsize_as_frac_prior_width=0.15
        )
         
        ## update run namelists with proposed params
        [j.update_nmls(new_params[i]) for i, j in enumerate(j_objs)]

        ## run jules with new parameter set at all sites
        #[j.run_all_sites(print_proc=False) for j in j_objs]
        [run_sites_parallel(j, ncpu=ncpu) for j in j_objs]
        
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
                chain_dfs[i].to_csv(inference_dir + f'/{chain_ids[i]}_pft_{PFT_RUN}.csv', index=False)
                
    ## finally, save chain
    for i, j in enumerate(j_objs):
        chain_dfs[i].to_csv(inference_dir + f'/{chain_ids[i]}_pft_{PFT_RUN}.csv', index=False)
        
    # aceptance probability
    # dat = pd.read_csv("./inference/chain_TZA6O2_pft_2.csv")
    # (dat.diff().iloc[1:].assign(accept = lambda x: x.tlow_io!=0).sum() / dat.shape[0]).accept
