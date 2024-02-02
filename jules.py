import subprocess
import os
import re
import sys
import f90nml
import configparser
import glob
import copy
import time
import numpy as np
import pandas as pd
from pathlib import Path

import experiment_setup as es

from nml_utils import (isnumber, fetch_val_from_config,
                       fill_site_specific_env_vars, replace_env_var,
                       clean_reformat_nml, create_site_ancil_and_config_files)

'''       
TODO:
    - get dummy runs working (DONE)
    - parse (with correct units) driving, ancillary and target data
        when flux data arrives (DONE)
    - extract output from site runs and define loss calculation with 
        respect to flux observations (DONE)
    - choose pft parameter list to optimize/sample, sensible priors/ranges (~Done?)
    - decide (based on run time?) whether to do full MCMC or
        something like MAP, or to go back to LAVENDAR and do 4DVAR (MCMC seems okay running parallel)
    - do runs!
'''

class Jules():
    def __init__(self, run_type='normal', prepare_nmls=True):
        '''
        run_type: ['normal', 'longspin']
        '''
        self.run_type = run_type        
        self.jules = es.model_exe
        self.base_nml_dic = self.read_nml(es.base_nml_dir)
        self.site_nmls = {}
        self.site_configs = {}
        self.site_env_vars = {}
        self.site_nml_patches = {}        
        self.site_list = es.site_list
        self.site_nml_path = es.nml_directory
        self.output_path = es.output_directory
        if prepare_nmls: self.prepare_site_nmls()

    def read_nml(self, nml_dir):
        ''' this is called in self.__init__() '''
        nml_dic = {}
        for f_nml in glob.glob(nml_dir + '/*.nml'):            
            nml_dic[f_nml.split('/')[-1][:-4]] = f90nml.read(f_nml)
        return nml_dic

    def prepare_site_nmls(self):
        ''' this is called in self.__init__() '''
        for sid in self.site_list:
            self.site_env_vars[sid] = fill_site_specific_env_vars(sid, self.run_type, self.output_path)
            create_site_ancil_and_config_files(sid, self.run_type, self.site_env_vars[sid])
            self.read_site_config(sid)
            self.fill_site_nml_patches_from_conf(self.base_nml_dic, sid,
                                                 self.site_configs[sid])
            self.create_site_nml_from_patches(sid)
            self.sub_in_env_vars_in_nmls(sid)
            self.site_nmls[sid] = clean_reformat_nml(self.site_nmls[sid])
    
    def read_site_config(self, sid):
        ''' this is called in self.prepare_site_nmls() '''
        # first load up site-specific config files
        conf_files = [es.config_directory + f'{t}_config-{sid}.conf' for t in ['ancil', 'drive']]
   
        ancil_config = configparser.ConfigParser(strict=False)
        ancil_config.read([f for f in conf_files if 'ancil' in f][0])
        
        drive_config = configparser.ConfigParser(strict=False)
        drive_config.read([f for f in conf_files if 'drive' in f][0])
        
        self.site_configs[sid] = {'ancil':ancil_config, 'drive':drive_config}

    def fill_site_nml_patches_from_conf(self, nml_dic, sid, configs,
                                        spiral_strings=[]):
        ''' this is called in self.prepare_site_nmls() '''
        if not (sid in self.site_nml_patches.keys()):
            self.site_nml_patches[sid] = []
        orig_strings = spiral_strings.copy()
        for ii in list(nml_dic.keys()):
            spiral_strings = orig_strings.copy()
            conf_var = False
            spiral_strings.append(ii)
            if type(nml_dic[ii])==f90nml.namelist.Namelist:
                self.fill_site_nml_patches_from_conf(
                    nml_dic[ii], sid, configs,
                    spiral_strings=spiral_strings)
            else:
                raw_vals = nml_dic[ii]
                new_vals = fetch_val_from_config(raw_vals, spiral_strings, configs)
                if not new_vals is None:
                    # fix integers
                    if (spiral_strings[-1] in es.int_vars and type(new_vals)==float):
                        new_vals = int(new_vals)
                    self.site_nml_patches[sid].append((spiral_strings, new_vals))  
    
    def create_site_nml_from_patches(self, sid):
        ''' this is called in self.prepare_site_nmls() '''
        self.site_nmls[sid] = copy.deepcopy(self.base_nml_dic)
        for elem in self.site_nml_patches[sid]:
            ids = elem[0]
            new_val = elem[1]
            if ids[2]=='file':                
                new_val = new_val.replace("$CYLC_SUITE_RUN_DIR", es.RSUITE_flux)                
            self.site_nmls[sid][ids[0]][ids[1]][ids[2]] = new_val

    def write_nml(self, sid, outpath):
        '''
        Function to write dictionary of stored nml data to nml files        
        '''        
        Path(outpath).mkdir(exist_ok=True, parents=True)
        for key in self.site_nmls[sid].keys():
            self.site_nmls[sid][key].write(outpath + key + '.nml', force=True)

    def write_all_site_nmls(self):        
        print("No output path specified in function definition")
        pass
        # for sid in self.site_list:
            # self.write_nml(sid)
            
    def sub_in_env_vars_in_nmls(self, sid):
        ''' this is called in self.prepare_site_nmls() '''        
        for i in self.site_nmls[sid].keys():
            for m in self.site_nmls[sid][i].keys():
                for k in self.site_nmls[sid][i][m].keys():
                    new_val = replace_env_var(self.site_nmls[sid][i][m][k],
                                              self.site_env_vars[sid])
                    if new_val=='.false.':
                        new_val = False
                    elif new_val=='.true.':
                        new_val = True
                    self.site_nmls[sid][i][m][k] = new_val
    
    def update_nmls(self, new_param_dict):
        ''' function to update the nmls with new parameters after,
        say, a metropolis-hastings step.
        The parameters to change are defined in
        experiment_setup.opt_params '''
        for sid in self.site_nmls.keys():
            for par in new_param_dict.keys():
                x = es.opt_namelist_struct[0]
                y = es.opt_namelist_struct[1]
                z = par
                self.site_nmls[sid][x][y][z] = new_param_dict[par]
                

    def run_jules(self, sid, print_proc=False):
        '''Run JULES in a subprocess. Check output for fatal errors.

        :return: stdout and stderr output from JULES model run.
        :rtype: tuple
        '''
        
        nml_path = self.site_nml_path + f'/{sid}_nmls/'
        Path(nml_path).mkdir(parents=True, exist_ok=True)
        for f in glob.glob(nml_path + '/*.nml'):
            os.remove(f)
            
        self.write_nml(sid, nml_path)
        cwd = os.getcwd()
        os.chdir(nml_path)
      
        cmd = []
        cmd.append(self.jules)
        
        a = time.time()
        if print_proc:
            p = subprocess.Popen(cmd, shell=False)
            p.communicate()
            out, err = [], []
        else:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = p.stdout.readlines()
            err = p.stderr.readlines()
        p.wait()
        b = time.time()
        print(f'{(b-a) / 60} minutes') # DE_Akm takes ~4 mins
        
        # catch "fatal" errors
        for line in out:
            if len(line.split()) == 0: continue
            if line.split()[0] == "[FATAL":
                print('>> sys.stderr, "*** runJules: caught fatal error in JULES run:"')
                print('>> sys.stderr, line,')
                sys.exit()

        # catch anything in stderr
        if len(err) > 0:
            for line in err:
                print('>> sys.stderr, "*** runJules: caught output on stderr in JULES run:"')
                print('>> sys.stderr, line,')
                # sys.exit()
        
        # move back to base
        os.chdir(cwd)
        return out, err
        
    def run_all_sites(self, print_proc=False):
        for sid in self.site_nmls.keys():
            out, err = self.run_jules(sid, print_proc=print_proc)
            
    # def calculate_site_loglik(self, sid, resample_str='1D'):
        # pred = xr.load_dataset(f'{self.site_env_vars[sid]["$OUTDIR"]}/{self.site_env_vars[sid]["$RUNID"]}.Daily_gb.nc')
        # data = pd.read_csv(es.site_info.loc[sid].observation_data_path)
        # pred_df = (pred[['gpp_gb', 'resp_p_gb', 'resp_s_gb']].to_dataframe()
            # .assign(nee_gb = lambda x: x.resp_p_gb + x.resp_s_gb - x.gpp_gb) # check signs
            # .reset_index()
            # .drop(['latitude', 'longitude', 'y', 'x'], axis=1)
            # .assign(DATE_TIME = lambda x: pd.to_datetime(x.time, utc=True))
        # )
        # jdat = (dat.assign(DATE_TIME = lambda x: pd.to_datetime(x.DATE_TIME, utc=True))
            # .merge(pred_df, on='DATE_TIME', how='left')
            # .set_index('DATE_TIME')
            # .drop(['time', 'gpp_gb', 'resp_p_gb', 'resp_s_gb'], axis=1)
        # )
        # del(data)
        # del(pred)
        # del(pred_df)
        
        # sc_dat = jdat / es.C_kg_p_mmco2
        # sc_dat = (sc_dat[['NEE', 'nee_gb']].resample(resample_str).mean().dropna()
            # .merge(sc_dat[['NEE_sd']].resample(resample_str).apply(resample_sd),
                   # on='DATE_TIME', how='left') # NEE_sd must resample as sqrt(sum(sd^2) / n)
        # )
        
        # if False:
            # import matplotlib.pyplot as plt

            # pltdat = (jdat[['NEE', 'nee_gb']].resample(resample_str).mean().dropna()
                # .merge(jdat[['NEE_sd']].resample(resample_str).apply(resample_sd),
                       # on='DATE_TIME', how='left') # NEE_sd must resample as sqrt(sum(sd^2) / n)
            # )
            
            # pltdat['NEE_upp'] = pltdat.NEE + pltdat.NEE_sd
            # pltdat['NEE_low'] = pltdat.NEE - pltdat.NEE_sd
            
            # fig, ax = plt.subplots()
            # plt.plot(pltdat.index, pltdat.NEE, '-', color='steelblue')
            # plt.plot(pltdat.index, pltdat.NEE_upp, '--', color='steelblue', alpha=0.5)
            # plt.plot(pltdat.index, pltdat.NEE_low, '--', color='steelblue', alpha=0.5)
            # plt.plot(pltdat.index, pltdat.nee_gb, '-', color='peru')
            # plt.show()        
            
            # fig, ax = plt.subplots()
            # ax.plot(pltdat.nee_gb, pltdat.NEE, 'o', color='steelblue')
            # ax.errorbar(pltdat.nee_gb, pltdat.NEE, yerr=pltdat.NEE_sd,
                        # fmt='none', color='steelblue', alpha=0.5)
            # xx = np.mean(ax.get_xlim())
            # ax.axline((xx,xx), slope=1, linestyle='--', color='k')
            # plt.show()
        
        # return gaussian_loglikelihood(sc_dat.nee_gb, sc_dat.NEE, sc_dat.NEE_sd).sum()

    # def calc_run_loglik(self, resample_str='1D'):
        # loglik = 0
        # for sid in self.site_nmls.keys():
            # loglik += self.calculate_site_loglik(sid, resample_str=resample_str)
        # return loglik
