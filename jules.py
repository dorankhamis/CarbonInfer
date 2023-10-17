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
from pathlib import Path

import experiment_setup as es
from nml_utils import (isnumber, fetch_val_from_config,
                       fill_site_specific_env_vars, replace_env_var,
                       clean_reformat_nml, create_site_ancil_and_config_files)

'''       
TODO:
    - get dummy runs working
    - parse (with correct units) driving, ancillary and target data
        when flux data arrives
    - extract output from site runs and define loss calculation with 
        respect to flux observations
    - choose pft parameter list to optimize/sample, sensible priors/ranges
    - decide (based on run time?) whether to do full MCMC or
        something like MAP
    - find out about the initial carbon state: will it have 4 pools??
    - do runs!
'''

class Jules():
    def __init__(self):
        self.nml_dir = es.nml_directory
        self.jules = es.model_exe
        self.base_nml_dic = self.read_nml(es.base_nml_dir)
        self.site_nmls = {}
        self.site_configs = {}
        self.site_env_vars = {}
        self.site_nml_patches = {}
        self.site_list = ['UK_Rdm', 'DE_Akm']
        self.prepare_site_nmls()

    def read_nml(self, nml_dir, fix_missing=False):
        ''' this is called in self.__init__() '''
        nml_dic = {}
        for f_nml in glob.glob(nml_dir + '/*.nml'):            
            nml_dic[f_nml.split('/')[-1][:-4]] = f90nml.read(f_nml)
        
        if fix_missing:
            # fix missing, by comparison with all_namelists
            nml_dic = fix_missing_nmls(nml_dic)
        return nml_dic

    def prepare_site_nmls(self):
        ''' this is called in self.__init__() '''
        for sid in self.site_list:
            create_site_ancil_and_config_files(sid)
            self.read_site_config(sid)
            self.fill_site_nml_patches_from_conf(self.base_nml_dic, sid,
                                                 self.site_configs[sid])
            self.create_site_nml_from_patches(sid)
            self.site_env_vars[sid] = fill_site_specific_env_vars(sid)
            self.sub_in_env_vars_in_nmls(sid)
            self.site_nmls[sid] = clean_reformat_nml(self.site_nmls[sid])
    
    def read_site_config(self, sid):
        ''' this is called in self.prepare_site_nmls() '''
        # first load up site-specific config files
        # hosts_process = subprocess.Popen(['find', es.RSUITE_flux+'/app/jules/opt/',
                                          # '-name', f'*{sid}*'], stdout=subprocess.PIPE)
        # hosts_out, hosts_err = hosts_process.communicate()
        # conf_files = [s.split('file=')[-1].replace("'", "") 
            # for s in hosts_out.decode("utf-8").split("\n")]
        # conf_files = [f for f in conf_files if len(f)>0]
        
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
                    if ((spiral_strings[-1]=='nvars' or 
                        spiral_strings[-1]=='timestep_len' or 
                        spiral_strings[-1]=='data_period') and
                        type(new_vals)==float):
                        new_vals = int(new_vals) # fix integers
                    self.site_nml_patches[sid].append((spiral_strings, new_vals))  
    
    def create_site_nml_from_patches(self, sid):
        ''' this is called in self.prepare_site_nmls() '''
        self.site_nmls[sid] = copy.deepcopy(self.base_nml_dic)
        for elem in self.site_nml_patches[sid]:
            ids = elem[0]
            new_val = elem[1]
            if ids[2]=='file':
                new_val = (new_val.replace("$FLUXNET2015_DRIVE_DIR", es.FN_DRIVE)
                                  .replace("$NEO_DRIVE_DIR", es.NEO_DRIVE)
                                  .replace("$LBA_DRIVE_DIR", es.LBA_DRIVE)
                                  .replace("$UKFLUX_DRIVE_DIR", es.UK_DRIVE)
                                  .replace("$CYLC_SUITE_RUN_DIR", es.RSUITE_flux)
                )                
            self.site_nmls[sid][ids[0]][ids[1]][ids[2]] = new_val

    def write_nml(self, sid, outpath):
        '''
        Function to write dictionary of stored nml data to nml files        
        '''        
        Path(outpath).mkdir(exist_ok=True, parents=True)
        for key in self.site_nmls[sid].keys():
            self.site_nmls[sid][key].write(outpath + key + '.nml', force=True)

    def write_all_site_nmls(self):
        for sid in self.site_list:
            self.write_nml(sid)
            
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
        pass

    def run_jules(self, sid):
        '''Run JULES in a subprocess. Check output for fatal errors.

        :return: stdout and stderr output from JULES model run.
        :rtype: tuple
        '''
        
        # add run-time nml sections
        #self.site_nmls[sid][key]
        #j.site_nmls[sid]['output']['jules_output']['run_id'] = sid
        #j.site_nmls[sid]['output']['jules_output']['output_dir'] = out_dir
        #j.site_nmls[sid]['timesteps']['jules_time']['main_run_start'] = str(self.year) + '-01-01 06:00:00'
        #j.site_nmls[sid]['timesteps']['jules_time']['main_run_end'] = str(self.year+1) + '-01-01 05:00:00'
        
        sid = 'DE_Akm' #'DE_Akm' # 'UK_Rdm'
        #j.site_nmls[sid]['timesteps']['jules_spinup']['max_spinup_cycles'] = 0  # 0 2  4
        
        # fix dummy.dat ancil for TOP model
        #j.site_nmls[sid]['ancillaries']['jules_top']['file'] = j.site_nmls['UK_Rdm']['ancillaries']['jules_top']['file']
        
        # set dummy surface tile fractions for new suite surfacetypes
        # j.site_nmls[sid]['ancillaries']['jules_frac']['file'] = \
            # '/home/users/doran/projects/carbon_prediction/dummy_surface_tile_fracs.dat'
        # j.site_nmls[sid]['ancillaries']['jules_frac']['file'] = \
            # '/home/users/doran/projects/carbon_prediction/dummy_transposed_surface_tile_fracs.dat'
        
        # move into namelist directory
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        
        nml_path = es.nml_directory + f'/{sid}_nmls/'
        for f in glob.glob(nml_path + '/*.nml'):
            os.remove(f)
        #self.write_nml(sid, nml_path)
        j.write_nml(sid, nml_path)
        cwd = os.getcwd()
        os.chdir(nml_path)
      
        cmd = []
        #cmd.append(self.jules)
        cmd.append(j.jules)
        
        a = time.time()
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
        return (out, err)

    # def run_jules_print(self):
        # '''Write all NML files to disk. Run JULES in a subprocess.

        # :return: string confirming model run (str)
        # :rtype: str
        # '''

        # # write all the nml files here so the
        # # user doesn't have to remember to...
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # cwd = os.getcwd()
        # os.chdir(dir_path+'/'+self.nml_dir)
        # self.write_nml()

        # # run JULES
        # cmd = []
        # cmd.append(self.jules)
        # proc = subprocess.Popen(cmd, shell=False)
        # proc.communicate()
        # os.chdir(cwd)
        # return 'Done', 'Done'
