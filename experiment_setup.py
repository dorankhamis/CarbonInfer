import os
import pandas as pd
from pathlib import Path

#import observations
#import plot

## opening sheets from an excel file
# xl = pd.ExcelFile('foo.xls')
# xl.sheet_names  # see all sheet names
# xl.parse(sheet_name)

# set directory for JULES output
output_directory = os.getcwd() + '/output/'
config_directory = os.getcwd() + '/configs/'
ancils_directory = os.getcwd() + '/ancils/'

# set directory containing JULES NML files
#base_nml_dir = '/home/users/doran/roses/nlists_u-cr886/'
base_nml_dir = os.getcwd() + '/base_namelists/'
nml_directory = os.getcwd() + '/run_namelists/'

# set model executable
model_exe = '/home/users/doran/MODELS/newest/build/bin/jules.exe'

# set suite directory, $CYLC_SUITE_RUN_DIR
RSUITE_flux = '/home/users/doran/roses/u-cr886/'
RSUITE_chess = '/home/users/doran/roses/u-cw537/'

# chess ancils path
chess_ancil_dir = '/gws/nopw/j04/hydro_jules/data/uk/ancillaries/'

# set driving data directory, (e.g. $FLUXNET2015_DRIVE_DIR)
DATA_DIR = '/gws/nopw/j04/ceh_generic/netzero/fluxdata/jules/'
FN_DRIVE = DATA_DIR + '/fluxnet/'
LBA_DRIVE = DATA_DIR + '/lba_drive/'
NEO_DRIVE = DATA_DIR + '/neo_drive/'
UK_DRIVE = DATA_DIR + '/uk_drive/'

# other environmental variables to set from site info files
site_info = pd.read_table(RSUITE_flux + '/var/info_table.tsv', sep=" ")
 
env_vars = {
    "$OUTDIR":'',
    "$RUNID":'',
    "$RUNBEGDATE":'',
    "$RUNENDDATE":'',
    "$SPINBEGDATE":'',
    "$SPINENDDATE":''
}
 
# set JULES parameters to optimised during data assimilation
opt_params = {'pft_params': {
                  'jules_pftparm': {
                      'neff_io': [7, 6.24155040e-04, (5e-05, 0.0015)],
                      'alpha_io': [7, 6.73249126e-02, (0, 1.0)],
                      'fd_io': [7, 8.66181324e-03, (0.0001, 0.1)]}},
              'crop_params': {
                  'jules_cropparm': {
                      'gamma_io': [2, 2.07047321e+01, (0.0, 40.0)],
                      'delta_io': [2, -2.97701647e-01, (-2.0, 0.0)],
                      'mu_io': [2, 2.37351160e-02, (0.0, 1.0)],
                      'nu_io': [2, 4.16006288e+00, (0.0, 20.0)]}}}

# set error on prior parameter estimates
prior_err = 0.25
# set size of ensemble to be used in data assimilation experiments
ensemble_size = 50
# set number of processors to use in parallel runs of JULES ensemble
num_processes = 100
# set seed value for any random number generation within experiments
seed_value = 0

# # plotting save function
# save_plots = plot.save_plots
# # plotting output director
# plot_output_dir = os.getcwd()+'/output/plot'

