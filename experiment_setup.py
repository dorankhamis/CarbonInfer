import os
import pandas as pd
import numpy as np
from pathlib import Path

# set directory for JULES output
output_directory = os.getcwd() + '/output/'
config_directory = os.getcwd() + '/configs/'
ancils_directory = os.getcwd() + '/ancils/'
data_directory = os.getcwd() + '/data/'

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
#DATA_DIR = '/gws/nopw/j04/ceh_generic/netzero/fluxdata/fluxnet_sites/'
DATA_DIR = '/gws/nopw/j04/ceh_generic/netzero/fluxdata/'
#FN_DRIVE = DATA_DIR + '/fluxnet/'
#LBA_DRIVE = DATA_DIR + '/lba_drive/'
#NEO_DRIVE = DATA_DIR + '/neo_drive/'
#UK_DRIVE = DATA_DIR + '/uk_drive/'

# other environmental variables to set from site info files
#site_info = pd.read_table(RSUITE_flux + '/var/info_table.tsv', sep=" ")
site_info = pd.read_csv(DATA_DIR + '/meta/site_info.csv')
site_info = site_info.set_index('SITE_ID')
site_list = list(site_info.index)

# move the Bioenergy landcovers LincsSRCWillow / LincsMiscanthus to BDT/C3Cr respectively
site_info.loc['LincsMiscanthus','landcover_simp'] = 'Agriculture'
site_info.loc['LincsSRCWillow','landcover_simp'] = 'Broadleaf tree'
 
env_vars = {
    "$OUTDIR":'',
    "$RUNID":'',
    "$RUNBEGDATE":'',
    "$RUNENDDATE":'',
    "$SPINBEGDATE":'',
    "$SPINENDDATE":''
}

int_vars = ['nvars', 'timestep_len', 'precip_disagg_method',
            'max_spinup_cycles', 'data_period']

# rescaling NEE from kg C / m2 / s to micromoles of CO2
C_aw = 12.0107
O_aw = 15.999
CO2_aw = C_aw + 2*O_aw
f_C = C_aw / CO2_aw # frac of CO2 that is carbon
co2_gpmm = CO2_aw * 1e-6 # grams per micromole of CO2
C_kg_p_mmco2 = 1e-3 * co2_gpmm * f_C # kilograms of C per micromole of CO2

# set JULES parameters to optimised during data assimilation
# we are using only PFT params so all will be inside the opt_namelist_struct
opt_namelist_struct = ['pft_params', 'jules_pftparm'] #{'pft_params':{'jules_pftparm':''}}
default_params = {
    'a_wl_io': [0.78, 0.8, 0.005, 0.13, 0.005], # Allometric coefficient relating the target woody biomass to the leaf area index (kg carbon m-2)
    'a_ws_io': [12, 10, 1, 13, 1], # Woody biomass as a multiple of live stem biomass (Clark et al., 2011; Table 7)
    'alnir_io': [0.45, 0.35, 0.58, 0.58, 0.58], # Leaf reflection coefficient for NIR
    'alpar_io': [0.1, 0.07, 0.1, 0.1, 0.1], # Leaf reflection coefficient for VIS (photosyntehtically active radiation)
    'alpha_io': [0.048, 0.08, 0.048, 0.048, 0.048], # Quantum efficiency of photosynthesis (mol CO2 per mol PAR photons)
    'b_wl_io': [1.667, 1.667, 1.667, 1.667, 1.667], # Allometric exponent relating the target woody biomass to the leaf area index
    'dqcrit_io': [0.09, 0.041, 0.051, 0.044, 0.051], # Critical humidity deficit (kg H2O per kg air).
    'eta_sl_io': [0.01, 0.01, 0.01, 0.01, 0.01], # Live stemwood coefficient (kg C/m/(m2 leaf))
    'f0_io': [0.892, 0.875, 0.931, 0.875, 0.931], # CI / CA for DQ = 0. See HCTN 24 Eq. 32
    'fd_io': [0.01, 0.015, 0.019, 0.015, 0.019], # Scale factor for dark respiration. See HCTN 24 Eq. 56    
    'gpp_st_io': [1.59e-07, 1.01e-07, 2.61e-07, 2.22e-07, 2.61e-07], # Gross primary production (GPP) at standard conditions (kgC m-2 s-1)
    'q10_leaf_io': [2.0, 2.0, 2.0, 2.0, 2.0], # Q10 factor for plant respiration
    'r_grow_io': [0.25, 0.25, 0.25, 0.25, 0.25], # Growth respiration fraction
    'tleaf_of_io': [233.15, 278.15, 278.15, 233.15, 278.15], # Temperature below which leaves are dropped (K)
    'tlow_io': [13, -10, 10, 0, 10], # Lower temperature parameter for photosynthesis (deg C)
    'tupp_io': [43, 26, 32, 36, 32] # Upper temperature parameter for photosynthesis (deg C)
    #'vint_io': [3.9, 6.32, 6.42, 14.71, 6.42] # Vcmax linear regression
    #'vsl_io': [28.4, 23.79, 40.96, 23.15, 40.96] # Vcmax linear regression       
}

default_stderr = default_params.copy()
for k in default_params.keys():
    default_stderr[k] = [np.max(abs(np.array(default_params[k]))) / 10. for i in range(len(default_params[k]))]


