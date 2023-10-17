import f90nml
import re
import configparser
import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path

import experiment_setup as es

def fetch_val_from_config(raw_vals, spiral_strings, configs):    
    for ll in range(len(spiral_strings)):
        if configs['ancil'].has_section(f'namelist:{spiral_strings[ll]}'):
            if spiral_strings[ll+1] in list(configs['ancil'][f'namelist:{spiral_strings[ll]}'].keys()):                
                new_vals = (configs['ancil']
                    [f'namelist:{spiral_strings[ll]}']
                    [spiral_strings[ll+1]])
            else: continue
        elif configs['drive'].has_section(f'namelist:{spiral_strings[ll]}'):
            if spiral_strings[ll+1] in list(configs['drive'][f'namelist:{spiral_strings[ll]}'].keys()):
                new_vals = (configs['drive']
                    [f'namelist:{spiral_strings[ll]}']
                    [spiral_strings[ll+1]])
            else: continue
        else:
            continue                        
        if type(raw_vals)==list:
            if ',' in new_vals:
                # explicit list
                new_vals = new_vals.split(',')
                new_vals = [f.strip().replace('\n','').replace('=','').replace("'","").replace('"','') for f in new_vals]
                new_vals = [float(f) if isnumber(f) else f for f in new_vals]
            elif '*' in new_vals:
                # implicit list with multiplication
                new_vals = new_vals.strip()
        else:
            new_vals = new_vals.strip()
            if isnumber(new_vals):
                new_vals = float(new_vals)
        return new_vals
    return None  

def extract_environment_vars(nml_dic):
    env_vars = []
    for k in list(nml_dic.keys()):
        for l in list(nml_dic[k].keys()):
            for m in list(nml_dic[k][l].keys()):
                if type(nml_dic[k][l][m])==str:
                    if "$" in nml_dic[k][l][m]:
                        env_vars.append((k, l, m, nml_dic[k][l][m]))
    return env_vars

def print_all_files(nml_dic):
    env_vars = []
    for k in list(nml_dic.keys()):
        for l in list(nml_dic[k].keys()):
            if 'file' in list(nml_dic[k][l].keys()):                
                print("%s; %s; %s\n" % (k, l, nml_dic[k][l]['file']))                

def add_land_dim(nc_dat, var='cs'):
    nc_dat = nc_dat.assign_coords({'land':[1]})
    nc_dat[var] = nc_dat[var].expand_dims(dim={'land':1})
    return nc_dat

def find_chess_tile(lat, lon, latlon_ref):
    # assumes equal length lat/lon vectors in latlon_ref
    dist_diff = np.sqrt(np.square(latlon_ref.lat.values - lat) +
                        np.square(latlon_ref.lon.values - lon))
    chesstile_yx = np.where(dist_diff == np.min(dist_diff))
    return chesstile_yx

def create_site_ancil_and_config_files(sid):
    '''
    - /gws/nopw/j04/hydro_jules/data/uk/ancillaries/chess-scape/chess-scape_uk_1km_topography.nc
        with "slope" variable if using pdm runoff
        or with "field900" == topographic index and "field900_1" == std topo index
        if using topmodel (set in jules_hydrology.nml and ancillaries.nml)
    - I have already hard-coded the values from the urban morphology file
        /gws/nopw/j04/hydro_jules/data/uk/ancillaries/chess-scape/chess-scape_uk_1km_morphology_U2T_prelim.nc
    - /gws/nopw/j04/hydro_jules/data/uk/ancillaries/chess-scape/chess-scape_uk_1km_soilcarbon_hwsd_vg.nc
        with carbon var "cs" for additional dims sclayer=1 and (I think) scpool=4
    - /gws/nopw/j04/hydro_jules/data/uk/ancillaries/chess-scape/chess-scape_uk_1km_soilparams_hwsd_vg.nc
        /gws/nopw/j04/hydro_jules/data/uk/ancillaries/chess-scape/chess-scape_uk_1km_soiltexture_1scl_hwsd.nc
        for soil props. if we don't get from site metadata?
        we must merge the vars ('oneovernminusone','oneoveralpha',
        'satcon','vsat','vcrit','vwilt','hcap','hcon') from
        ...params_hwsd_vg with the single "clay" var from ...texture_1scl_hwsd
    - /gws/nopw/j04/hydro_jules/data/uk/ancillaries/chess-scape/chess-scape_uk_1km_landcover2000_U2T_prelim.nc
        land cover, if we don't get from site metadata?
    '''
    
    # find site lat lon and calculate chess pixel yx, e.g.
    #lat = es.site_info.loc[sid].LATITUDE
    #lon = es.site_info.loc[sid].LONGITUDE
    #met_drive_path = es.site_info.loc[sid].DRIVING_DATA_PATH
    #data_start = str(es.site_info.loc[sid].DATA_START)
    #data_end = str(es.site_info.loc[sid].DATA_END)
    #data_period = es.site_info.loc[sid].DATA_RESOLUTION_SECONDS
    
    # test vals, these should be accessible in es.site_info
    sid = 'DE_Akm'
    lat, lon = 52.4429, 0.4195
    met_drive_path = es.FN_DRIVE + '/DE_Akm-met.dat'    
    data_start = '2008-12-31 23:00:00'
    data_end = '2014-12-31 22:30:00'
    data_period = 1800
        
    topo_out = es.ancils_directory + f'/topog_ancil-{sid}.nc'
    soil_out = es.ancils_directory + f'/soil_ancil-{sid}.nc'
    #frac_out = es.ancils_directory + f'/frac_ancil-{sid}.nc'
    frac_out = es.ancils_directory + f'/frac_ancil-{sid}.dat'
    carb_out = es.ancils_directory + f'/carbon_init-{sid}.nc'
    a_config_out = es.config_directory + f'/ancil_config-{sid}.conf'
    d_config_out = es.config_directory + f'/drive_config-{sid}.conf'
    
    if ((not Path(topo_out).exists()) or (not Path(soil_out).exists()) or
        (not Path(frac_out).exists()) or (not Path(carb_out).exists())):
        latlon_ref = xr.open_dataset(es.chess_ancil_dir + 'chess_lat_lon.nc').load()
        y_ind, x_ind = find_chess_tile(lat, lon, latlon_ref)
        latlon_ref.close()
        del(latlon_ref)

    # use that pixel to extract all the required ancil data from 
    # the various chess .nc files
    if not Path(topo_out).exists():
        topo = xr.open_dataset(es.chess_ancil_dir + 'chess-scape/chess-scape_uk_1km_topography.nc')
        topo = topo.isel(y=y_ind[0], x=x_ind[0]).expand_dims({"land":1})
        topo.to_netcdf(topo_out)
        topo.close()
        del(topo)
    
    if not Path(soil_out).exists():
        soil_ps = xr.open_dataset(es.chess_ancil_dir + '/chess-scape/chess-scape_uk_1km_soilparams_hwsd_vg.nc')
        soil_ps = soil_ps.isel(y=y_ind[0], x=x_ind[0]).expand_dims({"land":1})
        soil_tex = xr.open_dataset(es.chess_ancil_dir + '/chess-scape/chess-scape_uk_1km_soiltexture_1scl_hwsd.nc')
        soil_tex = soil_tex.isel(y=y_ind[0], x=x_ind[0]).expand_dims({"land":1})
        soil_tex = soil_tex.rename({'z':'sclayer'})
        soil_ps = (soil_ps.drop('cs')
            .assign({'clay': (("land", "sclayer"), soil_tex.clay.values)})
            .rename({'z':'soil'})
            .transpose('soil', ..., 'land')
        )
        soil_tex.close()
        del(soil_tex)
        soil_ps.to_netcdf(soil_out)
        soil_ps.close()
        del(soil_ps)
        
    if not Path(frac_out).exists():
        header = '# BDT\tNET\tC3G\tDSh\tC3Cr\tIW\tBS\tUrC\tUrR\n'
        lcm = xr.open_dataset(es.chess_ancil_dir + '/chess-scape/chess-scape_uk_1km_landcover2000_U2T_prelim.nc')
        lcm = lcm.isel(y=y_ind[0], x=x_ind[0]).expand_dims({"land":1})        
        fvals = lcm.frac.values.flatten()
        with open(frac_out, 'w') as fo:
            fo.write(header)
            fo.write('%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t' % 
                (fvals[0], fvals[1], fvals[2], fvals[3], fvals[4], fvals[5], fvals[6], fvals[7], fvals[8]))
        
        # if '_NCProperties' in lcm.attrs.keys(): # netcdf3 error
            # lcm.attrs.pop('_NCProperties')
        # lcm.to_netcdf(frac_out)
        lcm.close()
        del(lcm)

    if not Path(carb_out).exists():        
        carb = xr.open_dataset('/gws/nopw/j04/ceh_generic/matwig/v28_JULES_Bench/1/Clim_SL1_J_Q_MBulk/Carbon_Eqstate_monthly_pool.nc', decode_times=False)
        carb = carb.isel(y=y_ind[0], x=x_ind[0])
        doy = (pd.to_datetime(data_start).dayofyear-1) % 365 # zero-index and set leap year to zero
        dist = abs(carb.time.values - doy)
        inds = np.argsort(dist)
        dist = dist[inds[0:2]]
        if dist[0]==0:            
            carb_vals = carb.isel(time=inds[0]).cs.values[:4][..., None, None] # scpool, sclayer, land
        else:
            weights = sum(dist) / dist
            weights = weights / sum(weights)            
            carb_vals = (weights[0] * carb.isel(time=inds[0]).cs.values[:4] + 
                weights[1] * carb.isel(time=inds[1]).cs.values[:4])
            carb_vals = carb_vals[..., None, None] # scpool, sclayer, land
        carb.close()
        del(carb)
        
        ds = xr.Dataset(
            data_vars=dict(
                cs=(["scpool", "sclayer", "land"], carb_vals),                
            ),
            coords=dict(
                scpool=(["scpool"], np.array([1.,2.,3.,4.], dtype=np.float32)),
            ),            
        )
        
        ds.to_netcdf(carb_out)
        
        # # temporary for carbon, would actually use Matt's new carbon states?
        # # would need start date/month to get correct carbon
        # carb = xr.open_dataset(es.chess_ancil_dir + '/chess-scape/chess-scape_uk_1km_soilcarbon_hwsd_vg.nc')
        # carb = carb.isel(y=y_ind[0], x=x_ind[0])
        # cs_array = carb.cs.values.flatten()
        # cs_array = np.repeat(cs_array[None, None, ...], 4, 0) / 4.
        
        # carb.close()
        # del(carb)
        
        # ds = xr.Dataset(
            # data_vars=dict(
                # cs=(["scpool", "sclayer", "land"], cs_array),                
            # ),
            # coords=dict(
                # scpool=(["scpool"], np.array([1.,2.,3.,4.], dtype=np.float32)),
            # ),            
        # )
        
        # # if '_NCProperties' in carb.attrs.keys(): # netcdf3 error
            # # carb.attrs.pop('_NCProperties')
        
    # create config files pointing to these files to patch into the name lists
    if not Path(a_config_out).exists():
        # ancil config
        conf_template_f = es.config_directory + 'rose-app-ancil-TEMPLATE.conf'
        config = configparser.ConfigParser(strict=False)
        config.read(conf_template_f)
        
        config['namelist:jules_frac']['file'] = f"'{frac_out}'"
        config['namelist:jules_latlon']['const_val'] = f'{lat},{lon}'
        config['namelist:jules_soil_props']['file'] = f"'{soil_out}'"
        config['namelist:jules_pdm']['file'] = f"'{topo_out}'"
        config['namelist:jules_initial']['file'] = f"'{carb_out}'"
        with open(a_config_out, 'w') as fo:
            config.write(fo)
    
    if not Path(d_config_out).exists():
        # drive config
        conf_template_f = es.config_directory + 'rose-app-drive-TEMPLATE.conf'
        config = configparser.ConfigParser(strict=False)
        config.read(conf_template_f)
        
        config['namelist:jules_drive']['file'] = f"'{met_drive_path}'"
        config['namelist:jules_drive']['data_start'] = f"'{data_start}'"
        config['namelist:jules_drive']['data_end'] = f"'{data_end}'"
        config['namelist:jules_drive']['data_period'] = f'{data_period}'
        config['namelist:jules_time']['timestep_len'] = f'{data_period}'
        with open(d_config_out, 'w') as fo:
            config.write(fo)

def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def fill_site_specific_env_vars(site):
    ## site specific (should set when looping through sites, not here)
    env_vars = es.env_vars.copy()
    env_vars["$OUTDIR"] = es.output_directory + f'/{site}/'
    Path(env_vars["$OUTDIR"]).mkdir(parents=True, exist_ok=True)
    env_vars["$RUNID"] = f'jrun_{site}'
    env_vars["$RUNBEGDATE"] = es.site_info.loc[es.site_info['SITE_ID']==site, 'START_DATE'].values[0]
    env_vars["$RUNENDDATE"] = es.site_info.loc[es.site_info['SITE_ID']==site, 'END_DATE'].values[0]
    env_vars["$SPINBEGDATE"] = es.site_info.loc[es.site_info['SITE_ID']==site, 'START_DATE'].values[0]
    env_vars["$SPINENDDATE"] = es.site_info.loc[es.site_info['SITE_ID']==site, 'END_DATE'].values[0]
    return env_vars

def replace_env_var(val, env_vars):
    if type(val)==str:
        for l in env_vars.keys():
            if l in val:
                if type(env_vars[l])==bool:
                    new_val = env_vars[l]
                elif type(env_vars[l])==int:
                    new_val = val.replace(l, str(env_vars[l]))
                    new_val = int(new_val)
                else:
                    new_val = val.replace(l, env_vars[l])
                return new_val
        return val
    else:
        return val

def boolerise(string):
    if string=='.false.': return False
    if string=='.true.': return True
    return string

def clean_reformat_nml(nml):    
    for i in nml.keys():
        for k in nml[i].keys():
            for l in nml[i][k].keys():
                if type(nml[i][k][l])==str:
                    if "," in nml[i][k][l]:
                        nml[i][k][l] = nml[i][k][l].split(",")                        
                    elif not "*" in nml[i][k][l]:
                        nml[i][k][l] = nml[i][k][l].replace("'","").replace('"', '')
                    elif "*" in nml[i][k][l]:
                        splat = nml[i][k][l].split("*")
                        if splat[-1]=='.false.': nml[i][k][l] = [False] * int(splat[0])
                        elif splat[-1]=='.true.': nml[i][k][l] = [True] * int(splat[0])
                        else: nml[i][k][l] = [splat[-1]] * int(splat[0])                            
                if type(nml[i][k][l])==list:
                    if type(nml[i][k][l][0])==str:
                        nml[i][k][l] = [boolerise(s.replace("'","").replace('"', '')) if (not "*" in s) else boolerise(s) for s in nml[i][k][l]]
    return nml

all_namelists = dict(
    ancillaries = {'jules_frac':{},
                   'jules_soil_props':{},
                   'jules_top':{},
                   'jules_agric':{},
                   'jules_crop_props':{},
                   'jules_irrig_props':{},
                   'jules_rivers_props':{},
                   'jules_co2':{}},
    cable_prognostics = {},
    cable_soilparm = {},
    cable_surface_types = {},
    crop_params = {'jules_cropparm':{}},
    drive = {'jules_drive':{}},
    fire = {'fire_switches':{}},
    imogen = {'imogen_onoff_switch':{},
              'imogen_run_list':{},
              'imogen_anlg_vals_list':{}},
    initial_conditions = {'jules_initial':{}},
    jules_deposition = {'jules_deposition':{},
                        'jules_deposition_species':{},
                        'jules_deposition_species_specific':{}},
    jules_hydrology = {'jules_hydrology':{}},
    jules_irrig = {'jules_irrig':{}},
    jules_prnt_control = {'jules_prnt_control':{}},
    jules_radiation = {'jules_radiation':{}},
    jules_rivers = {'jules_rivers':{}},
    jules_snow = {'jules_snow':{}},
    jules_soil = {'jules_soil':{}},
    jules_soil_biogeochem = {'jules_soil_biogeochem':{}},
    jules_soil_ecosse = {'jules_soil_ecosse':{}},
    jules_soilparm_cable = {},
    jules_surface = {'jules_surface':{}},
    jules_surface_types = {'jules_surface_types':{}},
    jules_vegetation = {'jules_vegetation':{}},
    jules_water_resources = {'jules_water_resources':{},
                             'jules_water_resources_props':{}},
    logging = {'logging':{}},
    model_environment = {'jules_model_environment':{}},
    model_grid = {'jules_input_grid':{},
                  'jules_latlon':{},
                  'jules_land_frac':{},
                  'jules_model_grid':{},
                  'jules_nlsizes':{},
                  'jules_surf_hgt':{}},
    nveg_params = {'jules_nvegparm':{}},
    oasis_rivers = {'oasis_rivers':{}},
    output = {'jules_output':{},
              'jules_output_profile':{}},
    pft_params = {'jules_pftparm':{}},
    prescribed_data = {'jules_prescribed':{},
                       'jules_prescribed_dataset':{}},
    science_fixes = {'jules_temp_fixes':{}},
    timesteps = {'jules_time':{},
                 'jules_spinup':{}},
    triffid_params = {'jules_triffid':{}},
    urban = {'jules_urban_switches':{},
             'jules_urban2t_param':{},
             'urban_properties':{}}
)

# def fix_missing_nmls(nml_dic):
    # ''' going through the missing namelists from the results of
        # np.setdiff1d(np.array(list(all_namelists.keys())),
                     # np.array(list(j.base_nml_dic.keys())))
        # and filling from rose-app.conf
    # '''
    # if not 'initial_conditions' in nml_dic.keys():
        # # nml_dic['initial_conditions'] = f90nml.namelist.Namelist([
            # # ('jules_initial', f90nml.namelist.Namelist([
                # # ('const_val', ''),
                # # ('dump_file', True),
                # # ('file', "$INITIALCONDITIONSFILE"),
                # # ('nvars', 4),
                # # ('total_snow', True),
                # # ('use_file', ''),
                # # ('var', ['cs','lai','canht','frac']),
                # # ('var_name', '')
            # # ]))
        # # ])
        # ## need REQUIRED inputs to have initial conditions!
        # nml_dic['initial_conditions'] = f90nml.namelist.Namelist([
            # ('jules_initial', f90nml.namelist.Namelist([
                # ('const_val', [1.0,276.78,12.1,0.0,50.0,0.749,3.0,0.0,0.0,0.0]),
                # ('dump_file', False),
                # ('file', ''),
                # ('nvars', 10),
                # ('total_snow', True),
                # ('use_file', [False]*10),
                # ('var', ['canopy','tstar_tile','cs','gs','rgrain','sthzw','zw','sthuf','t_soil','snow_tile'])
            # ]))
        # ])
    # if not 'jules_hydrology' in nml_dic.keys():
        # nml_dic['jules_hydrology'] = f90nml.namelist.Namelist([
            # ('jules_hydrology', f90nml.namelist.Namelist([
                # ('l_limit_gsoil', False),
                # ('l_pdm', False),
                # ('l_top', "$L_TOP"),
                # ('l_wetland_unfrozen', False),
                # ('nfita', 30),                
                # ('ti_max', 10.0),
                # ('ti_wetl', 1.5),
                # ('zw_max', 6.0)
            # ]))
        # ])
    # if not 'jules_soil_biogeochem' in nml_dic.keys():
        # nml_dic['jules_soil_biogeochem'] = f90nml.namelist.Namelist([
            # ('jules_soil_biogeochem', f90nml.namelist.Namelist([
                # ('bio_hum_cn', 10.0),
                # ('ch4_cpow', 1.0),
                # ('ch4_substrate', 1),
                # ('const_ch4_cs', 7.41e-12),
                # ('const_ch4_npp', 9.99e-3),
                # ('const_ch4_resps', 4.36e-3),
                # ('kaps', 0.5e-8),
                # ('kaps_4pool', [3.22e-7,9.65e-9,2.12e-8,6.43e-10]),
                # ('l_ch4_interactive', False),
                # ('l_ch4_microbe', False),
                # ('l_ch4_tlayered', False),
                # ('l_layeredc', False),
                # ('l_q10', True),
                # ('l_soil_resp_lev2', True),
                # ('n_inorg_turnover', 1.0),
                # ('q10_ch4_cs', 3.7),
                # ('q10_ch4_npp', 1.5),
                # ('q10_ch4_resps', 1.5),
                # ('q10_soil', 2.0),
                # ('soil_bgc_model', "$SOIL_BGC_MODEL"),
                # ('sorp', 10.0),
                # ('t0_ch4', 273.15),
                # ('tau_lit', 5.0)
            # ]))
        # ])
    # if not 'jules_vegetation' in nml_dic.keys():
        # nml_dic['jules_vegetation'] = f90nml.namelist.Namelist([
            # ('jules_vegetation', f90nml.namelist.Namelist([
                # ('can_model', 4),
                # ('can_rad_mod', 6),
                # ('frac_min', 1.0e-6),
                # ('frac_seed', 0.01),
                # ('fsmc_shape', 0),
                # ('ilayers', 10),
                # ('l_bvoc_emis', False),
                # ('l_croprotate', False),
                # ('l_gleaf_fix', True),
                # ('l_ht_compete', False),
                # ('l_inferno', False),
                # ('l_landuse', False),
                # ('l_leaf_n_resp_fix', True),
                # ('l_limit_canhc', False),
                # ('l_nitrogen', False),
                # ('l_o3_damage', False),
                # ('l_phenol', "$L_PHENOL"),
                # ('l_scale_resp_pm', False),
                # ('l_spec_veg_z0', False),
                # ('l_stem_resp_fix', True),
                # ('l_trait_phys', True),
                # ('l_trif_crop', False),
                # ('l_trif_eq', False),
                # ('l_trif_fire', False),
                # ('l_triffid', "$L_TRIFFID"),
                # ('l_use_pft_psi', False),
                # ('l_veg_compete', False),
                # ('l_vegcan_soilfx', True),
                # ('l_vegdrag_pft', 9*[False]),
                # ('phenol_period', 1),
                # ('photo_model', 1),
                # ('pow', 20.0),
                # ('stomata_model', 1),
                # ('triffid_period', 1)
            # ]))
        # ])
    # if not 'output' in nml_dic.keys():
        # nml_dic['output'] = f90nml.namelist.Namelist([
            # ('jules_output', f90nml.namelist.Namelist([
                # ('dump_period', 1),
                # ('dump_period_unit', 'Y'),
                # ('nprofiles', "$NPROFILES"),
                # ('output_dir', "$OUTPUT_FOLDER"),
                # ('run_id', "$OUTSTRING")
            # ])),
            # ('jules_output_profile(1)', f90nml.namelist.Namelist([ # this was (:) namelist
                # ('file_period', -2),
                # ('l_land_frac', False),
                # ('nvars', 20),
                # ('output_main_run', True),
                # ('output_period', 86400),
                # ('output_spinup', False),
                # ('output_type', ['N'] + 19*['M']),
                # ('profile_name', 'D'),
                # ('var', ['t1p5m_gb','gpp_gb','resp_p_gb','resp_s_gb','ftl_gb','latent_heat','rad_net','sw_down','precip','t1p5m_gb','q1p5m_gb','canht','lai','fsmc_gb','smc_avail_top','fqw_gb','et_stom_gb','sthu','sthf','npp_gb']),
                # ('var_name', ['min_t1p5m_gb']+19*[''])
            # ])),
            # ('jules_output_profile(2)', f90nml.namelist.Namelist([
                # ('file_period', 0),
                # ('l_land_frac', False),
                # ('nvars', 10),
                # ('output_main_run', True),
                # ('output_period', 3600),
                # ('output_spinup', False),
                # ('output_type', 10*['M']),
                # ('profile_name', 'H'),
                # ('var', ['gpp_gb','resp_p_gb','resp_s_gb','ftl_gb','latent_heat','rad_net','sw_down','precip','t1p5m_gb','q1p5m_gb']),
                # ('var_name', 10*[''])
            # ]))
        # ])
    # if not 'prescribed_data' in nml_dic.keys():
        # nml_dic['prescribed_data'] = f90nml.namelist.Namelist([
            # ('jules_prescribed', f90nml.namelist.Namelist([
                # ('n_datasets', "$NPRESCRIBED")
            # ])),
            # #('jules_prescribed_dataset(1)', f90nml.namelist.Namelist([ # this was (:) namelist
            # ('jules_prescribed_dataset', f90nml.namelist.Namelist([ # this was (:) namelist
                # ('data_end', '2024-01-01 00:00:00'),
                # ('data_period', -2),
                # ('data_start', '1980-01-01 00:00:00'),
                # ('file', es.DATA_DIR+'/prescribed_co2mmr/NOAA_co2mmr_1980_2023byhand.txt'),
                # ('interp', 'nf'),
                # ('is_climatology', False),
                # ('nvars', 1),
                # ('read_list', False),
                # ('tpl_name', ''),
                # ('var', 'co2_mmr'),
                # ('var_name', 'co2_mmr')
            # ]))
        # ])
    # if not 'timesteps' in nml_dic.keys():
        # nml_dic['timesteps'] = f90nml.namelist.Namelist([
            # ('jules_time', f90nml.namelist.Namelist([
                # ('l_360', 'see conf'),
                # ('l_leap', 'see conf'),
                # ('l_local_solar_time', False),
                # ('main_run_end', "$RUNENDDATE 00:00:00"),
                # ('main_run_start', "$RUNBEGDATE 00:00:00"),
                # ('print_step', 48),
                # ('timestep_len', 'drive_timestep')
            # ])),
            # ('jules_spinup', f90nml.namelist.Namelist([
                # ('max_spinup_cycles', "$NSPIN"),
                # ('nvars', 1),
                # ('spinup_end', "$RUNENDDATE 00:00:00"), # $SPINENDDATE
                # ('spinup_start', "$RUNBEGDATE 00:00:00"), # $SPINBEGDATE
                # ('terminate_on_spinup_fail', False),
                # ('tolerance', -3.0),
                # ('use_percent', True),
                # ('var', 'smcl') 
            # ]))
        # ])
    # return nml_dic



