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

def create_site_ancil_and_config_files(sid, run_type, env_vars):
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
    lat = es.site_info.loc[sid].lat
    lon = es.site_info.loc[sid].lon
    data_start = str(es.site_info.loc[sid].data_start_date) + ' 00:00:00' # daily data fudge
    data_end = str(es.site_info.loc[sid].data_end_date) + ' 00:00:00'
    data_period = es.site_info.loc[sid].data_period
    met_drive_path = es.site_info.loc[sid].driving_data_path
    y_ind = es.site_info.loc[sid].chess_y
    x_ind = es.site_info.loc[sid].chess_x
    # and we might also want to load up the Landcover_j from the site_info
    # rather than read the land cover fraction from the chess_scape ancilliary file
    landcover_text = es.site_info.loc[sid].landcover_simp

    topo_out = es.ancils_directory + f'/topog_ancil-{sid}.nc'
    soil_out = es.ancils_directory + f'/soil_ancil-{sid}.nc'
    frac_out = es.ancils_directory + f'/frac_ancil-{sid}.dat'
    carb_out = es.ancils_directory + f'/carbon_init-{sid}.nc'
    a_config_out = es.config_directory + f'/ancil_config-{sid}.conf'
    d_config_out = es.config_directory + f'/drive_config-{sid}.conf'
    
    spinend_datestring = env_vars["$SPINENDDATE"].replace('-','')[:8]
    dump_file_path = es.output_directory + f'/{sid}/longspin_dump/{env_vars["$RUNID"]}.dump.{spinend_datestring}.0.nc'
    
    if False:
        if ((not Path(topo_out).exists()) or (not Path(soil_out).exists()) or
            (not Path(frac_out).exists()) or (not Path(carb_out).exists())):
            latlon_ref = xr.open_dataset(es.chess_ancil_dir + 'chess_lat_lon.nc').load()
            y_ind, x_ind = find_chess_tile(lat, lon, latlon_ref)
            latlon_ref.close()
            del(latlon_ref)

    ## use that pixel to extract all the required ancil data from the various chess .nc files
    if not Path(topo_out).exists():
        topo = xr.open_dataset(es.chess_ancil_dir + 'chess-scape/chess-scape_uk_1km_topography.nc')
        topo = topo.isel(y=y_ind, x=x_ind).expand_dims({"land":1})
        topo.to_netcdf(topo_out)
        topo.close()
        del(topo)
    
    if not Path(soil_out).exists():
        soil_ps = xr.open_dataset(es.chess_ancil_dir + '/chess-scape/chess-scape_uk_1km_soilparams_hwsd_vg.nc')
        soil_ps = soil_ps.isel(y=y_ind, x=x_ind).expand_dims({"land":1})
        soil_tex = xr.open_dataset(es.chess_ancil_dir + '/chess-scape/chess-scape_uk_1km_soiltexture_1scl_hwsd.nc')
        soil_tex = soil_tex.isel(y=y_ind, x=x_ind).expand_dims({"land":1})
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
        '''
            BDT        Broadleaf deciduous tree
            NET        Needleleaf evergreen tree
            C3G        C3 grass
            DSh        Deciduous shrub
            C3Cr       C3 Crop
            IW         Inland water
            BS         Bare soil
            UrC        Urban canyon
            UrR        Urban roof
        '''
        header = '# BDT\tNET\tC3G\tDSh\tC3Cr\tIW\tBS\tUrC\tUrR\n'
        
        if False:
            lcm = xr.open_dataset(es.chess_ancil_dir + '/chess-scape/chess-scape_uk_1km_landcover2000_U2T_prelim.nc')
            lcm = lcm.isel(y=y_ind, x=x_ind).expand_dims({"land":1})        
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
            
        # use simplified landcover from site metadata (from Hollie)
        fvals = np.zeros(9, dtype=np.float32)
        if (landcover_text=='Grassland' or 
            landcover_text=='Fen' or 
            landcover_text=='Bog'): # possibly want to introduce a new type of grassland for bog/fen
                fvals[2] = 1
        elif (landcover_text=='Agriculture' or # need to know whether agriculture means grazing grass or crops?
              landcover_text=='Bioenergy'):
                fvals[4] = 1
        elif (landcover_text=='Shrubland'):
                fvals[3] = 1
        with open(frac_out, 'w') as fo:
                fo.write(header)
                fo.write('%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t%1.8g\t' % 
                    (fvals[0], fvals[1], fvals[2], fvals[3], fvals[4], fvals[5], fvals[6], fvals[7], fvals[8]))

    if not Path(carb_out).exists():
        if True: # Matt's carbon output format    
            #carb = xr.open_dataset('/gws/nopw/j04/ceh_generic/matwig/v28_JULES_Bench/1/Clim_SL1_J_Q_MBulk/Carbon_Eqstate_monthly_pool.nc', decode_times=False)
            carb = xr.open_dataset('/gws/nopw/j04/ceh_generic/netzero/cs_states/Carbon_Eqstate_monthly_pool_bc.nc', decode_times=False)
            # with scpools: (decomposable plant material, resistant plant material, biomass, humus) andf Total
            carb = carb.isel(y=y_ind, x=x_ind)
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
        else:
            # place holder carbon
            carb = xr.open_dataset(es.chess_ancil_dir + '/chess-scape/chess-scape_uk_1km_soilcarbon_hwsd_vg.nc')
            carb = carb.isel(y=y_ind, x=x_ind)
            cs_array = carb.cs.values.flatten()
            cs_array = np.repeat(cs_array[None, None, ...], 4, 0) / 4.
            
            carb.close()
            del(carb)
            
            ds = xr.Dataset(
                data_vars=dict(
                    cs=(["scpool", "sclayer", "land"], cs_array),                
                ),
                coords=dict(
                    scpool=(["scpool"], np.array([1.,2.,3.,4.], dtype=np.float32)),
                ),            
            )
            ds.to_netcdf(carb_out)
            
            # if '_NCProperties' in carb.attrs.keys(): # netcdf3 error
                # carb.attrs.pop('_NCProperties')
        
    ### create config files pointing to these files to patch into the name lists
    ## ancil config
    conf_template_f = es.data_directory + 'rose-app-ancil-TEMPLATE.conf'
    config = configparser.ConfigParser(strict=False)
    config.read(conf_template_f)
    
    config['namelist:jules_frac']['file'] = f"'{frac_out}'"
    config['namelist:jules_latlon']['const_val'] = f'{lat},{lon}'
    config['namelist:jules_soil_props']['file'] = f"'{soil_out}'"
    config['namelist:jules_pdm']['file'] = f"'{topo_out}'"
    
    if run_type=='longspin':
        config['namelist:jules_initial']['file'] = f"'{carb_out}'"
        #config['namelist:jules_initial']['const_val'] = "0.9,0.0,0.0,50.0,275.0,278.0,10.0,0.0,1.0,2.0,0.5,3.0"
        config['namelist:jules_initial']['dump_file'] = ".false."
        config['namelist:jules_initial']['nvars'] = "12"
        config['namelist:jules_initial']['use_file'] = ".false.,.false.,.false.,.false.,.false.,.false.,.true.,.false.,.false.,.false.,.false.,.false."
        config['namelist:jules_initial']['var'] = "'sthuf','canopy','snow_tile','rgrain','tstar_tile','t_soil','cs','gs','lai','canht','sthzw','zw'"
        config['namelist:jules_initial']['var_name'] = "'sthuf','canopy','snow_tile','rgrain','tstar_tile','t_soil','cs','gs','lai','canht','sthzw','zw'"
    else:
        # we load from the dump file
        config['namelist:jules_initial']['file'] = f"'{dump_file_path}'"
        #config['namelist:jules_initial']['const_val'] = ""
        config['namelist:jules_initial']['dump_file'] = ".true."
        config['namelist:jules_initial']['nvars'] = "0"
        config['namelist:jules_initial']['use_file'] = ".true."
        config['namelist:jules_initial']['var'] = ""
        config['namelist:jules_initial']['var_name'] = ""
        
    with open(a_config_out, 'w') as fo:
        config.write(fo)

    ## drive config
    conf_template_f = es.data_directory + 'rose-app-drive-TEMPLATE.conf'
    config = configparser.ConfigParser(strict=False)
    config.read(conf_template_f)
    
    config['namelist:jules_drive']['file'] = f"'{met_drive_path}'"
    config['namelist:jules_drive']['data_start'] = f"'{data_start}'"
    config['namelist:jules_drive']['data_end'] = f"'{data_end}'"
    config['namelist:jules_drive']['data_period'] = f'{data_period}' # driving data time step
    config['namelist:jules_time']['timestep_len'] = "3600" # model time step, 30 or 60 mins
    if data_period==86400:
        # using daily data
        # check the interpolation/averaging flags
        config['namelist:jules_drive']['var'] = "'t','sw_down','lw_down','q','pstar','precip','wind','dt_range'"
        config['namelist:jules_drive']['nvars'] = "8"
        config['namelist:jules_drive']['interp'] = f"8*'nf'" # not used if daily disagg is true
        config['namelist:jules_drive']['l_daily_disagg'] = ".true."
        config['namelist:jules_drive']['l_disagg_const_rh'] = ".true."
        config['namelist:jules_drive']['dur_conv_rain'] = "7200"
        config['namelist:jules_drive']['dur_ls_rain'] = "18000"
        config['namelist:jules_drive']['dur_conv_snow'] = "3600"
        config['namelist:jules_drive']['dur_ls_snow'] = "18000"
        config['namelist:jules_drive']['precip_disagg_method'] = "2"
        config['namelist:jules_drive']['tpl_name'] = "8*''"
        config['namelist:jules_drive']['var_name'] = "8*''"
    else:
        config['namelist:jules_drive']['var'] = "'t','sw_down','lw_down','q','pstar','precip','wind'"
        config['namelist:jules_drive']['nvars'] = "7"
        config['namelist:jules_drive']['interp'] = f"7*'nb'" # backward-averaged from 30min end?
        config['namelist:jules_drive']['l_daily_disagg'] = ".false."
        config['namelist:jules_drive']['l_disagg_const_rh'] = ".false."
        config['namelist:jules_drive']['dur_conv_rain'] = ""
        config['namelist:jules_drive']['dur_ls_rain'] = ""
        config['namelist:jules_drive']['dur_conv_snow'] = ""
        config['namelist:jules_drive']['dur_ls_snow'] = ""
        config['namelist:jules_drive']['precip_disagg_method'] = ""
        config['namelist:jules_drive']['tpl_name'] = "7*''"
        config['namelist:jules_drive']['var_name'] = "7*''"
    
    if run_type=='longspin':
        config['namelist:jules_spinup']['max_spinup_cycles'] = "40"
        config['namelist:jules_spinup']['terminate_on_spinup_fail'] = ".true."
        config['namelist:jules_spinup']['tolerance'] = "-1.0,-1.0,-1.0,-1.0"
        # also set initial conditions to be defaults?
    else:
        config['namelist:jules_spinup']['max_spinup_cycles'] = "4"
        config['namelist:jules_spinup']['terminate_on_spinup_fail'] = ".false."
        config['namelist:jules_spinup']['tolerance'] = "0.1,0.01,0.001,0.001"
        # also set initial conditions to be the dump file from the long spin up
            
    with open(d_config_out, 'w') as fo:
        config.write(fo)

def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def fill_site_specific_env_vars(site, run_type, output_path):
    ## site specific
    env_vars = es.env_vars.copy()
    if run_type=='longspin':
        env_vars["$OUTDIR"] = output_path + f'/{site}/longspin_dump/'
    else:
        env_vars["$OUTDIR"] = output_path + f'/{site}/'
    Path(env_vars["$OUTDIR"]).mkdir(parents=True, exist_ok=True)
    env_vars["$RUNID"] = f'jrun_{site}'
    env_vars["$RUNBEGDATE"] = es.site_info.loc[site, 'mainrun_start_date'] + ' 00:00:00' # daily data fudge
    env_vars["$RUNENDDATE"] = es.site_info.loc[site, 'mainrun_end_date'] + ' 00:00:00'
    env_vars["$SPINBEGDATE"] = es.site_info.loc[site, 'spinup_start_date'] + ' 00:00:00'
    env_vars["$SPINENDDATE"] = es.site_info.loc[site, 'spinup_end_date'] + ' 00:00:00'
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
