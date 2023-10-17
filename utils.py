import xarray as xr
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from fluxtower_data import FluxTowerData

proj_dir = '/home/users/doran/projects/carbon_prediction/'
data_dir = '/home/users/doran/data_dump/flux_tower'

def read_one_fluxtower_site(lookup_row, missing_val=-9999.0):
    data = FluxTowerData(lookup_row)
    data.read_metadata()
    data.read_data(missing_val)
    return data

def provide_fluxtower_data(metadata, aggregate='daily', forcenew=False):
    Path(proj_dir+'/data/').mkdir(exist_ok=True, parents=True)
    fname = proj_dir+'/data/dynamics_at_fluxtower_sites.pkl'
    if forcenew is False and Path(fname).is_file():
        with open(fname, 'rb') as fo:
            dynamics = pickle.load(fo)
        dynamic_data = dynamics.pop('data')        
    else:
        # read data for each site, aggregating 
        # sub-daily data to daily means or sums (precip)
        dynamic_data = {}        
        for i in range(metadata.lookup.shape[0]):
            data = read_one_fluxtower_site(metadata.lookup.iloc[i], missing_val=-9999.0)            
            if data.dt_name.upper()=='DATE':
                ## already daily
                daily_dynamics = data.data
                daily_dynamics.index = daily_dynamics.index.rename('DATE_TIME')                
                if data.SID in dynamic_data.keys():
                    dynamic_data[data.SID] = pd.concat([dynamic_data[data.SID], data.data], axis=0)
                else:
                    dynamic_data[data.SID] = data.data
            else:
                if aggregate=='daily':
                    ## resample to daily
                    try:
                        precip_name = data.find_name('PRECIP')
                        daily_rainfall = data.data[[precip_name]].resample('D').sum()
                    except:
                        precip_name = None
                        daily_rainfall = pd.DataFrame()                
                    other_names = np.setdiff1d(data.metadata.Name, precip_name)
                    daily_others = (data.data[list(np.setdiff1d(data.data.columns, precip_name))]
                                        .resample('D').mean())
                    if precip_name is None:
                        daily_dynamics = daily_others
                    else:
                        daily_dynamics = pd.merge(daily_rainfall, daily_others, on=data.dt_name)
                    daily_dynamics.index = daily_dynamics.index.rename('DATE_TIME')
                    if data.SID in dynamic_data.keys():
                        dynamic_data[data.SID] = pd.concat([dynamic_data[data.SID], daily_dynamics], axis=0)
                    else:
                        dynamic_data[data.SID] = daily_dynamics
                else:
                    dynamics = data.data
                    dynamics.index = dynamics.index.rename('DATE_TIME')                
                    if data.SID in dynamic_data.keys():
                        dynamic_data[data.SID] = pd.concat([dynamic_data[data.SID], dynamics], axis=0)
                    else:
                        dynamic_data[data.SID] = dynamics
        dynamics = dict(data=dynamic_data)
        with open(fname, 'wb') as fs:
            pickle.dump(dynamics, fs)
    return dynamic_data
