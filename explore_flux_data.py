import pickle
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from plotnine import (ggplot, aes, geom_point, geom_line,
                      theme_light, facet_wrap)

from pathlib import Path

from fluxtower_data import FluxTowerMetaData
from utils import provide_fluxtower_data, proj_dir

metadata = FluxTowerMetaData()

# load all data
data = provide_fluxtower_data(metadata, aggregate=None, forcenew=False)

subdat = data['EF-LN'][['NEE', 'Ta', 'SWin']].reset_index().melt(id_vars='DATE_TIME')
scaled_vals = subdat.value / subdat.groupby('variable').value.transform(np.max)
subdat['value'] = scaled_vals
ggplot(data=subdat) + geom_line(aes(x='DATE_TIME', y='value', color='variable'), alpha=0.5)

subdat = data['EF-LN'].query('SWin>0')[['NEE', 'Ta']].dropna()
ggplot(data=subdat) + geom_point(aes(x='Ta', y='NEE')) + theme_light()

# combine sites, subset to daytime data
subdat = pd.DataFrame()
for k in data:
    if 'NEE' in data[k].columns:
        SW_name = [s for s in data[k].columns if s.casefold()=='SWIN'.casefold()]    
        if 'Ta' in data[k]:
            thisdat = data[k][data[k][SW_name[0]]>0][['NEE', 'Ta']].dropna()
        elif 'TA' in data[k]:
            thisdat = data[k][data[k][SW_name[0]]>0][['NEE', 'TA']].dropna()
        elif 'Tair' in data[k]:
            thisdat = data[k][data[k][SW_name[0]]>0][['NEE', 'Tair']].dropna()
        elif 'TAir' in data[k]:
            thisdat = data[k][data[k][SW_name[0]]>0][['NEE', 'TAir']].dropna()
        else:
            continue        
        thisdat.columns = ['NEE', 'TA']
        subdat = pd.concat([subdat, thisdat.assign(SITE_ID=k)], axis=0)
subdat = subdat.reset_index()

(ggplot(data=subdat) + 
    geom_point(aes(x='TA', y='NEE', color='SITE_ID'), alpha=0.2) + 
    theme_light())

(ggplot(data=subdat) + 
    facet_wrap(facets='~SITE_ID', scales="free_y") + 
    geom_point(aes(x='TA', y='NEE', color='SITE_ID'), alpha=0.7))






