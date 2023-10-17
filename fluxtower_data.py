import numpy as np
import pandas as pd

from soil_moisture.utils import preprocess, data_dir
fldr = data_dir + '/flux_tower/ceh_flux_towers/'

class FluxTowerData():

    def __init__(self, lookup_row):
        self.info = lookup_row
        self.SID = self.info['SITE_ID']
        self.data_file = fldr + self.info['DATA_FILE']
        self.metadata_file = fldr + '/metadata/' + self.info['METADATA_FILE']
        self.lat = self.info['LATITUDE']
        self.lon = self.info['LONGITUDE']
        self.data = None

    def read_metadata(self):
        self.metadata = pd.read_csv(self.metadata_file)

    def read_data(self, missing_val=np.nan):
        self.data = pd.read_csv(self.data_file, index_col=False)
        gotdate = False
        try:
            mname, self.dt_name = self.find_name('DATE')
            gotdate = True            
        except:
            pass
        
        if gotdate is False:
            if 'HyTES_Campaign' in self.data_file:
                self.dt_name = 'DateTime'
                self.data = self.data.assign(DateTime = pd.date_range('2019-06-22 00:30', '2019-07-06 00:00', freq='30min'))
                self.data = self.data.drop(['TIME_START', 'TIME_END'], axis=1)
        
        if self.dt_name.upper() == 'DATE':
            self.data = preprocess(self.data, missing_val, self.dt_name, add_stamp=' 00:00:00')
        else:
            self.data = preprocess(self.data, missing_val, self.dt_name)

    def find_name(self, STEM):
        m_name = np.asarray(
            self.metadata.Name[[STEM in self.metadata.Name[i].upper() for i in range(self.metadata.shape[0])]]
            )[0]
        d_name = np.asarray(
            self.data.columns[[STEM in self.data.columns[i].upper() for i in range(len(self.data.columns))]]
            )[0]        
        return m_name, d_name
    
    def read_QC_flags(self):
        # placeholder
        pass


class FluxTowerMetaData():

    def __init__(self):
        self.metadata_path = fldr + '/metadata/'
        self.lookup = pd.read_csv(fldr + '/metadata/site_lookup.csv')
        self.depths = pd.read_csv(fldr + '/metadata/soil_measurement_depth_lookup.csv')
        self.depths = self.depths.assign(depth_cm = lambda x: (x.depth * 100).astype(np.int32))

