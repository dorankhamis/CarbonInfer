#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: latin-1 -*-

# Python script to derive namelists

# Garry Hayman
# UK Centre for Ecology & Hydrology
# November 2022

# Import standard python modules

import os

import sys
import glob
import cartopy.crs as ccrs

import datetime as dt
import numpy as np
import pandas as pd

import data_netCDF
import write_netCDF_py3


# In[2]:


ALL_SITE_CODES = [         #"LBA-BAN",  "LBA-K34",  "LBA-K83",  "LBA-RJA",  "LBA-K67", \
        #"LBA-K77",  "LBA-FNS",  "LBA-PDG", \ 
        #"NEO-SCBI", "NEO-ABBY", "NEO-CPER", "NEO-BLAN", "NEO-BART", \
        "AT-Neu", "AU-Fog", "BE-Vie", "BR-Sa1", "BR-Sa3", "CA-Oas", "CG-Tch", "CH-Cha", "CH-Oe1", "CH-Oe2", \
        "CN-Cha", "CN-Cng", "CN-Dan", "CN-Din", "CN-Du2", "CN-Du3", "CN-HaM", "CN-Ha2", "CN-Qia", "CN-Sw2", \
        "CZ-wet", "DE-Akm", "DE-SfN", "DE-Spw", "DE-Tha", "DE-Zrk", "GL-NuF", "GL-ZaF", "ES-Amo", "FI-Hyy", \
        "FI-Lom", "FR-Gri", "FR-Pue", "GF-Guy", "IT-CA1", "IT-Col", "IT-Cpz", "IT-Cp2", "IT-Noe", "IT-Ren", \
        "IT-SRo", "SJ-Adv", "RU-Che", "RU-SkP", "SD-Dem", "SE-St1", \
        "UK-AMo", "UK-Arn", "UK-Bam", "UK-BBB", "UK-BBC", "UK-BnB", "UK-CLs", "UK-Cst", "UK-Dke", "UK-DkF", \
        "UK-EBu", "UK-Ech", "UK-EHd", "UK-ESa", "UK-GaB", "UK-Gnn", "UK-Gri", "UK-Gst", "UK-GtF", "UK-Gwr", \
        "UK-Ham", "UK-Har", "UK-Her", "UK-LBT", "UK-Lns", "UK-MrH", "UK-Myg", "UK-PL1", "UK-PL2", "UK-PL3", \
        "UK-Png", "UK-Po1", "UK-Po2", "UK-Po3", "UK-Pob", "UK-Rdm", "UK-Rsh", "UK-Stm", "UK-Swt", "UK-Tad", \
        "UK-WdC", "UK-Wdd", "UK-Wot", \
        "US-Atq", "US-Blo", "US-Ha1", "US-Ivo", "US-Los", "US-MMS", "US-Myb", "US-Ne1", "US-Ne2", "US-Ne3", \
        "US-ORv", "US-PFa", "US-SRG", "US-SRM", "US-Ton", "US-Tw1", "US-Tw4", "US-UMB", "US-Var", "US-WCr", \
        "US-Whs", "US-Wkg", "US-WPT", "ZA-Kru", "ZM-Mon"
                 ]


# In[3]:


# Parse date
# Uses the datetime module (imported as dt)
# https://docs.python.org/2/library/datetime.html#

def parse_datetime(DATE,iOPT):

    # Date format YYYY-MM-DD
    if iOPT == 1:
        YEAR,MONTH,DAY     = int(DATE[0:4]),int(DATE[5:7]),int(DATE[8:10])
        oDATE         = dt.datetime(YEAR,MONTH,DAY,0,0,0)
        return YEAR,MONTH,DAY,oDATE

    # Date format YYYY-MM-DD HH:MM:SS
    elif iOPT == 2:
        YEAR,MONTH,DAY     = int(DATE[0:4]),int(DATE[5:7]),int(DATE[8:10])
        HOUR,MINUTE,SECOND = int(DATE[11:13]),int(DATE[14:16]),int(DATE[18:19])
        oDATE         = dt.datetime(YEAR,MONTH,DAY,HOUR,MINUTE,SECOND)
        return YEAR,MONTH,DAY,HOUR,MINUTE,SECOND,oDATE

    # Date format dd/MM/YYYY
    if iOPT == 3:
        YEAR,MONTH,DAY     = int(DATE[6:10]),int(DATE[3:5]),int(DATE[0:2])
        oDATE         = dt.datetime(YEAR,MONTH,DAY,0,0,0)
        return YEAR,MONTH,DAY,oDATE

    # Date format dd/MM/YYYY HH:MM:SS
    if iOPT == 4:
        YEAR,MONTH,DAY     = int(DATE[6:10]),int(DATE[3:5]),int(DATE[0:2])
        HOUR,MINUTE,SECOND = int(DATE[11:13]),int(DATE[14:16]),int(DATE[18:19])
        oDATE         = dt.datetime(YEAR,MONTH,DAY,HOUR,MINUTE,SECOND)
        return YEAR,MONTH,DAY,HOUR,MINUTE,SECOND,oDATE


# In[4]:


# User-defined functions using the pandas module (imported as pd)
# See https://pandas.pydata.org/docs/user_guide/index.html
# See cheat sheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf 

# Get data from a csv file into a data frame 
def get_df_from_csv(DIR,FILE):
    DF           = pd.read_csv(DIR+FILE)
    return DF

# Get data from an excel spreadsheet into a data frame 
def get_df_from_excel(DIR,FILE,SHEET):
    DF           = pd.read_excel(DIR+FILE,SHEET)
    return DF


# In[5]:


# User-defined functions to convert vapour pressure deficit to humidty

Tref         = 273.15
Pref         = 101325.0
RMM_water    = 18.0160 # molecular weight of water
RMM_air      = 28.9660 # molecular weight of dry air

def SVP_from_T(T):
    '''Derive saturated vapour pressure (units [Pa]) for a given air temperature (units [K])'''
    logT_Tref    = np.log10(T/Tref)
    
    SVP          = Pref*10**(10.79586*(1-Tref/T)-5.02808*logT_Tref+                    1.50474*1e-4*(1.-10**(-8.29692*(T/Tref-1)))+                    0.42873*1e-3*(10**(4.76955*(1-Tref/T))-1)-2.2195983)
    return SVP

def RH_from_VPD_T(VPD, T):
    # VPD Units = Pa
    # T Units = Kelvin
    SVP          = SVP_from_T(T)
    RH           = (1.0-(VPD/SVP))*100.0
    return RH

def RH_2_mixing_ratio(RH, P, T):
    '''Convert relative humidity to water vapour mixing ratio'''
    SVP          = SVP_from_T(T)
    return (RMM_water/RMM_air)*(RH/100.0)*SVP/(P-(RH/100.0)*SVP)*1000.0

def mixing_ratio_2_specific_humidity(MR_water):
    '''Convert mixing ratio (units [kg/kg]) to specific humidity (units also [kg/kg])'''
    return MR_water/(1.0+MR_water)

def RH_2_specific_humidity(RH, P, T):
    '''conversion from relative humidity (units %) to specific humidity (units [kg/kg])'''
    MR_water     = RH_2_mixing_ratio(RH, P, T)
    return mixing_ratio_2_specific_humidity(MR_water*1.0e-3)


# In[6]:


def get_site_metadata(DIR, SITE_CODES, DEBUG):
    
    SITE_DATA    = { SITE_CODE:{                         'site_name':'',   'site_code':'',
                        'latitude':'',    'longitude':'',   'easting':'',     'northing':'', \
                        'drive_start':'', 'drive_end':'',   'drive_tstep':'', \
                        'jules_start':'', 'jules_end':'',   'jules_tstep':'', \
                        'lai_start':'',   'lai_end':'',     'lai_tstep':'',   \
                        'sthuf_start':'', 'sthuf_end':'',   'sthuf_tstep':'', \
                        'presc_data':'',  'presc_levels':'','top_mod_opt':'', \
                        'file_top':'',    'file_fracs':'',  'file_soil':'',   \
                        'file_drive':'',  'file_flux':'',   'file_lai':''   \
                               } for SITE_CODE in SITE_CODES }

    # Get metadata
    DF_META          = get_df_from_excel(DIR, 'MotherShip_Site_Data_202211.xlsx', 'Site_Meta')
    DF_ANCIL         = get_df_from_excel(DIR, 'MotherShip_Site_Data_202211.xlsx', 'Site_Ancil')
    DF_DRIVE         = get_df_from_excel(DIR, 'MotherShip_Site_Data_202211.xlsx', 'Site_Drive')
    
    # Convert time to 'YYYY-MM-DD HH:MM:SS'
    DF_DRIVE['JULES Start'] = DF_DRIVE['JULES Start'].dt.strftime('%Y-%m-%d %H:%M:%S')
    DF_DRIVE['JULES End']   = DF_DRIVE['JULES End'].dt.strftime('%Y-%m-%d %H:%M:%S')
    DF_DRIVE['Drive Start'] = DF_DRIVE['Drive Start'].dt.strftime('%Y-%m-%d %H:%M:%S')
    DF_DRIVE['Drive End']   = DF_DRIVE['Drive End'].dt.strftime('%Y-%m-%d %H:%M:%S')
    DF_DRIVE['STHUF Start'] = DF_DRIVE['STHUF Start'].dt.strftime('%Y-%m-%d %H:%M:%S')
    DF_DRIVE['STHUF End']   = DF_DRIVE['STHUF End'].dt.strftime('%Y-%m-%d %H:%M:%S')
    DF_DRIVE['LAI Start']   = DF_DRIVE['LAI Start'].dt.strftime('%Y-%m-%d %H:%M:%S')
    DF_DRIVE['LAI End']     = DF_DRIVE['LAI End'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract metadata for sites of interest
    for iSITE, SITE_CODE in enumerate(SITE_CODES):
        if DEBUG == 'Y':
            print(SITE_CODE)

        INDEX        = np.where(DF_META['Site Code'].values == SITE_CODE)[0]
        SITE_DATA[SITE_CODE]['site_code'] = SITE_CODE.replace('-','_')
        SITE_DATA[SITE_CODE]['site_name'] = DF_META['Site Name'][INDEX].values[0]
        SITE_DATA[SITE_CODE]['latitude']  = float(DF_META['Site Latitude'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['longitude'] = float(DF_META['Site Longitude'][INDEX].values[0])

        INDEX        = np.where(DF_ANCIL['Site Code'].values == SITE_CODE)[0]
        SITE_DATA[SITE_CODE]['file_fracs']  = str(DF_ANCIL['File Frac'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['file_soil']   = str(DF_ANCIL['File Soil'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['file_top']    = str(DF_ANCIL['File Top'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['top_mod_opt'] = float(DF_ANCIL['TopModel Option'][INDEX].values[0])
        
        INDEX        = np.where(DF_DRIVE['Site Code'].values == SITE_CODE)[0]
        SITE_DATA[SITE_CODE]['file_drive']  = str(DF_DRIVE['File Met Data'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['file_flux']   = str(DF_DRIVE['File Flux Data'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['file_lai']    = str(DF_DRIVE['File LAI Data'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['jules_start'] = str(DF_DRIVE['JULES Start'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['jules_end']   = str(DF_DRIVE['JULES End'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['jules_tstep'] = '%0d' % (DF_DRIVE['JULES Tstep'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['drive_start'] = str(DF_DRIVE['Drive Start'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['drive_end']   = str(DF_DRIVE['Drive End'][INDEX].values[0])
        SITE_DATA[SITE_CODE]['drive_tstep'] = '%0d' % (DF_DRIVE['Drive Tstep'][INDEX].values[0])

        SITE_DATA[SITE_CODE]['presc_data']  = '%0d' % (DF_DRIVE['Prescribed Data'][INDEX].values[0])
        if int(SITE_DATA[SITE_CODE]['presc_data']) >= 1:
            SITE_DATA[SITE_CODE]['sthuf_start'] = str(DF_DRIVE['STHUF Start'][INDEX].values[0])
            SITE_DATA[SITE_CODE]['sthuf_end']   = str(DF_DRIVE['STHUF End'][INDEX].values[0])
            SITE_DATA[SITE_CODE]['sthuf_tstep'] = '%0d' % (DF_DRIVE['STHUF Tstep'][INDEX].values[0])
            SITE_DATA[SITE_CODE]['presc_levels']= DF_DRIVE['Prescribed Levels'][INDEX].values[0]
        
        if int(SITE_DATA[SITE_CODE]['presc_data']) >= 2:
            SITE_DATA[SITE_CODE]['lai_start']   = str(DF_DRIVE['LAI Start'][INDEX].values[0])
            SITE_DATA[SITE_CODE]['lai_end']     = str(DF_DRIVE['LAI End'][INDEX].values[0])
            SITE_DATA[SITE_CODE]['lai_tstep']   = '%0d' % (DF_DRIVE['LAI Tstep'][INDEX].values[0])

    # Convert latitudes and longitudes to OSGB36 grid for UK sites        
    for iSITE, SITE_CODE in enumerate(SITE_CODES):
        if 'UK' in SITE_CODE:
            LON                = SITE_DATA[SITE_CODE]['longitude']
            LAT                = SITE_DATA[SITE_CODE]['latitude']
            OS_GRID            = ccrs.OSGB().transform_point( LON, LAT, ccrs.PlateCarree() )

            SITE_DATA[SITE_CODE]['easting']    = OS_GRID[0]
            SITE_DATA[SITE_CODE]['northing']   = OS_GRID[1]
            print('Metadata: ',SITE_CODE, LON, LAT,                  SITE_DATA[SITE_CODE]['easting'], SITE_DATA[SITE_CODE]['northing'])

    return SITE_DATA


# In[7]:


def output_namelist_ancil(DIR, SITE_CODE, SITE_DATA, DEBUG):
    
    NML_JULES = [         "# automatically generated file: do not edit manually",
        "",
        "[namelist:jules_frac]",
        "file='$CYLC_SUITE_RUN_DIR/ancil/tilefracs/FILE_SITE_FRACS'",
        "read_from_dump=.false.",
        "",
        "[namelist:jules_latlon]",
        "const_val=sLAT,sLON",
        "",
        "[namelist:jules_soil_props]",
        "!!const_val=10*0.0",
        "const_z=.true.",
        "file='$CYLC_SUITE_RUN_DIR/ancil/soil/FILE_SITE_SOIL'",
        "nvars=10",
        "read_from_dump=.false.",
        "tpl_name=10*''",
        "use_file=10*.true.",
        "var='b','sathh','satcon','sm_sat','sm_crit','sm_wilt','hcap',",
        "   ='hcon','albsoil', 'clay'",
        "var_name='','','','','','','','','',''",
        "",    
        "[namelist:jules_top]",
        "const_val=1.0,8.066711,2.067616",
        "file='FILE_TOP'",
        "nvars=3",
        "read_from_dump=.false.",
        "tpl_name='','',''",
        "use_file=.false.,.true.,.true.",
        "var='fexp','ti_mean','ti_sig'",
        "var_name='field900_2','field900','field900_1'",
        ""
                ]

    NML_SOIL  = [         "[namelist:jules_soil]\n",
        "l_vg_soil=.true.\n",
        "\n",
        "[namelist:jules_soil_props]",
                ]
    
    # Writing output
    NML_FILE_OUT       = DIR+'app/jules/opt/rose-app-ancil-'+SITE_DATA[SITE_CODE]['site_code']+'.conf'
    print("Writing to "+NML_FILE_OUT)
    NML_FID            = open(NML_FILE_OUT,'w')

    for LINE in NML_JULES:
        OUTLINE            = LINE
        
        if "namelist:jules_soil_props" in LINE and 'UK' in SITE_CODE:
            OUTLINE            = ''
            for LINE_SOIL in NML_SOIL:
                OUTLINE           = OUTLINE+LINE_SOIL

        if "FILE_SITE_FRACS" in LINE:
            FILE_SITE_FRACS    = SITE_DATA[SITE_CODE]['file_fracs']
            OUTLINE            = OUTLINE.replace("FILE_SITE_FRACS",FILE_SITE_FRACS) 

        if "FILE_SITE_SOIL" in LINE:
            if 'UK' in SITE_CODE:
                FILE_SITE_SOIL     = SITE_DATA[SITE_CODE]['file_soil']+"/"+SITE_CODE.replace("-","_")+"_soil_"+                                      SITE_DATA[SITE_CODE]['file_soil']+"_vg.dat"
            else:
                FILE_SITE_SOIL     = SITE_DATA[SITE_CODE]['file_soil']+"/"+SITE_CODE.replace("-","_")+"_soil_"+                                      SITE_DATA[SITE_CODE]['file_soil']+".dat"
                
            OUTLINE            = OUTLINE.replace("FILE_SITE_SOIL",FILE_SITE_SOIL) 

        if "FILE_SOIL" in LINE:
            FILE_SOIL          = SITE_DATA[SITE_CODE]['file_soil']
            OUTLINE            = OUTLINE.replace("FILE_SOIL",FILE_SOIL) 

        if "FILE_TOP" in LINE:
            TOP_MOD_OPT        = int(SITE_DATA[SITE_CODE]['top_mod_opt'])
            if TOP_MOD_OPT == 1:
                FILE_TOP           = "dummy.dat"
            if TOP_MOD_OPT == 2:
                FILE_TOP           = "$CYLC_SUITE_RUN_DIR/ancil/topmodel/"+SITE_DATA[SITE_CODE]['file_top']
            OUTLINE            = OUTLINE.replace("FILE_TOP",FILE_TOP) 

        if "sLAT,sLON" in LINE:
            sLAT               = '%.4f' % SITE_DATA[SITE_CODE]['latitude']
            sLON               = '%.4f' % SITE_DATA[SITE_CODE]['longitude']
            OUTLINE            = OUTLINE.replace("sLAT,sLON",sLAT+","+sLON)
            
        NML_FID.write(OUTLINE+'\n')
            
    NML_FID.close()

    return


# In[8]:


def output_namelist_drive(DIR, SITE_CODE, SITE_DATA, DEBUG):
    
    NML_JULES = [         "# automatically generated file: do not edit manually",
        "",
        "[namelist:jules_drive]",
        "data_end='DRIVE_END'",
        "data_period=DRIVE_TSTEP",
        "data_start='DRIVE_START'",
        "file='$FLUXNET2015_DRIVE_DIR/SITE_CODE-met.dat'",
        "interp=7*'nf'",
        "var='t','sw_down','lw_down','q','pstar','precip','wind'",
        "",
        "[namelist:jules_time]",
        "timestep_len=JULES_TSTEP",
        "",
        "[namelist:jules_time]",
        "l_360=.false.",
        "l_leap=.true."
                      ]

    # Writing output
    NML_FILE_OUT       = DIR+'app/jules/opt/rose-app-drive-'+SITE_DATA[SITE_CODE]['site_code']+'.conf'
    print("Writing to "+NML_FILE_OUT)
    NML_FID            = open(NML_FILE_OUT,'w')

    for LINE in NML_JULES:
        OUTLINE            = LINE 

        if "SITE_CODE" in LINE:
            SITE_CODE_U        = SITE_DATA[SITE_CODE]['site_code']
            OUTLINE            = OUTLINE.replace("SITE_CODE",SITE_CODE_U) 
        if "DRIVE_START" in LINE:
            DRIVE_START        = SITE_DATA[SITE_CODE]['drive_start']
            OUTLINE            = OUTLINE.replace("DRIVE_START",DRIVE_START) 
        if "DRIVE_END" in LINE:
            DRIVE_END          = SITE_DATA[SITE_CODE]['drive_end']
            OUTLINE            = OUTLINE.replace("DRIVE_END",DRIVE_END) 
        if "DRIVE_TSTEP" in LINE:
            DRIVE_TSTEP        = SITE_DATA[SITE_CODE]['drive_tstep']
            OUTLINE            = OUTLINE.replace("DRIVE_TSTEP",DRIVE_TSTEP) 
        if "JULES_TSTEP" in LINE:
            JULES_TSTEP        = SITE_DATA[SITE_CODE]['drive_tstep']
            OUTLINE            = OUTLINE.replace("JULES_TSTEP",JULES_TSTEP) 
            
        NML_FID.write(OUTLINE+'\n')
            
    NML_FID.close()

    return


# In[9]:


def output_namelist_presc_sthuf(DIR, SITE_CODE, SITE_DATA, DEBUG):
    
    NML_JULES = [         "# automatically generated file: do not edit manually",
        "",
        "[namelist:jules_prescribed_dataset(2)]",
        "data_end='STHUF_END'",
        "data_period=STHUF_TSTEP",
        "data_start='STHUF_START'",
        "file='$STHUF_DIR/prescribed_sthuf_SITE_CODE.txt'",
        "interp='nf'",
        "is_climatology=.false.",
        "!!nfiles=0",
        "nvars=1",
        "prescribed_levels=PRESC_LEVELS",
        "read_list=.false.",
        "tpl_name=''",
        "var='sthuf'",
        "var_name='sthuf'"
                      ]

    # Writing output
    NML_FILE_OUT       = DIR+'app/jules/opt/rose-app-presc-sthuf-'+SITE_DATA[SITE_CODE]['site_code']+'.conf'
    print("Writing to "+NML_FILE_OUT)
    NML_FID            = open(NML_FILE_OUT,'w')

    for LINE in NML_JULES:
        OUTLINE            = LINE 

        if "SITE_CODE" in LINE:
            #SITE_CODE_U        = SITE_DATA[SITE_CODE]['site_code']
            #OUTLINE            = OUTLINE.replace("SITE_CODE",SITE_CODE_U)
            OUTLINE            = OUTLINE.replace("SITE_CODE",SITE_CODE)
        if "STHUF_START" in LINE:
            STHUF_START        = SITE_DATA[SITE_CODE]['sthuf_start']
            OUTLINE            = OUTLINE.replace("STHUF_START",STHUF_START) 
        if "STHUF_END" in LINE:
            STHUF_END          = SITE_DATA[SITE_CODE]['sthuf_end']
            OUTLINE            = OUTLINE.replace("STHUF_END",STHUF_END) 
        if "STHUF_TSTEP" in LINE:
            STHUF_TSTEP        = SITE_DATA[SITE_CODE]['sthuf_tstep']
            OUTLINE            = OUTLINE.replace("STHUF_TSTEP",STHUF_TSTEP) 
        if "PRESC_LEVELS" in LINE:
            PRESC_LEVELS       = SITE_DATA[SITE_CODE]['presc_levels']
            OUTLINE            = OUTLINE.replace("PRESC_LEVELS",PRESC_LEVELS[1:-1]) 
            
        NML_FID.write(OUTLINE+'\n')
            
    NML_FID.close()

    return


# In[10]:


def output_namelist_presc_lai(DIR, SITE_CODE, SITE_DATA, DEBUG):
    
    NML_JULES = [         "# automatically generated file: do not edit manually",
        "",
        "[namelist:jules_prescribed_dataset(3)]",
        "data_end='LAI_END'",
        "data_period=LAI_TSTEP",
        "data_start='LAI_START'",
        "file='$LAI_DIR/LAI_FILE.txt'",
        "interp='nf'",
        "is_climatology=.false.",
        "!!nfiles=0",
        "nvars=1",
        "read_list=.false.",
        "tpl_name=''",
        "var='lai'",
        "var_name=''"
                      ]

    # Writing output
    NML_FILE_OUT       = DIR+'app/jules/opt/rose-app-presc-lai-'+SITE_DATA[SITE_CODE]['site_code']+'.conf'
    print("Writing to "+NML_FILE_OUT)
    NML_FID            = open(NML_FILE_OUT,'w')

    for LINE in NML_JULES:
        OUTLINE            = LINE 

        if "LAI_FILE" in LINE:
            LAI_FILE           = SITE_DATA[SITE_CODE]['file_lai']
            OUTLINE            = OUTLINE.replace("LAI_FILE",LAI_FILE) 
        if "LAI_START" in LINE:
            LAI_START        = SITE_DATA[SITE_CODE]['lai_start']
            OUTLINE            = OUTLINE.replace("LAI_START",LAI_START) 
        if "LAI_END" in LINE:
            LAI_END          = SITE_DATA[SITE_CODE]['lai_end']
            OUTLINE            = OUTLINE.replace("LAI_END",LAI_END) 
        if "LAI_TSTEP" in LINE:
            LAI_TSTEP        = SITE_DATA[SITE_CODE]['lai_tstep']
            OUTLINE            = OUTLINE.replace("LAI_TSTEP",LAI_TSTEP) 
            
        NML_FID.write(OUTLINE+'\n')
            
    NML_FID.close()

    return


# In[11]:


def output_var_info(DIR, ALL_SITE_CODES, INFO_UK_SITES, SITE_DATA, DEBUG):
    
    VAR_START = [         '',
        '# n.b. change IT_Cp2 presc_avail to 1 when prescribed_sthuf_IT-Cp2.txt goes in to next data version',
        '',
        '{%- set site_info = {',
        '        "LBA_BAN" : {"run_dates" : ["2004-01-02", "2006-10-31"], "presc_avail" : 2},',
        '        "LBA_K34" : {"run_dates" : ["2003-01-02", "2005-10-15"], "presc_avail" : 2},',
        '        "LBA_K83" : {"run_dates" : ["2001-01-02", "2003-08-12"], "presc_avail" : 2},',
        '        "LBA_RJA" : {"run_dates" : ["2000-02-03", "2002-09-13"], "presc_avail" : 2},',
        '        "LBA_K67" : {"run_dates" : ["2002-01-02", "2003-11-18"], "presc_avail" : 2},',
        '        "LBA_K77" : {"run_dates" : ["2001-01-02", "2005-12-31"], "presc_avail" : 0},',
        '        "LBA_FNS" : {"run_dates" : ["1999-01-02", "2001-12-31"], "presc_avail" : 0},',
        '        "LBA_PDG" : {"run_dates" : ["2002-01-02", "2003-12-31"], "presc_avail" : 0},',
        '        "NEO_SCBI": {"run_dates" : ["2017-03-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_ABBY": {"run_dates" : ["2017-10-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_CPER": {"run_dates" : ["2016-12-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_BLAN": {"run_dates" : ["2017-03-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_BART": {"run_dates" : ["2017-03-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_GUAN": {"run_dates" : ["2018-08-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_HARV": {"run_dates" : ["2017-09-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_TALL": {"run_dates" : ["2017-10-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_WREF": {"run_dates" : ["2018-06-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_CLBJ": {"run_dates" : ["2017-12-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_KONZ": {"run_dates" : ["2017-09-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_NIWO": {"run_dates" : ["2017-10-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_ONAQ": {"run_dates" : ["2017-10-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_ORNL": {"run_dates" : ["2017-09-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_BONA": {"run_dates" : ["2017-11-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_OSBS": {"run_dates" : ["2017-03-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_PUUM": {"run_dates" : ["2019-06-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_SJER": {"run_dates" : ["2018-10-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_TOOL": {"run_dates" : ["2017-10-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_YELL": {"run_dates" : ["2018-10-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_SRER": {"run_dates" : ["2017-10-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_UNDE": {"run_dates" : ["2017-03-02", "2021-12-31"], "presc_avail" : 0},',
        '        "NEO_WOOD": {"run_dates" : ["2017-10-02", "2021-12-31"], "presc_avail" : 0},',
                      ]
    
    VAR_END   = [         '     }%}',
        '',
        '     {%- set wetland_sites = ["LBA_BAN", "AU-Fog", "CN-Ha2",',
        '             "CZ_wet", "DE_Akm", "DE_SfN", "DE_Spw", "DE_Zrk", "DK_NuF", "DK_ZaF", "FI_Lom",',
        '             "NO_Adv", "RU-Che", "SE_St1", "US_Atq", "US_Ivo", "US_Los", "US_Myb", "US_ORv", "US_Tw1",',
        '             "US_Tw4", "US_WPT"]',
        '      %}',
        '',
        '     {%- set deposition_sites = ["CH_Oe1", "CH_Oe2", "FI_Hyy", "FR_Gri", "IT_Cpz", "IT_Cp2", "US_Ha1"]',
        '      %}',
        '',
        '     {%- set uk_flux_sites = XXXX %}',
        ''
                      ]

    # Writing output
    VAR_FILE_OUT       = DIR+'var/info.inc'
    print("Writing to "+VAR_FILE_OUT)
    VAR_FID            = open(VAR_FILE_OUT,'w')

    for LINE in VAR_START:
        VAR_FID.write(LINE+'\n')

    for SITE_CODE in ALL_SITE_CODES:
        RUN_START          = SITE_DATA[SITE_CODE]['jules_start'][:10]
        RUN_END            = SITE_DATA[SITE_CODE]['jules_end'][:10]
        PRESC_CODE         = SITE_DATA[SITE_CODE]['presc_data']
        
        OUTLINE            = '        "'+SITE_CODE.replace("-","_")+'"  : {"run_dates" : ["'+             RUN_START+'", "'+RUN_END+'"], "presc_avail" : '+PRESC_CODE+'},'

        VAR_FID.write(OUTLINE+'\n')
        
    UK_SITES           = '['
    for SITE_CODE in INFO_UK_SITES:
        UK_SITES           = UK_SITES+'"'+SITE_CODE.replace('-','_')+'", '
    UK_SITES           = UK_SITES[0:-2]+']'
    
    for LINE in VAR_END:
        if 'XXXX' in LINE:
            VAR_FID.write(LINE.replace('XXXX',UK_SITES)+'\n')
        else:
            VAR_FID.write(LINE+'\n')

    VAR_FID.close()

    return


# In[12]:


def input_ancil_soil_uk(SOIL_MODEL, VARS_OUT, DEBUG):
    
    # Get soil properties from CHESS data
    DIR_CHESS     = '/prj/chess/data/1km/v1.2/ancil_uncompressed/'
    FILE_TEXTURE  = 'chess_soil_texture_1km.nc'

    # Soil and grid variables
    if SOIL_MODEL == 'BC':
        print(''); print('Soil parameters: Brooks & Corey')
        FILE_SOIL     = 'chess_soilparams_hwsd_bc.nc'
        SOIL_VARS_IN  = ['b', 'sathh', 'satcon', 'vsat', 'vcrit', 'vwilt', 'hcap', 'hcon' ]
    elif SOIL_MODEL == 'VG':
        print(''); print('Soil parameters: Van Genuchten')
        FILE_SOIL     = 'chess_soilparams_hwsd_vg.nc'
        SOIL_VARS_IN  = ['oneovernminusone', 'oneoveralpha', 'satcon', 'vsat', 'vcrit', 'vwilt', 'hcap', 'hcon' ]

    GRID_VARS_IN  = ['x', 'y', 'lat', 'lon' ]

    VARS_IN       = GRID_VARS_IN +SOIL_VARS_IN

    DATA_SOIL     = { VAR:0.0 for VAR in VARS_OUT } 
    
    # Get soil properties from CHESS data
    for iVAR,VAR in enumerate(VARS_OUT):
        if VAR == 'albsoil':
            print('Albsoil')
        elif VAR == 'clay':
            print('Clay')
            DIMS,DATA_SOIL[VAR] = data_netCDF.data_netCDF_array_var(DIR_CHESS+FILE_TEXTURE,VAR)
        else:
            print(VAR)
            DIMS,DATA_SOIL[VAR] = data_netCDF.data_netCDF_array_var(DIR_CHESS+FILE_SOIL,VARS_IN[iVAR])

    return DATA_SOIL


# In[13]:


def output_ancil_soil_uk(DIR, SITE_CODE, SITE_DATA, DATA_SOIL, SOIL_VARS_OUT, SOIL_MODEL, DEBUG):

    FILE_SITE_SOIL     = SITE_DATA[SITE_CODE]['file_soil']+"/"+SITE_CODE.replace("-","_")+"_soil_"+                          SITE_DATA[SITE_CODE]['file_soil']+"_"+SOIL_MODEL+".dat"

    iEAST              = int(SITE_DATA[SITE_CODE]['easting']/1000)
    iNORTH             = int(SITE_DATA[SITE_CODE]['northing']/1000)
    print(SITE_CODE, iEAST, iNORTH)

    # Writing output
    NML_FILE_OUT       = DIR+'ancil/soil/'+FILE_SITE_SOIL
    print("Writing to "+NML_FILE_OUT)
    NML_FID            = open(NML_FILE_OUT,'w')

    NML_FID.write("#            'b'       'sathh'      'satcon'      'sm_sat'     'sm_crit'")
    NML_FID.write("     'sm_wilt'        'hcap'        'hcon'     'albsoil'        'clay'"+"\n")
    NML_FID.write("# automatically generated file: do not edit manually"+"\n")
    
    OUTLINE            = ''
    for VAR in SOIL_VARS_OUT:

        if VAR == 'albsoil':
            DATA_VAR      = 0.15
        else:
            DATA_VAR      = DATA_SOIL[VAR][0,iNORTH,iEAST]

        OUTLINE            = OUTLINE+('%14.6g' % DATA_VAR)

    NML_FID.write(OUTLINE+'\n')
            
    NML_FID.close()

    return


# In[14]:


def input_ancil_top(TOP_VARS, DEBUG):
    
    # Get soil properties from CHESS data
    DIR_TOP     = '/data/grp/eow/garr/Projects/MotherShip/Data/'
    FILE_TOP    = 'HadGEM2ES_Ancil.nc'

    print(''); print('Top Model parameters')
    DATA_TOP     = { VAR:0.0 for VAR in TOP_VARS } 
    
    # Get soil properties from CHESS data
    for iVAR,VAR in enumerate(TOP_VARS):
        print(VAR)
        DIMS,DATA_TOP[VAR] = data_netCDF.data_netCDF_array_var(DIR_TOP+FILE_TOP,VAR)

    return DATA_TOP


# In[15]:


def output_ancil_top(DIR, SITE_CODE, SITE_DATA, DATA_TOP, TOP_VARS, DEBUG):
    
    # Create filename from template
    FILE_CDF        = SITE_CODE.replace("-","_")+"_TOP.nc"
    DIR_CDF         = DIR+'ancil/topmodel/'
    NETCDF          = 'netCDF4'
    ATTRIB          = 'JULES_UK_Flux_Sites_namelist_generator.ipynb'
    
    VAR_META_DATA   = {                        'latitude':   {'name':'latitude',               'type':'float32', 'units':'degrees_north',                                       '_FillValue':float('NaN'),       'dependence': ['land']},                                          'longitude':  {'name':'longitude',              'type':'float32', 'units':'degrees_east',                                        '_FillValue':float('NaN'),       'dependence': ['land']},                                          'field900':   {'name':'mean topographic index', 'type':'float32', 'units':' ',                                                   '_FillValue':float('NaN'),       'dependence': ['land']},                                          'field900_1': {'name':'std topographic index',  'type':'float32', 'units':' ',                                                   '_FillValue':float('NaN'),       'dependence': ['land']},                                         }

    DIM_INFO        = [ [ 1, 'land'] ]
    
    VAR_INFO_ALL, VAR_DATA_ALL, VAR_DEP_ALL = [], [], []
    
    for VAR in TOP_VARS:

        VAR_INFO_ALL.append([VAR, VAR_META_DATA[VAR]['type'], VAR_META_DATA[VAR]['units'],                              '', VAR_META_DATA[VAR]['_FillValue'], VAR_META_DATA[VAR]['name'], '', '' ])
        VAR_DEP_ALL.append(VAR_META_DATA[VAR]['dependence'])

        if VAR == 'latitude':
            xLAT           = SITE_DATA[SITE_CODE]['latitude']
            iLAT           = int((90.0+xLAT)/1.25)
            VAR_DATA_ALL.append(xLAT)
        elif VAR == 'longitude':
            xLON           = SITE_DATA[SITE_CODE]['longitude']
            if xLON >= 0.0:
                iLON           = int(xLON/1.875)
            else:
                iLON           = int((360.0+xLON)/1.875)
            VAR_DATA_ALL.append(xLON)
        else:        
            VAR_DATA_ALL.append(DATA_TOP[VAR][iLAT,iLON])
    
    print(SITE_CODE, xLAT, xLON, iLAT, iLON, DATA_TOP['field900'][iLAT,iLON], DATA_TOP['field900_1'][iLAT,iLON])
    
    print('Writing to '+DIR_CDF+FILE_CDF)
    write_netCDF_py3.write_netCDF_multi_new4(DIR_CDF+FILE_CDF, DIM_INFO, VAR_INFO_ALL, VAR_DEP_ALL, VAR_DATA_ALL,         ATTRIB, NETCDF, DEBUG)

    return


# In[16]:


def get_write_met_data(DIR_IN, DIR_OUT, SITE_CODE, SITE_DATA, DEBUG):
    
    # Define filename for site met data
    # If filename = 'n/a', return
    FILE_MET         = SITE_DATA[SITE_CODE]['file_drive']
    if FILE_MET == "Not_Needed":
        return
    
    Time_Step        = float(SITE_DATA[SITE_CODE]['drive_tstep'])
    
    # Get met data
    DF_MET           = get_df_from_csv(DIR_IN, FILE_MET+'.csv')
    
    if 'FLUXNET' in FILE_MET:
        iSTART           = 0
        T_Name, SW_Name, LW_Name, P_Name, PR_Name, VPD_Name, RH_Name, WS_Name =             'TA_F', 'SW_IN_F', 'LW_IN_F', 'PA_F', 'P_F', 'VPD_F', 'RH', 'WS_F'
    else:
        iSTART           = 1 # UK data has units on second line    
        T_Name, SW_Name, LW_Name, P_Name, PR_Name, VPD_Name, RH_Name, WS_Name =             'TA_F_MDS', 'SW_IN_F_MDS', 'LW_IN_F_MDS', 'PA_F_MDS', 'P', 'VPD_F_MDS', 'RH_F_MDS', 'WS_F_MDS'
    
    if DEBUG == 'Y':
        nDATA            = 10
    else:
        nDATA            = len(DF_MET[T_Name])

    FILE_OUT         = DIR_OUT+SITE_CODE.replace('-','_')+'-met.dat'
    FILE_ID          = open(FILE_OUT,'w')
    print('Writing to: '+FILE_OUT)
    
    for iDATA in range(iSTART,nDATA):
        # Met data for JULES needs to be complete, use gap-filled
        T_K       = float(DF_MET[T_Name][iDATA])+273.15 # Convert from C to K
        SW_Down   = float(DF_MET[SW_Name][iDATA])
        LW_Down   = float(DF_MET[LW_Name][iDATA])
        P_Pa      = float(DF_MET[P_Name][iDATA])*1000.0 # Convert from kPa to Pa
        Precip    = float(DF_MET[PR_Name][iDATA])
        WS        = float(DF_MET[WS_Name][iDATA])
        VPD       = float(DF_MET[VPD_Name][iDATA])*100.0 # Convert from hPa to Pa
        RH_obs    = float(DF_MET[RH_Name][iDATA])
        # Derive humidity (kg kg-1) from vapour pressure deficit
        RH_mod    = RH_from_VPD_T(VPD, T_K)
        Q_water   = RH_2_specific_humidity(RH_mod, P_Pa, T_K)

        if DEBUG == 'Y':
            print(iDATA, VPD, RH_mod, RH_obs, Precip, Is_Nan, ':', OUTPUT)

        if np.isnan(Precip):
            Precip    = 0.0
            Is_Nan    = True
        else:
            Precip    = Precip/Time_Step
            Is_Nan    = False

        # Output
        # (a) Surface temperature (K)
        # (b) Downward short-wave radiation (W m-2)
        # (c) Downward long-wave radiation (W m-2)
        # (d) Humidity (kg kg-1)
        # (e) Surface pressure (Pa)
        # (f) Precipitation (convert mm per timestep to kg m-2 s-1)
        # (g) Wind (m s-1)
        #OUTPUT    = ('%.3f,%.4f,%.4f,%.8f,%.2f,%.8f,%.3f' % \
        #    (T_K, SW_Down, LW_Down, float(Q_water), P_Pa, Precip, WS))
        OUTPUT    = ('%s,%s,%s,%s,%s,%s,%s' %             (str(T_K), str(SW_Down), str(LW_Down), str(Q_water), str(P_Pa), str(Precip), str(WS)))
        
        FILE_ID.write(OUTPUT+'\n')
        
    return


# In[17]:


def get_write_flux_data_iris(DIR_IN, DIR_OUT, SITE_CODE, SITE_DATA, DEBUG):
    
    # Define filename for site met data
    # If filename = 'n/a', return
    FILE_FLUX        = SITE_DATA[SITE_CODE]['file_flux']
    if FILE_FLUX == "Not_Needed":
        return
    
    Time_Step        = float(SITE_DATA[SITE_CODE]['drive_tstep'])
    
    # Get met data
    DF_FLUX          = get_df_from_csv(DIR_IN, FILE_FLUX+'.csv')

    if 'FLUXNET' in FILE_FLUX:
        iSTART           = 0
        os.environ['NETWORK'] = 'FLUXNET Site'
        
        FILE_HEADINGS_subdaily_energy  = [             'TIMESTAMP_START', 'TIMESTAMP_END', 'G_F_MDS', 'LE_F_MDS', 'LE_CORR',             'LE_CORR_25', 'LE_CORR_75', 'LE_RANDUNC', 'H_F_MDS', 'H_CORR',             'H_CORR_25', 'H_CORR_75', 'H_RANDUNC'                                          ]

        FILE_HEADINGS_subdaily_carbon = ['TIMESTAMP_START', 'TIMESTAMP_END', 'NEE_VUT_REF',             'NEE_VUT_REF_RANDUNC', 'NEE_VUT_25', 'NEE_VUT_50', 'NEE_VUT_75',             'RECO_NT_VUT_REF', 'RECO_NT_VUT_25', 'RECO_NT_VUT_50', 'RECO_NT_VUT_75',             'GPP_NT_VUT_REF', 'GPP_NT_VUT_25', 'GPP_NT_VUT_50', 'GPP_NT_VUT_75',             'RECO_DT_VUT_REF', 'RECO_DT_VUT_25', 'RECO_DT_VUT_50', 'RECO_DT_VUT_75',             'GPP_DT_VUT_REF', 'GPP_DT_VUT_25', 'GPP_DT_VUT_50', 'GPP_DT_VUT_75' 
                                         ]
    else:
        iSTART           = 1 # UK data has units on second line    
        os.environ['NETWORK'] = 'UK Site'

        FILE_HEADINGS_subdaily_energy  = [             'TIMESTAMP_START', 'TIMESTAMP_END', 'G_F_MDS_1', 'G_F_MDS_2',             'LE_F_MDS', 'LE_RANDUNC', 'H_F_MDS', 'H_RANDUNC'                                          ]

        FILE_HEADINGS_subdaily_carbon = ['TIMESTAMP_START', 'TIMESTAMP_END',             'NEE_VUT_REF', 'NEE_VUT_REF_RANDUNC',  'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF',             'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF'                                          ]
    
    if DEBUG == 'Y':
        nDATA            = 10
    else:
        nDATA            = len(DF_FLUX['TIMESTAMP_START'])

    for EWC in ['energy', 'carbon']: 

        FILE_INT             = DIR_IN+'subdaily_obs/'+SITE_CODE.replace('-','_')+'-'+EWC+'.dat'
        FILE_ID              = open(FILE_INT,'w')
        print('Writing to: '+FILE_INT)

        if EWC == 'energy':
            FILE_HEADINGS        = FILE_HEADINGS_subdaily_energy
        elif EWC == 'carbon':
            FILE_HEADINGS        = FILE_HEADINGS_subdaily_carbon
            
        for iDATA in range(iSTART,nDATA):
            # Subdaily flux data

            FIRST     = True
            OUTPUT    = ''
            for VAR_NAME in FILE_HEADINGS:
                #if 'TIMESTAMP' in VAR_NAME:
                #    DF_FLUX[VAR_NAME][iDATA] = str(DF_FLUX[VAR_NAME][iDATA])+'00'
                if FIRST:
                    OUTPUT    = OUTPUT+'%s' % str(DF_FLUX[VAR_NAME][iDATA])
                    FIRST     = False
                else:
                    OUTPUT    = OUTPUT+',%s' % str(DF_FLUX[VAR_NAME][iDATA])
                
            if DEBUG == 'Y':
                print(iDATA, OUTPUT)

            FILE_ID.write(OUTPUT+'\n')

        FILE_ID.close()

    os.environ['OBS_FOLDER_PRE2015'] = DIR_IN
    os.environ['OBS_FOLDER']         = DIR_IN
    os.environ['OBS_FOLDER_LBA']     = DIR_IN
    import fluxnet_evaluation

    UTC_OFFSET       = 0
    fluxnet_evaluation.create_fluxnet2015_dailyUTC_files(SITE_CODE.replace('-','_'), UTC_OFFSET, DIR_OUT)

    return


# In[18]:


def get_write_flux_data_pandas(DIR_IN, DIR_OUT, SITE_CODE, SITE_DATA, DEBUG):
    
    DATETIME_format  = '%Y%m%d%H%M'
    TIME_convertfunc = lambda x: dt.datetime.strptime(x, DATETIME_format)
    CONVERTERS       = {'TIMESTAMP_START':TIME_convertfunc,
                        'TIMESTAMP_END':TIME_convertfunc}

    VAR_NAMES        = ['GPP_NT_VUT_REF','RECO_NT_VUT_REF','NEE_VUT_REF',                         'H_F_MDS','LE_F_MDS']
    
    HEADER           = ['GPP', 'Reco', 'NEE', 'SH', 'LE']
    
    CONV_C           = 12.011*1e-06*float(60*60*24)
    CONV_FACTORS     = [CONV_C, CONV_C, CONV_C, 1.0, 1.0]
    
    # Define filename for site met data
    # If filename = 'n/a', return
    FILE_FLUX        = SITE_DATA[SITE_CODE]['file_flux']
    if FILE_FLUX == "Not_Needed":
        return

    # Need to replace TIMESTAMP_START
    OS_CMD = "sed -e '2s/-/# -/' < "+DIR_IN+FILE_FLUX+'.csv > '+DIR_IN+FILE_FLUX+'_mod.csv'
    os.system(OS_CMD)
    
    DF_DATA_ALL      = pd.read_csv(DIR_IN+FILE_FLUX+'_mod.csv',converters=CONVERTERS,comment='#')
    DF_FLUX          = DF_DATA_ALL[['TIMESTAMP_START','TIMESTAMP_END']+VAR_NAMES]
    if DEBUG == 'Y':
        print('1: ',DF_FLUX)
    
    nDATA            = len(DF_FLUX['TIMESTAMP_START'])
    INDICES          = []
    for iDATA in range(nDATA):
        DATE_TIME        = DF_FLUX['TIMESTAMP_START'][iDATA]
        IS_MIDNIGHT      = (DATE_TIME.hour, DATE_TIME.minute) == (0,0)
        if IS_MIDNIGHT:
            print(iDATA, INDICES)
            DF_FLUX          = DF_FLUX.drop(index=INDICES)
            break
        else:
            INDICES.append(iDATA)

    if DEBUG == 'Y':
        print('2: ',iDATA,nDATA)
        print(DF_FLUX)
    
    nDATA            = len(DF_FLUX['TIMESTAMP_END'])
    INDICES          = []
    for iDATA in range(nDATA-1,0,-1):
        TIME             = DF_FLUX['TIMESTAMP_END'][iDATA+1]
        IS_MIDNIGHT      = (DATE_TIME.hour, DATE_TIME.minute) == (0,0)
        if IS_MIDNIGHT:
            print(iDATA, INDICES)
            DF_FLUX          = DF_FLUX.drop(index=INDICES)
            break
        else:
            INDICES.append(iDATA)
        
    if DEBUG == 'Y':
        print('3: ',iDATA,nDATA)
        print(DF_FLUX)
    
    DF_FLUX_DAILY    = DF_FLUX.resample('D', on='TIMESTAMP_START').mean()
    if DEBUG == 'Y':
        print('4: ',DF_FLUX_DAILY)

    FILE_OUT         = DIR_OUT+SITE_CODE.replace('-','_')+'-energyandcarbon-dailyUTC_unscaled.dat'
    print('Writing to: '+FILE_OUT)
    DF_FLUX_DAILY.to_csv(FILE_OUT,header=HEADER,date_format='%Y%m%d')
    
    #Convert carbon fluxes from micromolCO2 m-2 s-1 to gC m-2 d-1
    for iVAR,VAR in enumerate(VAR_NAMES):
        if CONV_FACTORS[iVAR] != 1.0:
            DF_FLUX_DAILY[VAR] = DF_FLUX_DAILY[VAR]*CONV_FACTORS[iVAR]
    
    if DEBUG == 'Y':
        print('5: ',DF_FLUX_DAILY)

    FILE_OUT         = DIR_OUT+SITE_CODE.replace('-','_')+'-energyandcarbon-dailyUTC.dat'
    print('Writing to: '+FILE_OUT)
    DF_FLUX_DAILY.to_csv(FILE_OUT,header=HEADER,date_format='%Y%m%d')
    
    # Need to replace TIMESTAMP_START
    OS_CMD = "sed -i '1s/TIMESTAMP_START/# YYYYMMDD UTC/' "+FILE_OUT
    os.system(OS_CMD)
    
    return


# In[19]:


def main():

    # Configure for python (.py) or Jupyter notebook

    if '-platform' in sys.argv:
        ARGLOC        = sys.argv.index('-platform')
        TEMP          = sys.argv.pop(ARGLOC)
        PLATFORM      = str(sys.argv.pop(ARGLOC))
        DEBUG         = sys.argv[1]
        INTERACTIVE   = sys.argv[2]
        SUITE         = sys.argv[3]
        OPTIONS       = sys.argv[4]
        PLOT_OPT      = sys.argv[5]
        sDIR          = sys.argv[6]
        sDATE         = sys.argv[6]
    else:
        PLATFORM      = 'notebook'
        DEBUG         = 'N'
        INTERACTIVE   = 'N'
        SUITE         = 'u-cr886'
        OPTIONS       = 'YYYYYYYY'
        PLOT_OPT      = '1'
        sDIR          = 'Test'
        sDATE         = '20221108'

    print('Platform = '+PLATFORM)

    # Set directories for the platform selected
    if PLATFORM == 'MONSOON':
        DIR_HOME     = '/projects/ecobrasil/'
        DIR_DATA     = DIR_HOME
        DIR_OUT      = DIR_HOME+'ghayma/'
    elif PLATFORM == 'CEH' or PLATFORM == 'notebook':
        DIR_HOME     = '/data/grp/eow/garr/'
        DIR_DATA     = DIR_HOME+'Projects/MotherShip/Data/'
        #DIR_OUT      = DIR_HOME+'Projects/MotherShip/Output/'+SUITE+'/'
        DIR_OUT      = '/users/eow/garr/roses/'+SUITE+'/'
        DIR_FLUX     = '/prj/MOYA/JULES/DEPOSITION/SITE/FLUXNET/DATA/testvn1.4/'

    DIR_PLOTS    = os.path.join(DIR_OUT, 'PLOTS/'+SUITE+'/')
    #SITE_CODES   = ['AT-Neu','FI-Hyy','UK-AMo','UK-Ham','US-Ne1','ZA-Kru']
    SITE_CODES   = ['UK-AMo', 'UK-Ham', 'UK-MrH', 'UK-Rdm', 'UK-Tad', 'CH-Oe1']
    SITES_INFO_UK= ['UK-MrH', 'UK-Rdm', 'UK-Tad']
    
    if not os.path.exists(DIR_OUT):
        os.system('mkdir -p '+DIR_OUT)

    # get site meta data
    SITE_DATA     = get_site_metadata(DIR=DIR_DATA, SITE_CODES=SITE_CODES, DEBUG=DEBUG)
    SITE_DATA_ALL = get_site_metadata(DIR=DIR_DATA, SITE_CODES=ALL_SITE_CODES, DEBUG=DEBUG)
    if DEBUG == 'Y':
        print(''); print('SITE_DATA:'); print(SITE_DATA)

    if OPTIONS[0] == 'Y':
        output_var_info(DIR_OUT, ALL_SITE_CODES, SITES_INFO_UK, SITE_DATA_ALL, DEBUG)    

    if OPTIONS[4] == 'Y':
        GRID_VARS_OUT = ['eastings', 'northings', 'latitude', 'longitude' ]
        SOIL_VARS_OUT = ['b', 'sathh', 'satcon', 'sm_sat', 'sm_crit', 'sm_wilt', 'hcap', 'hcon', 'albsoil', 'clay' ]
        SOIL_MODELS = ['BC', 'VG']
        DATA_SOIL   = { VAR:0.0 for VAR in SOIL_MODELS}

        for SOIL_MODEL in SOIL_MODELS:
            DATA_SOIL[SOIL_MODEL] = input_ancil_soil_uk(SOIL_MODEL, GRID_VARS_OUT+SOIL_VARS_OUT, DEBUG)

    if OPTIONS[5] == 'Y':
        TOP_VARS      = ['latitude', 'longitude', 'field900', 'field900_1' ]
        DATA_TOP      = input_ancil_top(TOP_VARS, DEBUG)

    # Loop over selected sites
    for SITE_CODE in SITE_CODES:

        print(""); print("Writing namelists for site: "+SITE_CODE)
        
        if OPTIONS[1] == 'Y':
            # output ancil namelists 
            output_namelist_ancil(DIR_OUT, SITE_CODE, SITE_DATA, DEBUG)

        if OPTIONS[2] == 'Y':
            # output drive namelists 
            output_namelist_drive(DIR_OUT, SITE_CODE, SITE_DATA, DEBUG)

        if OPTIONS[3] == 'Y':
            if int(SITE_DATA[SITE_CODE]['presc_data']) >= 1:
                # output prescribed data namelist: sthuf 
                output_namelist_presc_sthuf(DIR_OUT, SITE_CODE, SITE_DATA, DEBUG)

            if int(SITE_DATA[SITE_CODE]['presc_data']) >= 2:
                # output prescribed data namelists 
                output_namelist_presc_lai(DIR_OUT, SITE_CODE, SITE_DATA, DEBUG)

        if OPTIONS[4] == 'Y' and 'UK' in SITE_CODE:
            
            # output files with ancillary soil parameters
            for SOIL_MODEL in SOIL_MODELS:
                output_ancil_soil_uk(DIR_OUT, SITE_CODE, SITE_DATA,                                      DATA_SOIL[SOIL_MODEL], SOIL_VARS_OUT,                                      SOIL_MODEL.lower(), DEBUG)

        if OPTIONS[5] == 'Y':
            # output files with ancillary topographic parameters 
            output_ancil_top(DIR_OUT, SITE_CODE, SITE_DATA, DATA_TOP, TOP_VARS, DEBUG)

        if OPTIONS[6] == 'Y':
            # output files with met driving data
            #get_met_data(DIR_DATA+'FLUX_DATA/Original/', DIR_MET, SITE_CODE, SITE_DATA, DEBUG)
            DEBUG = 'N'
            if sDIR[0].lower() == 't':   # Write to test folder
                DIR_MET_OUT  = DIR_DATA+'FLUX_DATA/Fluxnet_Met/'
            else:                        # Write to folder with fluxnet met driving data
                DIR_MET_OUT  = DIR_FLUX+'fluxnet/'
            get_write_met_data(DIR_DATA+'FLUX_DATA/Original/', DIR_MET_OUT,                          SITE_CODE, SITE_DATA, DEBUG)
            
        if OPTIONS[7] == 'Y':
            # output files with flux data
            DEBUG = 'N'
            DIR_FLUX_SUB = DIR_DATA+'FLUX_DATA/Original/'
            
            if sDIR[0].lower() == 't':
                DIR_FLUX_OUT = DIR_DATA+'FLUX_DATA/Fluxnet_Fluxes/'
            else:
                DIR_FLUX_OUT = DIR_FLUX+'fluxnet_obs/'

            get_write_flux_data_pandas(DIR_FLUX_SUB, DIR_FLUX_OUT, SITE_CODE, SITE_DATA, DEBUG)
            #get_write_flux_data_iris(DIR_FLUX_SUB, DIR_FLUX_OUT, SITE_CODE, SITE_DATA, DEBUG)

    return


# In[20]:


if __name__ == '__main__':
    main()


# In[ ]:


#Auchencorth Moss
#AURN_Easting,  AURN_Northing   =  322166, 656128
#AURN_Latitude, AURN_Longitude  =  55.792160, -3.242900
#Easting, Northing = ccrs.OSGB().transform_point( AURN_Longitude, AURN_Latitude, ccrs.PlateCarree() )
#print(Easting, Northing, int(Easting/1000), int(Northing/1000))

#Yarner Wood
AURN_Easting, AURN_Northing    = 278611, 78949
AURN_Latitude, AURN_Longitude  =  50.597600, -3.716510
Easting, Northing = ccrs.OSGB().transform_point( AURN_Longitude, AURN_Latitude, ccrs.PlateCarree() )
print(Easting, Northing, int(Easting/1000), int(Northing/1000))


# In[ ]:


np.__version__


# In[ ]:


import iris


# In[ ]:


iris.__version__


# In[ ]:




