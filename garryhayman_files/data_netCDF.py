#
# Python module to input Sciamachy data
#
# Garry Hayman
# Centre for Ecology and Hydrology
# February 2011
#
import numpy as np
import numpy.ma as ma
import sys
from netCDF4 import Dataset
#
def data_netCDF_getDIMNAMES(FILE_DATA):
#
	FID        = Dataset(FILE_DATA,'r')
	DIM_NAMES  = FID.dimensions.keys()
	FID.close()
#
	return DIM_NAMES
#
def data_netCDF_getDIMVALUE(FILE_DATA,DIM_NAME):
#
	FID        = Dataset(FILE_DATA,'r')
	DIM_VALUE  = FID.dimensions[DIM_NAME]
	FID.close()
#
	return DIM_VALUE
#
def data_netCDF_getVARNAMES(FILE_DATA):
#
	FID        = Dataset(FILE_DATA,'r')
	VAR_NAMES  = FID.variables.keys()
	FID.close()
#
	return VAR_NAMES
#
def data_netCDF_getVARDIMNAMES(FILE_DATA,VAR_NAME):
#
	FID        = Dataset(FILE_DATA,'r')
	DIM_NAMES  = FID.variables[VAR_NAME].dimensions
	FID.close()
#
	return DIM_NAMES
#
def data_netCDF_getVAR_UNITS(FILE_DATA,VAR_NAME):
#
	FID        = Dataset(FILE_DATA,'r')
	VAR_UNITS  = FID.variables[VAR_NAME].units
	FID.close()
#
	return VAR_UNITS
#
def data_netCDF_getVAR_MISSDATA(FILE_DATA,VAR_NAME):
#
	FID        = Dataset(FILE_DATA,'r')
	VAR_MISS   = FID.variables[VAR_NAME].missing_value
	FID.close()
#
	return VAR_MISS
#
def data_netCDF_array(FILE_DATA,DATA_NAME,TIME_NAME):
#
	FID        = Dataset(FILE_DATA,'r')
	VAR_NAMES  = FID.variables.keys()
#
	if len(FID.variables[DATA_NAME])==0:
		VAR_DATA   = []
		TIME_DATA  = []
		DIMS       = np.zeros(1)
	else:
		VAR_DATA   = FID.variables[DATA_NAME][:]
		TIME_DATA  = FID.variables[TIME_NAME][:]
		DIMS       = np.array(VAR_DATA.shape)
#
	print(DIMS)
#
	FID.close()
#
	return DIMS,VAR_DATA,TIME_DATA
#
def data_netCDF_array_var(FILE_DATA,DATA_NAME):
#
	FID        = Dataset(FILE_DATA,'r')
	VAR_NAMES  = FID.variables.keys()
#
	if len(FID.variables[DATA_NAME])==0:
		VAR_DATA   = []
		DIMS       = np.zeros(1)
	else:
		VAR_DATA   = FID.variables[DATA_NAME][:]
		DIMS       = np.array(VAR_DATA.shape)
#
	print(DIMS)
#
	FID.close()
#
	return DIMS,VAR_DATA
#
def data_netCDF_array_land_var(FILE_DATA,DATA_NAME,LAT_NAME,LONG_NAME, \
	START_YEAR,END_YEAR,NLONG,NLAT,LONG_START,LAT_START,MISS_DATA,GRIDDED,DEBUG):
#
	if 'LAND' in GRIDDED:
		NYEARS     = END_YEAR-START_YEAR+1
		NTIMES     = 12*NYEARS
		iTIME      = 0
#
		VAR_DATA   = np.zeros((NTIMES,NLAT,NLONG))
		VAR_DATA[:,:,:] = MISS_DATA
#
		if 'MON' in GRIDDED:
#
			for YEAR in range(START_YEAR,END_YEAR+1):
				for MONTH in range(12):
#
					SDATE      = '%4d%02d' % (YEAR,MONTH+1)
					FILENAME   = FILE_DATA.replace('XXXXXX',SDATE)
					print(iTIME,SDATE,FILENAME)
#
					FID        = Dataset(FILENAME,'r')
#
# Get data
					VAR_TEMP   = FID.variables[DATA_NAME][:]
#
# On first pass, get latitude and longitude
#
					if iTIME == 0:
						LAT        = FID.variables[LAT_NAME][:]
						LONG       = FID.variables[LONG_NAME][:]
						NLAND      = LAT.shape[1]
						print(NLAND)
#
						VAR_LAND   = np.zeros((NTIMES,NLAND))
#
					VAR_LAND[iTIME,:]  = VAR_TEMP
#
					FID.close()
#
					iTIME += 1
#
		elif 'ANN' in GRIDDED:
#
			for YEAR in range(START_YEAR,END_YEAR+1):
#
				SDATE      = '%4d' % (YEAR)
				FILENAME   = FILE_DATA.replace('XXXX',SDATE)
				print(iTIME,SDATE,FILENAME)
#
				FID        = Dataset(FILENAME,'r')
#
# Get data
				VAR_TEMP   = FID.variables[DATA_NAME][:]
#
# On first pass, get latitude and longitude
#
				if iTIME == 0:
					LAT        = FID.variables[LAT_NAME][:]
					LONG       = FID.variables[LONG_NAME][:]
					NLAND      = LAT.shape[0]
					print(NLAND)
#
					VAR_LAND   = np.zeros((NTIMES,NLAND))
#
				VAR_LAND[iTIME:iTIME+12,:]  = VAR_TEMP
#
				FID.close()
#
				iTIME += 12
#
		elif 'ALL' in GRIDDED:
#
			FID        = Dataset(FILE_DATA,'r')
#
# Get data
			VAR_TEMP   = FID.variables[DATA_NAME][:]
			VAR_TEMP   = VAR_TEMP.squeeze()
#
# On first pass, get latitude and longitude
#
			LAT        = FID.variables[LAT_NAME][:].squeeze()
			LONG       = FID.variables[LONG_NAME][:].squeeze()
			NLAND      = LAT.shape[0]
			print(NLAND)
#
			VAR_LAND   = np.zeros((NTIMES,NLAND))
			VAR_LAND[:,:] = MISS_DATA
			print(VAR_TEMP.shape,VAR_LAND.shape)
			VAR_LAND[0:VAR_TEMP.shape[0],:] = VAR_TEMP[:,:]
#
			FID.close()
#
# Regrid from 1-D land array to lat-lon
#
		LONG       = LONG.squeeze()
		LAT        = LAT.squeeze()
#
		for iLAND in range(NLAND):
			iLON       = int((LONG[iLAND]-LONG_START)*NLONG/360.0)
			iLAT       = int((LAT[iLAND] -LAT_START )*NLAT/180.0)
			VAR_DATA[:,iLAT,iLON] = VAR_LAND[:,iLAND]
#
		DIMS       = VAR_DATA.shape 
#
	else:
		FID        = Dataset(FILE_DATA,'r')
		VAR_NAMES  = FID.variables.keys()
#
		if len(FID.variables[DATA_NAME])==0:
			VAR_DATA   = []
			DIMS       = np.zeros(1)
		else:
			VAR_DATA   = FID.variables[DATA_NAME][:]
			DIMS       = np.array(VAR_DATA.shape)
#
		print(DIMS)
#
		FID.close()
#
	return DIMS,VAR_DATA
#
def data_netCDF_array2D(FILE_DATA,DATA_NAME):
#
	FID        = Dataset(FILE_DATA,'r')
	VAR_NAMES  = FID.variables.keys()
#
	if len(FID.variables[DATA_NAME])==0:
		VAR_DATA   = []
		DIMS       = np.zeros(1)
	else:
		VAR_DATA   = FID.variables[DATA_NAME][:]
		DIMS       = VAR_DATA.shape
#
	print(DIMS)
#
	FID.close()
#
	return DIMS,VAR_DATA
#
def data_netCDF_3D(iREVERSE,DIMS,VAR_DATA,TIME_DATA,FACTOR,MISS_DATA,DEBUG):
#
	TIME       = []
	DATA       = []
	NUM        = []
	VALID_DATA = []
#
	MISS_DATA  = float('%.2f' % MISS_DATA)
#
# Input 3D array (time,lat,long)
#
	NTIMES     = DIMS[0]
	NLAT       = DIMS[1]
	NLONG      = DIMS[2]
#
	if iREVERSE == 'Y':
		LAT_START  = NLAT-1
		LAT_END    = -1
		LAT_INCR   = -1
	else:
		LAT_START  =  0
		LAT_END    = NLAT
		LAT_INCR   =  1
#
	for iLAT in range(LAT_START,LAT_END,LAT_INCR):
#
		if DEBUG == 'Y': print(iLAT,LAT_START,LAT_END,LAT_INCR)
#
		TIME1         = []
		DATA1         = []
		NUM1          = []
		VALID1        = []
#
		for iLONG in range(NLONG):
#
			TIME2         = []
			DATA2         = []
			VALID2        = []
			COUNT         = 0 
#
			for iTIME in range(NTIMES):
				VALUE         = float('%.2f' % (VAR_DATA[iTIME,iLAT,iLONG]*FACTOR))
				if VALUE == MISS_DATA:
					VALID2.append(0)
				else:
					VALID2.append(1)
					COUNT        +=  1
					TIME2.append(TIME_DATA[iTIME])
					DATA2.append(VALUE)
#
			TIME1.append(TIME2)
			DATA1.append(DATA2)
			NUM1.append(COUNT)
			VALID1.append(VALID2)
#
		TIME.append(TIME1)
		DATA.append(DATA1)
		NUM.append(NUM1)
		VALID_DATA.append(VALID1)
#
	return TIME, DATA, NUM, VALID_DATA
#
def data_netCDF_4D(iREVERSE,DIMS,VAR_DATA,TIME_DATA,FACTOR,iLEVEL,MISS_DATA,DEBUG):
#
	TIME       = []
	DATA       = []
	NUM        = []
	VALID_DATA = []
#
	MISS_DATA  = float('%.2f' % MISS_DATA)
#
# Input 4D array (time,level,lat,long)
#
	NTIMES     = DIMS[0]
	NLEVELS    = DIMS[1]
#	NLAT       = 5
#	NLONG      = 5
	NLAT       = DIMS[2]
	NLONG      = DIMS[3]
#
	if iREVERSE == 'Y':
		LAT_START  = NLAT-1
		LAT_END    = -1
		LAT_INCR   = -1
	else:
		LAT_START  =  0
		LAT_END    = NLAT
		LAT_INCR   =  1
#
	for iLAT in range(LAT_START,LAT_END,LAT_INCR):
#
		if DEBUG == 'Y': print(iLAT,LAT_START,LAT_END,LAT_INCR)
#
		TIME1         = []
		DATA1         = []
		NUM1          = []
		VALID1        = []
#
		for iLONG in range(NLONG):
#
			TIME2         = []
			DATA2         = []
			VALID2        = []
			COUNT         = 0 
#
			for iTIME in range(NTIMES):
				VALUE         = float('%.2f' % (VAR_DATA[iTIME,iLEVEL,iLAT,iLONG]*FACTOR))
				if VALUE == MISS_DATA:
					VALID2.append(0)
				else:
					VALID2.append(1)
					COUNT        +=  1
					TIME2.append(TIME_DATA[iTIME])
					DATA2.append(VALUE)
#
			TIME1.append(TIME2)
			DATA1.append(DATA2)
			NUM1.append(COUNT)
			VALID1.append(VALID2)
#
		TIME.append(TIME1)
		DATA.append(DATA1)
		NUM.append(NUM1)
		VALID_DATA.append(VALID1)
#
	return TIME,DATA,NUM,VALID_DATA
#
def data_netCDF_count(DATA,NLONG,NLAT,FACTOR,MISS_DATA,DEBUG):
#
	NUM            = np.zeros((NLAT,NLONG))
#
	for iLAT in range(NLAT):
		for iLONG in range(NLONG):
#
			TEMP            = DATA[:,iLAT,iLONG]
			NUM[iLAT,iLONG] = len(TEMP[ma.getdata(TEMP) != MISS_DATA])
#			print(iLAT,iLONG,TEMP,NUM[iLAT,iLONG])
			TEMP[ma.getdata(TEMP) != MISS_DATA] = TEMP[ma.getdata(TEMP) != MISS_DATA]*FACTOR
#
			DATA[:,iLAT,iLONG] = TEMP
#
	return DATA,NUM
#
