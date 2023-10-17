#
# Python module to write to a netCDF file
#
# Garry Hayman
# Centre for Ecology and Hydrology
# February 2011, updated August 2017
#
import sys
import datetime
from netCDF4 import Dataset
from numpy   import dtype
#
# Contains
#
def write_netCDF(FILE_CDF,nDIMS,DIM_INFO,DIM_DATA,VAR_INFO,VAR_DEP,VAR_DATA,MISS_DATA,DEBUG='N'):
#
# Write out netCDF file
#
	print(FILE_CDF)
	NC_FID     = Dataset(FILE_CDF,'w',format='NETCDF3_CLASSIC')
	setattr(NC_FID,'history','netCDF file from Python')
#
# Define the co-ordinate variables
#
	for iDIM in range(nDIMS):
		if DIM_INFO[iDIM][0] == 0:
			NC_FID.createDimension(DIM_INFO[iDIM][1],None)
		else:
			NC_FID.createDimension(DIM_INFO[iDIM][1],DIM_INFO[iDIM][0])
		DIM_CDF       = NC_FID.createVariable(DIM_INFO[iDIM][1],dtype(DIM_INFO[iDIM][2]).char,(DIM_INFO[iDIM][1]))
		DIM_CDF.units = DIM_INFO[iDIM][3]
		DIM_CDF[:]    = DIM_DATA[iDIM]
#
# Create and write data to variable.
#
	if DEBUG == 'Y': print(VAR_DEP)
	DATA_CDF    = NC_FID.createVariable(VAR_INFO[0],dtype(VAR_INFO[1]).char,(VAR_DEP))
#
	if DEBUG == 'Y': print(DIM_INFO[0][0])
#	for iDIM in range(DIM_INFO[0][0]):
#		DATA_CDF[iDIM,:,:] = VAR_DATA[iDIM,:,:]
	DATA_CDF[:] = VAR_DATA
#
	setattr(DATA_CDF,'missing_value',MISS_DATA)
#
# Close the file.
#
	NC_FID.close()
#
	TEXT	= '*** SUCCESS writing file '+FILE_CDF
	print(TEXT)
#
	return
#
def write_netCDF_new(FILE_CDF,nDIMS,DIM_INFO,DIM_DATA,VAR_INFO,VAR_DEP,VAR_DATA,MISS_DATA,DEBUG='N'):
#
# Write out netCDF file
#
	DATE	  = datetime.datetime.today()
	DATE	  = datetime.datetime.strftime(DATE,'%d/%m/%Y %H:%M')
#
	print(FILE_CDF)
	NC_FID     = Dataset(FILE_CDF,'w',format='NETCDF3_CLASSIC')
	setattr(NC_FID,'history','netCDF file from Python')
	setattr(NC_FID,'created',DATE)
	setattr(NC_FID,'author','Garry Hayman (Centre for Ecology and Hydrology, UK)')
	setattr(NC_FID,'contact','tel: +44-1491-692527, e-mail: garr@ceh.ac.uk')
#
# Define the co-ordinate variables
#
	for iDIM in range(nDIMS):
		if DIM_INFO[iDIM][0] == 0:
			NC_FID.createDimension(DIM_INFO[iDIM][1],None)
		else:
			NC_FID.createDimension(DIM_INFO[iDIM][1],DIM_INFO[iDIM][0])
		DIM_CDF       = NC_FID.createVariable(DIM_INFO[iDIM][1],dtype(DIM_INFO[iDIM][2]).char,(DIM_INFO[iDIM][1]))
		DIM_CDF.units = DIM_INFO[iDIM][3]
		DIM_CDF[:]    = DIM_DATA[iDIM]
#
# Create and write data to variable.
#
	if DEBUG == 'Y': print(VAR_DEP)
	DATA_CDF    = NC_FID.createVariable(VAR_INFO[0],dtype(VAR_INFO[1]).char,(VAR_DEP))
#
	if DEBUG == 'Y': print(DIM_INFO[0][0])
#	for iDIM in range(DIM_INFO[0][0]):
#		DATA_CDF[iDIM,:,:] = VAR_DATA[iDIM,:,:]
	DATA_CDF[:] = VAR_DATA
#
	setattr(DATA_CDF,'missing_value',MISS_DATA)
#
# Close the file.
#
	NC_FID.close()
#
	TEXT	= '*** SUCCESS writing file '+FILE_CDF
	print(TEXT)
#
	return
#
def write_netCDF_multi(FILE_CDF,nDIMS,DIM_INFO,DIM_DATA,nVARS,VAR_INFO_ALL,VAR_DEP_ALL,VAR_DATA_ALL,MISS_DATA,DEBUG='N'):
#
# Write out netCDF file
#
	NC_FID     = Dataset(FILE_CDF,'w',format='NETCDF3_CLASSIC')
	setattr(NC_FID,'history','netCDF file from Python')
	if DEBUG == 'Y': print
#
# Define the co-ordinate variables
#
	for iDIM in range(nDIMS):

		if DEBUG == 'Y': print('DIM_INFO: ',DIM_INFO[iDIM])
		if DIM_INFO[iDIM][0] == 0:
			NC_FID.createDimension(DIM_INFO[iDIM][1],None)
		else:
			NC_FID.createDimension(DIM_INFO[iDIM][1],DIM_INFO[iDIM][0])
		DIM_CDF       = NC_FID.createVariable( \
			DIM_INFO[iDIM][1],dtype(DIM_INFO[iDIM][2]).char,(DIM_INFO[iDIM][1]))

		if len(DIM_INFO[iDIM]) > 3:
			if len(DIM_INFO[iDIM][3]) != 0:
				DIM_CDF.name = DIM_INFO[iDIM][3]
		if len(DIM_INFO[iDIM]) > 4:
			if len(DIM_INFO[iDIM][4]) != 0:
				DIM_CDF.units = DIM_INFO[iDIM][4]
		if len(DIM_DATA) > 0:
			if DEBUG == 'Y': print('DIM_INFO: ',DIM_DATA[iDIM].shape)
			DIM_CDF[:]    = DIM_DATA[iDIM]
#
# Create and write data to variable.
#
	if DEBUG == 'Y': print
#
	for iVAR in range(nVARS):
#
		VAR_DEP       = VAR_DEP_ALL[iVAR]
		if DEBUG == 'Y': print('VAR_DEP:  ',VAR_DEP)
#
		VAR_INFO      = VAR_INFO_ALL[iVAR]
		if DEBUG == 'Y': print('VAR_INFO: ',VAR_INFO)
		DATA_CDF      = NC_FID.createVariable(VAR_INFO[0],dtype(VAR_INFO[1]).char,(VAR_DEP))
#
		if DEBUG == 'Y': print('VAR_DATA:  ',VAR_DATA_ALL[iVAR].shape)
		DATA_CDF[:] = VAR_DATA_ALL[iVAR]
#
		setattr(DATA_CDF,'missing_value',MISS_DATA)
		if len(VAR_INFO) > 2:
			if len(VAR_INFO[2]) != 0:
				setattr(DATA_CDF,'name',VAR_INFO[2])
		if len(VAR_INFO) > 3:
			if len(VAR_INFO[3]) != 0:
				setattr(DATA_CDF,'units',VAR_INFO[3])
#
# Close the file.
#
	NC_FID.close()
#
	TEXT	= '*** SUCCESS writing file '+FILE_CDF
	print(TEXT)
#
	return
#
def write_netCDF_multi_new(FILE_CDF,nDIMS,DIM_INFO,DIM_DATA,nVARS,VAR_INFO_ALL,VAR_DEP_ALL,VAR_DATA_ALL,MISS_DATA_ALL,DEBUG='N'):
#
# Write out netCDF file
#
	NC_FID     = Dataset(FILE_CDF,'w',format='NETCDF3_CLASSIC')
	setattr(NC_FID,'history','netCDF file from Python')
#
# Define the co-ordinate variables
#
	for iDIM in range(nDIMS):
		if DEBUG == 'Y': print(DIM_INFO[iDIM])
		if DIM_INFO[iDIM][0] == 0:
			NC_FID.createDimension(DIM_INFO[iDIM][1],None)
		else:
			NC_FID.createDimension(DIM_INFO[iDIM][1],DIM_INFO[iDIM][0])
		DIM_CDF       = NC_FID.createVariable(DIM_INFO[iDIM][1],dtype(DIM_INFO[iDIM][2]).char,(DIM_INFO[iDIM][1]))
		DIM_CDF.units = DIM_INFO[iDIM][3]
		DIM_CDF[:]    = DIM_DATA[iDIM]
#
# Create and write data to variable.
#
	for iVAR in range(nVARS):
#
		if DEBUG == 'Y': print(iVAR)
		VAR_DEP       = VAR_DEP_ALL[iVAR]
		if DEBUG == 'Y': print(VAR_DEP)
#
		VAR_INFO      = VAR_INFO_ALL[iVAR]
		if DEBUG == 'Y': print(VAR_INFO)
		DATA_CDF      = NC_FID.createVariable(VAR_INFO[0],dtype(VAR_INFO[1]).char,(VAR_DEP))
#
		if DEBUG == 'Y': print(DIM_INFO[0][0])
		DATA_CDF[:] = VAR_DATA_ALL[iVAR]
#
		setattr(DATA_CDF,'missing_value',MISS_DATA_ALL[iVAR])
		if len(VAR_INFO[2]) != 0:
			setattr(DATA_CDF,'units',VAR_INFO[2])
#
# Close the file.
#
	NC_FID.close()
#
	TEXT	= '*** SUCCESS writing file '+FILE_CDF
	print(TEXT)
#
	return
#
def write_netCDF_variable_append(FILE_CDF,VAR_INFO,VAR_DEP,VAR_DATA,MISS_DATA,DEBUG='N'):
#
# Write out netCDF file
#
	NC_FID     = Dataset(FILE_CDF,'a',format='NETCDF3_CLASSIC')
#
# Create and write data to variable.
#
	if DEBUG == 'Y': print(VAR_DEP)
	DATA_CDF    = NC_FID.createVariable(VAR_INFO[0],dtype(VAR_INFO[1]).char,(VAR_DEP))
	DATA_CDF[:] = VAR_DATA
#
	setattr(DATA_CDF,'missing_value',MISS_DATA)
	setattr(DATA_CDF,'units',VAR_INFO[2])
#
# Close the file.
#
	NC_FID.close()
#
	TEXT	= '*** SUCCESS writing file '+FILE_CDF
	print(TEXT)
#
	return

def write_netCDF_multi_new2(FILE_IN,FILE_OUT,DIM_INFO,VAR_INFO,NETCDF='netCDF4',DEBUG='N'):

# Python module to write netCDF file

# Garry Hayman
# Centre for Ecology and Hydrology
# July 2015

	if DEBUG == 'Y': print; print(FILE_IN)

	DATE       = datetime.datetime.today()
	DATE       = datetime.datetime.strftime(DATE,'%d/%m/%Y %H:%M')

	if NETCDF == 'netCDF3':
		FID_OUT    = Dataset(FILE_OUT,"w", format = "NETCDF3_CLASSIC")
	else:
		FID_OUT    = Dataset(FILE_OUT,"w")

	ATTR       = [ \
			['history','file '+FILE_IN.split('/')[-1]+' converted to netCDF classic 3 and renamed to '+FILE_OUT.split('/')[-1]], \
			['comment','Using python modules' ], \
			['created',DATE], \
			['author','Garry Hayman (Centre for Ecology and Hydrology, UK)'], \
			['contact','tel: +44-1491-692527, e-mail: garr@ceh.ac.uk'] \
		     ]

# Copy dimension names and information

	DIM_NAMES   = []; DIM_SIZES   = []

	for DIM_NAME in sorted(DIM_INFO.keys()):

		DIM_SIZE    = int(DIM_INFO[DIM_NAME])
		DIM_NAMES.append(DIM_NAME)
		DIM_SIZES.append(DIM_SIZE)
		
		FID_OUT.createDimension(DIM_NAME,DIM_SIZE)
		if DEBUG == 'Y': print(DIM_NAME,DIM_SIZE)

# Copy variables, attributes and data

	for VAR_NAME in sorted(VAR_INFO.keys()):

		VAR_DTYPE  = VAR_INFO[VAR_NAME][0]
		VAR_DIMS   = VAR_INFO[VAR_NAME][1]
		VAR_TITLE  = VAR_INFO[VAR_NAME][2]
		VAR_UNITS  = VAR_INFO[VAR_NAME][3]
		VAR_MISS   = VAR_INFO[VAR_NAME][4]
		VAR_DATA   = VAR_INFO[VAR_NAME][5]
		VAR_DEP    = []

		for VAR_SIZE in VAR_DIMS:
			VAR_DEP.append(DIM_NAMES[DIM_SIZES.index(VAR_SIZE)])

		if DEBUG == 'Y':
			print
			print(VAR_NAME)
			print(VAR_TITLE)
			print(VAR_DTYPE)
			print(VAR_DEP)
			print(VAR_DIMS)
			print(VAR_UNITS)
			print(VAR_MISS)

		VAR_ID     = FID_OUT.createVariable(VAR_NAME,VAR_DTYPE,VAR_DEP)
		FID_OUT.variables[VAR_NAME][:] = VAR_DATA

		if VAR_TITLE != '': setattr(VAR_ID,'title',VAR_TITLE)
		if VAR_UNITS != '': setattr(VAR_ID,'units',VAR_UNITS)
		setattr(VAR_ID,'missing_value',VAR_MISS)

# Write out attributes

	for iATTR in range(len(ATTR)):
		setattr(FID_OUT,ATTR[iATTR][0],ATTR[iATTR][1])

# Close the files

	FID_OUT.close()

	TEXT	= '*** SUCCESS writing file '+FILE_OUT
	print; print(TEXT)

	return
#
def write_netCDF_multi_new3(FILE_CDF,nDIMS,DIM_INFO,DIM_DATA,nVARS,VAR_INFO_ALL,VAR_DEP_ALL,VAR_DATA_ALL,MISS_DATA_ALL,FILL_VALUE_ALL=[],ATTRIB='',NETCDF='netCDF4',DEBUG='N'):

# Python module to write netCDF file

# Garry Hayman
# Centre for Ecology and Hydrology
# April 2017

	DATE       = datetime.datetime.today()
	DATE       = datetime.datetime.strftime(DATE,'%d/%m/%Y %H:%M')

# Write out netCDF file

	if NETCDF == 'netCDF3':
		NC_FID     = Dataset(FILE_CDF,"w", format = "NETCDF3_CLASSIC")
	else:
		NC_FID     = Dataset(FILE_CDF,"w")

	ATTR       = [ \
			['history','Created from '+ATTRIB ], \
			['comment','Using python modules' ], \
			['created',DATE], \
			['author','Garry Hayman (Centre for Ecology and Hydrology, UK)'], \
			['contact','tel: +44-1491-692527, e-mail: garr@ceh.ac.uk'] \
		     ]

# Define the co-ordinate variables

	for iDIM in range(nDIMS):
		if DEBUG == 'Y': print(DIM_INFO[iDIM])
		if DIM_INFO[iDIM][0] == 0:
			NC_FID.createDimension(DIM_INFO[iDIM][1],None)
		else:
			NC_FID.createDimension(DIM_INFO[iDIM][1],DIM_INFO[iDIM][0])
		DIM_CDF       = NC_FID.createVariable(DIM_INFO[iDIM][1],dtype(DIM_INFO[iDIM][2]).char,(DIM_INFO[iDIM][1]))
		DIM_CDF.units = DIM_INFO[iDIM][3]
		DIM_CDF[:]    = DIM_DATA[iDIM]

# Create and write data to variable.

	for iVAR in range(nVARS):

		if DEBUG == 'Y': print(iVAR)
		VAR_DEP       = VAR_DEP_ALL[iVAR]
		if DEBUG == 'Y': print(VAR_DEP)

		VAR_INFO      = VAR_INFO_ALL[iVAR]
		if DEBUG == 'Y': print(VAR_INFO)
		DATA_CDF      = NC_FID.createVariable(VAR_INFO[0],dtype(VAR_INFO[1]).char,(VAR_DEP))

		if DEBUG == 'Y': print(DIM_INFO[0][0])
		DATA_CDF[:] = VAR_DATA_ALL[iVAR]

		setattr(DATA_CDF,'missing_value',MISS_DATA_ALL[iVAR])
		if len(VAR_INFO[2]) != 0:
			setattr(DATA_CDF,'units',VAR_INFO[2])

		if len(FILL_VALUE_ALL) != 0:
			setattr(DATA_CDF,'fill_value',FILL_VALUE_ALL[iVAR])

# Write out attributes

	for iATTR in range(len(ATTR)):
		setattr(NC_FID,ATTR[iATTR][0],ATTR[iATTR][1])

# Close the file.

	NC_FID.close()

	TEXT	= '*** SUCCESS writing file '+FILE_CDF
	print(TEXT)

	return
#
#
def write_netCDF_multi_new4(FILE_CDF,DIM_INFO,VAR_INFO_ALL,VAR_DEP_ALL,VAR_DATA_ALL,ATTRIB='',NETCDF='netCDF4',DEBUG='N'):

# Python module to write netCDF file

# Garry Hayman
# Centre for Ecology and Hydrology
# April 2017

# Modified from write_netCDF_multi_new3
# February 2020 

	DATE       = datetime.datetime.today()
	DATE       = datetime.datetime.strftime(DATE,'%d/%m/%Y %H:%M')

# Write out netCDF file

	if NETCDF == 'netCDF3':
		NC_FID     = Dataset(FILE_CDF,"w", format = "NETCDF3_CLASSIC")
	else:
		NC_FID     = Dataset(FILE_CDF,"w")

	ATTR       = [ \
			['history','Created from '+ATTRIB ], \
			['comment','Using python modules' ], \
			['created',DATE], \
			['author','Garry Hayman (Centre for Ecology and Hydrology, UK)'], \
			['contact','tel: +44-1491-692527, e-mail: garr@ceh.ac.uk'] \
		     ]

# Define the dimensions

	nDIMS      = len(DIM_INFO)

	for iDIM in range(nDIMS):
		if DEBUG == 'Y': print(DIM_INFO[iDIM])
		if DIM_INFO[iDIM][0] == 0:
			NC_FID.createDimension(DIM_INFO[iDIM][1],None)
		else:
			NC_FID.createDimension(DIM_INFO[iDIM][1],DIM_INFO[iDIM][0])

# Create variables and write data

	nVARS      = len(VAR_INFO_ALL)

	for iVAR in range(nVARS):

		if DEBUG == 'Y': print(iVAR)
		VAR_INFO      = VAR_INFO_ALL[iVAR]
		if DEBUG == 'Y': print(VAR_INFO)
		VAR_DEP       = VAR_DEP_ALL[iVAR]
		if DEBUG == 'Y': print(VAR_DEP)

		if VAR_INFO[4] != '': # fill value - needs to be set when creating the variable
			DATA_CDF    = NC_FID.createVariable(varname=VAR_INFO[0], \
				datatype=dtype(VAR_INFO[1]).char,dimensions=(VAR_DEP), \
				fill_value=float(VAR_INFO[4]))
		else:
			DATA_CDF    = NC_FID.createVariable(VAR_INFO[0],dtype(VAR_INFO[1]).char, \
				(VAR_DEP))
		DATA_CDF[:] = VAR_DATA_ALL[iVAR]

# Add variable attributes

		for iATTR in range(2,7):

			if iATTR == 2 and VAR_INFO[iATTR] != '': # units
				setattr(DATA_CDF,'units',VAR_INFO[iATTR])

			if iATTR == 3 and VAR_INFO[iATTR] != '': # missing value
				setattr(DATA_CDF,'missing_value',float(VAR_INFO[iATTR]))

			if iATTR == 5 and VAR_INFO[iATTR] != '': # long name
				setattr(DATA_CDF,'long_name',VAR_INFO[iATTR])

			if iATTR == 6 and VAR_INFO[iATTR] != '': # CF name
				setattr(DATA_CDF,'CF_name',VAR_INFO[iATTR])

# Write out global attributes

	for iATTR in range(len(ATTR)):
		setattr(NC_FID,ATTR[iATTR][0],ATTR[iATTR][1])

# Close the file.

	NC_FID.close()

	TEXT	= '*** SUCCESS writing file '+FILE_CDF
	print(TEXT)

	return
#
