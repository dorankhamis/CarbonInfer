#!/usr/bin/env python2.7
# -*- coding: iso-8859-1 -*-
import os
import datetime
import glob
import ast

import numpy as np

# Set the matplotlib backend.
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats

import iris
import iris.coord_categorisation
import iris.quickplot as qplt
import iris.plot as iplt

import jules

import make_time_coord
import parallelise


'''
Plots the results from the FLUXNET runs against observations.

Karina Williams
'''


OBS_FOLDER_PRE2015 = os.path.expandvars('$OBS_FOLDER_PRE2015')
OBS_FOLDER_FLUXNET2015 = os.path.expandvars('$OBS_FOLDER')
OBS_FOLDER_LBA = os.path.expandvars('$OBS_FOLDER_LBA')

print('OBS_FOLDER_PRE2015: ',OBS_FOLDER_PRE2015)
print('OBS_FOLDER_FLUXNET2015: ',OBS_FOLDER_FLUXNET2015)
print('OBS_FOLDER_LBA: ',OBS_FOLDER_LBA)

# keys are the variables, 'obs_var_name_pre2015' and 'model_var_name' are the var names
# for the observations and the model output resp.
VAR_NAMES_DICT = {
                'GPP': {'obs_var_name_pre2015':'GPP', 'model_var_name':'gpp_gb'},
                'Reco': {'obs_var_name_pre2015':'Reco', 'model_var_name':'Reco'},
                'NEE': {'obs_var_name_pre2015':'NEE', 'model_var_name':'NEE'},
                'SH': {'obs_var_name_pre2015':'Qh', 'model_var_name':'ftl_gb'},
                'LE': {'obs_var_name_pre2015':'Qlh', 'model_var_name':'latent_heat'},
                'Transpiration': {'model_var_name':'et_stom_gb'},
                'Bowen_Ratio': {'obs_var_name_pre2015':'Bowen_Ratio', 'model_var_name':'Bowen_Ratio'},
                #'NPP': {'obs_var_name_pre2015':'NPP', 'model_var_name':'NPP'},
                'Rauto': {'obs_var_name_pre2015':'Rauto', 'model_var_name':'resp_p_gb'} #check this one!
               }

def add_hour(cube, coord, name='hour'):
    """Add a categorical hour coordinate, values 0..23.
    Taken from myutils.py (KW) 20.7.16."""
    iris.coord_categorisation.add_categorised_coord(
        cube, name, coord,
        lambda coord, x: coord.units.num2date(x).hour
        )


def jules_output_cube(cube, field, filename):
    '''
    Callback to add an x and y coord (otherwise load returns a cube with anonymous 
    dimensions, which can't be concatenated).
    '''
    
    n = cube.ndim
    
    try:
        if cube.coord_dims('time') == (0,): # assume if time is a dimcoord, it is at position 0
            n -= 1
    except iris.exceptions.CoordinateNotFoundError:
        pass
    
    if n >= 1:
        x_coord = iris.coords.DimCoord(range(cube.shape[-1]), var_name='x')
        xdim = cube.ndim - 1
        cube.add_dim_coord(x_coord, (xdim, ))
    
    if n >= 2: 
        y_coord = iris.coords.DimCoord([0], var_name='y')
        ydim = cube.ndim - 2
        cube.add_dim_coord(y_coord, (ydim, ))
    
    return


def convert_micromolCO2_to_gC(val):
    molar_mass_of_carbon = 12.011 # g / mol
    return val * 1E-6 * molar_mass_of_carbon


def convert_micromolCO2_per_m2_per_s_to_gC_per_m2_per_day(val):
    # roughly the same anyway: 1 micromol CO2/m2/s = 1.037 g/m2/day
    seconds_in_day = 60 * 60 * 24
    return convert_micromolCO2_to_gC(val) * seconds_in_day


def chop_off_partial_days(cube, check_masked=True):
    '''
    Chop off any partial days at the beginning or end of the time series.
    Works this out by making sure the first bound of the first timestep and the last bound of the last timestep
    are both at midnight (and also checks that these data points are not masked unless check_masked=False). 

    Args:
     * cube: should have one dim_coord, called 'time'. This time coord should have bounds.
     
    KWargs:
     * check_masked: False means decision about which timesteps to chop does not depend on
      data array. True means that it enforces that the beginning and end of resulting cube
      is not masked data.

    Returns:
     * cube: the inputted cube, but with any partial days and masked data points at the beginning or end stripped out.

    '''

    if not cube.coord('time').has_bounds():
         raise UserWarning('the time coord in the cube passed to '
                           'chop_off_partial_days should already have bounds')

    if [coord.name() for coord in cube.coords(dim_coords=True)] != ['time']:
         raise UserWarning('expected only one dim coord, and that should be called time')

    time_units = cube.coord('time').units

    def dt_valid_midnight(dt, data_point):
        if check_masked:
            is_valid = (dt.hour, dt.minute, dt.second) == (0,0,0) and not data_point is np.ma.masked
        else:
            is_valid = (dt.hour, dt.minute, dt.second) == (0,0,0)
        return is_valid

    print(cube.coord('time').cell(0))
    print(cube.coord('time').cell(-2))
    print(cube.coord('time').cell(-1)) 

    time_beg = datetime.datetime(1970,1,1,0,0,0)+datetime.timedelta(hours=cube.coord('time').cell(0).bound[0])
    print('time begin: ',time_beg)
    time_end = datetime.datetime(1970,1,1,0,0,0)+datetime.timedelta(hours=cube.coord('time').cell(-1).bound[1])
    print('time end: ',time_end)

   #with iris.FUTURE.context(cell_datetime_objects=True):

   #    time_beg = cube.coord('time').cell(0).bound[0]
   #    print('time begin: ',time_beg)
   #    time_end = cube.coord('time').cell(-1).bound[1]
   #    print('time end: ',time_end) 

   #    if not dt_valid_midnight(time_beg, cube.data[0]):  
   #        for i, cell in enumerate(cube.coord('time').cells()):
   #            if dt_valid_midnight(cell.bound[0], cube.data[i]): 
   #                cube = cube[i:]
   #                break
   #   
   #    if not dt_valid_midnight(time_end, cube.data[-1]):  
   #        cell_list = [cell for cell in cube.coord('time').cells()]
   #        for i, cell in reversed(list(enumerate(cell_list))):              
   #            if dt_valid_midnight(cell.bound[-1], cube.data[i]):
   #                cube = cube[:i + 1]
   #                break

    if not dt_valid_midnight(time_beg, cube.data[0]):  
        for i, cell in enumerate(cube.coord('time').cells()):
            time = datetime.datetime(1970,1,1,0,0,0)+datetime.timedelta(hours=cell.bound[0])
            if dt_valid_midnight(time, cube.data[i]):
                print('Data chopped from beginning to:',time,i) 
                cube = cube[i:]
                break
   
    if not dt_valid_midnight(time_end, cube.data[-1]):  
        cell_list = [cell for cell in cube.coord('time').cells()]
        for i, cell in reversed(list(enumerate(cell_list))):              
            time = datetime.datetime(1970,1,1,0,0,0)+datetime.timedelta(hours=cell.bound[-1])
            if dt_valid_midnight(time, cube.data[i]):
                print('Data chopped from end to:',time,i) 
                cube = cube[:i + 1]
                break

    return cube


def unify_time_units(cubes):
    '''
    Wraps iris.utils.unify_time_units but also checks for different calendars first
    (iris.utils.unify_time_units does not unify time units for time coords with different calendars)
    '''
    
    calendar = None
    for cube in cubes:
        for time_coord in cube.coords():
            if time_coord.units.is_time_reference():
                if calendar is None:
                    calendar = time_coord.units.calendar
                else:
                    if calendar != time_coord.units.calendar:
                         raise UserWarning('time coords have different calendars')
                         
    iris.util.unify_time_units(cubes)
    

def chop_off_partial_months(cube):
    '''
    Chop off any partial months at the beginning or end of the time series.
    Works this out by making sure the first bound of the first timestep and the last bound of the last timestep
    are both at midnight on the 1st of the month (and also checks that these data points are not masked). 

    Args:
     * cube: should have one dim_coord, called 'time'. This time coord should have bounds.

    Returns:
     * cube: the inputted cube, but with any partial months and masked data points at the beginning or end stripped out.

    '''

    if not cube.coord('time').has_bounds():
         raise UserWarning('the time coord in the cube passed to '
                           'chop_off_partial_months should already have bounds')

    if [coord.name() for coord in cube.coords(dim_coords=True)] != ['time']:
         raise UserWarning('expected only one dim coord, and that should be called time')

    time_units = cube.coord('time').units

    with iris.FUTURE.context(cell_datetime_objects=True):

        time_beg = cube.coord('time').cell(0).bound[0]
        time_end = cube.coord('time').cell(-1).bound[1] 

        if (time_beg.day, time_beg.hour, time_beg.minute, time_beg.second) != (1,0,0,0) or \
          cube.data[0] is np.ma.masked:
            for i, cell in enumerate(cube.coord('time').cells()):
                bound = cell.bound[0]
                if (bound.day, bound.hour, bound.minute, bound.second) == (1,0,0,0) and \
                   not cube.data[i] is np.ma.masked:
                    cube = cube[i:]
                    break

        if (time_end.day, time_end.hour, time_end.minute, time_end.second) != (1,0,0,0) or \
          cube.data[-1] is np.ma.masked:
            cell_list = [cell for cell in cube.coord('time').cells()]
            for i, cell in reversed(list(enumerate(cell_list))):
                bound = cell.bound[-1]
                if (bound.day, bound.hour, bound.minute, bound.second) == (1,0,0,0) and \
                  not cube.data[i] is np.ma.masked:
                    cube = cube[:i + 1]
                    break

    return cube


def time_units_are_equal(cubes):
    """
    Checks that the time units are all the same.

    Arg:

    * cubes:
        An iterable containing iris.cube.Cube instances.

    Returns:

    * equal_time_units: 
        True if the time units are the same, False otherwise.

    """
    epochs = {}

    equal_time_units = True

    time_units = None
    for cube in cubes:
        for time_coord in cube.coords():
            if time_coord.units.is_time_reference():
                if time_units is None:
                    time_units = time_coord.units
                else:
                    if time_units != time_coord.units:
                        equal_time_units = False

    return equal_time_units 


def cut_to_overlapping_time_period(cubes):
    """
    Cuts the cubes down, keeping only the time period that
    overlaps.
    Works this out using the first bound of the first day and the last bound of the last day.

    Arg:

    * cubes:
        An iterable containing iris.cube.Cube instances.
        Each cube needs to have a time coord called 'time', with bounds.

    Returns:

    * cubes:
        The inputted cubes reduced to the same time period as eachother. Returns an
        empty cubelist if there is no overlapping time period.

    """
    for cube in cubes:
        if not cube.coord('time').has_bounds():
            raise UserWarning('the time coord in the cube passed to '
                              'cut_to_overlapping_time_period should already have bounds')

    with iris.FUTURE.context(cell_datetime_objects=True):

        time_beg = cubes[0].coord('time').cell(0).bound[0]
        time_end = cubes[0].coord('time').cell(-1).bound[1] 

        for cube in cubes[1:]:
            if time_beg < cube.coord('time').cell(0).bound[0]:
                time_beg = cube.coord('time').cell(0).bound[0]

            if time_end > cube.coord('time').cell(-1).bound[1]:
                time_end = cube.coord('time').cell(-1).bound[1]

        time_beg_constraint = iris.Constraint(time=lambda cell: cell.bound[0] >= time_beg)
        cubes = iris.cube.CubeList(cubes).extract(time_beg_constraint)
        time_end_constraint = iris.Constraint(time=lambda cell: cell.bound[1] <= time_end)
        cubes = cubes.extract(time_end_constraint)

    return cubes


def check_all_time_coords_are_the_same(cubes):
    """
    Checks that the time coords are identical.
    Where they are different, prints some more information to screen.

    Arg:

    * cubes:
        An iterable containing iris.cube.Cube instances.
        Each cube needs to have a time coord called 'time'.

    Returns:
     * all_same: True if all the time coords are identical, 
       False otherwise.

    """

    all_same = True

    time_coord = cubes[0].coord('time')

    for cube in cubes[1:]:
        if time_coord != cube.coord('time'):
            all_same = False

            if not np.array_equal(time_coord.points, cube.coord('time').points):
                 print 'points are not the same' 
            if not np.array_equal(time_coord.bounds, cube.coord('time').bounds):
                 print 'bounds are not the same' 
            if time_coord.units != cube.coord('time').units:
                 print 'units are not the same' 

    return all_same


def reduce_month_coord_string(original_string):
    '''
    Sometimes iris operations can end up with the month
    being lots of 3 letter month strings separated by
    '|'. If all the three letter month strings are 
    the same, this function replaces the whole string with 
    just a single three letter month string.

    e.g. if the argument is 'Jan|Jan|Jan|Jan'
    then the function returns 'Jan'

    Args:
    * original_string:

    Returns:
    * reduced_string:
    '''

    reduced_string = original_string

    split_string = original_string.split('|')
    if len(split_string) > 1:
        if len(set(split_string)) == 1:
            reduced_string = split_string[0]

    return reduced_string


def read_model_dump(var_name, model_output_folder, run_id=None, timestamp=None):
    '''
    Reads a variable from a model dump file. 

    Arg:
       * var_name: var_name of the variable, as written in the netCDF model output file 
       * model_output_folder: folder containing the model output              

    Kwarg:
       * run_id: JULES run_id (from the output profile namelist) e.g. JP_Tak-vn4.6_trunk
       * timestamp: The timestamp in the dump file name e.g. "20020101.0" (
         beginning of the 1st Jan 2002).

    Returns
       * cube: The model dump file variable in a cube.
    '''

    model_filename = os.path.join(model_output_folder, run_id + ".dump." + timestamp + ".nc")

    var_name_constraint = iris.Constraint(cube_func=lambda x: x.var_name == var_name)

    cube = jules.load_cube(model_filename, var_name_constraint, conv_to_grid=False)

    return cube


def read_model_output(var_name, model_output_folder, run_id=None, profile_name=None):
    '''
    Reads the output from the model for an output profile.
    Does a bit of tidying: makes sure the time points are at the midpoints of the bounds
    (not the endpoints) and converts units for some of the variables, and makes sure some
    variables do not go below zero.
        
    Arg:
       * var_name: var_name of the variable, as written in the netCDF model output file e.g. "gpp_gb"         
       * model_output_folder: folder containing the model output             

    Kwarg:
       * run_id: JULES run_id (from the output profile namelist) e.g. JP_Tak-vn4.6_trunk
       * profile_name: Should be 'H' (hourly) or 'D' (daily).

    Returns
       * cube: The model data in a cube.

    '''
    if profile_name not in ['H', 'D']:
        raise UserWarning('profile_name not recognised')

    model_filename = os.path.join(model_output_folder, run_id + "." + profile_name + "*.nc")

    var_name_constraint = iris.Constraint(cube_func=lambda x: x.var_name == var_name)
    print 'reading ' + model_filename
    cubelist = jules.load(model_filename, var_name_constraint, conv_to_grid=False, callback=jules_output_cube)
    unify_time_units(cubelist)

    cube = cubelist.concatenate_cube()
    cube.data # make sure the data is read in right away, otherwise get in to trouble with biggus
    # (it gets confused about which points should be masked)

    cube = iris.util.squeeze(cube)

    if profile_name == 'D':
        # move the time points so that they are at the middle of the time period, not
        # the end
        cube.coord('time').points = 0.5 * ( cube.coord('time').bounds[:,0] + cube.coord('time').bounds[:,1] ) 
    elif profile_name == 'H':
        # move the time points so that they are at the beginning of the time period, not
        # the end
        cube.coord('time').points = cube.coord('time').bounds[:,0]

    if var_name in ["gpp_gb", "resp_s_gb", "resp_p_gb"]:
        cube *= 86400 * 1000 
        cube.units = "g/m2/day"

        cube.data = np.ma.maximum(cube.data, 0.0)

    elif var_name in ["et_stom_gb", "ecan_gb", "esoil_gb", "fqw_gb"]:
        cube *= 86400 
        cube.units = "mm/day"

    if cube.coord('time').units.calendar == 'gregorian':
        pass
    elif cube.coord('time').units.calendar == 'standard':
       cube.coord('time').units = make_time_coord.wrap_unit(cube.coord('time').units.origin, 'gregorian')
    elif profile_name == 'D' and cube.coord('time').units.calendar == '365_day':  
       # Converts calendar to gregorian and interpolates for day 366.
       cube = make_time_coord.convert_calendar_from_365_day_to_gregorian(cube)       
       cube = make_time_coord.interpolate_to_get_daily_cube(cube)
       cube.coord('time').guess_bounds(bound_position=0.5) 
    else:
       raise Exception('calendar has not been recogised')
       
    return cube
    

def read_subdaily_pre2015_obs(site, var_name):
    '''
    Reads the processed subdaily pre-FLUXNET2015 observation files.
    Does a bit of tidying: adds bounds (assuming time point labels the beginning of the bound) 
    and converts units for some of the variables, and makes sure some
    variables do not go below zero. Also sets the calendar and adds a long_name for the
    time coord.
    
    Have not checked the timezone.
        
    Arg:
       * site: name of the FLUXNET site e.g. "JP_Tak"
       * var_name: var_name of the variable, as written in the processed obs file e.g. "GPP" 

    Returns
       * cube: The subdaily observations in a cube.

    '''

    obs_filename = os.path.join(OBS_FOLDER_PRE2015, site + "_hh.csv_filled.nc")

    var_name_constraint = iris.Constraint(cube_func=lambda x: x.var_name == var_name)
    cube = iris.load_cube(obs_filename, var_name_constraint) 
    cube.data # make sure the data is read in right away, otherwise get in to trouble with biggus
    # (it gets confused about which points should be masked)

    # add some time bounds
    cube.coord('time').guess_bounds(bound_position=0.0)  #is the correct bound_position to use for this data?

    # set the calendar to be "gregorian"
    new_unit = make_time_coord.wrap_unit(cube.coord('time').units.origin, 'gregorian')
    cube.coord('time').units = new_unit

    cube.coord('time').long_name = "Time of data" # to be consistent with the JULES output

    cube = iris.util.squeeze(cube)
    if var_name in ["NEE", "GPP", "Reco"]:
        cube.data = convert_micromolCO2_per_m2_per_s_to_gC_per_m2_per_day(cube.data)
        cube.units = "g/m2/day" 

    if var_name in ["GPP", "Reco"]:
        cube.data = np.ma.maximum(cube.data, 0.0)

    return cube


def get_processed_daily_model_data(var_name, model_output_folder, run_id=None, 
                                   use_daily_output_if_possible=False):
    '''
    Calls function to read in subdaily model data and converts it to daily data
    for a particular variable.
    Partial days at the beginning and end of the time series are removed before
    the daily means are calculated. For var_name='NEE', the mean is subtracted. 
        
    Arg:
       * var_name: var_name of the variable, as written in the netCDF model output file e.g. "gpp_gb" 
       * model_output_folder: folder containing the model output       

    Kwarg:
       * run_id: JULES run_id (from the output profile namelist) e.g. JP_Tak-vn4.6_trunk

    Returns
       * cube: The daily model data in a cube.

    '''
    if var_name in ["24hNPP_per_24hGPP", "24hNPP_per_24hAPAR", "24hGPP_per_24hAPAR"]: 
        profile_name = 'D'
        if not use_daily_output_if_possible:
            raise UserWarning(var_name + ' needs daily output')
    elif not use_daily_output_if_possible or var_name in ["Bowen_Ratio", "daytimeCUE"]: 
        # Bowen ratio and CUE always need to be calculated with hourly output
        profile_name = 'H'
    else:    
        profile_name = 'D'
        
    # Read model output. In some cases, variable is not outputted directly
    # by model but is calculated from a combination of other variables.
    if var_name == "Reco":
        resp_p_cube = read_model_output('resp_p_gb', model_output_folder, run_id=run_id, profile_name=profile_name)
        resp_s_cube = read_model_output('resp_s_gb', model_output_folder, run_id=run_id, profile_name=profile_name)
        cube = resp_p_cube + resp_s_cube
    elif var_name == 'NEE': # mean will be subtracted later
        resp_p_cube = read_model_output('resp_p_gb', model_output_folder, run_id=run_id, profile_name=profile_name)
        resp_s_cube = read_model_output('resp_s_gb', model_output_folder, run_id=run_id, profile_name=profile_name)
        gpp_cube = read_model_output('gpp_gb', model_output_folder, run_id=run_id, profile_name=profile_name)
        cube = resp_p_cube + resp_s_cube - gpp_cube
    elif var_name == 'NEP': 
        resp_p_cube = read_model_output('resp_p_gb', model_output_folder, run_id=run_id, profile_name=profile_name)
        resp_s_cube = read_model_output('resp_s_gb', model_output_folder, run_id=run_id, profile_name=profile_name)
        gpp_cube = read_model_output('gpp_gb', model_output_folder, run_id=run_id, profile_name=profile_name)
        cube = gpp_cube - resp_p_cube - resp_s_cube
    elif var_name == 'NPP':
        resp_p_cube = read_model_output('resp_p_gb', model_output_folder, run_id=run_id, profile_name=profile_name)
        gpp_cube = read_model_output('gpp_gb', model_output_folder, run_id=run_id, profile_name=profile_name)
        cube = gpp_cube - resp_p_cube
    elif var_name == "Bowen_Ratio":
        le_cube = read_model_output("latent_heat", model_output_folder, run_id=run_id, profile_name=profile_name)
        sh_cube = read_model_output("ftl_gb", model_output_folder, run_id=run_id, profile_name=profile_name)
        cube = sh_cube / le_cube
    elif var_name == "24hNPP_per_24hGPP": 
        resp_p_cube = read_model_output("resp_p_gb", model_output_folder, run_id=run_id, profile_name=profile_name)
        gpp_cube = read_model_output("gpp_gb", model_output_folder, run_id=run_id, profile_name=profile_name)
        cube = resp_p_cube / gpp_cube * (-1.0) + 1.0
    #elif var_name == "daytimeCUE": # trying to implement as in Marthews et al 2012, but too sensitive to sw_down_threshol
    #    resp_p_cube = read_model_output("resp_p_gb", model_output_folder, run_id=run_id, profile_name=profile_name)
    #    gpp_cube = read_model_output("gpp_gb", model_output_folder, run_id=run_id, profile_name=profile_name)
    #    sw_down = read_model_output("sw_down", model_output_folder, run_id=run_id, profile_name=profile_name)
    #    cube = resp_p_cube / gpp_cube * (-1.0) + 1.0
    #
    #    # now get rid of nighttime values
    #    sw_down_threshold = 10.0 # Wm-2
    #    cube.data = np.ma.masked_where(sw_down.data < sw_down_threshold, cube.data)
    elif var_name == "24hGPP_per_24hAPAR": 
        gpp_cube = read_model_output("gpp_gb", model_output_folder, run_id=run_id, profile_name=profile_name)      
        apar_cube = read_model_output("apar_gb", model_output_folder, run_id=run_id, profile_name=profile_name)      
        cube = gpp_cube / apar_cube
    elif var_name == "24hNPP_per_24hAPAR": 
        resp_p_cube = read_model_output("resp_p_gb", model_output_folder, run_id=run_id, profile_name=profile_name)
        gpp_cube = read_model_output("gpp_gb", model_output_folder, run_id=run_id, profile_name=profile_name)      
        apar_cube = read_model_output("apar_gb", model_output_folder, run_id=run_id, profile_name=profile_name)      
        npp_cube = gpp_cube - resp_p_cube
        cube = npp_cube / apar_cube
    else:
        cube = read_model_output(var_name, model_output_folder, run_id=run_id, profile_name=profile_name)

    # add some extra aux_coords
    iris.coord_categorisation.add_year(cube, 'time')
    iris.coord_categorisation.add_month(cube, 'time')
    iris.coord_categorisation.add_day_of_year(cube, 'time')
    
    if profile_name == 'H':
        if var_name == 'daytimeCUE': # this is only for daytime (so be aware that there might be partial days
                              # that we've not taken into account)
            mdtol = None
        else:
            # chop off partial days at beginning or end
            cube = chop_off_partial_days(cube)

            # If there's more than 3% missing data in a day, mask that day.
            mdtol = 0.03
         
        # convert to daily data.
        cube = wrap_aggregated_by_with_mdtol(cube, ['year', 'day_of_year'], iris.analysis.MEAN, mdtol=mdtol)

        # tidy month coord
        cube.coord('month').points = map(reduce_month_coord_string, cube.coord('month').points)  
    
    # for NEE, need to subtract the mean
    if var_name == 'NEE':
        cube = cube - cube.collapsed('time', iris.analysis.MEAN)

    return cube


def get_daily_pre2015_obs_data(site, var_name, regenerate_files=False):
    '''
    Calls function to read in processed subdaily pre-FLUXNET2015 observations and converts to daily data
    for a particular variable.
    Partial days at the beginning and end of the time series are removed before
    the daily means are calculated. For var_name='NEE', the mean is subtracted. 
        
    Have not checked what the time zone is.    
        
    Arg:
       * site: name of the FLUXNET site e.g. "JP_Tak"
       * var_name: var_name of the variable, as written in the processed obs file e.g. "GPP"       

    Returns
       * cube: The daily obs data in a cube.

    '''

    processed_daily_from_subdaily_obs_filename = os.path.join(OBS_FOLDER_PRE2015, 
        "processed-" + site + '_' + var_name + "_hh.csv_filled.nc")
    
    try:  
        # quick hack: should implement this option with its own exception class 
        if regenerate_files:
            raise IOError

        cube  =  iris.load_cube(processed_daily_from_subdaily_obs_filename)

        # get rid of some of the coord var_names
        for coord_name in ["year", "month", "day_of_year"]:
            cube.coord(coord_name).var_name = None

        cube.data # make sure the data is read in right away, otherwise get in to trouble with biggus
        # (it gets confused about which points should be masked)

    except IOError:

        # Read processed subdaily observations. In some cases, variable is not explicitly in the obs
        # file but is calculated from a combination of other variables.

        if var_name == "Bowen_Ratio":
            le_cube = read_subdaily_pre2015_obs(site, "Qlh")
            sh_cube = read_subdaily_pre2015_obs(site, "Qh")
            cube = sh_cube / le_cube
        elif var_name == "NPP":
            raise Exception('still need to implement this!')
        elif var_name == "Rauto":
            raise Exception('still need to implement this!')
        else:
            cube = read_subdaily_pre2015_obs(site, var_name)

        # add some extra aux_coords
        iris.coord_categorisation.add_year(cube, 'time')
        iris.coord_categorisation.add_month(cube, 'time')
        iris.coord_categorisation.add_day_of_year(cube, 'time')

        ## Convert from half-hourly data to hourly data.
        ## Not doing this conversion at the moment, but it can be added back in if needed.
        #add_hour(cube, 'time')
        #cube = cube.aggregated_by(["year", "day_of_year", "hour"], iris.analysis.MEAN)

        # chop off partial days at beginning or end
        cube = chop_off_partial_days(cube)

        # convert to daily data. If there's more than 3% missing data in a day, mask that day.
        cube = wrap_aggregated_by_with_mdtol(cube, ['year', 'day_of_year'], iris.analysis.MEAN, mdtol=0.03)
        
        # tidy month coord
        cube.coord('month').points = map(reduce_month_coord_string, cube.coord('month').points)

        # for NEE, need to subtract the mean
        if var_name == 'NEE':
            cube = cube - cube.collapsed('time', iris.analysis.MEAN)

        iris.save(cube, processed_daily_from_subdaily_obs_filename)

    return cube


def get_subdaily_fluxnet2015_obs_data(site, var_name, local_time=True):
        '''
    Reads the subdaily FLUXNET2015 observation files.
    Also sets the calendar and adds a long_name for the
    time coord.
        
    Arg:
       * site: name of the FLUXNET site e.g. "FI_Hyy"
       * var_name: var_name of the variable (as given in VAR_NAMES_DICT.keys()) 
       * local_time: True means times are local standard time 
         (as in original FLUXNET2015 files, no daylight saving time),
         False means UTC.

    Returns
       * cube: The subdaily observations in a cube.

    '''

        if not local_time:
            raise Exception('have not implemented this yet')

        delimiter = ','

        dt_format = '%Y%m%d%H%M'

        NETWORK = os.environ['NETWORK']
        print('Network: ',NETWORK)

        if 'UK' in NETWORK:
            subdaily_energy_file_headings  = [ \
                'TIMESTAMP_START', 'TIMESTAMP_END', 'G_F_MDS_1', 'G_F_MDS_2', \
                'LE_F_MDS', 'LE_RANDUNC', 'H_F_MDS', 'H_RANDUNC' \
                                             ]

            subdaily_carbon_file_headings = ['TIMESTAMP_START', 'TIMESTAMP_END', \
                'NEE_VUT_REF', 'NEE_VUT_REF_RANDUNC',  'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', \
                'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF' \
                                             ]
        else:
            subdaily_energy_file_headings = [ 
                'TIMESTAMP_START', 'TIMESTAMP_END', 'G_F_MDS', 'LE_F_MDS', 'LE_CORR',
                'LE_CORR_25', 'LE_CORR_75', 'LE_RANDUNC', 'H_F_MDS', 'H_CORR', 
                'H_CORR_25', 'H_CORR_75', 'H_RANDUNC']

            subdaily_carbon_file_headings = ['TIMESTAMP_START', 'TIMESTAMP_END', 'NEE_VUT_REF',
                'NEE_VUT_REF_RANDUNC', 'NEE_VUT_25', 'NEE_VUT_50', 'NEE_VUT_75', 
                'RECO_NT_VUT_REF', 'RECO_NT_VUT_25', 'RECO_NT_VUT_50', 'RECO_NT_VUT_75', 
                'GPP_NT_VUT_REF', 'GPP_NT_VUT_25', 'GPP_NT_VUT_50', 'GPP_NT_VUT_75', 
                'RECO_DT_VUT_REF', 'RECO_DT_VUT_25', 'RECO_DT_VUT_50', 'RECO_DT_VUT_75', 
                'GPP_DT_VUT_REF', 'GPP_DT_VUT_25', 'GPP_DT_VUT_50', 'GPP_DT_VUT_75'
                ]

        var_dict = {
            'GPP': {'obs_var_name_fluxnet2015':'GPP_NT_VUT_REF'},
            'Reco': {'obs_var_name_fluxnet2015':'RECO_NT_VUT_REF'},
            'NEE': {'obs_var_name_fluxnet2015':'NEE_VUT_REF'},
            'SH': {'obs_var_name_fluxnet2015':'H_F_MDS'},
            'LE': {'obs_var_name_fluxnet2015':'LE_F_MDS'},
            }

        if var_dict[var_name]['obs_var_name_fluxnet2015'] in subdaily_energy_file_headings:
            filename = os.path.join(OBS_FOLDER_FLUXNET2015, 'subdaily_obs', site + '-energy.dat')
            names = subdaily_energy_file_headings
        elif var_dict[var_name]['obs_var_name_fluxnet2015'] in subdaily_carbon_file_headings:
            filename = os.path.join(OBS_FOLDER_FLUXNET2015, 'subdaily_obs', site + '-carbon.dat')
            names = subdaily_carbon_file_headings            
        else:
            raise Exception('obs_var_name_fluxnet2015 is not recognised')

        timestamp_convertfunc = lambda x: datetime.datetime.strptime(x, dt_format)

        #rename 'TIMESTAMP_START' as 'TIMESTAMP' so make_time_coord recognises it
        names = [x.replace('TIMESTAMP_START', 'TIMESTAMP') for x in names]
        
        converters = {'TIMESTAMP':timestamp_convertfunc, 
                      'TIMESTAMP_END':timestamp_convertfunc}

        print(filename, names)

        data = np.genfromtxt(filename, names=names, converters=converters,
                             dtype=None, deletechars='', delimiter=delimiter,
                             skip_header=0, skip_footer=0, comments='#',
                             usemask=True, missing_values=-9999) 

        time_coord = make_time_coord.make_time_coord(data)
        time_coord.var_name = 'time'

        cube = iris.cube.Cube(data[var_dict[var_name]['obs_var_name_fluxnet2015']], 
                              var_name=var_name)
        cube.add_dim_coord(time_coord, (0,))

        cube.data # make sure the data is read in right away, otherwise get in to trouble with biggus
        # (it gets confused about which points should be masked)

        # add some time bounds
        cube.coord('time').guess_bounds(bound_position=0.0)  

        # set the calendar to be "gregorian"
        new_unit = make_time_coord.wrap_unit(cube.coord('time').units.origin, 'gregorian')
        cube.coord('time').units = new_unit

        cube.coord('time').long_name = "Time of data" # to be consistent with the JULES output

        return cube


def create_fluxnet2015_dailyUTC_files(site, utc_offset, output_folder=None):
    '''
    Creates daily mean files where days are UTC, rather than local time.
    Note no unit conversion. 
    
    args:
      * site: name of FLUXNET2015 site.
      * utc_offset: UTC offset of site, as given in the FLUXNET2015 supplementary info. 
        Positive offset means site is east of Grenwich, negative offset means 
        site is west of Grenwich.
        could get this with the function
        utc_offset = generate_conf_files.read_site_info(site, 'UTC_OFFSET')
    
    kwargs:
      * output folder: where to save the new text file
    '''    
    var_name_list = ['GPP', 'Reco', 'NEE', 'SH', 'LE']

    cubedict = {}
    
    for i,var_name in enumerate(var_name_list):
    
        cube = get_subdaily_fluxnet2015_obs_data(site, var_name, local_time=True)
        
        if i == 0:
            local_dt_array = cube.coord('time').units.num2date(cube.coord('time').points)
            gmt_dt_array = local_dt_array - datetime.timedelta(hours=utc_offset) 
            gmt_time_coord = make_time_coord.create_time_coord_from_array_of_datetimes(gmt_dt_array)
            gmt_time_coord.guess_bounds(bound_position=0.0)
        else:
            tmp_local_dt_array = cube.coord('time').units.num2date(cube.coord('time').points)
            if not np.array_equal(local_dt_array, tmp_local_dt_array):
                print var_name
                print local_dt_array
                print tmp_local_dt_array
                raise Exception('was expecting these to be the same (1)')
                
        cube.remove_coord('time')
        cube.add_dim_coord(gmt_time_coord, (0,))
        iris.coord_categorisation.add_year(cube, 'time')
        iris.coord_categorisation.add_day_of_year(cube, 'time')

        print('utc offset: ',utc_offset)
        print(cube.coord('time').cell(0))
        print(cube.coord('time').cell(-2))
        print(cube.coord('time').cell(-1)) 

        cube = chop_off_partial_days(cube, check_masked=False)
        cube = wrap_aggregated_by_with_mdtol(cube, ['year', 'day_of_year'], iris.analysis.MEAN, mdtol=0.03)
        
        if i != 0:
            if cube.ndim != cubedict[var_name_list[0]].ndim:
                raise Exception('was expecting these to be the same (2)')
            if cube.coord('time') != cubedict[var_name_list[0]].coord('time'):
                print cube.coord('time')
                print cubedict[var_name_list[0]].coord('time')
                raise Exception('was expecting these to be the same (3)')
        
        cubedict[var_name] = cube
    
    filename = os.path.join(output_folder, site + '-energyandcarbon-dailyUTC.dat')
    
    dt_format = '%Y%m%d'
    with open(filename, 'w') as f:
        f.write('# YYYYMMDD_UTC, ' + ', '.join(var_name_list) + ' \n')
        cube0 = cubedict[var_name_list[0]]
        for i in range(len(cube0.data)):
            dt = cube0.coord('time').units.num2date(cube0.coord('time').points[i])
            print dt
            f.write(dt.strftime(dt_format) + ', ')
            f.write(', '.join(str(cubedict[var_name].data[i]) for var_name in var_name_list))
            f.write('\n')
        
    return
    

def get_daily_obs_data(site, var, obs=None):
    '''
    Wrapper to the function that reads in the daily observations.

    KWargs:
      * obs: obs dataset. 'peg', 'lba', 'all' or None.
    '''

    if obs == 'peg':
        if var in VAR_NAMES_DICT.keys():
            obs_cube = get_daily_fluxnet2015_obs_data(site, var, local_time=False)
        else:
            obs_cube = None
    elif obs == 'lba':
        if var in VAR_NAMES_DICT.keys():
            obs_cube = get_daily_lba_obs_data(site, var, local_time=True) # should be local_time=False but have not created these files yet: waiting for confirmation that the original LBA version 2 files are in local time, despite saying 'UTM'.
        else:
            obs_cube = None
    elif obs == 'all':
        if var in VAR_NAMES_DICT.keys():
            if 'lba' in site.lower():
                obs_cube = get_daily_lba_obs_data(site, var, local_time=True)
            else:    
                obs_cube = get_daily_fluxnet2015_obs_data(site, var, local_time=False)
        else:
            obs_cube = None
    elif obs is None:
        obs_cube = None
    elif obs == 'user_defined':
        obs_cube = get_user_defined_daily_obs_data(site, var)
    else:
        raise UserWarning('subset has not been recognised')

    if obs_cube is not None:
        if np.ma.is_masked(obs_cube.data):
            if obs_cube.data.count() == 0:
                # all data is masked
                obs_cube = None

    return obs_cube


def get_daily_lba_obs_data(site, var_name, regenerate_files=False, local_time=True,
    use_processed_data=True, unprocessed_file_path=None):
    '''
    Reads in daily LBA observations and converts to a cube
    for a particular variable.
    For var_name='NEE', the mean is subtracted. 
        
    Arg:
       * site: name of the FLUXNET site e.g. "LBA_RJA"
       * var_name: var_name of the variable (as given in VAR_NAMES_DICT.keys())  
       * local_time: True means times are local standard time 
         (as in original FLUXNET2015 files, no daylight saving time),
         False means UTC.
         Assumes LBA version 2 files are in local time, despite saying
         UTM (see notes on email conversation)
       * use_processed_data: whether to use the raw file downloaded from saleskaweb or
         a processed version. 

    if use_processed_data=False, then there is an additional option:
       * unprocessed_file_path: path to the raw file downloaded from saleskaweb.

    Returns
       * cube: The daily obs data in a cube.

    '''
    if not local_time:
        raise Exception('have not implemented this yet')
    
    if var_name == 'Bowen_Ratio':
        raise Exception('still need to implement this!')
    elif var_name in ["NPP", "Rauto"]:
        raise Exception('still need to implement this!')
    else:
        # Read daily LBA observations. 

        delimiter = ','

        daily_energy_file_headings = [
            'Year_LBAMIP', 'DoY_LBAMIP', 'Hour_LBAMIP',
            'FG', 'Leraw', 'LE', 'Hraw', 'H', 'Rn'
            ] 

        daily_carbon_file_headings = [
            'Year_LBAMIP', 'DoY_LBAMIP', 'Hour_LBAMIP',
            'NEE', 'NEEf', 'NEE_model',  
            'Re_5day_ust_Sco2_LUT', 'GEP_model', 'par_fill', 'VPD', 
            'mrs'
            ]

        var_dict = {
            'GPP': {'obs_var_name_lba':'GEP_model'},
            'Reco': {'obs_var_name_lba':'Re_5day_ust_Sco2_LUT'},
            'NEE': {'obs_var_name_lba':'NEE_model'},
            'SH': {'obs_var_name_lba':'H'},
            'LE': {'obs_var_name_lba':'LE'},
            }

        if var_name in var_dict:
            obs_var_name_lba = var_dict[var_name]['obs_var_name_lba']
        else:
            obs_var_name_lba = var_name

        if use_processed_data == False:
            filename = os.path.join(unprocessed_file_path, site[4:] + 'day_CfluxBF.csv')
            names = True
        elif obs_var_name_lba in daily_energy_file_headings:
            filename = os.path.join(OBS_FOLDER_LBA, site[4:] + 'day-energy.dat')
            names = daily_energy_file_headings
        elif obs_var_name_lba in daily_carbon_file_headings:
            filename = os.path.join(OBS_FOLDER_LBA, site[4:] + 'day-carbon.dat')
            names = daily_carbon_file_headings
        else:
            raise Exception('obs_var_name_lba is not recognised')
        
        data = np.genfromtxt(filename, names=names, 
                             dtype=None, deletechars='', delimiter=delimiter,
                             skip_header=0, skip_footer=0,
                             usemask=True, missing_values=-9999) 

        # add 12 hours to Hour_LBAMIP to put the time at midday
        for i, hour in enumerate(data['Hour_LBAMIP']):
            data['Hour_LBAMIP'][i] += 12

        time_coord = make_time_coord.make_time_coord(data)
        time_coord.var_name = 'time'
        
        var_data = data[obs_var_name_lba]
        if var_data.dtype == bool: # happens when all points are masked
            var_data = var_data.astype(np.float32)
        
        cube = iris.cube.Cube(var_data, 
                              var_name=var_name)
        cube.add_dim_coord(time_coord, (0,))
        
        # Do a bit of tidying: adds bounds 
        # and converts units for some of the variables, and makes sure some
        # variables do not go below zero. Also sets the calendar and adds a long_name for the
        #time coord.

        cube.data # make sure the data is read in right away, otherwise get in to trouble with biggus
        # (it gets confused about which points should be masked)

        # add some time bounds
        cube.coord('time').guess_bounds(bound_position=0.5)  

        # set the calendar to be "gregorian"
        new_unit = make_time_coord.wrap_unit(cube.coord('time').units.origin, 'gregorian')
        cube.coord('time').units = new_unit

        cube.coord('time').long_name = "Time of data" # to be consistent with the JULES output

        # add some extra aux_coords
        iris.coord_categorisation.add_year(cube, 'time')
        iris.coord_categorisation.add_month(cube, 'time')
        iris.coord_categorisation.add_day_of_year(cube, 'time')

        cube = iris.util.squeeze(cube)

    if var_name in ["NEE", "GPP", "Reco"]:
        # carbon fluxes in original LBA files are in micromol CO2 m-2 s-1
        # See ftp://saleskalab.eebweb.arizona.edu/pub/BrasilFlux_Data/Version.2.0/Documentation/README_Files.txt
        cube.data = convert_micromolCO2_per_m2_per_s_to_gC_per_m2_per_day(cube.data) 
        cube.units = "g/m2/day" 

    if var_name in ["SH", "LE"]:
        cube.units = make_time_coord.wrap_unit('W m-2')

    if var_name in ["GPP", "Reco"]:
        cube.data = np.ma.maximum(cube.data, 0.0)

    # for NEE, need to subtract the mean
    if var_name == 'NEE':
        cube = cube - cube.collapsed('time', iris.analysis.MEAN)
        
    return cube


def get_daily_fluxnet2015_obs_data(site, var_name, regenerate_files=False, local_time=False,
    use_processed_data=True, unprocessed_filename=None):
    '''
    Reads in daily FLUXNET2015 observations and converts to a cube
    for a particular variable.
    For var_name='NEE', the mean is subtracted. 
        
    Arg:
       * site: name of the FLUXNET site e.g. "JP_Tak"
       * var_name: var_name of the variable (as given in VAR_NAMES_DICT.keys()) 
       * local_time: True means times are local standard time 
         (as in original FLUXNET2015 files, no daylight saving time),
         False means UTC.
       * use_processed_data: whether to use the raw file downloaded from FLUXNET or
         a processed version. Must be True if local_time=False. 

    if use_processed_data=False, then there is an additional option:
       * unprocessed_filename: path to the raw file downloaded from FLUXNET. 
         The filename should contain the string '_DD_'

    Returns
       * cube: The daily obs data in a cube.

    '''

    if use_processed_data == False:
        if local_time == False:
            raise UserWarning('Can only use the raw FLUXNET files if using local time')
        if unprocessed_filename is None:
            raise UserWarning('Need to give path to raw FLUXNET file if use_processed_data=False')
        else:
            if '_DD_' not in os.path.basename(unprocessed_filename):
                raise UserWarning('Name of raw FLUXNET file shold contain the string _DD_ i.e. should be daily')
 
    if var_name == 'Bowen_Ratio':

        if local_time == False or use_processed_data == False:
            raise UserWarning('have not yet implemented this case')
            
        processed_daily_from_subdaily_obs_filename = os.path.join(OBS_FOLDER_FLUXNET2015, 
            "processed-" + site + '_' + var_name + "_from_FLUXNET2015_subdaily.nc")
    
        try:  
            # quick hack: should implement this option with its own exception class 
            if regenerate_files:
                raise IOError

            cube  =  iris.load_cube(processed_daily_from_subdaily_obs_filename)

            cube.data # make sure the data is read in right away, otherwise get in to trouble with biggus
            # (it gets confused about which points should be masked)
            
            # remove the var_name of these coords, which were added by saving as netCDF and 
            # reading in again
            for coord_str in ['year', 'day_of_year', 'month']:
                cube.coord(coord_str).var_name = None

        except IOError:
            le_cube = get_subdaily_fluxnet2015_obs_data(site, 'LE')
            sh_cube = get_subdaily_fluxnet2015_obs_data(site, 'SH')
            cube = sh_cube / le_cube

            # add some extra aux_coords
            iris.coord_categorisation.add_year(cube, 'time')
            iris.coord_categorisation.add_month(cube, 'time')
            iris.coord_categorisation.add_day_of_year(cube, 'time')

            # chop off partial days at beginning or end
            cube = chop_off_partial_days(cube)

            # convert to daily data. If there's more than 3% missing data in a day, mask that day.
            cube = wrap_aggregated_by_with_mdtol(cube, ['year', 'day_of_year'], iris.analysis.MEAN, mdtol=0.03)

            # tidy month coord
            cube.coord('month').points = map(reduce_month_coord_string, cube.coord('month').points)

            cube.var_name = var_name
            cube.units = make_time_coord.wrap_unit('1')

            iris.save(cube, processed_daily_from_subdaily_obs_filename)

    elif var_name in ["NPP", "Rauto"]:
        raise Exception('still need to implement this!')
    else:
        # Read daily FLUXNET 2015 observations. 

        delimiter = ','

        dt_format = '%Y%m%d'
        
        if local_time:
            daily_energy_file_headings = [
                'TIMESTAMP','G_F_MDS','LE_F_MDS','LE_CORR','LE_CORR_25','LE_CORR_75',
                'LE_RANDUNC','H_F_MDS','H_CORR','H_CORR_25','H_CORR_75','H_RANDUNC'] 

            daily_carbon_file_headings = [
                'TIMESTAMP','NEE_VUT_REF','NEE_VUT_REF_RANDUNC','NEE_VUT_25','NEE_VUT_50','NEE_VUT_75',
                'RECO_NT_VUT_REF','RECO_NT_VUT_25','RECO_NT_VUT_50','RECO_NT_VUT_75',
                'GPP_NT_VUT_REF','GPP_NT_VUT_25','GPP_NT_VUT_50','GPP_NT_VUT_75',
                'RECO_DT_VUT_REF','RECO_DT_VUT_25','RECO_DT_VUT_50','RECO_DT_VUT_75',
                'GPP_DT_VUT_REF','GPP_DT_VUT_25','GPP_DT_VUT_50','GPP_DT_VUT_75']

            var_dict = {
                'GPP': {'obs_var_name_fluxnet2015':'GPP_NT_VUT_REF'},
                'Reco': {'obs_var_name_fluxnet2015':'RECO_NT_VUT_REF'},
                'NEE': {'obs_var_name_fluxnet2015':'NEE_VUT_REF'},
                'SH': {'obs_var_name_fluxnet2015':'H_F_MDS'},
                'LE': {'obs_var_name_fluxnet2015':'LE_F_MDS'},
                }

            if var_name in var_dict:
                obs_var_name_fluxnet2015 = var_dict[var_name]['obs_var_name_fluxnet2015']
            else:
                obs_var_name_fluxnet2015 = var_name

            if use_processed_data == False:
                filename = unprocessed_filename
                names = True
            elif obs_var_name_fluxnet2015 in daily_energy_file_headings:
                filename = os.path.join(OBS_FOLDER_FLUXNET2015, 'daily_obs', site + '-energy-daily.dat')
                names = daily_energy_file_headings
            elif obs_var_name_fluxnet2015 in daily_carbon_file_headings:
                # carbon fluxes are in gC/m2/day 
                # (were not converted after downloading: https://fluxnet.fluxdata.org/data/fluxnet2015-dataset/fullset-data-product/)
                filename = os.path.join(OBS_FOLDER_FLUXNET2015, 'daily_obs', site + '-carbon-daily.dat')
                names = daily_carbon_file_headings
            else:
                raise Exception('obs_var_name_fluxnet2015 is not recognised')
            missing_values = -9999     
        else:
            filename = os.path.join(OBS_FOLDER_FLUXNET2015, 'daily_obs', site + '-energyandcarbon-dailyUTC.dat')
            # carbon fluxes are in micromolCO2 m-2 s-1
            # see conversion later in function
            # (were not converted after downloading: https://fluxnet.fluxdata.org/data/fluxnet2015-dataset/fullset-data-product/)
            names = ['TIMESTAMP', 'GPP', 'Reco', 'NEE', 'SH', 'LE']
            obs_var_name_fluxnet2015 = var_name
            missing_values = '--' # need to check this     
    
        timestamp_convertfunc = lambda x: datetime.datetime.strptime(x, dt_format)
        
        data = np.genfromtxt(filename, names=names, converters={'TIMESTAMP':timestamp_convertfunc},
                             dtype=None, deletechars='', delimiter=delimiter,
                             skip_header=0, skip_footer=0,
                             usemask=True, missing_values=missing_values) 

        # add 12 hours to the datetimes
        for i, dt in enumerate(data['TIMESTAMP']):
            data['TIMESTAMP'][i] = dt + datetime.timedelta(hours=12)

        time_coord = make_time_coord.make_time_coord(data)
        time_coord.var_name = 'time'

        cube = iris.cube.Cube(data[obs_var_name_fluxnet2015], 
                              var_name=var_name)
        cube.add_dim_coord(time_coord, (0,))

        # Do a bit of tidying: adds bounds 
        # and converts units for some of the variables, and makes sure some
        # variables do not go below zero. Also sets the calendar and adds a long_name for the
        #time coord.

        cube.data # make sure the data is read in right away, otherwise get in to trouble with biggus
        # (it gets confused about which points should be masked)

        # add some time bounds
        cube.coord('time').guess_bounds(bound_position=0.5)  

        # set the calendar to be "gregorian"
        new_unit = make_time_coord.wrap_unit(cube.coord('time').units.origin, 'gregorian')
        cube.coord('time').units = new_unit

        cube.coord('time').long_name = "Time of data" # to be consistent with the JULES output

        # add some extra aux_coords
        iris.coord_categorisation.add_year(cube, 'time')
        iris.coord_categorisation.add_month(cube, 'time')
        iris.coord_categorisation.add_day_of_year(cube, 'time')

        cube = iris.util.squeeze(cube)

    if var_name in ["NEE", "GPP", "Reco"]:
        if not local_time: # see note above about units in UTC files
            cube.data = convert_micromolCO2_per_m2_per_s_to_gC_per_m2_per_day(cube.data) 
        cube.units = "g/m2/day" 

    if var_name in ["SH", "LE"]:
        cube.units = make_time_coord.wrap_unit('W m-2')

    if var_name in ["GPP", "Reco"]:
        cube.data = np.ma.maximum(cube.data, 0.0)

    # for NEE, need to subtract the mean
    if var_name == 'NEE':
        cube = cube - cube.collapsed('time', iris.analysis.MEAN)

    return cube


def calc_rmse(cube1, cube2):
    '''
    Root mean square error
   
    Args:
    * cube1, cube2: iris cubes. Should each only have one dim coord, called 'time'. These
      time coords should be the same.
  
    Returns: rmse (float).
    '''    

    for cube in [cube1, cube2]:
        if [coord.name() for coord in cube.coords(dim_coords=True)] != ['time']:
            raise UserWarning('expected only one dim coord, and that should be called time')

    # iris checks the coords are all the same before it does the subtraction
    sq_diff = (cube1 - cube2) ** 2.0
    mean_sq_diff = sq_diff.collapsed('time', iris.analysis.MEAN)
    rmse = mean_sq_diff ** 0.5 

    if rmse.ndim != 0:
        raise UserWarning('rmse cube has not collapsed as expected')

    # float applied to a masked point gives nan
    return float(rmse.data)


def get_site_list_from_info_inc(filename, only_lba=False, only_fluxnet=False):
    '''
    loads list of sites from info.inc
    '''
    with open(filename) as f:
        lines = f.readlines()

    # remove comment lines
    lines = [x for x in lines if not x.startswith('#')]

    mystr = "".join(lines)
    mystr = mystr.partition("%}")[0]
    mystr = mystr.replace('{%- set site_info = ', '')
    d = ast.literal_eval(mystr)
    list_sites = list(d.keys())
    list_sites.sort()

    if only_lba:
        list_sites = [s for s in list_sites if 'lba' in s.lower()]

    if only_fluxnet:
        list_sites = [s for s in list_sites if 'lba' not in s.lower()]

    return list_sites


def get_site_list(filename):
    '''
    Read the file containing the fluxnet site IDs and return as an array.
    Will ignore empty lines and lines starting with a hash.
    Any hyphens will be converted to underscores.
    '''
    
    with open(filename, 'r') as f:
        site_list = f.readlines()
      
    # get rid of newline character and also empty lines
    # if there are any hyphens, convert them to underscores
    site_list  = [site.strip().replace('-', '_') for site in site_list if site != '\n' and site[0] != '#']    
    
    return site_list


def wrap_aggregated_by_with_mdtol(cube, coord_names, aggregator, mdtol=None):
    '''
    Hack to get around https://github.com/SciTools/iris/issues/3190
    
    args:
     * cube: cube to aggregated
     * coord_names: should be either ['year', 'day_of_year'], ['year', 'month']
       or ['year', 'month_number']
     * aggregator: Iris aggregator object e.g. iris.analysis.MEAN
     * mdtol: tolerance of missing data to pass on to the Iris aggregated_by function  
    '''
    
    allowed_coord_names = (['year', 'day_of_year'], ['year', 'month'],
        ['year', 'month_number'], ['month'])
	
    coord_names_are_allowed = False
    for cn in allowed_coord_names:
        if coord_names == cn:
	    coord_names_are_allowed = True	
	
    if not coord_names_are_allowed:
        raise UserWarning('coord_names is not one of the allowed combinations: ' 
	    + str(allowed_coord_names))

    print(cube,coord_names)

    try:
        agg_cube = cube.aggregated_by(coord_names, aggregator, mdtol=mdtol)

    except (AttributeError, ValueError):

        def make_str_arr(in_cube, coord_names):
            '''
            Makes a list of the form 
	    ['DOY-YYYY', 'DOY-YYYY',....] e.g. ['001-2000', '002-2000', ...]
	    or	    
	    ['MM-YYYY', 'MM-YYYY',....] e.g. ['01-2000', '02-2000', ...]
            or
	    ['MM', 'MM',....] e.g. ['01', '02', ...]
            '''
	    n = len(in_cube.coord('year').points)
	  
	    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
		          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

	    if len(coord_names) == 1:
	        if coord_names[0] == 'month':
	            str_c0 = [format(month_list.index(in_cube.coord(coord_names[0]).points[i]) + 1, '02') 
		              for i in range(n)]
                    mylist = [str_c0[i] for i in range(n)]
            elif len(coord_names) == 2:
	        if coord_names[0] == 'year':
	            str_c0 = [format(in_cube.coord(coord_names[0]).points[i], '03') for i in range(n)]
                else:
	            raise Exception('have not implemented this')
	    
	        if coord_names[1] == 'day_of_year':	    
		    str_c1 = [format(in_cube.coord(coord_names[1]).points[i], '03') 
		              for i in range(n)]    
                elif coord_names[1] == 'month_number':
	            str_c1 = [format(in_cube.coord(coord_names[1]).points[i], '02')    
		              for i in range(n)] 
                elif coord_names[1] == 'month':
	            str_c1 = [format(month_list.index(in_cube.coord(coord_names[1]).points[i]) + 1, '02') 
		              for i in range(n)]
                else:
	            raise Exception('have not implemented this')		
						
                mylist = [str_c1[i] + '-' + str_c0[i] for i in range(n)]
            else:
                raise UserWarning('have not considered this case')
            return mylist

        # do the aggregation without mdtol
        agg_cube_no_mdtol_used = cube.aggregated_by(coord_names, aggregator)

        # create cube to store final result (after correct mdtol), but fill with nans for now
        agg_cube = agg_cube_no_mdtol_used.copy()
        agg_cube *= np.nan

        if agg_cube_no_mdtol_used.data.count() == 0:
            # all of data is missing
            pass 
        else:
            # get rid of masked days/months (i.e. days/months with no valid data)
            agg_cube_no_mdtol_used = agg_cube_no_mdtol_used[~agg_cube_no_mdtol_used.data.mask]

            # get list of days/months which have at least one data point
            str_arr_after_agg = list(set(make_str_arr(agg_cube_no_mdtol_used, coord_names)))

            # get list of strings for each time step in original cube
            str_arr_before_agg = make_str_arr(cube, coord_names)

            # has_data indicates, for each timestep in the original cube, whether it's part of a day/month with at least some data
            has_data = []
            for i in range(len(str_arr_before_agg)):
                if str_arr_before_agg[i] in str_arr_after_agg:
                   has_data.append(True)
                else:
                    has_data.append(False)

            # take out timesteps from days/months with no data
            new_cube = cube[has_data]

            # try the aggregation with mdtol again
            agg_cube_not_yet_filled = new_cube.aggregated_by(coord_names, aggregator, mdtol=mdtol)

            # put the valid data into agg_cube
            j = 0

            for i in range(agg_cube.shape[0]):
                c0_i = agg_cube.coord(coord_names[0]).points[i]
                c1_i = agg_cube.coord(coord_names[1]).points[i]

                c0_j = agg_cube_not_yet_filled.coord(coord_names[0]).points[j]
                c1_j = agg_cube_not_yet_filled.coord(coord_names[1]).points[j]

                if (c0_i == c0_j) and (c1_i == c1_j):
                    agg_cube.data[i] = agg_cube_not_yet_filled.data[j]
                    j += 1

    return agg_cube


def get_model_cube(var, model_output_folder,
                   run_id=None, pft=None, cpft=None, soil=None,
                   model_year_constraint=None):
    '''
    Function to get the model cube for use in plotting scripts
    '''

    if var in VAR_NAMES_DICT.keys():
        model_var_name = VAR_NAMES_DICT[var]['model_var_name']
    else:
        model_var_name = var

    model_cube = get_processed_daily_model_data(model_var_name, model_output_folder,
        run_id=run_id, use_daily_output_if_possible=True)
                    
    if model_year_constraint is not None:
        model_cube = model_cube.extract(model_year_constraint)

    coord_names = [coord.name() for coord in model_cube.coords(dim_coords=True)]
           
    if 'pft' in coord_names:
        if pft is None:
            raise UserWarning('need to give pft number')
        else:
            model_cube = model_cube.extract(iris.Constraint(pft=pft))

    if 'type' in coord_names: 
        if pft is None: # veg tiles come before non-veg tiles so can use pft here
            raise UserWarning('need to give pft number')
        else:
            model_cube = model_cube.extract(iris.Constraint(type=pft))

    if 'tile' in coord_names: 
        if pft is None: # veg tiles come before non-veg tiles so can use pft here
            raise UserWarning('need to give pft number')
        else:
            model_cube = model_cube.extract(iris.Constraint(tile=pft))
           
    if 'cpft' in coord_names:
        if cpft is None:
            raise UserWarning('need to give cpft number')
        else:
            model_cube = model_cube.extract(iris.Constraint(cpft=cpft))
           
    if 'soil' in coord_names:
        if soil is None:
            raise UserWarning('need to give soil layer number')
        else:
            model_cube = model_cube.extract(iris.Constraint(soil=soil))

    # check all the missing data is masked 
    model_cube.data = np.ma.masked_array(model_cube.data, np.isnan(model_cube.data))
   
    return model_cube


def make_plots(vars_to_plot=None,
               run_id_list=None, site_list=None, title_list=None,
               nx=None, ny=None, pft=None, cpft=None, soil=None, obs=None,
               l_include_rmse_and_corr=False, 
               plot_beginning=None, plot_ending='pdf',
               model_output_folder=None, plot_output_folder=None,
               extra_wide=False, ylim=None, model_year_constraint=None, 
               parallel=False):
    '''
    Create the benchmark plots for a list of variables. 

    Args:
       * vars_to_plot: list of variables to plot

    KWargs:
       * run_id_list: list of run_ids. 
       * site_list: list of sites (will be used to pick out the obs for this site if
         obs are available). Should be one per item in run_id_list.
       * title_list: list of the plot titles. Should be one per item in run_id_list.
         If None, use site_list.
       * nx: number of columns of plots
       * ny: number of rows of plots
       * pft: which pft to plot if the variable has a pft dimension
       * cpft: which cpft to plot if the variable has a cpft dimension
       * soil: which soil layer to plot if the variable has a soil layer dimension
       * obs: which obs to add to the plot. 'peg', 'lba', 'all' or None
       * l_include_rmse_and_corr: whether to include the rmse and corr on the plots
       * plot_beginning: string to start the plot names with
       * plot_ending: e.g. 'pdf', 'png'
       * model_output_folder: folder containing the model output
       * plot_output_folder: folder to put the finished plots in
       * extra_wide: False means plot ratio is roughly 4:3, True means wider plots
       * parallel: True: plot each variable as a separate process, in parallel.
         False: go through the variables one at a time.
       * model_year_constraint: iris constraint on year
    '''

    parallelise.wrap_map(make_plots_one_variable, 
               vars_to_plot,
               run_id_list=run_id_list, site_list=site_list, title_list=title_list,
               nx=nx, ny=ny, pft=pft, cpft=cpft, soil=soil, obs=obs,
               l_include_rmse_and_corr=l_include_rmse_and_corr, 
               plot_beginning=plot_beginning, plot_ending=plot_ending,
               model_output_folder=model_output_folder, plot_output_folder=plot_output_folder,
               extra_wide=extra_wide, 
               model_year_constraint=model_year_constraint,
               parallel=parallel)

    return


def make_plots_one_variable(
               var, run_id_list=None,
               site_list=None, title_list=None, 
               nx=None, ny=None, pft=None, cpft=None, soil=None, obs=None,
               l_include_rmse_and_corr=False, 
               plot_beginning=None, plot_ending='pdf',
               model_output_folder=None, plot_output_folder=None,
               extra_wide=False, ylim=None, model_year_constraint=None):
    '''
    Create the benchmark plots for one variable. 

    Args:
       * var: variable to plot (string)

    KWargs:
       * run_id_list: list of run_ids. 
       * site_list: list of sites (will be used to pick out the obs for this site if
         obs are available). Should be one per item in run_id_list.
       * title_list: list of the plot titles. Should be one per item in run_id_list.
         If None, use site_list.
       * nx: number of columns of plots
       * ny: number of rows of plots
       * pft: which pft to plot if the variable has a pft dimension
       * cpft: which cpft to plot if the variable has a cpft dimension
       * soil: which soil layer to plot if the variable has a soil layer dimension
       * obs: which obs to add to the plot. 'peg', 'lba', 'all' or None
       * l_include_rmse_and_corr: whether to include the rmse and corr on the plots
       * plot_beginning: string to start the plot names with
       * plot_ending: e.g. 'pdf', 'png'
       * model_output_folder: folder containing the model output
       * plot_output_folder: folder to put the finished plots in
       * extra_wide: False means plot ratio is roughly 4:3, True means wider plots
       * ylim: gives the option to specify the ylim 
       * model_year_constraint: iris constraint on year
    '''

    if len(site_list) == len(run_id_list):
        original_site_list = site_list
        original_run_id_list = run_id_list
    else:
        raise UserWarning('site_list and run_id_list should be the same length')

    if title_list is None:
        original_title_list = site_list   
    else:
        if len(title_list) == len(run_id_list):
            original_title_list = title_list
        else:
            raise UserWarning('if given, title_list should be the same length as run_id_list')

    # get rid of sites from list if there's no daily model output for them for this simulation
    site_list = []
    title_list = []
    run_id_list = []
    for i,run_id in enumerate(original_run_id_list):
        filestr = os.path.join(model_output_folder, run_id + ".D*.nc")
        filenames = glob.glob(filestr)
        if len(filenames) > 0:
            run_id_list.append(run_id)
            site_list.append(original_site_list[i])
            title_list.append(original_title_list[i])
        else:
            print 'no files found for ' + filestr

    if True: # used to be a loop over vars_to_plot. Keeping the indent for now while testing.
        i_page = 1
        j_page = 0
        fig_filename = None

        if extra_wide:
 	    fig = plt.figure(figsize=[6.0 * nx * 3.0, 4.0 * ny / 1.5])
 	else:
 	    fig = plt.figure(figsize=[6.0 * nx, 4.0 * ny])

        data_for_taylor_diagram = []

        for i, run_id in enumerate(run_id_list):            
            j_page += 1
            print var, run_id

            model_cube = get_model_cube(var, model_output_folder,
                    run_id=run_id, pft=pft, cpft=cpft, soil=soil, 
                    model_year_constraint=model_year_constraint)

            site = site_list[i]
            obs_cube =  get_daily_obs_data(site, var, obs=obs)
        
            if obs_cube is not None:
                obs_cube.data = np.ma.masked_array(obs_cube.data, np.isnan(obs_cube.data))

                # don't use iris.utils.unify_time_units because it won't do anything if they have a different
                # calendar
                unify_time_units([model_cube, obs_cube]) 
            
                if not time_units_are_equal([model_cube, obs_cube]):
                    raise UserWarning('time units are not all the same.')
   
                res = cut_to_overlapping_time_period([model_cube, obs_cube]) 
            
                if len(res) == 0: # i.e. there's no overlapping time period           
                    j_page -= 1
                    continue
            
                [model_cube, obs_cube] = res
   
                if not check_all_time_coords_are_the_same([model_cube, obs_cube]):
                    print model_cube.coord('time')[0]
                    print obs_cube.coord('time')[0]
                    raise UserWarning('time coords are not all the same.')

                if l_include_rmse_and_corr:
                    #rmse is based on the daily time series
                    #print 'obs', obs_cube
                    #print obs_cube.coord('year')
                    #print 'model', model_cube
                    #print model_cube.coord('year')
                    if 'lba' in site.lower():
                        # not calculating for the moment, until we get confirmation of time zone issue
                        rmse = np.ma.masked
                    else:    
                        rmse = calc_rmse(model_cube, obs_cube)

            # get rid of partial months before getting monthly means.
            # I'm not sure if we need this stage anymore, now that mdtol is being specified
            model_cube = chop_off_partial_months(model_cube)
            if obs_cube is not None:
                obs_cube = chop_off_partial_months(obs_cube)               

                res = cut_to_overlapping_time_period([model_cube, obs_cube]) 
            
                if len(res) == 0: # i.e. there's no overlapping time period          
                    j_page -= 1
                    continue
                        
                [model_cube, obs_cube] = res
            
                # mask days in one cube when the same day in the other cube is masked
                model_cube.data = np.ma.masked_where(np.ma.getmask(obs_cube.data), model_cube.data)
                obs_cube.data = np.ma.masked_where(np.ma.getmask(model_cube.data), obs_cube.data)

            # Now calculate the monthly means and the monthly climatology
            # Mask any months where the missing data is more than 50% 
            mdtol = 0.5

            model_cube_mean = wrap_aggregated_by_with_mdtol(model_cube, ["year", "month"], iris.analysis.MEAN, mdtol=mdtol)
            model_cube_clim = wrap_aggregated_by_with_mdtol(model_cube, ["month"], iris.analysis.MEAN, mdtol=mdtol)

            if obs_cube is not None:
                obs_cube_mean = wrap_aggregated_by_with_mdtol(obs_cube, ["year", "month"], iris.analysis.MEAN, mdtol=mdtol)
                obs_cube_clim = wrap_aggregated_by_with_mdtol(obs_cube, ["month"], iris.analysis.MEAN, mdtol=mdtol)

                # for the obs, we also need the standard deviation, based on the daily compared to the monthly
                # time series
                obs_cube_stddev = wrap_aggregated_by_with_mdtol(obs_cube, ["year", "month"], iris.analysis.STD_DEV, mdtol=mdtol)
          
                # corr is based on the monthly time series
                corr, pvalue = scipy.stats.mstats.pearsonr(model_cube_mean.data, obs_cube_mean.data)
               
                month_obs_stddev = float(obs_cube_mean.collapsed("time", iris.analysis.STD_DEV).data)
                month_model_stddev = float(model_cube_mean.collapsed("time", iris.analysis.STD_DEV).data)
                
                obs_diff = np.max(obs_cube_mean.data) - np.min(obs_cube_mean.data)

            model_diff = np.max(model_cube_mean.data) - np.min(model_cube_mean.data)

            if obs_cube is not None:
                if ( not (obs_diff > 0.0) ) and ( not (model_diff > 0.0) ):          
                    j_page -= 1
                    continue #in this case the plotting script would get no information about the size of the y axis
            
            ax = fig.add_subplot(ny, nx, j_page) #args: nrows, ncols, plot_number
            
            model_label = 'model'
            title = title_list[i]

            if obs_cube is not None:
                #print obs_cube_mean
                #print obs_cube_mean.coord('time')
                #print obs_cube_mean.coord('time').units
                qplt.plot(obs_cube_mean, 'b-', label='obs (dashes: $\pm 1\sigma$)', linewidth=3 ) 
                qplt.plot(obs_cube_mean + obs_cube_stddev, 'b--', linewidth=2)
                qplt.plot(obs_cube_mean - obs_cube_stddev, 'b--', linewidth=2)

                if l_include_rmse_and_corr:
                    if rmse is not np.nan:
                        title += ', rmse=' + "{:.2f}".format(rmse)
                    if corr is not np.ma.masked: 
                        title += ', corr=' + "{:.2f}".format(corr) 
                        
                data_for_taylor_diagram.append([month_obs_stddev, month_model_stddev, corr, site])
            
            qplt.plot(model_cube_mean, 'k-', label=model_label, linewidth=3) 
                      
            # probably want to take these out later: but good at the moment for testing  
            if obs_cube is not None:        
                qplt.plot(obs_cube, 'b-', linewidth=1, alpha=0.2)
            qplt.plot(model_cube, 'k-', linewidth=1, alpha=0.2)

            # Still to do: 
            #  * need to plot seasonal cycle not time series 
            #  * standard deviation of daily obs on a monthly basis needs to go on plot as error bars
            #  * lots of tidying of code and plots
            
            if ylim is None:
                if var in ['24hNPP_per_24hGPP']: # impose limits on y axis
                    ax.set_ylim([0.0, 1.0]) 
            else:
                ax.set_ylim(ylim) 

            plt.title(title)
            plt.ylabel(var + ' in ' + str(model_cube.units)) 
            plt.xlabel('')

            if i_page == 1 and j_page == 1: 
                plt.legend(fontsize=10)
                
            plt.xticks(rotation=30)

            fig_filename = os.path.join(plot_output_folder, 
                                        plot_beginning + '_' + var + '_page' + str(i_page) + '.' + plot_ending)
            plt.tight_layout()
            
            if j_page == nx * ny:
                plt.savefig(fig_filename)
                plt.clf()
                i_page += 1
                j_page = 0
                
        if j_page != 0 and fig_filename is not None: # i.e. if last plot has not just been written to file 
                                                     # and there's at least one plot on page
            print 'about to output',fig_filename
            plt.savefig(fig_filename)
            plt.clf()
        
        if len(data_for_taylor_diagram) > 0:
            filename_taylor_diagram_data = os.path.join(plot_output_folder, 
                plot_beginning + '_' + var + '_summary.dat')
                
            with open(filename_taylor_diagram_data, 'w') as f:
                f.write('# obs_stddev, model_stddev, corr, label\n')
                for i in range(len(data_for_taylor_diagram)):
                    line_data = ','.join([str(j) for j in data_for_taylor_diagram[i]]) 
                    line_data += '\n'
                    f.write(line_data)
        
    return
