"""Obtain rainfall accumulation for a given time interval.

Author: KMF 4/10/20

Description:

Input data are read from two NetCDF files that include the
start and end point of the time interval. This scripts uses
the hourly WRF output files (one netcdf file per day). A file
with all season from the 10-year simulations are saved in a netcdf file.

Usage:
    within ipython use %run get_wet_season_precip_from_hourly.py ENS
    where ENS is an argument for the ensemble member (integer number 1-20)
    Example:
    %run get_wet_season_precip_from_hourly.py 1 will process the files in subfolder

    /network/rit/lab/elisontimmlab_rit/DATA/WRF/hires/WRF1/hourly/


This script is was developed for wet season Nov-Apr
precipitation accumulation calculations. Adjustments need to be made
for other season definitions (e.g. dry season May-Oct)

It can be used as a starting point for monthly calculations, however,
one should add one more loop inside the year loop to process each month
in a year, and take care of the Dec to Jan transition between years.

If you want to run the script or develop a script for other seasons, make sure
to change the variable DPATH_OUT to a local directory with write permissions.

Variables needed from the NetCDF files:

    `I_RAINNC`  Buck counts (one bucket is 100mm precipitation)
    `RAINNC`    the accumulation to be added to the buckets in range (0, 100)

The shape of the accumulation variables is 3-dim (time,lat,lon)
with 24 time steps.
"""

import xarray
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import sys

# which ensemble member (1-20)
try:
    member=int(sys.argv[1])
except:
    print ("call the script with arguments for ensemble member number (1-20)")
    sys.exit(1)

ens='WRF'+str(member)
print("Process ensemble simulation "+ens)

# use variable ens to choose an ensemble member
# DPATH is the path to the directory for hires input data
# DPATH_OUT the directory for writing the output netcdf file

# Neutral phase data is located in a different location
#DPATH='/network/rit/lab/elisontimmlab_rit/DATA/WRF/hires/'+ens+'/hourly/'
DPATH='/data/elisontimm_scr/DATA/WRF_PDO_neutral/'+ens+'/hourly/'

# Change outpath for neutral phase data
DPATH_OUT='/network/rit/lab/elisontimmlab_rit/kf835882/python/seasonal/neutral/'

# CHANGE HERE FOR WRF11-20  and adjust for dry season (e.g. append 2006 or 2035)
#if 10<member<=20:
#    years=[2026,2027,2028,2029,2030,2031,2032,2033,2034,2035]
#    filename_yrs='2026-2035'
#else:
#    years=[1996,1997,1998,1999,2000,2001,2002,2003,2004,2005] # wet season 2005/2006 incomplete)
#    filename_yrs='1996-2005'

# Neutral phase years
if 5<member<=10:
    years=[2026,2027,2028,2029,2030,2031,2032,2033,2034,2035]
    filename_yrs='2026-2035'
else:
    years=[1996,1997,1998,1999,2000,2001,2002,2003,2004,2005]
    filename_yrs='1996-2005'

# change the file name for other seasons or variables
# convention in years for wet season Nov1996-Apr1997 is 1997
# change name for neutral phase
#ncout_filename=DPATH_OUT+ens+'_RAINNC_ndjfma_'+filename_yrs+'.nc'
ncout_filename=DPATH_OUT+ens+'neutral_RAINNC_mjjaso_'+filename_yrs+'.nc'

# BUCKET size for precipitation accumulation totals
BUCKET=100 # mm

#######################################################################
# get lon-lat grid dimension before entering the loop
# and create 3-d numpy array for storing all time steps
#######################################################################
# variable x3d is a numpy array with dimensions in order time,lat,lon

# get the lat and lon dimension information from input netcdf file:
reference_file="/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/WRF1/" + \
    "wrfout_d02_monthly_mean_2005_12.nc"
# NOTE: check the names in the attributes for your variable:
# staggered grids are used (e.g. XLONG_U XLAT_U for wind variables)
#

# used at end of script for creating the netcdf output
ncref=xarray.open_dataset(reference_file)
NLAT=ncref['XLAT'].shape[1]
NLON=ncref['XLONG'].shape[2]

NTIME=len(years)
x3d=np.zeros((NTIME,NLAT,NLON))

itime=0
time=[]
for startyear in years:
    endyear=startyear # endyear = startyear + 1 for wet season
    startdate=dt.datetime(startyear,5,1,00,00,00)
    enddate=dt.datetime(endyear,11,1,00,00,00)
    time_delta=enddate-startdate
    stime1=dt.datetime.strftime(startdate,"%Y-%m-%d_%H_%M_%S") # naming convention changes for neutral phase
    file1="surface_d02_"+stime1
    stime2=dt.datetime.strftime(enddate,"%Y-%m-%d_%H_%M_%S")
    file2="surface_d02_"+stime2
    info="Total precipitation accumulation from " \
    +startdate.strftime('%Y-%m-%d_%H_%M_%S') \
    +" to "+enddate.strftime('%Y-%m-%d_%H_%M_%S')
    print (info)

    nc1=xarray.open_dataset(DPATH+str(startdate.year)+'/'+file1)
    nc2=xarray.open_dataset(DPATH+str(enddate.year)+'/'+file2)

    i_rainnc1=nc1['I_RAINNC'].values[0,:,:]
    rainnc1=nc1['RAINNC'].values[0,:,:]
    accum1=i_rainnc1*BUCKET+rainnc1
    i_rainnc2=nc2['I_RAINNC'].values[0,:,:]
    rainnc2=nc2['RAINNC'].values[0,:,:]
    accum2=i_rainnc2*BUCKET+rainnc2

    # Print some summary statistics to check the calculation
    print ("accumulation during the time interval:")
    print (20*"=")
    print(stime1)
    print(20*"-")
    print("buckets: ",np.min(i_rainnc1),np.max(i_rainnc1))
    print("rainnc:  ",np.min(rainnc1),np.max(rainnc1))
    print("accum:   ",np.min(accum1),np.max(accum1))
    print("")
    print(stime2)
    print(20*"-")
    print("buckets: ",np.min(i_rainnc2),np.max(i_rainnc2))
    print("rainnc:  ",np.min(rainnc2),np.max(rainnc2))
    print("accum:   ",np.min(accum2),np.max(accum2))
    print (20*"-")
    print (np.min(accum2-accum1),np.max(accum2-accum1))
    print(20*"=")

    # store result in output numpy array
    # add the time: Here I use the end date of the accumulation period
    # NOTE: Currently, the netcdf file misses information of the startdate
    # so this should be improved (e.g. by adding a second variable)
    # that includes all start dates. (TODO task)
    x3d[itime,:,:]=np.reshape(np.array(accum2-accum1),(NLAT,NLON))
    time.append(enddate)
    itime+=1
    # end of loop


############################################################################
# export output to netcdf file
#
# structure of this netcdf output is 3-dim: time,lat,lon
# single variable (currently), and no vertical level dimension
# Use one reference netdf output file from WRF model that has the
# longitude and latitude information included
############################################################################

ref_lat=ncref['XLAT']
ref_lon=ncref['XLONG']
dim_names = nc1['RAINNC'].dims
var_units= nc1['RAINNC'].attrs['units']

############################################################################
# create a xarray DataArray structure
# by adding the dimensions
# and attributes
############################################################################

# use data_out as a generic variable name below
# assign the numpy array that you want to export to netcdf file

# to data_out (variable x3d in this script)
data_out=x3d


ntime=np.size(time)
lat=[i+1 for i in np.arange(NLAT)] # simple index counter (starting at 1)
lon=[i+1 for i in np.arange(NLON)] # simple index counter (starting at 1)


ncout_var=xarray.DataArray(x3d,coords=(time,lat,lon),dims=('time',dim_names[1],dim_names[2]))
ncout_var.attrs['name']='RAINNC'
ncout_var.attrs['standard_name']='precipitation'
ncout_var.attrs['long_name']='monthly accumulated total grid scale precipitation'
ncout_var.attrs['description']='monthly accumulated total grid scale precipitation'#+startdate.strftime('%Y-%m-%d %H:%M:%S')
ncout_var.attrs['units']=var_units

# create support variables for latitude and longitude grid coordinates
ncout_lat=xarray.DataArray(ref_lat.values.squeeze(),coords=(lat,lon),dims=dim_names[1:3])
ncout_lat.attrs['name']='XLAT'
ncout_lat.attrs['units']=ref_lat.attrs['units']
ncout_lat.attrs['description']=ref_lat.attrs['description']
ncout_lat.attrs['stagger']=ref_lat.attrs['stagger']
ncout_lat.attrs['cell_method']=ref_lat.attrs['cell_methods']

ncout_lon=xarray.DataArray(ref_lon.values.squeeze(),coords=(lat,lon),dims=dim_names[1:3])
ncout_lat.attrs['name']='XLONG'
ncout_lon.attrs['units']=ref_lon.attrs['units']
ncout_lon.attrs['description']=ref_lon.attrs['description']
ncout_lon.attrs['stagger']=ref_lon.attrs['stagger']
ncout_lon.attrs['cell_method']=ref_lon.attrs['cell_methods']

# put together all output variable and create the netcdf file object
# dictionary holds key:value pairs: variable names (keys)
# and the xarray DataArray object (values)
dsout=xarray.Dataset({'RAINNC':ncout_var,'XLAT':ncout_lat,'XLONG':ncout_lon})
print(dsout)


dsout.to_netcdf(ncout_filename,format="NETCDF4")


print (40*"*")
print ("wrote output to file:\n"+ncout_filename)
