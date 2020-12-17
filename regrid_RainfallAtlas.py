# Script to regrid high resolution Rainfall Atlas data to coarser resolution

from sympy import symbols
import xarray
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats
from mpl_toolkits.basemap import interp
from netCDF4 import Dataset
from pandas import DataFrame as df
from mpl_toolkits.basemap import Basemap, shiftgrid, cm

path1 = '/network/rit/lab/elisontimmlab_rit/DATA/HI_DAILY_PRECIP/' # path to daily rainfall data
path2 = '/network/rit/lab/elisontimmlab_rit/DATA/HI_DAILY/tmax_tmin_precip_gridded/' # path to daily temp data

years = [str(x) for x in range(1990,2015)]

low_res = np.zeros(shape = [254,327])

for year in years:
    for n in range(0,365):
        print(year, n)

        ncfile1 = 'Hawaii_Precipitation_UH_7.5sec_dTS'+year+'.nc' # daily rainfall
        ncfile2 = 'UH_1990.nc4'

        with Dataset(path1+ncfile1, mode='r') as fh1:
            with Dataset(path2+ncfile2, mode='r') as fh2:
                lons = fh1.variables['longitude'][:]
                lats = fh1.variables['latitude'][:]
                hgt = fh2.variables['elevation'][:]
                #rain = fh1.variables['precipitation'][n,:,:]
                tmax_C = fh2.variables['tmax'][n,:,:] # in degrees C
                tmin_C = fh2.variables['tmin'][n,:,:]
                tmax_K = tmax_C + 273.15 # Convert to degrees K
                tmin_K = tmin_C + 273.15
                tavg_K = (tmax_K + tmin_K) / 2

        lons_sub, lats_sub = np.meshgrid(lons[::7], lats[::6])

        #coarse_rain = interp(rain, lons, lats, lons_sub, lats_sub, order=1)
        coarse_temp = interp(hgt, lons, lats, lons_sub, lats_sub, order=1)

        low_res[:] = coarse_temp[:]
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/longman_elevation_low_res.npy', low_res)
        #np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(year)+'_'+str(n)+'_low_res.npy', low_res)
        #np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(year)+'_'+str(n)+'_low_res_T2MAX.npy', low_res)
        #np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(year)+'_'+str(n)+'_low_res_T2avg.npy', low_res)

#######################################################################################################################################
#######################################################################################################################################