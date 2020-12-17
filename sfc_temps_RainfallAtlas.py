from sympy import symbols
import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats

# Code to calculate average 2m temps 
years = [str(x) for x in range(1990,2015)]

for year in years:
    annual_temps = np.zeros(shape=[365,254,327])
    daily_temps_wet = np.zeros(shape=[181,254,327])
    daily_temps_dry = np.zeros(shape=[184,254,327])

    annual_days = 0
    wet_days = 0
    dry_days = 0

    for n in range(0,365):
        print(year,n,annual_days,dry_days,wet_days)
        temps = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(year)+'_'+str(n)+'_low_res_T2avg.npy')

        annual_temps[n] = temps

        if n >= 0 and n <= 119:
            daily = daily_temps_wet
            days = wet_days
            wet_days = wet_days + 1
        elif n >= 120 and n <= 303:
            daily = daily_temps_dry
            days = dry_days
            dry_days = dry_days + 1
        elif n >= 304 and n <= 364:
            daily = daily_temps_wet
            days = wet_days
            wet_days = wet_days + 1

        daily[days] = temps
        annual_days = annual_days + 1

    annual_temps = np.mean(annual_temps, axis = 0)
    daily_temps_wet = np.mean(daily_temps_wet, axis = 0)
    daily_temps_dry = np.mean(daily_temps_dry, axis = 0)

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_2m_temps_annual.npy', annual_temps)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_2m_temps_wet.npy', daily_temps_wet)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_2m_temps_dry.npy', daily_temps_dry)

annual_T = np.zeros(shape=[25,254,327])
wet_T = np.zeros(shape=[25,254,327])
dry_T = np.zeros(shape=[25,254,327])

n = 0
for year in years:
    print(year,n)
    annual = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_2m_temps_annual.npy')
    wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_2m_temps_wet.npy')
    dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_2m_temps_dry.npy')

    annual_T[n] = annual
    wet_T[n] = wet
    dry_T[n] = dry

    n = n + 1

annual_T = np.mean(annual_T, axis = 0)
wet_T = np.mean(wet_T, axis = 0)
dry_T = np.mean(dry_T, axis = 0)

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/obs_2mtemps_ann.npy', annual_T)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/obs_2mtemps_wet.npy', wet_T)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/obs_2mtemps_dry.npy', dry_T)
