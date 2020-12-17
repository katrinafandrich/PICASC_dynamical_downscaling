from sympy import symbols
import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats

# Code to calculate extreme temps 
years = [str(x) for x in range(1990,2015)]
'''
for year in years:
    daily_temps_annual = np.zeros(shape=[365,254,327])
    daily_temps_wet = np.zeros(shape=[181,254,327])
    daily_temps_dry = np.zeros(shape=[184,254,327])

    annual_days = 0
    wet_days = 0
    dry_days = 0

    for n in range(0,365):
        print(year,n,annual_days,dry_days,wet_days)
        temps = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(year)+'_'+str(n)+'_low_res_T2MAX.npy')

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

        for k in range(0,254):
            for j in range(0,327):
                daily_temps_annual[annual_days,k,j] = temps[k,j]
                daily[days,k,j] = temps[k,j]
        
        annual_days = annual_days + 1

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_temps_annual.npy', daily_temps_annual)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_temps_wet.npy', daily_temps_wet)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_temps_dry.npy', daily_temps_dry)
'''
'''
seasons = ['annual','wet','dry']
years = [str(x) for x in range(1990,2015)]

for season in seasons:
    for year in years:
        print(season, year)
        daily_temps = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_temps_'+season+'.npy')

        temps_90th = np.zeros(shape=[254,327])
        temps_99th = np.zeros(shape=[254,327])

        for k in range(0,254):
            for l in range(0,327):
                daily = daily_temps[:,k,l]
                dsort = np.sort(daily[:])
                percent90 = dsort[-(round(len(dsort)*0.1)):]
                percent99 = dsort[-(round(len(dsort)*0.01)):]

                temps_90th[k,l] = np.mean(percent90, axis = 0)
                temps_99th[k,l] = np.mean(percent99, axis = 0)

        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_90th_'+season+'.npy', temps_90th)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_99th_'+season+'.npy', temps_99th)
'''

seasons = ['annual','wet','dry']
years = [str(x) for x in range(1990,2015)]

percent90 = np.zeros(shape = [25,254,327])
percent99 = np.zeros(shape = [25,254,327])

for season in seasons:
    n = 0
    for year in years:
        temps90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_90th_'+season+'.npy')
        temps99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_99th_'+season+'.npy')
        percent90[n] = temps90
        percent99[n] = temps99
        n = n + 1

    avg_percent90 = np.mean(percent90, axis = 0)
    avg_percent99 = np.mean(percent99, axis = 0)

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_90th_temps_'+season+'.npy', avg_percent90)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_99th_temps_'+season+'.npy', avg_percent99)
