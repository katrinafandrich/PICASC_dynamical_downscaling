
from sympy import symbols
import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats


# Code to calculate extreme precip 
years = [str(x) for x in range(1990,2015)]

for year in years:
    daily_rain_wet = np.zeros(shape=[181,254,327])
    daily_rain_dry = np.zeros(shape=[184,254,327])

    wet_days = 0
    dry_days = 0

    for n in range(0,365):
        print(year,n,dry_days,wet_days)
        rain = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_'+str(n)+'_low_res.npy')
        
        if n >= 0 and n <= 119:
            daily = daily_rain_wet
            days = wet_days
            wet_days = wet_days + 1
        elif n >= 120 and n <= 303:
            daily = daily_rain_dry
            days = dry_days
            dry_days = dry_days + 1
        elif n >= 304 and n <= 364:
            daily = daily_rain_wet
            days = wet_days
            wet_days = wet_days + 1

        for k in range(0,254):
            for j in range(0,327):
                if rain[k,j] > 0.1:
                    daily[days,k,j] = rain[k,j]
                else:
                    daily[days,k,j] = np.nan

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_wet.npy', daily_rain_wet)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_dry.npy', daily_rain_dry)
    print()


seasons = ['wet','dry']
years = [str(x) for x in range(1990,2015)]

for season in seasons:
    for year in years:
        print(season, year)
        rain_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_'+season+'.npy')

        rain_90th = np.zeros(shape=[254,327])
        rain_99th = np.zeros(shape=[254,327])

        for k in range(0,254):
            for l in range(0,327):
                daily = rain_days[:,k,l]
                no_nans = daily[np.logical_not(np.isnan(daily))]
                dsort = np.sort(no_nans[:])
                percent90 = dsort[-(round(len(dsort)*0.1)):]
                percent99 = dsort[-(round(len(dsort)*0.01)):]

                if len(percent90) > 0:
                    rain_90th[k,l] = np.mean(percent90, axis = 0)
                else:
                    rain_90th[k,l] = 0

                if len(percent99) > 0:
                    rain_99th[k,l] = np.mean(percent99, axis = 0)
                else:
                    rain_99th[k,l] = 0

        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_90th_'+season+'.npy', rain_90th)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_99th_'+season+'.npy', rain_99th)


seasons = ['wet','dry']
years = [str(x) for x in range(1990,2015)]

percent90 = np.zeros(shape = [25,254,327])
percent99 = np.zeros(shape = [25,254,327])

for season in seasons:
    n = 0
    for year in years:
        rain90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_90th_'+season+'.npy')
        rain99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_99th_'+season+'.npy')
        percent90[n] = rain90
        percent99[n] = rain99
        n = n + 1

    avg_percent90 = np.mean(percent90, axis = 0)
    avg_percent99 = np.mean(percent99, axis = 0)

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_90th_'+season+'.npy', avg_percent90)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_99th_'+season+'.npy', avg_percent99)
