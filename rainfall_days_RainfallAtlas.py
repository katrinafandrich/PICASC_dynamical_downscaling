# Script to calculate total rainfall days (annual, wet season, dry season) for Rainfall Atlas data

import xarray
import numpy as np

'''
# Part 1: Loop thru each year and give dry days "1" and rain days "0"
years = [str(x) for x in range(1990,2015)]

for year in years:
    rain_days = np.zeros(shape=[365,254,327])

    for n in range(0,365):
        print(year,n)
        rain = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_'+str(n)+'_low_res.npy')
        
        for k in range(0,254):
            for j in range(0,327):
                if rain[k,j] > 0.1:
                    rain_days[n,k,j] = 1
                else:
                    rain_days[n,k,j] = 0

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_rain_days_RainAtlas.npy', rain_days)

for year in years:
    total_rain_days = np.zeros(shape=[254,327])

    for n in range(0,365):
        print(year,n)
        rain_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_rain_days_RainAtlas.npy')
        
        for k in range(0,254):
            for j in range(0,327):
                total_rain_days[k,j] = np.sum(rain_days[:,k,j], axis = 0)

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_total_rain_days_RainAtlas.npy', total_rain_days)

average = np.zeros((25,254,327))

n = 0
for year in years:
    total_rain_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_total_rain_days_RainAtlas.npy')
    average[n,:,:] = total_rain_days[:]
    n = n + 1

annual_average = np.mean(average, axis = 0)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_RainAtlas.npy', annual_average)
'''

# Part 1: Loop thru SEASONS of each year and give dry days "0" and rain days "1"
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
                    daily[days,k,j] = 1
                else:
                    daily[days,k,j] = 0

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_rain_days_wet_atlas.npy', daily_rain_wet)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_rain_days_dry_atlas.npy', daily_rain_dry)

# Part 2
for year in years:
    total_rain_days_wet = np.zeros(shape=[254,327])
    total_rain_days_dry = np.zeros(shape=[254,327])

    for n in range(0,365):
        print(year,n)
        rain_days_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_rain_days_wet_atlas.npy')
        rain_days_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_rain_days_dry_atlas.npy')
        for k in range(0,254):
            for j in range(0,327):
                total_rain_days_wet[k,j] = np.sum(rain_days_wet[:,k,j], axis = 0)
                total_rain_days_dry[k,j] = np.sum(rain_days_dry[:,k,j], axis = 0)

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_total_rain_days_wet_atlas.npy', total_rain_days_wet)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_total_rain_days_dry_atlas.npy', total_rain_days_dry)

# Part 3
average_wet = np.zeros((25,254,327))
average_dry = np.zeros((25,254,327))

n = 0
for year in years:
    total_rain_days_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_total_rain_days_wet_atlas.npy')
    total_rain_days_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_total_rain_days_dry_atlas.npy')
    average_wet[n,:,:] = total_rain_days_wet[:]
    average_dry[n,:,:] = total_rain_days_dry[:]
    n = n + 1

wet_average = np.mean(average_wet, axis = 0)
dry_average = np.mean(average_dry, axis = 0)

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_atlas.npy', wet_average)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_atlas.npy', dry_average)