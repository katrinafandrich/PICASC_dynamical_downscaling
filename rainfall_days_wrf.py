# Script to calculate total rainfall days (annual, wet season, dry season) for Rainfall Atlas data

import xarray
import numpy as np

'''
# Part 1: Loop thru each year in each WRF ens and give dry days "0" and rain days "1"
WRF_list = ['WRF11','WRF12','WRF13','WRF14','WRF15','WRF16','WRF17','WRF18','WRF19','WRF20']

present = [str(x) for x in range(1996,2006)]
future = [str(x) for x in range(2026,2036)]


for ens in WRF_list:
    for year in future:
        rain_days = np.zeros((365,240,360))
        RAIN = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dailyrain_'+year+'_'+ens+'.npy')
        for n in range(0,365):
            for k in range(0,240):
                for j in range(0,360):
                    if RAIN[n,k,j] > 0.1:
                        rain_days[n,k,j] = 1
                    else:
                        rain_days[n,k,j] = 0
            print(ens, year, n)
        filename = 'rain_days_'+year+'_'+ens+'.npy'
        print(filename)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+filename, rain_days)
'''
'''
# Part 2: Calculate the total rainfall days per year
for ens in WRF_list:
    for year in future:
        total_rain_days = np.zeros(shape=[240,360])
        rain_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/rain_days_'+year+'_'+ens+'.npy')
        print(ens,year)
        for k in range(0,240):
            for j in range(0,360):
                total_rain_days[k,j] = np.sum(rain_days[:,k,j], axis = 0)
           
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_'+year+'_'+ens+'.npy', total_rain_days)
'''

'''
# Part 3: Calculate the ensemble average of annual rainfall days
for ens in WRF_list:
    avg = np.zeros((10,240,360))
    n = 0
    for year in future:
        print(ens, year, n)
        annual_rain = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_'+year+'_'+ens+'.npy')
        avg[n,:,:] = annual_rain[:]
        n = n + 1

    ens_avg = np.mean(avg, axis = 0)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_'+ens+'.npy', ens_avg)
'''
'''
# Part 4: Calculate the average annual rain days for ensemble groups
WRF_list = ['WRF16','WRF17','WRF18','WRF19','WRF20']

ens_all = np.zeros((5,240,360))

n = 0
for ens in WRF_list:
    print(ens, n)
    rain_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_'+ens+'.npy')
    ens_all[n,:,:] = rain_days[:]
    n = n + 1

ens_avg = np.mean(ens_all, axis = 0)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_WRF16-20.npy', ens_avg)
'''
'''
# Repeat but for wet and dry season
# Part 2: Calculate the total rainfall days per season
for ens in WRF_list:
    for year in future:
        total_rain_days_wet = np.zeros(shape=[240,360])
        total_rain_days_dry = np.zeros(shape=[240,360])
        rain_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/rain_days_'+year+'_'+ens+'.npy')
        print(ens,year)
        for k in range(0,240):
            for j in range(0,360):
                wet1 = np.sum(rain_days[0:120,k,j], axis = 0)
                wet2 = np.sum(rain_days[303:365,k,j], axis = 0)
                total_rain_days_wet[k,j] = wet1 + wet2
                total_rain_days_dry[k,j] = np.sum(rain_days[120:304,k,j], axis = 0)

        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_'+year+'_'+ens+'.npy', total_rain_days_wet)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_'+year+'_'+ens+'.npy', total_rain_days_dry)
'''
'''
# Part 3: Calculate the ensemble average of wet/dry season total rainfall days
for ens in WRF_list:
    avg_wet = np.zeros((10,240,360))
    avg_dry = np.zeros((10,240,360))
    n = 0
    for year in future:
        print(ens, year, n)
        wet_rain = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_'+year+'_'+ens+'.npy')
        dry_rain = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_'+year+'_'+ens+'.npy')
        avg_wet[n,:,:] = wet_rain[:]
        avg_dry[n,:,:] = dry_rain[:]
        n = n + 1

    ens_wet_avg = np.mean(avg_wet, axis = 0)
    ens_dry_avg = np.mean(avg_dry, axis = 0)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_'+ens+'.npy', ens_wet_avg)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_'+ens+'.npy', ens_dry_avg)
'''

# Part 4: Calculate the average annual rain days for ensemble groups
WRF_list = ['WRF16','WRF17','WRF18','WRF19','WRF20']

ens_wet = np.zeros((5,240,360))
ens_dry = np.zeros((5,240,360))

n = 0
for ens in WRF_list:
    print(ens, n)
    rain_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_'+ens+'.npy')
    rain_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_'+ens+'.npy')
    ens_wet[n,:,:] = rain_wet[:]
    ens_dry[n,:,:] = rain_dry[:]
    n = n + 1

ens_wet_avg = np.mean(ens_wet, axis = 0)
ens_dry_avg = np.mean(ens_dry, axis = 0)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_WRF16-20.npy', ens_wet_avg)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_WRF16-20.npy', ens_dry_avg)