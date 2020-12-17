# Script to calculate total days (annual, wet season, dry season) over 90 and 95 degrees F

import xarray
import numpy as np


# Part 1: Loop thru each year in each WRF ens and give cool days "0" and days >= 90 and >= 95 degrees "1"
WRF_list = ['WRF11','WRF12','WRF13','WRF14','WRF15','WRF16','WRF17','WRF18','WRF19','WRF20']

present = [str(x) for x in range(1996,2006)]
future = [str(x) for x in range(2026,2036)]
'''
for ens in WRF_list:
    for year in future:
        days_90deg = np.zeros((365,240,360))
        days_95deg = np.zeros((365,240,360))
        TEMPS = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dailytemps_'+year+'_'+ens+'.npy')
        for n in range(0,365):
            for k in range(0,240):
                for j in range(0,360):

                    if TEMPS[n,k,j] >= 90:
                        days_90deg[n,k,j] = 1
                    else:
                        days_90deg[n,k,j] = 0

                    if TEMPS[n,k,j] >= 95:
                        days_95deg[n,k,j] = 1
                    else:
                        days_95deg[n,k,j] = 0

            print(ens, year, n)
        filename_90 = '90deg_days_'+year+'_'+ens+'.npy'
        filename_95 = '95deg_days_'+year+'_'+ens+'.npy'
        print(filename_90, filename_95)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+filename_90, days_90deg)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+filename_95, days_95deg)
'''
'''
# Part 2: Calculate the total 90 and 95 degree days per year
for ens in WRF_list:
    for year in future:
        total_90deg_days = np.zeros(shape=[240,360])
        total_95deg_days = np.zeros(shape=[240,360])
        deg90_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/90deg_days_'+year+'_'+ens+'.npy')
        deg95_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/95deg_days_'+year+'_'+ens+'.npy')
        print(ens,year)
        for k in range(0,240):
            for j in range(0,360):
                total_90deg_days[k,j] = np.sum(deg90_days[:,k,j], axis = 0)
                total_95deg_days[k,j] = np.sum(deg95_days[:,k,j], axis = 0)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_90deg_days_'+year+'_'+ens+'.npy', total_90deg_days)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_95deg_days_'+year+'_'+ens+'.npy', total_95deg_days)
'''
'''
# Part 3: Calculate the ensemble average of annual 90 and 95 degree days
for ens in WRF_list:
    avg90 = np.zeros((10,240,360))
    avg95 = np.zeros((10,240,360))
    n = 0
    for year in future:
        print(ens, year, n)
        annual_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_90deg_days_'+year+'_'+ens+'.npy')
        annual_95 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_95deg_days_'+year+'_'+ens+'.npy')
        avg90[n,:,:] = annual_90[:]
        avg95[n,:,:] = annual_95[:]
        n = n + 1

    ens_avg90 = np.mean(avg90, axis = 0)
    ens_avg95 = np.mean(avg95, axis = 0)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_90deg_days_'+ens+'.npy', ens_avg90)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_95deg_days_'+ens+'.npy', ens_avg95)
'''
'''
# Part 4: Calculate the average annual 90 and 95 degree days for ensemble groups
WRF_list = ['WRF16','WRF17','WRF18','WRF19','WRF20']

ens_all90 = np.zeros((5,240,360))
ens_all95 = np.zeros((5,240,360))

n = 0
for ens in WRF_list:
    print(ens, n)
    days90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_90deg_days_'+ens+'.npy')
    days95 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_95deg_days_'+ens+'.npy')
    ens_all90[n,:,:] = days90[:]
    ens_all95[n,:,:] = days95[:]
    n = n + 1

ens_avg90 = np.mean(ens_all90, axis = 0)
ens_avg95 = np.mean(ens_all95, axis = 0)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_90F_days_WRF16-20.npy', ens_avg90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_95F_days_WRF16-20.npy', ens_avg95)
'''
'''
# Repeat but for wet and dry season
# Part 2: Calculate the total rainfall days per season
for ens in WRF_list:
    for year in future:
        total_days_wet90 = np.zeros(shape=[240,360])
        total_days_dry90 = np.zeros(shape=[240,360])
        total_days_wet95 = np.zeros(shape=[240,360])
        total_days_dry95 = np.zeros(shape=[240,360])
        days90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/90deg_days_'+year+'_'+ens+'.npy')
        days95 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/95deg_days_'+year+'_'+ens+'.npy')
        print(ens,year)
        for k in range(0,240):
            for j in range(0,360):
                wet90_1 = np.sum(days90[0:120,k,j], axis = 0)
                wet90_2 = np.sum(days90[303:365,k,j], axis = 0)
                total_days_wet90[k,j] = wet90_1 + wet90_2
                total_days_dry90[k,j] = np.sum(days90[120:304,k,j], axis = 0)

                wet95_1 = np.sum(days95[0:120,k,j], axis = 0)
                wet95_2 = np.sum(days95[303:365,k,j], axis = 0)
                total_days_wet95[k,j] = wet95_1 + wet95_2
                total_days_dry95[k,j] = np.sum(days95[120:304,k,j], axis = 0)

        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/90deg_wet_days_'+year+'_'+ens+'.npy', total_days_wet90)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/90deg_dry_days_'+year+'_'+ens+'.npy', total_days_dry90)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/95deg_wet_days_'+year+'_'+ens+'.npy', total_days_wet95)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/95deg_dry_days_'+year+'_'+ens+'.npy', total_days_dry95)
'''
'''
# Part 3: Calculate the ensemble average of wet/dry season total 90 and 95 degree days
for ens in WRF_list:
    avg_wet90 = np.zeros((10,240,360))
    avg_wet95 = np.zeros((10,240,360))
    avg_dry90 = np.zeros((10,240,360))
    avg_dry95 = np.zeros((10,240,360))
    n = 0
    for year in future:
        print(ens, year, n)
        wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/90deg_wet_days_'+year+'_'+ens+'.npy')
        wet_95 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/95deg_wet_days_'+year+'_'+ens+'.npy')
        dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/90deg_dry_days_'+year+'_'+ens+'.npy')
        dry_95 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/95deg_dry_days_'+year+'_'+ens+'.npy')
        avg_wet90[n,:,:] = wet_90[:]
        avg_wet95[n,:,:] = wet_95[:]
        avg_dry90[n,:,:] = dry_90[:]
        avg_dry95[n,:,:] = dry_95[:]

        n = n + 1

    ens_wet90_avg = np.mean(avg_wet90, axis = 0)
    ens_wet95_avg = np.mean(avg_wet95, axis = 0)
    ens_dry90_avg = np.mean(avg_dry90, axis = 0)
    ens_dry95_avg = np.mean(avg_dry95, axis = 0)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_90deg_days_'+ens+'.npy', ens_wet90_avg)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_95deg_days_'+ens+'.npy', ens_wet95_avg)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_90deg_days_'+ens+'.npy', ens_dry90_avg)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_95deg_days_'+ens+'.npy', ens_dry95_avg)
'''
'''
# Part 4: Calculate the average wet/dry season 90 and 95 degree days for ensemble groups
WRF_list = ['WRF16','WRF17','WRF18','WRF19','WRF20']

ens_wet90 = np.zeros((5,240,360))
ens_wet95 = np.zeros((5,240,360))
ens_dry90 = np.zeros((5,240,360))
ens_dry95 = np.zeros((5,240,360))

n = 0
for ens in WRF_list:
    print(ens, n)
    temps_wet90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_90deg_days_'+ens+'.npy')
    temps_wet95 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_95deg_days_'+ens+'.npy')
    temps_dry90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_90deg_days_'+ens+'.npy')
    temps_dry95 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_95deg_days_'+ens+'.npy')
    ens_wet90[n,:,:] = temps_wet90[:]
    ens_wet95[n,:,:] = temps_wet95[:]
    ens_dry90[n,:,:] = temps_dry90[:]
    ens_dry95[n,:,:] = temps_dry95[:]
    n = n + 1

ens_wet90_avg = np.mean(ens_wet90, axis = 0)
ens_wet95_avg = np.mean(ens_wet95, axis = 0)
ens_dry90_avg = np.mean(ens_dry90, axis = 0)
ens_dry95_avg = np.mean(ens_dry95, axis = 0)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_90deg_days_WRF16-20.npy', ens_wet90_avg)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_95deg_days_WRF16-20.npy', ens_wet95_avg)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_90deg_days_WRF16-20.npy', ens_dry90_avg)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_95deg_days_WRF16-20.npy', ens_dry95_avg)
'''

