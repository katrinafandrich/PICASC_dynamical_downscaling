# Calculate record consecutive dry days for WRF simulations

import xarray
import numpy as np

'''
# Part 1: Loop thru each year in each WRF ens and give dry days "1" and rain days "0"
WRF_list = ['WRF']

present = [str(x) for x in range(1996,2006)]
future = [str(x) for x in range(2026,2036)]

for ens in WRF_list:
    for year in future:
        total_days = 0
        consecutive_dry = np.zeros((365,240,360))
        RAIN = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dailyrain_'+year+'_'+ens+'.npy')
        for n in range(0,365):
            for k in range(0,240):
                for j in range(0,360):
                    if RAIN[n,k,j] < 0.1:
                        consecutive_dry[total_days,k,j] = 1
                    else:
                        consecutive_dry[total_days,k,j] = 0
            print(ens, year, n)
            total_days = total_days + 1
        filename = 'dry_days_'+year+'_'+ens+'.npy'
        print(filename)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+filename, consecutive_dry)


# Part 2: Count CDDs in each year for each WRF ens
present = [str(x) for x in range(1996,2006)]
future = [str(x) for x in range(2026,2036)]

years = present

WRF_list = ['WRF']

total_consecutive_dry = np.zeros((365,240,360))

for ens in WRF_list:
    for year in years:
        consecutive_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_days_'+year+'_'+ens+'.npy')
        for i in range(1,len(consecutive_dry)):
            print(ens,year,i)
            for k in range(0,240):
                for j in range(0,360):
                    if consecutive_dry[i,k,j] == 1:
                        consecutive_dry[i,k,j] += consecutive_dry[i-1,k,j]

                    total_consecutive_dry[i,k,j] = consecutive_dry[i,k,j]
           
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_'+year+'_'+ens+'.npy', total_consecutive_dry)


# Part 3: Find max CDDs in each year for each WRF ens
present = [str(x) for x in range(1996,2006)]
future = [str(x) for x in range(2026,2036)]

WRF_list = ['WRF20']

record_dry_days = np.zeros((10,240,360))

for ens in WRF_list:
    yr = 0
    for year in future:
        total_dry_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_'+year+'_'+ens+'.npy')
        print(ens,year,yr)
        for k in range(0,240):
            for j in range(0,360):
                record_dry_days[yr,k,j] = total_dry_days[:,k,j].max()
        yr = yr + 1

record_dry_days = np.mean(record_dry_days, axis = 0)

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/record_CDD_'+ens+'_.npy', record_dry_days)


# Part 4: Calculate average CDDs for WRF groups
WRF_list = ['WRF16','WRF17','WRF18','WRF19','WRF20']

WRFgroup = np.zeros((5,240,360))

n=0
for ens in WRF_list:
    record_CDD = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/record_CDD_'+ens+'_.npy')
    WRFgroup[n,:,:] = record_CDD[:]
    n = n + 1

average = np.mean(WRFgroup, axis = 0)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/record_CDD_WRF16-20.npy', average)
'''
'''
# Part 1: Loop thru each year and each season in each WRF ens and give dry days "1" and rain days "0"
WRF_list = ['WRF11','WRF12','WRF13','WRF14','WRF15','WRF16','WRF17','WRF18','WRF19','WRF20']

present = [str(x) for x in range(1996,2006)]
future = [str(x) for x in range(2026,2036)]

for ens in WRF_list:
    for year in future:
        cdd_wet = np.zeros((181,240,360))
        cdd_dry = np.zeros((184,240,360))

        wet_days = 0
        dry_days = 0

        RAIN = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dailyrain_'+year+'_'+ens+'.npy')
        for n in range(0,365):
            print(ens,year,n,dry_days,wet_days)
            if n >= 0 and n <= 119:
                daily = cdd_wet
                days = wet_days
                wet_days = wet_days + 1
            elif n >= 120 and n <= 303:
                daily = cdd_dry
                days = dry_days
                dry_days = dry_days + 1
            elif n >= 304 and n <= 364:
                daily = cdd_wet
                days = wet_days
                wet_days = wet_days + 1
            for k in range(0,240):
                for j in range(0,360):
                    if RAIN[n,k,j] < 0.1:
                        daily[days,k,j] = 1
                    else:
                        daily[days,k,j] = 0
        filename1 = 'cdd_wet_'+year+'_'+ens+'.npy'
        filename2 = 'cdd_dry_'+year+'_'+ens+'.npy'
        print(filename1, filename2)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+filename1, cdd_wet)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+filename2, cdd_dry)
'''
'''
# Part 2: Count CDDs in each year for each WRF ens
present = [str(x) for x in range(1996,2006)]
future = [str(x) for x in range(2026,2036)]

years = future 

WRF_list = ['WRF11','WRF12','WRF13','WRF14','WRF15','WRF16','WRF17','WRF18','WRF19','WRF20']

for ens in WRF_list:
    for year in years:
        consecutive_wet = np.zeros((181,240,360))
        consecutive_dry = np.zeros((184,240,360))
        dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/cdd_dry_'+year+'_'+ens+'.npy')
        wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/cdd_wet_'+year+'_'+ens+'.npy')
        for i in range(1,len(dry)):
            print(ens,year,i)
            for k in range(0,240):
                for j in range(0,360):
                    if dry[i,k,j] == 1:
                        dry[i,k,j] += dry[i-1,k,j]
                    consecutive_dry[i,k,j] = dry[i,k,j]

        for i in range(1,len(wet)):
            print(ens,year,i)
            for k in range(0,240):
                for j in range(0,360):
                    if wet[i,k,j] == 1:
                        wet[i,k,j] += wet[i-1,k,j]
                    consecutive_wet[i,k,j] = wet[i,k,j]
                    
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_dry_'+year+'_'+ens+'.npy', consecutive_dry)
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_wet_'+year+'_'+ens+'.npy', consecutive_wet)
'''
'''
# Part 3: Find max CDDs in each season of each year for each WRF ens
present = [str(x) for x in range(1996,2006)]
future = [str(x) for x in range(2026,2036)]

WRF_list = ['WRF11','WRF12','WRF13','WRF14','WRF15','WRF16','WRF17','WRF18','WRF19','WRF20']

for ens in WRF_list:
    yr = 0
    wet_season_cdd = np.zeros((10,240,360))
    dry_season_cdd = np.zeros((10,240,360))
    for year in future:
        cdd_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_wet_'+year+'_'+ens+'.npy')
        cdd_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_dry_'+year+'_'+ens+'.npy')
        print(ens,year,yr)
        for k in range(0,240):
            for j in range(0,360):
                wet_season_cdd[yr,k,j] = cdd_wet[:,k,j].max()
                dry_season_cdd[yr,k,j] = cdd_dry[:,k,j].max()
        yr = yr + 1

    cdd_wet_season = np.mean(wet_season_cdd, axis = 0)
    cdd_dry_season = np.mean(dry_season_cdd, axis = 0)

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_wet_'+ens+'_.npy', cdd_wet_season)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_dry_'+ens+'_.npy', cdd_dry_season)

'''

# Part 4: Calculate average CDDs for WRF groups
WRF_list = ['WRF16','WRF17','WRF18','WRF19','WRF20']

wet = np.zeros((5,240,360))
dry = np.zeros((5,240,360))

n=0
for ens in WRF_list:
    record_CDD_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_wet_'+ens+'_.npy')
    record_CDD_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_dry_'+ens+'_.npy')
    wet[n,:,:] = record_CDD_wet[:]
    dry[n,:,:] = record_CDD_dry[:]
    n = n + 1

average_wet = np.mean(wet, axis = 0)
average_dry = np.mean(dry, axis = 0)

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_wet_WRF16-20.npy', average_wet)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_dry_WRF16-20.npy', average_dry)