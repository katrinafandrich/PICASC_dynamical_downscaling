import xarray
import numpy as np

'''
# Part 1: Loop thru each year and give dry days "1" and rain days "0"
years = [str(x) for x in range(1990,2015)]

for year in years:
    dry_days = np.zeros(shape=[365,254,327])

    for n in range(0,365):
        print(year,n)
        rain = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_'+str(n)+'_low_res.npy')
        
        for k in range(0,254):
            for j in range(0,327):
                if rain[k,j] < 0.1:
                    dry_days[n,k,j] = 1
                else:
                    dry_days[n,k,j] = 0

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_dry_days_RainAtlas.npy', dry_days)


# Part 2: Count CDDs in each year
years = [str(x) for x in range(1990,2015)]

total_consecutive_dry = np.zeros(shape = [365,254,327])

for year in years:
    n = 1
    consecutive_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_dry_days_RainAtlas.npy')
    for i in range(1,len(consecutive_dry)):
        print(year,i,n)
        for k in range(0,254):
            for j in range(0,327):
                if consecutive_dry[i,k,j] == 1:
                    consecutive_dry[i,k,j] += consecutive_dry[i-1,k,j]
                total_consecutive_dry[n-1,k,j] = consecutive_dry[i,k,j]
        n = n + 1
                    
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_'+year+'_RainAtlas.npy', total_consecutive_dry)


# Part 3: Find max CDDs in each year
years = [str(x) for x in range(1990,2015)]
record_dry_days = np.zeros(shape = [25,254,327])

n = 0
for year in years:
    total_dry_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_'+year+'_RainAtlas.npy')
    print(year,n)
    for k in range(0,254):
        for j in range(0,327):
            record_dry_days[n,k,j] = total_dry_days[:,k,j].max()
    n = n + 1

average = np.mean(record_dry_days, axis = 0)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_CDDs_1990-2014_RainAtlas.npy', average)

print('DONE')
'''
'''
# Part 1: Loop thru SEASONS of each year and give dry days "1" and rain days "0"
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
                if rain[k,j] < 0.1:
                    daily[days,k,j] = 1
                else:
                    daily[days,k,j] = 0

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_wet.npy', daily_rain_wet)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_dry.npy', daily_rain_dry)
'''
'''
# Part 2: Count CDDs in each year
years = [str(x) for x in range(1990,2015)]

for year in years:
    total_consecutive_wet = np.zeros(shape=[181,254,327])
    total_consecutive_dry = np.zeros(shape=[184,254,327])
    consecutive_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_wet.npy')
    consecutive_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+year+'_daily_dry.npy')
    n = 1
    for i in range(1,len(consecutive_dry)):
        print(year,i,n)
        for k in range(0,254):
            for j in range(0,327):
                if consecutive_dry[i,k,j] == 1:
                    consecutive_dry[i,k,j] += consecutive_dry[i-1,k,j]
                total_consecutive_dry[n-1,k,j] = consecutive_dry[i,k,j]
        n = n + 1
    
    n = 1
    for i in range(1,len(consecutive_wet)):
        print(year,i,n)
        for k in range(0,254):
            for j in range(0,327):
                if consecutive_wet[i,k,j] == 1:
                    consecutive_wet[i,k,j] += consecutive_wet[i-1,k,j]
                total_consecutive_wet[n-1,k,j] = consecutive_wet[i,k,j]
        n = n + 1
                    
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_'+year+'_wet_RainAtlas.npy', total_consecutive_wet)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_'+year+'_dry_RainAtlas.npy', total_consecutive_dry)
'''

# Part 3: Find max CDDs in each year
years = [str(x) for x in range(1990,2015)]
record_dry_days = np.zeros(shape = [25,254,327])

n = 0

for year in years:
    total_dry_days = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_'+year+'_dry_RainAtlas.npy')
    print(year,n)
    for k in range(0,254):
        for j in range(0,327):
            record_dry_days[n,k,j] = total_dry_days[:,k,j].max()
    n = n + 1

average = np.mean(record_dry_days, axis = 0)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_CDDs_1990-2014_RainAtlas.npy', average)


print('DONE')
                
