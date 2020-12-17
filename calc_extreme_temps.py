# Code to calculate 90th and 99th percentile temperature days
# KMF 7/1/20

import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Loop through each day in each simulation and store daily rainfall
WRF_list = ['WRF11','WRF12','WRF13','WRF14','WRF15','WRF16','WRF17','WRF18','WRF19','WRF20']

#years = ['1996','1997','1998','1999','2000','2001','2002','2003','2004','2005']
years = ['2026','2027','2028','2029','2030','2031','2032','2033','2034','2035']

months = ['01','02','03','04','05','06','07','08','09','10','11','12']
dry_mon = ['11','12','01','02','03','04']
dry_mon = ['05','06','07','08','09','10']

days_jan = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
days_31 = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
days_30 = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
days_28 = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']


for ens in WRF_list:
    for year in years:
        total_days = 0
        daily_temps = np.zeros(shape=[365,240,360])
        for month in months:
            if month == '01' and year == '2026':
                days = days_jan 
            elif month == '01' and year != '2026':
                days = days_31
            elif month == '03' or month == '05' or month == '07' or month == '08' or month == '10' or month =='12':
                days = days_31
            elif month == '04' or month == '06' or month == '09' or month == '11':
                days = days_30
            else:
                days = days_28
            
            '''
            # Use this section for dry season
            if month == '01' and year != '2005':
                oldyear = years.index(year)
                year = years[oldyear+1]
            elif month == '02':
                oldyear = years.index(year)
                year = years[oldyear]
            elif month == '03':
                oldyear = years.index(year)
                year = years[oldyear]
            elif month == '04':
                oldyear = years.index(year)
                year = years[oldyear]
            elif month == '11' and year == '2005':
                break
            else:
                oldyear = years.index(year)
                year = years[oldyear]
            '''
            
            for day in days:
                path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/hires/'+ens+'/daily/'+year
                ncfile = '/wrfxtrm_d02_'+year+'-'+month+'-'+day+'.nc'
                print(ens,year,month,day)
                data = xarray.open_dataset(path+ncfile)

                T_kelvin = data.T2MAX[:]
                T_C = T_kelvin[0,:,:] - 273 # Convert K to degrees C
                T_F = (T_C[:]*1.8) + 32 # Convert to degrees F

                daily_temps[total_days] = T_F[:,:]

                total_days = total_days + 1
                data.close()
        
        np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dailytemps_'+year+'_'+ens+'.npy', daily_temps)
    #np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/daily_temps_dry_'+ens+'.npy', daily_temps)
    #np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/daily_temps_dry_'+ens+'.npy', daily_temps)

'''
# Load in daily data and calculate 90th and 99th percentile temp extremes
ens = 'WRF20'
print(ens)
daily_data = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/daily_temps_dry_'+ens+'.npy')

temps_90th = np.zeros(shape=[240,360])
temps_99th = np.zeros(shape=[240,360])

for k in range(0,240):
    for l in range(0,360):
        dsort = np.sort(daily_data[:,k,l])

        percent90 = dsort[-(round(len(dsort)*0.1)):]
        percent99 = dsort[-(round(len(dsort)*0.01)):]

        temps_90th[k,l] = np.mean(percent90, axis = 0)
        temps_99th[k,l] = np.mean(percent99, axis = 0)

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+ens+'_90th_temps_dry.npy', temps_90th)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+ens+'_99th_temps_dry.npy', temps_99th)
'''


# Make seperate script to load these in and plot
wrf1_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF1_90th_temps_dry.npy')
wrf1_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF1_99th_temps_dry.npy')
wrf2_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF2_90th_temps_dry.npy')
wrf2_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF2_99th_temps_dry.npy')
wrf3_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF3_90th_temps_dry.npy')
wrf3_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF3_99th_temps_dry.npy')
wrf4_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF4_90th_temps_dry.npy')
wrf4_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF4_99th_temps_dry.npy')
wrf5_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF5_90th_temps_dry.npy')
wrf5_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF5_99th_temps_dry.npy')

percent90_wrf1_5 = (wrf1_90th + wrf2_90th + wrf3_90th + wrf4_90th + wrf5_90th) / 5
percent99_wrf1_5 = (wrf1_99th + wrf2_99th + wrf3_99th + wrf4_99th + wrf5_99th) / 5

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF1-5_90th_temps_dry.npy', percent90_wrf1_5)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF1-5_99th_temps_dry.npy', percent99_wrf1_5)

wrf6_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF6_90th_temps_dry.npy')
wrf6_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF6_99th_temps_dry.npy')
wrf7_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF7_90th_temps_dry.npy')
wrf7_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF7_99th_temps_dry.npy')
wrf8_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF8_90th_temps_dry.npy')
wrf8_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF8_99th_temps_dry.npy')
wrf9_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF9_90th_temps_dry.npy')
wrf9_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF9_99th_temps_dry.npy')
wrf10_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF10_90th_temps_dry.npy')
wrf10_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF10_99th_temps_dry.npy')

percent90_wrf6_10 = (wrf6_90th + wrf7_90th + wrf8_90th + wrf9_90th + wrf10_90th) / 5
percent99_wrf6_10 = (wrf6_99th + wrf7_99th + wrf8_99th + wrf9_99th + wrf10_99th) / 5

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF6-10_90th_temps_dry.npy', percent90_wrf6_10)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF6-10_99th_temps_dry.npy', percent99_wrf6_10)

wrf11_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF11_90th_temps_dry.npy')
wrf11_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF11_99th_temps_dry.npy')
wrf12_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF12_90th_temps_dry.npy')
wrf12_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF12_99th_temps_dry.npy')
wrf13_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF13_90th_temps_dry.npy')
wrf13_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF13_99th_temps_dry.npy')
wrf14_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF14_90th_temps_dry.npy')
wrf14_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF14_99th_temps_dry.npy')
wrf15_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF15_90th_temps_dry.npy')
wrf15_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF15_99th_temps_dry.npy')

percent90_wrf11_15 = (wrf11_90th + wrf12_90th + wrf13_90th + wrf14_90th + wrf15_90th) / 5
percent99_wrf11_15 = (wrf11_99th + wrf12_99th + wrf13_99th + wrf14_99th + wrf15_99th) / 5

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF11-15_90th_temps_dry.npy', percent90_wrf11_15)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF11-15_99th_temps_dry.npy', percent99_wrf11_15)

wrf16_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF16_90th_temps_dry.npy')
wrf16_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF16_99th_temps_dry.npy')
wrf17_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF17_90th_temps_dry.npy')
wrf17_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF17_99th_temps_dry.npy')
wrf18_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF18_90th_temps_dry.npy')
wrf18_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF18_99th_temps_dry.npy')
wrf19_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF19_90th_temps_dry.npy')
wrf19_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF19_99th_temps_dry.npy')
wrf20_90th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF20_90th_temps_dry.npy')
wrf20_99th = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF20_99th_temps_dry.npy')

percent90_wrf16_20 = (wrf16_90th + wrf17_90th + wrf18_90th + wrf19_90th + wrf20_90th) / 5
percent99_wrf16_20 = (wrf16_99th + wrf17_99th + wrf18_99th + wrf19_99th + wrf20_99th) / 5

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF16-20_90th_temps_dry.npy', percent90_wrf16_20)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF16-20_99th_temps_dry.npy', percent99_wrf16_20)
