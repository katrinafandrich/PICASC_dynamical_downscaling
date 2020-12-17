# Extreme precip histogram
# KMF
# 5/4/20

from sympy import symbols
import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats

'''
# Loop through each day in each simulation and store daily rainfall
WRF_list1 = ['WRF1','WRF2','WRF3','WRF4','WRF5']
WRF_list2 = ['WRF6','WRF7','WRF8','WRF9','WRF10']
WRF_list3 = ['WRF11','WRF12','WRF13','WRF14','WRF15']
WRF_list4 = ['WRF16','WRF17','WRF18','WRF19','WRF20']

#years = ['1996','1997','1998','1999','2000','2001','2002','2003','2004','2005']
years = ['2026','2027','2028','2029','2030','2031','2032','2033','2034','2035']

months = ['01','02','03','04','05','06','07','08','09','10','11','12']
wet_mon = ['11','12','01','02','03','04']
dry_mon = ['05','06','07','08','09','10']

days_jan = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
days_31 = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
days_30 = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
days_28 = days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']

#daily_rain = np.zeros(shape=[3649*10]) # annual days

lihue_rain = np.zeros(shape=[1629*5]) # wet season days
hon_rain = np.zeros(shape=[1629*5])
bog_rain = np.zeros(shape=[1629*5])
hilo_rain = np.zeros(shape=[1629*5])

#lihue_rain = np.zeros(shape=[1840*5]) # dry season days
#hon_rain = np.zeros(shape=[1840*5])
#bog_rain = np.zeros(shape=[1840*5])
#hilo_rain = np.zeros(shape=[1840*5])

total_days = 0
for ens in WRF_list4:
    for year in years:
        for month in wet_mon:
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

            if month == '01' and year != '2035':
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
            elif month == '11' and year == '2035':
                break
            else:
                oldyear = years.index(year)
                year = years[oldyear]

            for day in days:
                path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/hires/'+ens+'/daily/'+year
                ncfile = '/wrfxtrm_d02_'+year+'-'+month+'-'+day+'.nc'
                print(ens,year,month,day)
                data = xarray.open_dataset(path+ncfile)
                
                RAINC = data.RAINCVMEAN[0,189,70]
                RAINNC = data.RAINNCVMEAN[0,189,70]
                RAIN = (RAINC + RAINNC)*86400 # Convert to mm/day
                if RAIN > 0.1:
                    lihue_rain[total_days] = RAIN 
                else:
                    lihue_rain[total_days] = np.nan 

                RAINC = data.RAINCVMEAN[0,152,143]
                RAINNC = data.RAINNCVMEAN[0,152,143]
                RAIN = (RAINC + RAINNC)*86400 # Convert to mm/day
                if RAIN > 0.1:
                    hon_rain[total_days] = RAIN
                else:
                    hon_rain[total_days] = np.nan

                RAINC = data.RAINCVMEAN[0,119,239]
                RAINNC = data.RAINNCVMEAN[0,119,239]
                RAIN = (RAINC + RAINNC)*86400 # Convert to mm/day
                if RAIN > 0.1:
                    bog_rain[total_days] = RAIN
                else:
                    bog_rain[total_days] = np.nan

                RAINC = data.RAINCVMEAN[0,64,293]
                RAINNC = data.RAINNCVMEAN[0,64,293]
                RAIN = (RAINC + RAINNC)*86400 # Convert to mm/day
                if RAIN > 0.1:
                    hilo_rain[total_days] = RAIN
                else:
                    hilo_rain[total_days] = np.nan      

                total_days = total_days + 1
                data.close()


np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_wet.npy', lihue_rain)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_wet.npy', hon_rain)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_wet.npy', bog_rain)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_wet.npy', hilo_rain)
'''

# Remove nan values and calc 90th/99th percentiles for each station
names = ['lihue','honolulu','bigbog','hilo']
sims = ['1-5','6-10','11-15','16-20']
seasons = ['wet','dry']

for season in seasons:
    for name in names:
        for sim in sims:
            data = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+name+sim+'_'+season+'.npy')
            no_nans = data[np.logical_not(np.isnan(data))]
            dsort = np.sort(no_nans[:])
            print(name, sim, season)
            percent90 = dsort[-(round(len(dsort)*0.1)):]
            percent99 = dsort[-(round(len(dsort)*0.01)):]

            np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+name+sim+'_90_'+season+'.npy', percent90)
            np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+name+sim+'_99_'+season+'.npy', percent99)
