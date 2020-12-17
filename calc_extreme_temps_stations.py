# Calculate 90th and 99th percentile temperatures for stations
# KMF 7/13/19

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
WRF_list = ['WRF20']

#years = ['1996','1997','1998','1999','2000','2001','2002','2003','2004','2005']
years = ['2026','2027','2028','2029','2030','2031','2032','2033','2034','2035']

months = ['01','02','03','04','05','06','07','08','09','10','11','12']
wet_mon = ['11','12','01','02','03','04']
dry_mon = ['05','06','07','08','09','10']

days_jan = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
days_31 = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
days_30 = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
days_28 = days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']


T_lihue = np.zeros(shape=[3649]) # annual days
T_honolulu = np.zeros(shape=[3649])
T_bigbog = np.zeros(shape=[3649])
T_hilo = np.zeros(shape=[3649])


T_lihue = np.zeros(shape=[1629]) # wet season days
T_honolulu = np.zeros(shape=[1629])
T_bigbog = np.zeros(shape=[1629])
T_hilo = np.zeros(shape=[1629])


T_lihue = np.zeros(shape=[1840]) # dry season days
T_honolulu = np.zeros(shape=[1840])
T_bigbog = np.zeros(shape=[1840])
T_hilo = np.zeros(shape=[1840])

total_days = 0
for ens in WRF_list:
    for year in years:
        for month in dry_mon:
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
            

            for day in days:
                path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/hires/'+ens+'/daily/'+year
                ncfile = '/wrfxtrm_d02_'+year+'-'+month+'-'+day+'.nc'
                print(ens,year,month,day)
                data = xarray.open_dataset(path+ncfile)

                T_kelvin = data.T2MAX[:]
                T_C = T_kelvin[0,:,:] - 273 # Convert K to degrees C
                T_F = (T_C[:]*1.8) + 32 # Convert to degrees F
                
                T_lihue[total_days] = T_F[189,70]

                T_honolulu[total_days] = T_F[152,143]

                T_bigbog[total_days] = T_F[119,239]

                T_hilo[total_days] = T_F[64,293]     

                total_days = total_days + 1
                data.close()


np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_'+ens+'_dry.npy', T_lihue)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_'+ens+'_dry.npy', T_honolulu)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_'+ens+'_dry.npy', T_bigbog)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_'+ens+'_dry.npy', T_hilo)
'''
'''
# Remove nan values and calc 90th/99th percentiles for each station
names = ['lihue','honolulu','bigbog','hilo']
sims = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
seasons = ['ann','wet','dry']

for season in seasons:
    for name in names:
        for sim in sims:
            data = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+name+'_WRF'+sim+'_'+season+'.npy')
            dsort = np.sort(data[:])
            print(name, sim, season)
            percent90 = dsort[-(round(len(dsort)*0.1)):]
            percent99 = dsort[-(round(len(dsort)*0.01)):]

            np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+name+'_WRF'+sim+'_P90temps_'+season+'.npy', percent90)
            np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+name+'_WRF'+sim+'_P99temps_'+season+'.npy', percent99)
'''
##########################################################################################################################
lihue1_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1_P90temps_ann.npy')
lihue2_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF2_P90temps_ann.npy')
lihue3_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF3_P90temps_ann.npy')
lihue4_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF4_P90temps_ann.npy')
lihue5_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF5_P90temps_ann.npy')

lihue1_5_ann_90 = (lihue1_ann_90 + lihue2_ann_90 + lihue3_ann_90 + lihue4_ann_90 + lihue5_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1-5_P90temps_ann.npy', lihue1_5_ann_90)

lihue1_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1_P99temps_ann.npy')
lihue2_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF2_P99temps_ann.npy')
lihue3_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF3_P99temps_ann.npy')
lihue4_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF4_P99temps_ann.npy')
lihue5_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF5_P99temps_ann.npy')

lihue1_5_ann_99 = (lihue1_ann_99 + lihue2_ann_99 + lihue3_ann_99 + lihue4_ann_99 + lihue5_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1-5_P99temps_ann.npy', lihue1_5_ann_99)

lihue1_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1_P90temps_wet.npy')
lihue2_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF2_P90temps_wet.npy')
lihue3_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF3_P90temps_wet.npy')
lihue4_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF4_P90temps_wet.npy')
lihue5_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF5_P90temps_wet.npy')

lihue1_5_wet_90 = (lihue1_wet_90 + lihue2_wet_90 + lihue3_wet_90 + lihue4_wet_90 + lihue5_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1-5_P90temps_wet.npy', lihue1_5_wet_90)

lihue1_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1_P99temps_wet.npy')
lihue2_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF2_P99temps_wet.npy')
lihue3_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF3_P99temps_wet.npy')
lihue4_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF4_P99temps_wet.npy')
lihue5_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF5_P99temps_wet.npy')

lihue1_5_wet_99 = (lihue1_wet_99 + lihue2_wet_99 + lihue3_wet_99 + lihue4_wet_99 + lihue5_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1-5_P99temps_wet.npy', lihue1_5_wet_99)

lihue1_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1_P90temps_dry.npy')
lihue2_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF2_P90temps_dry.npy')
lihue3_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF3_P90temps_dry.npy')
lihue4_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF4_P90temps_dry.npy')
lihue5_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF5_P90temps_dry.npy')

lihue1_5_dry_90 = (lihue1_dry_90 + lihue2_dry_90 + lihue3_dry_90 + lihue4_dry_90 + lihue5_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1-5_P90temps_dry.npy', lihue1_5_dry_90)

lihue1_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1_P99temps_dry.npy')
lihue2_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF2_P99temps_dry.npy')
lihue3_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF3_P99temps_dry.npy')
lihue4_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF4_P99temps_dry.npy')
lihue5_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF5_P99temps_dry.npy')

lihue1_5_dry_99 = (lihue1_dry_99 + lihue2_dry_99 + lihue3_dry_99 + lihue4_dry_99 + lihue5_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF1-5_P99temps_dry.npy', lihue1_5_dry_99)
############################################################################################################################
honolulu1_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1_P90temps_ann.npy')
honolulu2_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF2_P90temps_ann.npy')
honolulu3_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF3_P90temps_ann.npy')
honolulu4_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF4_P90temps_ann.npy')
honolulu5_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF5_P90temps_ann.npy')

honolulu1_5_ann_90 = (honolulu1_ann_90 + honolulu2_ann_90 + honolulu3_ann_90 + honolulu4_ann_90 + honolulu5_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1-5_P90temps_ann.npy', honolulu1_5_ann_90)

honolulu1_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1_P99temps_ann.npy')
honolulu2_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF2_P99temps_ann.npy')
honolulu3_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF3_P99temps_ann.npy')
honolulu4_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF4_P99temps_ann.npy')
honolulu5_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF5_P99temps_ann.npy')

honolulu1_5_ann_99 = (honolulu1_ann_99 + honolulu2_ann_99 + honolulu3_ann_99 + honolulu4_ann_99 + honolulu5_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1-5_P99temps_ann.npy', honolulu1_5_ann_99)

honolulu1_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1_P90temps_wet.npy')
honolulu2_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF2_P90temps_wet.npy')
honolulu3_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF3_P90temps_wet.npy')
honolulu4_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF4_P90temps_wet.npy')
honolulu5_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF5_P90temps_wet.npy')

honolulu1_5_wet_90 = (honolulu1_wet_90 + honolulu2_wet_90 + honolulu3_wet_90 + honolulu4_wet_90 + honolulu5_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1-5_P90temps_wet.npy', honolulu1_5_wet_90)

honolulu1_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1_P99temps_wet.npy')
honolulu2_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF2_P99temps_wet.npy')
honolulu3_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF3_P99temps_wet.npy')
honolulu4_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF4_P99temps_wet.npy')
honolulu5_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF5_P99temps_wet.npy')

honolulu1_5_wet_99 = (honolulu1_wet_99 + honolulu2_wet_99 + honolulu3_wet_99 + honolulu4_wet_99 + honolulu5_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1-5_P99temps_wet.npy', honolulu1_5_wet_99)

honolulu1_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1_P90temps_dry.npy')
honolulu2_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF2_P90temps_dry.npy')
honolulu3_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF3_P90temps_dry.npy')
honolulu4_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF4_P90temps_dry.npy')
honolulu5_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF5_P90temps_dry.npy')

honolulu1_5_dry_90 = (honolulu1_dry_90 + honolulu2_dry_90 + honolulu3_dry_90 + honolulu4_dry_90 + honolulu5_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1-5_P90temps_dry.npy', honolulu1_5_dry_90)

honolulu1_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1_P99temps_dry.npy')
honolulu2_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF2_P99temps_dry.npy')
honolulu3_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF3_P99temps_dry.npy')
honolulu4_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF4_P99temps_dry.npy')
honolulu5_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF5_P99temps_dry.npy')

honolulu1_5_dry_99 = (honolulu1_dry_99 + honolulu2_dry_99 + honolulu3_dry_99 + honolulu4_dry_99 + honolulu5_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF1-5_P99temps_dry.npy', honolulu1_5_dry_99)
###################################################################################################################################
bigbog1_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1_P90temps_ann.npy')
bigbog2_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF2_P90temps_ann.npy')
bigbog3_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF3_P90temps_ann.npy')
bigbog4_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF4_P90temps_ann.npy')
bigbog5_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF5_P90temps_ann.npy')

bigbog1_5_ann_90 = (bigbog1_ann_90 + bigbog2_ann_90 + bigbog3_ann_90 + bigbog4_ann_90 + bigbog5_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1-5_P90temps_ann.npy', bigbog1_5_ann_90)

bigbog1_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1_P99temps_ann.npy')
bigbog2_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF2_P99temps_ann.npy')
bigbog3_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF3_P99temps_ann.npy')
bigbog4_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF4_P99temps_ann.npy')
bigbog5_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF5_P99temps_ann.npy')

bigbog1_5_ann_99 = (bigbog1_ann_99 + bigbog2_ann_99 + bigbog3_ann_99 + bigbog4_ann_99 + bigbog5_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1-5_P99temps_ann.npy', bigbog1_5_ann_99)

bigbog1_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1_P90temps_wet.npy')
bigbog2_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF2_P90temps_wet.npy')
bigbog3_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF3_P90temps_wet.npy')
bigbog4_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF4_P90temps_wet.npy')
bigbog5_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF5_P90temps_wet.npy')

bigbog1_5_wet_90 = (bigbog1_wet_90 + bigbog2_wet_90 + bigbog3_wet_90 + bigbog4_wet_90 + bigbog5_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1-5_P90temps_wet.npy', bigbog1_5_wet_90)

bigbog1_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1_P99temps_wet.npy')
bigbog2_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF2_P99temps_wet.npy')
bigbog3_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF3_P99temps_wet.npy')
bigbog4_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF4_P99temps_wet.npy')
bigbog5_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF5_P99temps_wet.npy')

bigbog1_5_wet_99 = (bigbog1_wet_99 + bigbog2_wet_99 + bigbog3_wet_99 + bigbog4_wet_99 + bigbog5_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1-5_P99temps_wet.npy', bigbog1_5_wet_99)

bigbog1_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1_P90temps_dry.npy')
bigbog2_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF2_P90temps_dry.npy')
bigbog3_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF3_P90temps_dry.npy')
bigbog4_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF4_P90temps_dry.npy')
bigbog5_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF5_P90temps_dry.npy')

bigbog1_5_dry_90 = (bigbog1_dry_90 + bigbog2_dry_90 + bigbog3_dry_90 + bigbog4_dry_90 + bigbog5_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1-5_P90temps_dry.npy', bigbog1_5_dry_90)

bigbog1_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1_P99temps_dry.npy')
bigbog2_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF2_P99temps_dry.npy')
bigbog3_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF3_P99temps_dry.npy')
bigbog4_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF4_P99temps_dry.npy')
bigbog5_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF5_P99temps_dry.npy')

bigbog1_5_dry_99 = (bigbog1_dry_99 + bigbog2_dry_99 + bigbog3_dry_99 + bigbog4_dry_99 + bigbog5_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF1-5_P99temps_dry.npy', bigbog1_5_dry_99)

###########################################################################################################################
hilo1_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1_P90temps_ann.npy')
hilo2_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF2_P90temps_ann.npy')
hilo3_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF3_P90temps_ann.npy')
hilo4_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF4_P90temps_ann.npy')
hilo5_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF5_P90temps_ann.npy')

hilo1_5_ann_90 = (hilo1_ann_90 + hilo2_ann_90 + hilo3_ann_90 + hilo4_ann_90 + hilo5_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1-5_P90temps_ann.npy', hilo1_5_ann_90)

hilo1_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1_P99temps_ann.npy')
hilo2_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF2_P99temps_ann.npy')
hilo3_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF3_P99temps_ann.npy')
hilo4_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF4_P99temps_ann.npy')
hilo5_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF5_P99temps_ann.npy')

hilo1_5_ann_99 = (hilo1_ann_99 + hilo2_ann_99 + hilo3_ann_99 + hilo4_ann_99 + hilo5_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1-5_P99temps_ann.npy', hilo1_5_ann_99)

hilo1_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1_P90temps_wet.npy')
hilo2_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF2_P90temps_wet.npy')
hilo3_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF3_P90temps_wet.npy')
hilo4_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF4_P90temps_wet.npy')
hilo5_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF5_P90temps_wet.npy')

hilo1_5_wet_90 = (hilo1_wet_90 + hilo2_wet_90 + hilo3_wet_90 + hilo4_wet_90 + hilo5_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1-5_P90temps_wet.npy', hilo1_5_wet_90)

hilo1_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1_P99temps_wet.npy')
hilo2_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF2_P99temps_wet.npy')
hilo3_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF3_P99temps_wet.npy')
hilo4_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF4_P99temps_wet.npy')
hilo5_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF5_P99temps_wet.npy')

hilo1_5_wet_99 = (hilo1_wet_99 + hilo2_wet_99 + hilo3_wet_99 + hilo4_wet_99 + hilo5_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1-5_P99temps_wet.npy', hilo1_5_wet_99)

hilo1_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1_P90temps_dry.npy')
hilo2_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF2_P90temps_dry.npy')
hilo3_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF3_P90temps_dry.npy')
hilo4_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF4_P90temps_dry.npy')
hilo5_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF5_P90temps_dry.npy')

hilo1_5_dry_90 = (hilo1_dry_90 + hilo2_dry_90 + hilo3_dry_90 + hilo4_dry_90 + hilo5_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1-5_P90temps_dry.npy', hilo1_5_dry_90)

hilo1_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1_P99temps_dry.npy')
hilo2_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF2_P99temps_dry.npy')
hilo3_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF3_P99temps_dry.npy')
hilo4_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF4_P99temps_dry.npy')
hilo5_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF5_P99temps_dry.npy')

hilo1_5_dry_99 = (hilo1_dry_99 + hilo2_dry_99 + hilo3_dry_99 + hilo4_dry_99 + hilo5_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF1-5_P99temps_dry.npy', hilo1_5_dry_99)

##########################################################################################################################
lihue6_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6_P90temps_ann.npy')
lihue7_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF7_P90temps_ann.npy')
lihue8_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF8_P90temps_ann.npy')
lihue9_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF9_P90temps_ann.npy')
lihue10_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF10_P90temps_ann.npy')

lihue6_10_ann_90 = (lihue6_ann_90 + lihue7_ann_90 + lihue8_ann_90 + lihue9_ann_90 + lihue10_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6-10_P90temps_ann.npy', lihue6_10_ann_90)

lihue6_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6_P99temps_ann.npy')
lihue7_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF7_P99temps_ann.npy')
lihue8_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF8_P99temps_ann.npy')
lihue9_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF9_P99temps_ann.npy')
lihue10_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF10_P99temps_ann.npy')

lihue6_10_ann_99 = (lihue6_ann_99 + lihue7_ann_99 + lihue8_ann_99 + lihue9_ann_99 + lihue10_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6-10_P99temps_ann.npy', lihue6_10_ann_99)

lihue6_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6_P90temps_wet.npy')
lihue7_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF7_P90temps_wet.npy')
lihue8_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF8_P90temps_wet.npy')
lihue9_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF9_P90temps_wet.npy')
lihue10_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF10_P90temps_wet.npy')

lihue6_10_wet_90 = (lihue6_wet_90 + lihue7_wet_90 + lihue8_wet_90 + lihue9_wet_90 + lihue10_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6-10_P90temps_wet.npy', lihue6_10_wet_90)

lihue6_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6_P99temps_wet.npy')
lihue7_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF7_P99temps_wet.npy')
lihue8_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF8_P99temps_wet.npy')
lihue9_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF9_P99temps_wet.npy')
lihue10_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF10_P99temps_wet.npy')

lihue6_10_wet_99 = (lihue6_wet_99 + lihue7_wet_99 + lihue8_wet_99 + lihue9_wet_99 + lihue10_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6-10_P99temps_wet.npy', lihue6_10_wet_99)

lihue6_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6_P90temps_dry.npy')
lihue7_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF7_P90temps_dry.npy')
lihue8_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF8_P90temps_dry.npy')
lihue9_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF9_P90temps_dry.npy')
lihue10_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF10_P90temps_dry.npy')

lihue6_10_dry_90 = (lihue6_dry_90 + lihue7_dry_90 + lihue8_dry_90 + lihue9_dry_90 + lihue10_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6-10_P90temps_dry.npy', lihue6_10_dry_90)

lihue6_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6_P99temps_dry.npy')
lihue7_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF7_P99temps_dry.npy')
lihue8_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF8_P99temps_dry.npy')
lihue9_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF9_P99temps_dry.npy')
lihue10_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF10_P99temps_dry.npy')

lihue6_10_dry_99 = (lihue6_dry_99 + lihue7_dry_99 + lihue8_dry_99 + lihue9_dry_99 + lihue10_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF6-10_P99temps_dry.npy', lihue6_10_dry_99)
############################################################################################################################
honolulu6_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6_P90temps_ann.npy')
honolulu7_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF7_P90temps_ann.npy')
honolulu8_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF8_P90temps_ann.npy')
honolulu9_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF9_P90temps_ann.npy')
honolulu10_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF10_P90temps_ann.npy')

honolulu6_10_ann_90 = (honolulu6_ann_90 + honolulu7_ann_90 + honolulu8_ann_90 + honolulu9_ann_90 + honolulu10_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6-10_P90temps_ann.npy', honolulu6_10_ann_90)

honolulu6_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6_P99temps_ann.npy')
honolulu7_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF7_P99temps_ann.npy')
honolulu8_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF8_P99temps_ann.npy')
honolulu9_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF9_P99temps_ann.npy')
honolulu10_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF10_P99temps_ann.npy')

honolulu6_10_ann_99 = (honolulu6_ann_99 + honolulu7_ann_99 + honolulu8_ann_99 + honolulu9_ann_99 + honolulu10_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6-10_P99temps_ann.npy', honolulu6_10_ann_99)

honolulu6_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6_P90temps_wet.npy')
honolulu7_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF7_P90temps_wet.npy')
honolulu8_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF8_P90temps_wet.npy')
honolulu9_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF9_P90temps_wet.npy')
honolulu10_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF10_P90temps_wet.npy')

honolulu6_10_wet_90 = (honolulu6_wet_90 + honolulu7_wet_90 + honolulu8_wet_90 + honolulu9_wet_90 + honolulu10_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6-10_P90temps_wet.npy', honolulu6_10_wet_90)

honolulu6_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6_P99temps_wet.npy')
honolulu7_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF7_P99temps_wet.npy')
honolulu8_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF8_P99temps_wet.npy')
honolulu9_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF9_P99temps_wet.npy')
honolulu10_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF10_P99temps_wet.npy')

honolulu6_10_wet_99 = (honolulu6_wet_99 + honolulu7_wet_99 + honolulu8_wet_99 + honolulu9_wet_99 + honolulu10_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6-10_P99temps_wet.npy', honolulu6_10_wet_99)

honolulu6_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6_P90temps_dry.npy')
honolulu7_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF7_P90temps_dry.npy')
honolulu8_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF8_P90temps_dry.npy')
honolulu9_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF9_P90temps_dry.npy')
honolulu10_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF10_P90temps_dry.npy')

honolulu6_10_dry_90 = (honolulu6_dry_90 + honolulu7_dry_90 + honolulu8_dry_90 + honolulu9_dry_90 + honolulu10_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6-10_P90temps_dry.npy', honolulu6_10_dry_90)

honolulu6_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6_P99temps_dry.npy')
honolulu7_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF7_P99temps_dry.npy')
honolulu8_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF8_P99temps_dry.npy')
honolulu9_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF9_P99temps_dry.npy')
honolulu10_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF10_P99temps_dry.npy')

honolulu6_10_dry_99 = (honolulu6_dry_99 + honolulu7_dry_99 + honolulu8_dry_99 + honolulu9_dry_99 + honolulu10_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF6-10_P99temps_dry.npy', honolulu6_10_dry_99)
###################################################################################################################################
bigbog6_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6_P90temps_ann.npy')
bigbog7_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF7_P90temps_ann.npy')
bigbog8_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF8_P90temps_ann.npy')
bigbog9_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF9_P90temps_ann.npy')
bigbog10_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF10_P90temps_ann.npy')

bigbog6_10_ann_90 = (bigbog6_ann_90 + bigbog7_ann_90 + bigbog8_ann_90 + bigbog9_ann_90 + bigbog10_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6-10_P90temps_ann.npy', bigbog6_10_ann_90)

bigbog6_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6_P99temps_ann.npy')
bigbog7_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF7_P99temps_ann.npy')
bigbog8_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF8_P99temps_ann.npy')
bigbog9_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF9_P99temps_ann.npy')
bigbog10_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF10_P99temps_ann.npy')

bigbog6_10_ann_99 = (bigbog6_ann_99 + bigbog7_ann_99 + bigbog8_ann_99 + bigbog9_ann_99 + bigbog10_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6-10_P99temps_ann.npy', bigbog6_10_ann_99)

bigbog6_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6_P90temps_wet.npy')
bigbog7_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF7_P90temps_wet.npy')
bigbog8_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF8_P90temps_wet.npy')
bigbog9_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF9_P90temps_wet.npy')
bigbog10_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF10_P90temps_wet.npy')

bigbog6_10_wet_90 = (bigbog6_wet_90 + bigbog7_wet_90 + bigbog8_wet_90 + bigbog9_wet_90 + bigbog10_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6-10_P90temps_wet.npy', bigbog6_10_wet_90)

bigbog6_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6_P99temps_wet.npy')
bigbog7_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF7_P99temps_wet.npy')
bigbog8_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF8_P99temps_wet.npy')
bigbog9_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF9_P99temps_wet.npy')
bigbog10_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF10_P99temps_wet.npy')

bigbog6_10_wet_99 = (bigbog6_wet_99 + bigbog7_wet_99 + bigbog8_wet_99 + bigbog9_wet_99 + bigbog10_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6-10_P99temps_wet.npy', bigbog6_10_wet_99)

bigbog6_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6_P90temps_dry.npy')
bigbog7_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF7_P90temps_dry.npy')
bigbog8_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF8_P90temps_dry.npy')
bigbog9_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF9_P90temps_dry.npy')
bigbog10_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF10_P90temps_dry.npy')

bigbog6_10_dry_90 = (bigbog6_dry_90 + bigbog7_dry_90 + bigbog8_dry_90 + bigbog9_dry_90 + bigbog10_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6-10_P90temps_dry.npy', bigbog6_10_dry_90)

bigbog6_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6_P99temps_dry.npy')
bigbog7_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF7_P99temps_dry.npy')
bigbog8_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF8_P99temps_dry.npy')
bigbog9_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF9_P99temps_dry.npy')
bigbog10_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF10_P99temps_dry.npy')

bigbog6_10_dry_99 = (bigbog6_dry_99 + bigbog7_dry_99 + bigbog8_dry_99 + bigbog9_dry_99 + bigbog10_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF6-10_P99temps_dry.npy', bigbog6_10_dry_99)

###########################################################################################################################
hilo6_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6_P90temps_ann.npy')
hilo7_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF7_P90temps_ann.npy')
hilo8_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF8_P90temps_ann.npy')
hilo9_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF9_P90temps_ann.npy')
hilo10_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF10_P90temps_ann.npy')

hilo6_10_ann_90 = (hilo6_ann_90 + hilo7_ann_90 + hilo8_ann_90 + hilo9_ann_90 + hilo10_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6-10_P90temps_ann.npy', hilo6_10_ann_90)

hilo6_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6_P99temps_ann.npy')
hilo7_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF7_P99temps_ann.npy')
hilo8_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF8_P99temps_ann.npy')
hilo9_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF9_P99temps_ann.npy')
hilo10_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF10_P99temps_ann.npy')

hilo6_10_ann_99 = (hilo6_ann_99 + hilo7_ann_99 + hilo8_ann_99 + hilo9_ann_99 + hilo10_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6-10_P99temps_ann.npy', hilo6_10_ann_99)

hilo6_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6_P90temps_wet.npy')
hilo7_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF7_P90temps_wet.npy')
hilo8_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF8_P90temps_wet.npy')
hilo9_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF9_P90temps_wet.npy')
hilo10_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF10_P90temps_wet.npy')

hilo6_10_wet_90 = (hilo6_wet_90 + hilo7_wet_90 + hilo8_wet_90 + hilo9_wet_90 + hilo10_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6-10_P90temps_wet.npy', hilo6_10_wet_90)

hilo6_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6_P99temps_wet.npy')
hilo7_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF7_P99temps_wet.npy')
hilo8_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF8_P99temps_wet.npy')
hilo9_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF9_P99temps_wet.npy')
hilo10_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF10_P99temps_wet.npy')

hilo6_10_wet_99 = (hilo6_wet_99 + hilo7_wet_99 + hilo8_wet_99 + hilo9_wet_99 + hilo10_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6-10_P99temps_wet.npy', hilo6_10_wet_99)

hilo6_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6_P90temps_dry.npy')
hilo7_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF7_P90temps_dry.npy')
hilo8_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF8_P90temps_dry.npy')
hilo9_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF9_P90temps_dry.npy')
hilo10_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF10_P90temps_dry.npy')

hilo6_10_dry_90 = (hilo6_dry_90 + hilo7_dry_90 + hilo8_dry_90 + hilo9_dry_90 + hilo10_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6-10_P90temps_dry.npy', hilo6_10_dry_90)

hilo6_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6_P99temps_dry.npy')
hilo7_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF7_P99temps_dry.npy')
hilo8_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF8_P99temps_dry.npy')
hilo9_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF9_P99temps_dry.npy')
hilo10_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF10_P99temps_dry.npy')

hilo6_10_dry_99 = (hilo6_dry_99 + hilo7_dry_99 + hilo8_dry_99 + hilo9_dry_99 + hilo10_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF6-10_P99temps_dry.npy', hilo6_10_dry_99)

##########################################################################################################################
lihue11_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11_P90temps_ann.npy')
lihue12_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF12_P90temps_ann.npy')
lihue13_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF13_P90temps_ann.npy')
lihue14_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF14_P90temps_ann.npy')
lihue15_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF15_P90temps_ann.npy')

lihue11_15_ann_90 = (lihue11_ann_90 + lihue12_ann_90 + lihue13_ann_90 + lihue14_ann_90 + lihue15_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11-15_P90temps_ann.npy', lihue11_15_ann_90)

lihue11_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11_P99temps_ann.npy')
lihue12_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF12_P99temps_ann.npy')
lihue13_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF13_P99temps_ann.npy')
lihue14_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF14_P99temps_ann.npy')
lihue15_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF15_P99temps_ann.npy')

lihue11_15_ann_99 = (lihue11_ann_99 + lihue12_ann_99 + lihue13_ann_99 + lihue14_ann_99 + lihue15_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11-15_P99temps_ann.npy', lihue11_15_ann_99)

lihue11_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11_P90temps_wet.npy')
lihue12_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF12_P90temps_wet.npy')
lihue13_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF13_P90temps_wet.npy')
lihue14_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF14_P90temps_wet.npy')
lihue15_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF15_P90temps_wet.npy')

lihue11_15_wet_90 = (lihue11_wet_90 + lihue12_wet_90 + lihue13_wet_90 + lihue14_wet_90 + lihue15_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11-15_P90temps_wet.npy', lihue11_15_wet_90)

lihue11_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11_P99temps_wet.npy')
lihue12_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF12_P99temps_wet.npy')
lihue13_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF13_P99temps_wet.npy')
lihue14_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF14_P99temps_wet.npy')
lihue15_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF15_P99temps_wet.npy')

lihue11_15_wet_99 = (lihue11_wet_99 + lihue12_wet_99 + lihue13_wet_99 + lihue14_wet_99 + lihue15_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11-15_P99temps_wet.npy', lihue11_15_wet_99)

lihue11_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11_P90temps_dry.npy')
lihue12_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF12_P90temps_dry.npy')
lihue13_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF13_P90temps_dry.npy')
lihue14_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF14_P90temps_dry.npy')
lihue15_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF15_P90temps_dry.npy')

lihue11_15_dry_90 = (lihue11_dry_90 + lihue12_dry_90 + lihue13_dry_90 + lihue14_dry_90 + lihue15_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11-15_P90temps_dry.npy', lihue11_15_dry_90)

lihue11_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11_P99temps_dry.npy')
lihue12_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF12_P99temps_dry.npy')
lihue13_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF13_P99temps_dry.npy')
lihue14_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF14_P99temps_dry.npy')
lihue15_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF15_P99temps_dry.npy')

lihue11_15_dry_99 = (lihue11_dry_99 + lihue12_dry_99 + lihue13_dry_99 + lihue14_dry_99 + lihue15_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF11-15_P99temps_dry.npy', lihue11_15_dry_99)
############################################################################################################################
honolulu11_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11_P90temps_ann.npy')
honolulu12_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF12_P90temps_ann.npy')
honolulu13_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF13_P90temps_ann.npy')
honolulu14_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF14_P90temps_ann.npy')
honolulu15_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF15_P90temps_ann.npy')

honolulu11_15_ann_90 = (honolulu11_ann_90 + honolulu12_ann_90 + honolulu13_ann_90 + honolulu14_ann_90 + honolulu15_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11-15_P90temps_ann.npy', honolulu11_15_ann_90)

honolulu11_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11_P99temps_ann.npy')
honolulu12_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF12_P99temps_ann.npy')
honolulu13_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF13_P99temps_ann.npy')
honolulu14_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF14_P99temps_ann.npy')
honolulu15_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF15_P99temps_ann.npy')

honolulu11_15_ann_99 = (honolulu11_ann_99 + honolulu12_ann_99 + honolulu13_ann_99 + honolulu14_ann_99 + honolulu15_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11-15_P99temps_ann.npy', honolulu11_15_ann_99)

honolulu11_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11_P90temps_wet.npy')
honolulu12_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF12_P90temps_wet.npy')
honolulu13_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF13_P90temps_wet.npy')
honolulu14_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF14_P90temps_wet.npy')
honolulu15_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF15_P90temps_wet.npy')

honolulu11_15_wet_90 = (honolulu11_wet_90 + honolulu12_wet_90 + honolulu13_wet_90 + honolulu14_wet_90 + honolulu15_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11-15_P90temps_wet.npy', honolulu11_15_wet_90)

honolulu11_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11_P99temps_wet.npy')
honolulu12_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF12_P99temps_wet.npy')
honolulu13_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF13_P99temps_wet.npy')
honolulu14_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF14_P99temps_wet.npy')
honolulu15_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF15_P99temps_wet.npy')

honolulu11_15_wet_99 = (honolulu11_wet_99 + honolulu12_wet_99 + honolulu13_wet_99 + honolulu14_wet_99 + honolulu15_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11-15_P99temps_wet.npy', honolulu11_15_wet_99)

honolulu11_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11_P90temps_dry.npy')
honolulu12_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF12_P90temps_dry.npy')
honolulu13_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF13_P90temps_dry.npy')
honolulu14_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF14_P90temps_dry.npy')
honolulu15_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF15_P90temps_dry.npy')

honolulu11_15_dry_90 = (honolulu11_dry_90 + honolulu12_dry_90 + honolulu13_dry_90 + honolulu14_dry_90 + honolulu15_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11-15_P90temps_dry.npy', honolulu11_15_dry_90)

honolulu11_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11_P99temps_dry.npy')
honolulu12_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF12_P99temps_dry.npy')
honolulu13_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF13_P99temps_dry.npy')
honolulu14_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF14_P99temps_dry.npy')
honolulu15_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF15_P99temps_dry.npy')

honolulu11_15_dry_99 = (honolulu11_dry_99 + honolulu12_dry_99 + honolulu13_dry_99 + honolulu14_dry_99 + honolulu15_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF11-15_P99temps_dry.npy', honolulu11_15_dry_99)
###################################################################################################################################
bigbog11_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11_P90temps_ann.npy')
bigbog12_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF12_P90temps_ann.npy')
bigbog13_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF13_P90temps_ann.npy')
bigbog14_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF14_P90temps_ann.npy')
bigbog15_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF15_P90temps_ann.npy')

bigbog11_15_ann_90 = (bigbog11_ann_90 + bigbog12_ann_90 + bigbog13_ann_90 + bigbog14_ann_90 + bigbog15_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11-15_P90temps_ann.npy', bigbog11_15_ann_90)

bigbog11_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11_P99temps_ann.npy')
bigbog12_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF12_P99temps_ann.npy')
bigbog13_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF13_P99temps_ann.npy')
bigbog14_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF14_P99temps_ann.npy')
bigbog15_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF15_P99temps_ann.npy')

bigbog11_15_ann_99 = (bigbog11_ann_99 + bigbog12_ann_99 + bigbog13_ann_99 + bigbog14_ann_99 + bigbog15_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11-15_P99temps_ann.npy', bigbog11_15_ann_99)

bigbog11_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11_P90temps_wet.npy')
bigbog12_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF12_P90temps_wet.npy')
bigbog13_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF13_P90temps_wet.npy')
bigbog14_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF14_P90temps_wet.npy')
bigbog15_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF15_P90temps_wet.npy')

bigbog11_15_wet_90 = (bigbog11_wet_90 + bigbog12_wet_90 + bigbog13_wet_90 + bigbog14_wet_90 + bigbog15_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11-15_P90temps_wet.npy', bigbog11_15_wet_90)

bigbog11_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11_P99temps_wet.npy')
bigbog12_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF12_P99temps_wet.npy')
bigbog13_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF13_P99temps_wet.npy')
bigbog14_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF14_P99temps_wet.npy')
bigbog15_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF15_P99temps_wet.npy')

bigbog11_15_wet_99 = (bigbog11_wet_99 + bigbog12_wet_99 + bigbog13_wet_99 + bigbog14_wet_99 + bigbog15_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11-15_P99temps_wet.npy', bigbog11_15_wet_99)

bigbog11_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11_P90temps_dry.npy')
bigbog12_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF12_P90temps_dry.npy')
bigbog13_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF13_P90temps_dry.npy')
bigbog14_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF14_P90temps_dry.npy')
bigbog15_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF15_P90temps_dry.npy')

bigbog11_15_dry_90 = (bigbog11_dry_90 + bigbog12_dry_90 + bigbog13_dry_90 + bigbog14_dry_90 + bigbog15_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11-15_P90temps_dry.npy', bigbog11_15_dry_90)

bigbog11_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11_P99temps_dry.npy')
bigbog12_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF12_P99temps_dry.npy')
bigbog13_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF13_P99temps_dry.npy')
bigbog14_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF14_P99temps_dry.npy')
bigbog15_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF15_P99temps_dry.npy')

bigbog11_15_dry_99 = (bigbog11_dry_99 + bigbog12_dry_99 + bigbog13_dry_99 + bigbog14_dry_99 + bigbog15_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF11-15_P99temps_dry.npy', bigbog11_15_dry_99)

###########################################################################################################################
hilo11_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11_P90temps_ann.npy')
hilo12_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF12_P90temps_ann.npy')
hilo13_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF13_P90temps_ann.npy')
hilo14_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF14_P90temps_ann.npy')
hilo15_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF15_P90temps_ann.npy')

hilo11_15_ann_90 = (hilo11_ann_90 + hilo12_ann_90 + hilo13_ann_90 + hilo14_ann_90 + hilo15_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11-15_P90temps_ann.npy', hilo11_15_ann_90)

hilo11_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11_P99temps_ann.npy')
hilo12_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF12_P99temps_ann.npy')
hilo13_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF13_P99temps_ann.npy')
hilo14_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF14_P99temps_ann.npy')
hilo15_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF15_P99temps_ann.npy')

hilo11_15_ann_99 = (hilo11_ann_99 + hilo12_ann_99 + hilo13_ann_99 + hilo14_ann_99 + hilo15_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11-15_P99temps_ann.npy', hilo11_15_ann_99)

hilo11_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11_P90temps_wet.npy')
hilo12_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF12_P90temps_wet.npy')
hilo13_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF13_P90temps_wet.npy')
hilo14_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF14_P90temps_wet.npy')
hilo15_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF15_P90temps_wet.npy')

hilo11_15_wet_90 = (hilo11_wet_90 + hilo12_wet_90 + hilo13_wet_90 + hilo14_wet_90 + hilo15_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11-15_P90temps_wet.npy', hilo11_15_wet_90)

hilo11_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11_P99temps_wet.npy')
hilo12_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF12_P99temps_wet.npy')
hilo13_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF13_P99temps_wet.npy')
hilo14_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF14_P99temps_wet.npy')
hilo15_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF15_P99temps_wet.npy')

hilo11_15_wet_99 = (hilo11_wet_99 + hilo12_wet_99 + hilo13_wet_99 + hilo14_wet_99 + hilo15_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11-15_P99temps_wet.npy', hilo11_15_wet_99)

hilo11_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11_P90temps_dry.npy')
hilo12_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF12_P90temps_dry.npy')
hilo13_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF13_P90temps_dry.npy')
hilo14_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF14_P90temps_dry.npy')
hilo15_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF15_P90temps_dry.npy')

hilo11_15_dry_90 = (hilo11_dry_90 + hilo12_dry_90 + hilo13_dry_90 + hilo14_dry_90 + hilo15_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11-15_P90temps_dry.npy', hilo11_15_dry_90)

hilo11_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11_P99temps_dry.npy')
hilo12_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF12_P99temps_dry.npy')
hilo13_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF13_P99temps_dry.npy')
hilo14_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF14_P99temps_dry.npy')
hilo15_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF15_P99temps_dry.npy')

hilo11_15_dry_99 = (hilo11_dry_99 + hilo12_dry_99 + hilo13_dry_99 + hilo14_dry_99 + hilo15_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF11-15_P99temps_dry.npy', hilo11_15_dry_99)

##########################################################################################################################
lihue16_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16_P90temps_ann.npy')
lihue17_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF17_P90temps_ann.npy')
lihue18_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF18_P90temps_ann.npy')
lihue19_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF19_P90temps_ann.npy')
lihue20_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF20_P90temps_ann.npy')

lihue16_20_ann_90 = (lihue16_ann_90 + lihue17_ann_90 + lihue18_ann_90 + lihue19_ann_90 + lihue20_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16-20_P90temps_ann.npy', lihue16_20_ann_90)

lihue16_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16_P99temps_ann.npy')
lihue17_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF17_P99temps_ann.npy')
lihue18_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF18_P99temps_ann.npy')
lihue19_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF19_P99temps_ann.npy')
lihue20_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF20_P99temps_ann.npy')

lihue16_20_ann_99 = (lihue16_ann_99 + lihue17_ann_99 + lihue18_ann_99 + lihue19_ann_99 + lihue20_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16-20_P99temps_ann.npy', lihue16_20_ann_99)

lihue16_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16_P90temps_wet.npy')
lihue17_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF17_P90temps_wet.npy')
lihue18_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF18_P90temps_wet.npy')
lihue19_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF19_P90temps_wet.npy')
lihue20_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF20_P90temps_wet.npy')

lihue16_20_wet_90 = (lihue16_wet_90 + lihue17_wet_90 + lihue18_wet_90 + lihue19_wet_90 + lihue20_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16-20_P90temps_wet.npy', lihue16_20_wet_90)

lihue16_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16_P99temps_wet.npy')
lihue17_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF17_P99temps_wet.npy')
lihue18_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF18_P99temps_wet.npy')
lihue19_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF19_P99temps_wet.npy')
lihue20_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF20_P99temps_wet.npy')

lihue16_20_wet_99 = (lihue16_wet_99 + lihue17_wet_99 + lihue18_wet_99 + lihue19_wet_99 + lihue20_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16-20_P99temps_wet.npy', lihue16_20_wet_99)

lihue16_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16_P90temps_dry.npy')
lihue17_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF17_P90temps_dry.npy')
lihue18_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF18_P90temps_dry.npy')
lihue19_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF19_P90temps_dry.npy')
lihue20_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF20_P90temps_dry.npy')

lihue16_20_dry_90 = (lihue16_dry_90 + lihue17_dry_90 + lihue18_dry_90 + lihue19_dry_90 + lihue20_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16-20_P90temps_dry.npy', lihue16_20_dry_90)

lihue16_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16_P99temps_dry.npy')
lihue17_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF17_P99temps_dry.npy')
lihue18_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF18_P99temps_dry.npy')
lihue19_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF19_P99temps_dry.npy')
lihue20_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF20_P99temps_dry.npy')

lihue16_20_dry_99 = (lihue16_dry_99 + lihue17_dry_99 + lihue18_dry_99 + lihue19_dry_99 + lihue20_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_WRF16-20_P99temps_dry.npy', lihue16_20_dry_99)
############################################################################################################################
honolulu16_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16_P90temps_ann.npy')
honolulu17_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF17_P90temps_ann.npy')
honolulu18_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF18_P90temps_ann.npy')
honolulu19_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF19_P90temps_ann.npy')
honolulu20_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF20_P90temps_ann.npy')

honolulu16_20_ann_90 = (honolulu16_ann_90 + honolulu17_ann_90 + honolulu18_ann_90 + honolulu19_ann_90 + honolulu20_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16-20_P90temps_ann.npy', honolulu16_20_ann_90)

honolulu16_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16_P99temps_ann.npy')
honolulu17_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF17_P99temps_ann.npy')
honolulu18_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF18_P99temps_ann.npy')
honolulu19_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF19_P99temps_ann.npy')
honolulu20_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF20_P99temps_ann.npy')

honolulu16_20_ann_99 = (honolulu16_ann_99 + honolulu17_ann_99 + honolulu18_ann_99 + honolulu19_ann_99 + honolulu20_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16-20_P99temps_ann.npy', honolulu16_20_ann_99)

honolulu16_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16_P90temps_wet.npy')
honolulu17_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF17_P90temps_wet.npy')
honolulu18_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF18_P90temps_wet.npy')
honolulu19_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF19_P90temps_wet.npy')
honolulu20_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF20_P90temps_wet.npy')

honolulu16_20_wet_90 = (honolulu16_wet_90 + honolulu17_wet_90 + honolulu18_wet_90 + honolulu19_wet_90 + honolulu20_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16-20_P90temps_wet.npy', honolulu16_20_wet_90)

honolulu16_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16_P99temps_wet.npy')
honolulu17_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF17_P99temps_wet.npy')
honolulu18_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF18_P99temps_wet.npy')
honolulu19_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF19_P99temps_wet.npy')
honolulu20_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF20_P99temps_wet.npy')

honolulu16_20_wet_99 = (honolulu16_wet_99 + honolulu17_wet_99 + honolulu18_wet_99 + honolulu19_wet_99 + honolulu20_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16-20_P99temps_wet.npy', honolulu16_20_wet_99)

honolulu16_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16_P90temps_dry.npy')
honolulu17_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF17_P90temps_dry.npy')
honolulu18_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF18_P90temps_dry.npy')
honolulu19_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF19_P90temps_dry.npy')
honolulu20_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF20_P90temps_dry.npy')

honolulu16_20_dry_90 = (honolulu16_dry_90 + honolulu17_dry_90 + honolulu18_dry_90 + honolulu19_dry_90 + honolulu20_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16-20_P90temps_dry.npy', honolulu16_20_dry_90)

honolulu16_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16_P99temps_dry.npy')
honolulu17_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF17_P99temps_dry.npy')
honolulu18_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF18_P99temps_dry.npy')
honolulu19_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF19_P99temps_dry.npy')
honolulu20_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF20_P99temps_dry.npy')

honolulu16_20_dry_99 = (honolulu16_dry_99 + honolulu17_dry_99 + honolulu18_dry_99 + honolulu19_dry_99 + honolulu20_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_WRF16-20_P99temps_dry.npy', honolulu16_20_dry_99)
###################################################################################################################################
bigbog16_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16_P90temps_ann.npy')
bigbog17_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF17_P90temps_ann.npy')
bibog18_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF18_P90temps_ann.npy')
bigbog19_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF19_P90temps_ann.npy')
bigbog20_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF20_P90temps_ann.npy')

bigbog16_20_ann_90 = (bigbog16_ann_90 + bigbog17_ann_90 + bibog18_ann_90 + bigbog19_ann_90 + bigbog20_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16-20_P90temps_ann.npy', bigbog16_20_ann_90)

bigbog16_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16_P99temps_ann.npy')
bigbog17_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF17_P99temps_ann.npy')
bibog18_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF18_P99temps_ann.npy')
bigbog19_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF19_P99temps_ann.npy')
bigbog20_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF20_P99temps_ann.npy')

bigbog16_20_ann_99 = (bigbog16_ann_99 + bigbog17_ann_99 + bibog18_ann_99 + bigbog19_ann_99 + bigbog20_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16-20_P99temps_ann.npy', bigbog16_20_ann_99)

bigbog16_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16_P90temps_wet.npy')
bigbog17_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF17_P90temps_wet.npy')
bibog18_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF18_P90temps_wet.npy')
bigbog19_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF19_P90temps_wet.npy')
bigbog20_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF20_P90temps_wet.npy')

bigbog16_20_wet_90 = (bigbog16_wet_90 + bigbog17_wet_90 + bibog18_wet_90 + bigbog19_wet_90 + bigbog20_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16-20_P90temps_wet.npy', bigbog16_20_wet_90)

bigbog16_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16_P99temps_wet.npy')
bigbog17_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF17_P99temps_wet.npy')
bibog18_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF18_P99temps_wet.npy')
bigbog19_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF19_P99temps_wet.npy')
bigbog20_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF20_P99temps_wet.npy')

bigbog16_20_wet_99 = (bigbog16_wet_99 + bigbog17_wet_99 + bibog18_wet_99 + bigbog19_wet_99 + bigbog20_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16-20_P99temps_wet.npy', bigbog16_20_wet_99)

bigbog16_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16_P90temps_dry.npy')
bigbog17_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF17_P90temps_dry.npy')
bibog18_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF18_P90temps_dry.npy')
bigbog19_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF19_P90temps_dry.npy')
bigbog20_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF20_P90temps_dry.npy')

bigbog16_20_dry_90 = (bigbog16_dry_90 + bigbog17_dry_90 + bibog18_dry_90 + bigbog19_dry_90 + bigbog20_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16-20_P90temps_dry.npy', bigbog16_20_dry_90)

bigbog16_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16_P99temps_dry.npy')
bigbog17_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF17_P99temps_dry.npy')
bibog18_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF18_P99temps_dry.npy')
bigbog19_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF19_P99temps_dry.npy')
bigbog20_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF20_P99temps_dry.npy')

bigbog16_20_dry_99 = (bigbog16_dry_99 + bigbog17_dry_99 + bibog18_dry_99 + bigbog19_dry_99 + bigbog20_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_WRF16-20_P99temps_dry.npy', bigbog16_20_dry_99)

###########################################################################################################################
hilo16_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16_P90temps_ann.npy')
hilo17_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF17_P90temps_ann.npy')
hilo18_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF18_P90temps_ann.npy')
hilo19_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF19_P90temps_ann.npy')
hilo20_ann_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF20_P90temps_ann.npy')

hilo16_20_ann_90 = (hilo16_ann_90 + hilo17_ann_90 + hilo18_ann_90 + hilo19_ann_90 + hilo20_ann_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16-20_P90temps_ann.npy', hilo16_20_ann_90)

hilo16_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16_P99temps_ann.npy')
hilo17_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF17_P99temps_ann.npy')
hilo18_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF18_P99temps_ann.npy')
hilo19_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF19_P99temps_ann.npy')
hilo20_ann_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF20_P99temps_ann.npy')

hilo16_20_ann_99 = (hilo16_ann_99 + hilo17_ann_99 + hilo18_ann_99 + hilo19_ann_99 + hilo20_ann_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16-20_P99temps_ann.npy', hilo16_20_ann_99)

hilo16_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16_P90temps_wet.npy')
hilo17_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF17_P90temps_wet.npy')
hilo18_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF18_P90temps_wet.npy')
hilo19_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF19_P90temps_wet.npy')
hilo20_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF20_P90temps_wet.npy')

hilo16_20_wet_90 = (hilo16_wet_90 + hilo17_wet_90 + hilo18_wet_90 + hilo19_wet_90 + hilo20_wet_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16-20_P90temps_wet.npy', hilo16_20_wet_90)

hilo16_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16_P99temps_wet.npy')
hilo17_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF17_P99temps_wet.npy')
hilo18_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF18_P99temps_wet.npy')
hilo19_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF19_P99temps_wet.npy')
hilo20_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF20_P99temps_wet.npy')

hilo16_20_wet_99 = (hilo16_wet_99 + hilo17_wet_99 + hilo18_wet_99 + hilo19_wet_99 + hilo20_wet_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16-20_P99temps_wet.npy', hilo16_20_wet_99)

hilo16_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16_P90temps_dry.npy')
hilo17_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF17_P90temps_dry.npy')
hilo18_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF18_P90temps_dry.npy')
hilo19_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF19_P90temps_dry.npy')
hilo20_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF20_P90temps_dry.npy')

hilo16_20_dry_90 = (hilo16_dry_90 + hilo17_dry_90 + hilo18_dry_90 + hilo19_dry_90 + hilo20_dry_90) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16-20_P90temps_dry.npy', hilo16_20_dry_90)

hilo16_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16_P99temps_dry.npy')
hilo17_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF17_P99temps_dry.npy')
hilo18_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF18_P99temps_dry.npy')
hilo19_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF19_P99temps_dry.npy')
hilo20_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF20_P99temps_dry.npy')

hilo16_20_dry_99 = (hilo16_dry_99 + hilo17_dry_99 + hilo18_dry_99 + hilo19_dry_99 + hilo20_dry_99) / 5
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_WRF16-20_P99temps_dry.npy', hilo16_20_dry_99)

print('DONE')