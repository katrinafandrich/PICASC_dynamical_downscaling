# Script to do extreme temp analysis with Rainfall Atlas data
# KMF 5/19/20

from sympy import symbols
import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats

path = '/network/rit/lab/elisontimmlab_rit/DATA/HI_DAILY/tmax_tmin_precip_gridded/'

years = ['1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']

lihue_wet = np.zeros(shape=[(181*25)])
honolulu_wet = np.zeros(shape = [(181*25)])
bigbog_wet = np.zeros(shape = [(181*25)])
hilo_wet = np.zeros(shape = [(181*25)])

lihue_dry = np.zeros(shape=[(184*25)])
honolulu_dry = np.zeros(shape = [(184*25)])
bigbog_dry = np.zeros(shape = [(184*25)])
hilo_dry = np.zeros(shape = [(184*25)])

wet_days = 0
dry_days = 0

for year in years:
    for n in range(0,365):
        ncfile = 'UH_'+year+'.nc4'
        data = xarray.open_dataset(path+ncfile)
        temp_C = data.tmax[:] # in degrees C
        #temp_F = (temp*1.8) + 32 # Convert to degrees F

        if n >= 0 and n <= 119:
            lihue = temp_C[n,1392,210]
            lihue_wet[wet_days] = lihue

            honolulu = temp_C[n,1098,833]
            honolulu_wet[wet_days] = honolulu

            bigbog = temp_C[n,834,1655]
            bigbog_wet[wet_days] = bigbog
            
            hilo = temp_C[n,387,2118]
            hilo_wet[wet_days] = hilo

            wet_days = wet_days + 1
            print(year, n)

        elif n >= 120 and n <= 303:
            lihue = temp_C[n,1392,210]
            lihue_dry[dry_days] = lihue

            honolulu = temp_C[n,1098,833]
            honolulu_dry[dry_days] = honolulu

            bigbog = temp_C[n,834,1655]
            bigbog_dry[dry_days] = bigbog
            
            hilo = temp_C[n,387,2118]
            hilo_dry[dry_days] = hilo

            dry_days = dry_days + 1
            print(year, n)

        elif n >= 304 and n <= 364:
            lihue = temp_C[n,1392,210]
            lihue_wet[wet_days] = lihue

            honolulu = temp_C[n,1098,833]
            honolulu_wet[wet_days] = honolulu

            bigbog = temp_C[n,834,1655]
            bigbog_wet[wet_days] = bigbog
            
            hilo = temp_C[n,387,2118]
            hilo_wet[wet_days] = hilo

            wet_days = wet_days + 1
            print(year, n)


# Calculate 90th and 99th percentiles and save as np arrays
lihue_wet = (lihue_wet*1.8) + 32 # Convert to degrees F
lihue_sorted = np.sort(lihue_wet)
lihue_wet_90 = lihue_sorted[-(round(len(lihue_sorted)*0.1)):]
lihue_wet_99 = lihue_sorted[-(round(len(lihue_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_wet_90_atlas_temps.npy', lihue_wet_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_wet_99_atlas_temps.npy', lihue_wet_99)

honolulu_wet = (honolulu_wet*1.8) + 32
honolulu_sorted = np.sort(honolulu_wet)
honolulu_wet_90 = honolulu_sorted[-(round(len(honolulu_sorted)*0.1)):]
honolulu_wet_99 = honolulu_sorted[-(round(len(honolulu_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_wet_90_atlas_temps.npy', honolulu_wet_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_wet_99_atlas_temps.npy', honolulu_wet_99)

bigbog_wet = (bigbog_wet*1.8) + 32
bigbog_sorted = np.sort(bigbog_wet)
bigbog_wet_90 = bigbog_sorted[-(round(len(bigbog_sorted)*0.1)):]
bigbog_wet_99 = bigbog_sorted[-(round(len(bigbog_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_wet_90_atlas_temps.npy', bigbog_wet_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_wet_99_atlas_temps.npy', bigbog_wet_99)

hilo_wet = (hilo_wet*1.8) + 32
hilo_sorted = np.sort(hilo_wet)
hilo_wet_90 = hilo_sorted[-(round(len(hilo_sorted)*0.1)):]
hilo_wet_99 = hilo_sorted[-(round(len(hilo_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_wet_90_atlas_temps.npy', hilo_wet_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_wet_99_atlas_temps.npy', hilo_wet_99)

lihue_dry = (lihue_dry*1.8) + 32
lihue_sorted = np.sort(lihue_dry)
lihue_dry_90 = lihue_sorted[-(round(len(lihue_sorted)*0.1)):]
lihue_dry_99 = lihue_sorted[-(round(len(lihue_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_dry_90_atlas_temps.npy', lihue_dry_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_dry_99_atlas_temps.npy', lihue_dry_99)

honolulu_dry = (honolulu_dry*1.8) + 32
honolulu_sorted = np.sort(honolulu_dry)
honolulu_dry_90 = honolulu_sorted[-(round(len(honolulu_sorted)*0.1)):]
honolulu_dry_99 = honolulu_sorted[-(round(len(honolulu_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_dry_90_atlas_temps.npy', honolulu_dry_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_dry_99_atlas_temps.npy', honolulu_dry_99)

bigbog_dry = (bigbog_dry*1.8) + 32
bigbog_sorted = np.sort(bigbog_dry)
bigbog_dry_90 = bigbog_sorted[-(round(len(bigbog_sorted)*0.1)):]
bigbog_dry_99 = bigbog_sorted[-(round(len(bigbog_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_dry_90_atlas_temps.npy', bigbog_dry_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_dry_99_atlas_temps.npy', bigbog_dry_99)

hilo_dry = (hilo_dry*1.8) + 32
hilo_sorted = np.sort(hilo_dry)
hilo_dry_90 = hilo_sorted[-(round(len(hilo_sorted)*0.1)):]
hilo_dry_99 = hilo_sorted[-(round(len(hilo_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_dry_90_atlas_temps.npy', hilo_dry_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_dry_99_atlas_temps.npy', hilo_dry_99)

print('DONE')