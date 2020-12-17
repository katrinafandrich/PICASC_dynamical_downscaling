# Script to do extreme rainfall analysis with Rainfall Atlas data
# KMF 5/19/20

from sympy import symbols
import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats

path = '/network/rit/lab/elisontimmlab_rit/DATA/HI_DAILY_PRECIP/'

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
        ncfile = 'Hawaii_Precipitation_UH_7.5sec_dTS'+year+'.nc'
        data = xarray.open_dataset(path+ncfile)

        if n >= 0 and n <= 119:
            lihue = data.precipitation[n,1392,210]
            if lihue > 0.1:
                lihue_wet[wet_days] = lihue
            else:
                lihue_wet[wet_days]= np.nan

            honolulu = data.precipitation[n,1098,833]
            if honolulu > 0.1:
                honolulu_wet[wet_days] = honolulu
            else:
                honolulu_wet[wet_days] = np.nan

            bigbog = data.precipitation[n,834,1655]
            if bigbog > 0.1:
                bigbog_wet[wet_days] = bigbog
            else:
                bigbog_wet[wet_days] = np.nan
            
            hilo = data.precipitation[n,387,2118]
            if hilo > 0.1:
                hilo_wet[wet_days] = hilo
            else:
                hilo_wet[wet_days] = np.nan

            wet_days = wet_days + 1
            print(year, n)

        elif n >= 120 and n <= 303:
            lihue = data.precipitation[n,1392,210]
            if lihue > 0.11:
                lihue_dry[dry_days] = lihue
            else:
                lihue_dry[dry_days] = np.nan

            honolulu = data.precipitation[n,1098,833]
            if honolulu > 0.1:
                honolulu_dry[dry_days] = honolulu
            else:
                honolulu_dry[dry_days] = np.nan

            bigbog = data.precipitation[n,834,1655]
            if bigbog > 0.1:
                bigbog_dry[dry_days] = bigbog
            else:
                bigbog_dry[dry_days] = np.nan
            
            hilo = data.precipitation[n,387,2118]
            if hilo > 0.1:
                hilo_dry[dry_days] = hilo
            else:
                hilo_dry[dry_days] = np.nan

            dry_days = dry_days + 1
            print(year, n)

        elif n >= 304 and n <= 364:
            lihue = data.precipitation[n,1392,210]
            if lihue > 0.1:
                lihue_wet[wet_days] = lihue
            else:
                lihue_wet[wet_days] = np.nan

            honolulu = data.precipitation[n,1098,833]
            if honolulu > 0.1:
                honolulu_wet[wet_days] = honolulu
            else:
                honolulu_wet[wet_days] = np.nan

            bigbog = data.precipitation[n,834,1655]
            if bigbog > 0.1:
                bigbog_wet[wet_days] = bigbog
            else:
                bigbog_wet[wet_days] = np.nan
            
            hilo = data.precipitation[n,387,2118]
            if hilo > 0.1:
                hilo_wet[wet_days] = hilo
            else:
                hilo_wet[wet_days] = np.nan

            wet_days = wet_days + 1
            print(year, n)

# Remove nan values from daily rainfall arrays
lihue_wet_new = lihue_wet[np.logical_not(np.isnan(lihue_wet))]
honolulu_wet_new = honolulu_wet[np.logical_not(np.isnan(honolulu_wet))]
bigbog_wet_new = bigbog_wet[np.logical_not(np.isnan(bigbog_wet))]
hilo_wet_new = hilo_wet[np.logical_not(np.isnan(hilo_wet))]

lihue_dry_new = lihue_dry[np.logical_not(np.isnan(lihue_dry))]
honolulu_dry_new = honolulu_dry[np.logical_not(np.isnan(honolulu_dry))]
bigbog_dry_new = bigbog_dry[np.logical_not(np.isnan(bigbog_dry))]
hilo_dry_new = hilo_dry[np.logical_not(np.isnan(hilo_dry))]

# Calculate 90th and 99th percentiles and save as np arrays
lihue_sorted = np.sort(lihue_wet_new)
lihue_wet_90 = lihue_sorted[-(round(len(lihue_sorted)*0.1)):]
lihue_wet_99 = lihue_sorted[-(round(len(lihue_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_wet_90_atlas.npy', lihue_wet_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_wet_99_atlas.npy', lihue_wet_99)

honolulu_sorted = np.sort(honolulu_wet_new)
honolulu_wet_90 = honolulu_sorted[-(round(len(honolulu_sorted)*0.1)):]
honolulu_wet_99 = honolulu_sorted[-(round(len(honolulu_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_wet_90_atlas.npy', honolulu_wet_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_wet_99_atlas.npy', honolulu_wet_99)

bigbog_sorted = np.sort(bigbog_wet_new)
bigbog_wet_90 = bigbog_sorted[-(round(len(bigbog_sorted)*0.1)):]
bigbog_wet_99 = bigbog_sorted[-(round(len(bigbog_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_wet_90_atlas.npy', bigbog_wet_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_wet_99_atlas.npy', bigbog_wet_99)

hilo_sorted = np.sort(hilo_wet_new)
hilo_wet_90 = hilo_sorted[-(round(len(hilo_sorted)*0.1)):]
hilo_wet_99 = hilo_sorted[-(round(len(hilo_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_wet_90_atlas.npy', hilo_wet_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_wet_99_atlas.npy', hilo_wet_99)

lihue_sorted = np.sort(lihue_dry_new)
lihue_dry_90 = lihue_sorted[-(round(len(lihue_sorted)*0.1)):]
lihue_dry_99 = lihue_sorted[-(round(len(lihue_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_dry_90_atlas.npy', lihue_dry_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_dry_99_atlas.npy', lihue_dry_99)

honolulu_sorted = np.sort(honolulu_dry_new)
honolulu_dry_90 = honolulu_sorted[-(round(len(honolulu_sorted)*0.1)):]
honolulu_dry_99 = honolulu_sorted[-(round(len(honolulu_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_dry_90_atlas.npy', honolulu_dry_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_dry_99_atlas.npy', honolulu_dry_99)

bigbog_sorted = np.sort(bigbog_dry_new)
bigbog_dry_90 = bigbog_sorted[-(round(len(bigbog_sorted)*0.1)):]
bigbog_dry_99 = bigbog_sorted[-(round(len(bigbog_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_dry_90_atlas.npy', bigbog_dry_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_dry_99_atlas.npy', bigbog_dry_99)

hilo_sorted = np.sort(hilo_dry_new)
hilo_dry_90 = hilo_sorted[-(round(len(hilo_sorted)*0.1)):]
hilo_dry_99 = hilo_sorted[-(round(len(hilo_sorted)*0.01)):]
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_dry_90_atlas.npy', hilo_dry_90)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_dry_99_atlas.npy', hilo_dry_99)

print('DONE')




