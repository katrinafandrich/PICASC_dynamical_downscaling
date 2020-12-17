# Author: KMF
# Date: 6/29/19
# Write code to read in seasonal rainfall netcdf files
# Calculate averages for each WRF run (1-20) and plot results
# Do this for both wet and dry seasons by making adjustments

import xarray
import matplotlib.pyplot as plt
import numpy as np
import cftime as cft

WRF_LIST1 = ['WRF1','WRF2','WRF3','WRF4','WRF5'] # historical (1996-2005), negative PDO
WRF_LIST2 = ['WRF6','WRF7','WRF8','WRF9','WRF10'] # historical, positive PDO
WRF_LIST3 = ['WRF11','WRF12','WRF13','WRF14','WRF15'] # rcp85 (2026-2035), negative PDO
WRF_LIST4 = ['WRF16','WRF17','WRF18','WRF19','WRF20'] # rcp85, positive PDO

WRF_LIST = ['WRF1','WRF2','WRF3','WRF4','WRF5','WRF6','WRF7','WRF8','WRF9','WRF10']

# Calculate dry season rainfall
ens=WRF_LIST
rain_list=[]
n = 0
for ensemble in ens:
    DPATH1='/network/rit/lab/elisontimmlab_rit/kf835882/python/seasonal/'
    file1=ensemble+'_RAINNC_mjjaso_1996-2005.nc'
    data1 = xarray.open_dataset(DPATH1+file1)
    rain1=data1['RAINNC'].mean(dim='time')
    rain_list.append(rain1)
    print(file1)

    if n <= 4:
        DPATH2='/network/rit/lab/elisontimmlab_rit/kf835882/python/seasonal/neutral/'
        file2=ensemble+'neutral_RAINNC_mjjaso_1996-2005.nc'
        data2 = xarray.open_dataset(DPATH2+file2)
        rain2 = data2['RAINNC'].mean(dim='time')
        rain_list.append(rain2)
        n = n + 1
        print(file2)

sum1 = rain_list[0] + rain_list[1] + rain_list[2] + rain_list[3] + rain_list[4] + rain_list[5] + rain_list[6] + rain_list[7] + rain_list[8] + rain_list[9] + rain_list[10] + rain_list[11] + rain_list[12] + rain_list[13] + rain_list[14]
average1 = sum1 / 15

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/present-day_rain_dry.npy', average1)

'''
# Calculate wet season rainfall
ens=WRF_LIST
rain_list=[]
n = 0
for ensemble in ens:
    DPATH1='/network/rit/lab/elisontimmlab_rit/DATA/WRF/seasonal/'
    file1=ensemble+'_RAINNC_ndjfma_1997-2005.nc'
    data1 = xarray.open_dataset(DPATH1+file1)
    rain1=data1['RAINNC'].mean(dim='time')
    rain_list.append(rain1)
    print(file1)

    if n <= 4:
        ensemble = WRF_LIST[n]
        DPATH2='/network/rit/lab/elisontimmlab_rit/kf835882/python/seasonal/neutral/'
        file2=ensemble+'neutral_RAINNC_ndjfma_1997-2005.nc'
        data2 = xarray.open_dataset(DPATH2+file2)
        rain2=data2['RAINNC'].mean(dim='time')
        rain_list.append(rain2)
        n = n + 1
        print(file2)

sum2 = rain_list[0] + rain_list[1] + rain_list[2] + rain_list[3] + rain_list[4] + rain_list[5] + rain_list[6] + rain_list[7] + rain_list[8] + rain_list[9] + rain_list[10] + rain_list[11] + rain_list[12] + rain_list[13] + rain_list[14]
average2 = sum2 / 15

# Save these arrays in DATA/np_arrays not home directory
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/present-day_rain_wet.npy', average2)
'''
print('DONE')

