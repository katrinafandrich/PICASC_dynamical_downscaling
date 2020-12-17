# Author: KMF
# 1/23/20
# Write script to run monte carlo simulations

import pandas as pd
import numpy as np
import xarray
from netCDF4 import Dataset
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load data
path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'

# Rainfall
rain_wet1 = np.load(path+'rain_wet_wrf1_5.npy')
rain_wet2 = np.load(path+'rain_wet_wrf6_10.npy')
rain_wet3 = np.load(path+'rain_wet_wrf11_15.npy')
rain_wet4 = np.load(path+'rain_wet_wrf16_20.npy')

rain_dry1 = np.load(path+'rain_dry_wrf1_5.npy')
rain_dry2 = np.load(path+'rain_dry_wrf6_10.npy')
rain_dry3 = np.load(path+'rain_dry_wrf11_15.npy')
rain_dry4 = np.load(path+'rain_dry_wrf16_20.npy')

P_wet1 = rain_wet1 / 180
P_wet2 = rain_wet2 / 180
P_wet3 = rain_wet3 / 180
P_wet4 = rain_wet4 / 180

P_dry1 = rain_dry1 / 180
P_dry2 = rain_dry2 / 180
P_dry3 = rain_dry3 / 180
P_dry4 = rain_dry4 / 180
'''
# Raindays
RD_wet1 =np.load(path+'wet_rain_days_WRF1-5.npy')
RD_wet2 =np.load(path+'wet_rain_days_WRF6-10.npy')
RD_wet3 =np.load(path+'wet_rain_days_WRF11-15.npy')
RD_wet4 =np.load(path+'wet_rain_days_WRF16-20.npy')

RD_dry1 =np.load(path+'dry_rain_days_WRF1-5.npy')
RD_dry2 =np.load(path+'dry_rain_days_WRF6-10.npy')
RD_dry3 =np.load(path+'dry_rain_days_WRF11-15.npy')
RD_dry4 = np.load(path+'dry_rain_days_WRF16-20.npy')

# Consecutive Dry days
CDD_wet1 = np.load(path+'CDD_wet_WRF1-5.npy')
CDD_wet2 = np.load(path+'CDD_wet_WRF6-10.npy')
CDD_wet3 = np.load(path+'CDD_wet_WRF11-15.npy')
CDD_wet4 = np.load(path+'CDD_wet_WRF16-20.npy')

CDD_dry1 = np.load(path+'CDD_dry_WRF1-5.npy')
CDD_dry2 = np.load(path+'CDD_dry_WRF6-10.npy')
CDD_dry3 = np.load(path+'CDD_dry_WRF11-15.npy')
CDD_dry4 = np.load(path+'CDD_dry_WRF16-20.npy')

#Temps
temps_wet1 = np.load(path+'temps_wet_wrf1_5.npy')
temps_wet2 = np.load(path+'temps_wet_wrf6_10.npy')
temps_wet3 = np.load(path+'temps_wet_wrf11_15.npy')
temps_wet4 = np.load(path+'temps_wet_wrf16_20.npy')

temps_dry1 = np.load(path+'temps_dry_wrf1_5.npy')
temps_dry2 = np.load(path+'temps_dry_wrf6_10.npy')
temps_dry3 = np.load(path+'temps_dry_wrf11_15.npy')
temps_dry4 = np.load(path+'temps_dry_wrf16_20.npy')

T99_wet1 = np.load(path+'WRF1-5_99th_temps_wet.npy')
T99_wet2 = np.load(path+'WRF6-10_99th_temps_wet.npy')
T99_wet3 = np.load(path+'WRF11-15_99th_temps_wet.npy')
T99_wet4 = np.load(path+'WRF16-20_99th_temps_wet.npy')

T99_dry1 = np.load(path+'WRF1-5_99th_temps_dry.npy')
T99_dry2 = np.load(path+'WRF6-10_99th_temps_dry.npy')
T99_dry3 = np.load(path+'WRF11-15_99th_temps_dry.npy')
T99_dry4 = np.load(path+'WRF16-20_99th_temps_dry.npy')

#MFC
mfc_wet1 = np.load(path+'mfc_wet_wrf1_5.npy')
mfc_wet2 = np.load(path+'mfc_wet_wrf6_10.npy')
mfc_wet3 = np.load(path+'mfc_wet_wrf11_15.npy')
mfc_wet4 = np.load(path+'mfc_wet_wrf16_20.npy')

mfc_dry1 = np.load(path+'mfc_dry_wrf1_5.npy')
mfc_dry2 = np.load(path+'mfc_dry_wrf6_10.npy')
mfc_dry3 = np.load(path+'mfc_dry_wrf11_15.npy')
mfc_dry4 = np.load(path+'mfc_dry_wrf16_20.npy')

#Evap
E_wet1 = np.load(path+'E_wet1_5.npy')
E_wet2 = np.load(path+'E_wet6_10.npy')
E_wet3 = np.load(path+'E_wet11_15.npy')
E_wet4 = np.load(path+'E_wet16_20.npy')

E_dry1 = np.load(path+'E_dry1_5.npy')
E_dry2 = np.load(path+'E_dry6_10.npy')
E_dry3 = np.load(path+'E_dry11_15.npy')
E_dry4 = np.load(path+'E_dry16_20.npy')


# Extreme Precip
P90_wet1 = np.load(path+'wrf1-5_90th_wet.npy')
P90_wet2 = np.load(path+'wrf6-10_90th_wet.npy')
P90_wet3 = np.load(path+'wrf11-15_90th_wet.npy')
P90_wet4 = np.load(path+'wrf16-20_90th_wet.npy')

P90_dry1 = np.load(path+'wrf1-5_90th_dry.npy')
P90_dry2 = np.load(path+'wrf6-10_90th_dry.npy')
P90_dry3 = np.load(path+'wrf11-15_90th_dry.npy')
P90_dry4 = np.load(path+'wrf16-20_90th_dry.npy')

P99_wet1 = np.load(path+'wrf1-5_99th_wet.npy')
P99_wet2 = np.load(path+'wrf6-10_99th_wet.npy')
P99_wet3 = np.load(path+'wrf11-15_99th_wet.npy')
P99_wet4 = np.load(path+'wrf16-20_99th_wet.npy')

P99_dry1 = np.load(path+'wrf1-5_99th_dry.npy')
P99_dry2 = np.load(path+'wrf6-10_99th_dry.npy')
P99_dry3 = np.load(path+'wrf11-15_99th_dry.npy')
P99_dry4 = np.load(path+'wrf16-20_99th_dry.npy')
'''
# These are the values we are testing for significance

dif_P1_wet = rain_wet4 - rain_wet2
dif_P2_wet = rain_wet3 - rain_wet1
dif_P3_wet = rain_wet4 - rain_wet3
dif_P4_wet = rain_wet2 - rain_wet1

dif_P1_dry = rain_dry4 - rain_dry2
dif_P2_dry = rain_dry3 - rain_dry1
dif_P3_dry = rain_dry4 - rain_dry3
dif_P4_dry = rain_dry2 - rain_dry1
'''
dif_T1_wet = temps_wet4 - temps_wet2
dif_T2_wet = temps_wet3 - temps_wet1
dif_T3_wet = temps_wet4 - temps_wet3
dif_T4_wet = temps_wet2 - temps_wet1

dif_T1_dry = temps_dry4 - temps_dry2
dif_T2_dry = temps_dry3 - temps_dry1
dif_T3_dry = temps_dry4 - temps_dry3
dif_T4_dry = temps_dry2 - temps_dry1

dif_E1_wet = E_wet4 - E_wet2
dif_E2_wet = E_wet3 - E_wet1
dif_E3_wet = E_wet4 - E_wet3
dif_E4_wet = E_wet2 - E_wet1

dif_E1_dry = E_dry4 - E_dry2
dif_E2_dry = E_dry3 - E_dry1
dif_E3_dry = E_dry4 - E_dry3
dif_E4_dry = E_dry2 - E_dry1

dif_PE1_wet = P_wet4 - E_wet2
dif_PE2_wet = P_wet3 - E_wet1
dif_PE3_wet = P_wet4 - E_wet3
dif_PE4_wet = P_wet2 - E_wet1

dif_PE1_dry = P_dry4 - E_dry2
dif_PE2_dry = P_dry3 - E_dry1
dif_PE3_dry = P_dry4 - E_dry3
dif_PE4_dry = P_dry2 - E_dry1

dif_P901_wet = P90_wet4 - P90_wet2
dif_P902_wet = P90_wet3 - P90_wet1
dif_P903_wet = P90_wet4 - P90_wet3
dif_P904_wet = P90_wet2 - P90_wet1

dif_P901_dry = P90_dry4 - P90_dry2
dif_P902_dry = P90_dry3 - P90_dry1
dif_P903_dry = P90_dry4 - P90_dry3
dif_P904_dry = P90_dry2 - P90_dry1

dif_P991_wet = P99_wet4 - P99_wet2
dif_P992_wet = P99_wet3 - P99_wet1
dif_P993_wet = P99_wet4 - P99_wet3
dif_P994_wet = P99_wet2 - P99_wet1

dif_P991_dry = P99_dry4 - P99_dry2
dif_P992_dry = P99_dry3 - P99_dry1
dif_P993_dry = P99_dry4 - P99_dry3
dif_P994_dry = P99_dry2 - P99_dry1

dif_RD1_wet = RD_wet4 - RD_wet2
dif_RD2_wet = RD_wet3 - RD_wet1
dif_RD3_wet = RD_wet4 - RD_wet3
dif_RD4_wet = RD_wet2 - RD_wet1

dif_RD1_dry = RD_dry4 - RD_dry2
dif_RD2_dry = RD_dry3 - RD_dry1
dif_RD3_dry = RD_dry4 - RD_dry3
dif_RD4_dry = RD_dry2 - RD_dry1

dif_CDD1_wet = CDD_wet4 - CDD_wet2
dif_CDD2_wet = CDD_wet3 - CDD_wet1
dif_CDD3_wet = CDD_wet4 - CDD_wet3
dif_CDD4_wet = CDD_wet2 - CDD_wet1

dif_CDD1_dry = CDD_dry4 - CDD_dry2
dif_CDD2_dry = CDD_dry3 - CDD_dry1
dif_CDD3_dry = CDD_dry4 - CDD_dry3
dif_CDD4_dry = CDD_dry2 - CDD_dry1

dif_mfc1_wet = mfc_wet4 - mfc_wet2
dif_mfc2_wet = mfc_wet3 - mfc_wet1
dif_mfc3_wet = mfc_wet4 - mfc_wet3
dif_mfc4_wet = mfc_wet2 - mfc_wet1

dif_mfc1_dry = mfc_dry4 - mfc_dry2
dif_mfc2_dry = mfc_dry3 - mfc_dry1
dif_mfc3_dry = mfc_dry4 - mfc_dry3
dif_mfc4_dry = mfc_dry2 - mfc_dry1
'''
###############################################################################
###############################################################################

# Select data and we want to test
actual_data = dif_P1_wet
name = 'dif_P1_wet'

# Select season
wet = 'wet'
dry = 'dry'
season = wet

# Select number of simulations
ens_all = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
num_simulations = 1000

# Loop through simulations
all_simulations = np.zeros((num_simulations,240,360))
for i in range(num_simulations):
    random = np.random.choice(ens_all, 10)
    group1 = np.random.choice(random, 5)
    group2 = np.random.choice(random, 5)

    print(group1)
    print(group2)
    print('===============================')

    sum1 = 0
    for n in range(0,5):
        sim = str(group1[n])
        WRF_ens = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/rain_'+str(season)+'_wrf'+sim+'.npy')
        #WRF_ens = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/temps_'+str(season)+'_wrf'+sim+'.npy')
        #WRF_ens = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/E_'+str(season)+'_wrf'+sim+'.npy')
        #WRF_ens = np.load(path+'wrf'+sim+'_99th_'+str(season)+'.npy')
        #WRF_ens = np.load(path+str(season)+'_rain_days_WRF'+sim+'.npy')
        #WRF_ens = np.load(path+'CDD_'+str(season)+'_WRF'+sim+'_.npy')
        #WRF_ens = np.load(path+'WRF'+sim+'_99th_temps_'+str(season)+'.npy')
        #WRF_ens = np.load(path+'mfc_'+season+'_wrfWRF'+sim+'.npy')
        sum1 = sum1 + WRF_ens
    avg1 = sum1 / 5

    sum2 = 0
    for n in range(0,5):
        sim = str(group2[n])
        WRF_ens = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/rain_'+str(season)+'_wrf'+sim+'.npy')
        #WRF_ens = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/temps_'+str(season)+'_wrf'+sim+'.npy')
        #WRF_ens = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/E_'+str(season)+'_wrf'+sim+'.npy')
        #WRF_ens = np.load(path+'wrf'+sim+'_99th_'+str(season)+'.npy')
        #WRF_ens = np.load(path+str(season)+'_rain_days_WRF'+sim+'.npy')
        #WRF_ens = np.load(path+'CDD_'+str(season)+'_WRF'+sim+'_.npy')
        #WRF_ens = np.load(path+'WRF'+sim+'_99th_temps_'+str(season)+'.npy')
        #WRF_ens = np.load(path+'mfc_'+season+'_wrfWRF'+sim+'.npy')
        sum2 = sum2 + WRF_ens
    avg2 = sum2 / 5

    dif = avg2 - avg1
    all_simulations[i] = dif  

# Function to find nearest value in an array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Calculate p-values by hand
p_values = np.zeros((240,360))
sig_p = np.zeros((240,360))
sig_gridpoints = 0
total_gridpoints = 0
for k in range(0,240):
    for l in range(0,360):
        total_gridpoints = total_gridpoints + 1
        dsort = np.sort(all_simulations[:,k,l])
        index = find_nearest(dsort, actual_data[k,l])
        prob = (index / num_simulations)
        p_values[k,l] = prob
        if (p_values[k,l] < 0.05):
            sig_p[k,l] = p_values[k,l]
            sig_gridpoints = sig_gridpoints + 1
        elif (p_values[k,l] > 0.95):
            sig_p[k,l] = p_values[k,l]
            sig_gridpoints = sig_gridpoints + 1
        else:
            sig_p[k,l] = np.nan
percent = (sig_gridpoints / total_gridpoints) * 100
print(percent)

#np.save(path+'p_vals_'+str(name)+'.npy', sig_p)

print('DONE')

plt.hist(all_simulations[:,64,293], bins = 25)
plt.xlabel('Rainfall (mm)')
plt.ylabel('Frequency')
plt.title('Hilo    Fut(+) - Pres(+)')
plt.show()



