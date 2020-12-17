# Calculate linear regression for temp vs. height plots
# 4/30/20

from sympy import symbols
import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim
from scipy import stats

path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'

#SW down
SW_wet1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/SWDOWN_wet_WRF1_5.npy')
SW_wet2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/SWDOWN_wet_WRF6_10.npy')
SW_wet3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/SWDOWN_wet_WRF11_15.npy')
SW_wet4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/SWDOWN_wet_WRF16_20.npy')

SW_dry1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/SWDOWN_dry_WRF1_5.npy')
SW_dry2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/SWDOWN_dry_WRF6_10.npy')
SW_dry3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/SWDOWN_dry_WRF11_15.npy')
SW_dry4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/SWDOWN_dry_WRF16_20.npy')
'''
dif_SW1_wet = SW_wet4 - SW_wet2
dif_SW2_wet = SW_wet3 - SW_wet1
dif_SW3_wet = SW_wet4 - SW_wet3
dif_SW4_wet = SW_wet2 - SW_wet1

dif_SW1_dry = SW_dry4 - SW_dry2
dif_SW2_dry = SW_dry3 - SW_dry1
dif_SW3_dry = SW_dry4 - SW_dry3
dif_SW4_dry = SW_dry2 - SW_dry1
'''
'''
# Open reference file
dataDIR = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/WRF1/'
filename = 'wrfout_d02_monthly_mean_1997_01.nc'
DS = xarray.open_dataset(dataDIR+filename)

height = DS.HGT[0,:,:]
landmask = DS.LANDMASK[0,:,:]

# Convert all arrays into lists for linregress
temp_dif = landmask

temp_difs = []
for k in range(0,240):
    for l in range(0,360):
        temp_difs.append(temp_dif[k,l])

np.save(path+'landmask_list.npy', temp_difs)
'''
'''

heights = np.load(path+'heights_list.npy')
longman_heights = np.load(path+'longman_hgt_list.npy')

T1wet = np.load(path+'Tdif1wet_list.npy')
T2wet = np.load(path+'Tdif2wet_list.npy')
T3wet = np.load(path+'Tdif3wet_list.npy')
T4wet = np.load(path+'Tdif4wet_list.npy')

T1dry = np.load(path+'Tdif1dry_list.npy')
T2dry = np.load(path+'Tdif2dry_list.npy')
T3dry = np.load(path+'Tdif3dry_list.npy')
T4dry = np.load(path+'Tdif4dry_list.npy')

q1wet = np.load(path+'dif_q1_wet_list.npy')
q2wet = np.load(path+'dif_q2_wet_list.npy')
q3wet = np.load(path+'dif_q3_wet_list.npy')
q4wet = np.load(path+'dif_q4_wet_list.npy')

q1dry = np.load(path+'dif_q1_dry_list.npy')
q2dry = np.load(path+'dif_q2_dry_list.npy')
q3dry = np.load(path+'dif_q3_dry_list.npy')
q4dry = np.load(path+'dif_q4_dry_list.npy')

bin1 = np.where((heights > 0) & (heights <= 500))
bin2 = np.where((heights >= 500) & (heights <= 1000))
bin3 = np.where((heights >= 1000) & (heights <= 1500))
bin4 = np.where((heights >= 1500) & (heights <= 2000))
bin5 = np.where((heights >= 2000) & (heights <= 2500))
bin6 = np.where((heights >= 2500) & (heights <= 3000))
bin7 = np.where((heights >= 3000) & (heights <= 3500))
bin8 = np.where((heights >= 3500) & (heights <= 4044))

# Plot temp changes as a function of height for wet season

f = plt.figure(figsize=(10,8), dpi = 150)
ax = f.add_subplot(241)
ax2 = f.add_subplot(242)
ax3 = f.add_subplot(243)
ax4 = f.add_subplot(244)
ax5 = f.add_subplot(245)
ax6 = f.add_subplot(246)
ax7 = f.add_subplot(247)
ax8 = f.add_subplot(248)

plt.subplot(2, 4, 1)
plt.scatter(heights, T1wet, s = 6, color = '#007fff', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(T1wet[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(T1wet[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(T1wet[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(T1wet[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(T1wet[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(T1wet[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(T1wet[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(T1wet[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(T1wet[bin1]),np.mean(T1wet[bin2]),np.mean(T1wet[bin3]),np.mean(T1wet[bin4]),np.mean(T1wet[bin5]),np.mean(T1wet[bin6]),np.mean(T1wet[bin7]),np.mean(T1wet[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.around(np.arange(T1wet.min(),1.4,0.1), decimals = 2))
ax.set_xticklabels('')
plt.ylabel('(K)')
plt.title('(a) wet    Fut(+) - Pres(+)', fontsize = 12)

plt.subplot(2, 4, 2)
plt.scatter(heights, T2wet, s = 6, color = '#007fff', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(T2wet[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(T2wet[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(T2wet[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(T2wet[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(T2wet[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(T2wet[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(T2wet[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(T2wet[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(T2wet[bin1]),np.mean(T2wet[bin2]),np.mean(T2wet[bin3]),np.mean(T2wet[bin4]),np.mean(T2wet[bin5]),np.mean(T2wet[bin6]),np.mean(T2wet[bin7]),np.mean(T2wet[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.around(np.arange(T1wet.min(),1.4,0.1), decimals = 2))
ax2.set_xticklabels('')
ax2.set_yticklabels('')
plt.title('(b) wet    Fut(-) - Pres(-)', fontsize = 12)

plt.subplot(2, 4, 5)
plt.scatter(heights, T3wet, s = 6, color = '#007fff', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(T3wet[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(T3wet[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(T3wet[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(T3wet[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(T3wet[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(T3wet[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(T3wet[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(T3wet[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(T3wet[bin1]),np.mean(T3wet[bin2]),np.mean(T3wet[bin3]),np.mean(T3wet[bin4]),np.mean(T3wet[bin5]),np.mean(T3wet[bin6]),np.mean(T3wet[bin7]),np.mean(T3wet[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.around(np.arange(T3wet.min(),0.1,0.05), decimals = 2))
plt.xlabel('Height (m)')
plt.ylabel('(K)')
plt.title('(e) wet    Fut(+) - Fut(-)', fontsize = 12)

plt.subplot(2, 4, 6)
plt.scatter(heights, T4wet, s = 6, color = '#007fff', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(T4wet[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(T4wet[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(T4wet[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(T4wet[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(T4wet[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(T4wet[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(T4wet[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(T4wet[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(T4wet[bin1]),np.mean(T4wet[bin2]),np.mean(T4wet[bin3]),np.mean(T4wet[bin4]),np.mean(T4wet[bin5]),np.mean(T4wet[bin6]),np.mean(T4wet[bin7]),np.mean(T4wet[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.around(np.arange(T3wet.min(),0.1,0.05), decimals = 2))
ax6.set_yticklabels('')
plt.xlabel('Height (m)')
plt.title('(f) wet    Pres(+) - Pres(-)', fontsize = 12)

plt.subplot(2, 4, 3)
plt.scatter(heights, T1dry, s = 6, color = '#dc143c', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(T1dry[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(T1dry[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(T1dry[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(T1dry[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(T1dry[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(T1dry[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(T1dry[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(T1dry[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(T1dry[bin1]),np.mean(T1dry[bin2]),np.mean(T1dry[bin3]),np.mean(T1dry[bin4]),np.mean(T1dry[bin5]),np.mean(T1dry[bin6]),np.mean(T1dry[bin7]),np.mean(T1dry[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.around(np.arange(T1wet.min(),1.4,0.1), decimals = 2))
ax3.set_xticklabels('')
ax3.set_yticklabels('')
plt.title('(c) dry    Fut(+) - Pres(+)', fontsize = 12)

plt.subplot(2, 4, 4)
plt.scatter(heights, T2dry, s = 6, color = '#dc143c', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(T2dry[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(T2dry[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(T2dry[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(T2dry[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(T2dry[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(T2dry[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(T2dry[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(T2dry[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(T2dry[bin1]),np.mean(T2dry[bin2]),np.mean(T2dry[bin3]),np.mean(T2dry[bin4]),np.mean(T2dry[bin5]),np.mean(T2dry[bin6]),np.mean(T2dry[bin7]),np.mean(T2dry[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.around(np.arange(T1wet.min(),1.4,0.1), decimals = 2))
ax4.set_xticklabels('')
ax4.set_yticklabels('')
plt.title('(d) dry    Fut(-) - Pres(-)', fontsize = 12)

plt.subplot(2, 4, 7)
plt.scatter(heights, T3dry, s = 6, color = '#dc143c', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(T3dry[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(T3dry[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(T3dry[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(T3dry[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(T3dry[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(T3dry[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(T3dry[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(T3dry[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(T3dry[bin1]),np.mean(T3dry[bin2]),np.mean(T3dry[bin3]),np.mean(T3dry[bin4]),np.mean(T3dry[bin5]),np.mean(T3dry[bin6]),np.mean(T3dry[bin7]),np.mean(T3dry[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.around(np.arange(T3wet.min(),0.1,0.05), decimals = 2))
ax7.set_yticklabels('')
plt.xlabel('Height (m)')
plt.title('(g) dry    Fut(+) - Fut(-)', fontsize = 12)

plt.subplot(2, 4, 8)
plt.scatter(heights, T4dry, s = 6, color = '#dc143c', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(T4dry[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(T4dry[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(T4dry[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(T4dry[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(T4dry[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(T4dry[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(T4dry[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(T4dry[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(T4dry[bin1]),np.mean(T4dry[bin2]),np.mean(T4dry[bin3]),np.mean(T4dry[bin4]),np.mean(T4dry[bin5]),np.mean(T4dry[bin6]),np.mean(T4dry[bin7]),np.mean(T4dry[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.around(np.arange(T3wet.min(),0.1,0.05), decimals = 2))
ax8.set_yticklabels('')
plt.xlabel('Height (m)')
plt.title('(h) dry    Pres(+) - Pres(-)', fontsize = 12)

f.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.01, wspace=0.01)
plt.tight_layout()
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/T_vs_hgt_dry.png')
plt.show()

'''
landmask = np.load(path+'landmask_list.npy')

heights = np.load(path+'heights_list.npy')

q1wet = np.load(path+'dif_SW1_wet_list.npy')
q2wet = np.load(path+'dif_SW2_wet_list.npy')
q3wet = np.load(path+'dif_SW3_wet_list.npy')
q4wet = np.load(path+'dif_SW4_wet_list.npy')

q1wet = np.ma.masked_where(landmask == False, q1wet)
q2wet = np.ma.masked_where(landmask == False, q2wet)
q3wet = np.ma.masked_where(landmask == False, q3wet)
q4wet = np.ma.masked_where(landmask == False, q4wet)

q1dry = np.load(path+'dif_SW1_dry_list.npy')
q2dry = np.load(path+'dif_SW2_dry_list.npy')
q3dry = np.load(path+'dif_SW3_dry_list.npy')
q4dry = np.load(path+'dif_SW4_dry_list.npy')

q1dry = np.ma.masked_where(landmask == False, q1dry)
q2dry = np.ma.masked_where(landmask == False, q2dry)
q3dry = np.ma.masked_where(landmask == False, q3dry)
q4dry = np.ma.masked_where(landmask == False, q4dry)

bin1 = np.where((heights > 0) & (heights <= 500))
bin2 = np.where((heights >= 500) & (heights <= 1000))
bin3 = np.where((heights >= 1000) & (heights <= 1500))
bin4 = np.where((heights >= 1500) & (heights <= 2000))
bin5 = np.where((heights >= 2000) & (heights <= 2500))
bin6 = np.where((heights >= 2500) & (heights <= 3000))
bin7 = np.where((heights >= 3000) & (heights <= 3500))
bin8 = np.where((heights >= 3500) & (heights <= 4044))

# Plot specific humidity changes as a function of height for wet season

f = plt.figure(figsize=(10,8), dpi = 150)
ax = f.add_subplot(241)
ax2 = f.add_subplot(242)
ax3 = f.add_subplot(243)
ax4 = f.add_subplot(244)
ax5 = f.add_subplot(245)
ax6 = f.add_subplot(246)
ax7 = f.add_subplot(247)
ax8 = f.add_subplot(248)

plt.subplot(2, 4, 1)
plt.scatter(heights, q1wet, s = 6, color = '#007fff', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(q1wet[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(q1wet[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(q1wet[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(q1wet[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(q1wet[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(q1wet[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(q1wet[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(q1wet[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(q1wet[bin1]),np.mean(q1wet[bin2]),np.mean(q1wet[bin3]),np.mean(q1wet[bin4]),np.mean(q1wet[bin5]),np.mean(q1wet[bin6]),np.mean(q1wet[bin7]),np.mean(q1wet[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
#plt.yticks(np.arange(0,0.01))
ax.set_xticklabels('')
plt.ylabel('(W/m^2)')
plt.title('(a) wet    Fut(+) - Pres(+)', fontsize = 12)

plt.subplot(2, 4, 2)
plt.scatter(heights, q2wet, s = 6, color = '#007fff', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(q2wet[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(q2wet[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(q2wet[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(q2wet[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(q2wet[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(q2wet[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(q2wet[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(q2wet[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(q2wet[bin1]),np.mean(q2wet[bin2]),np.mean(q2wet[bin3]),np.mean(q2wet[bin4]),np.mean(q2wet[bin5]),np.mean(q2wet[bin6]),np.mean(q2wet[bin7]),np.mean(q2wet[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
#plt.yticks(np.arange(0,0.01))
ax2.set_xticklabels('')
ax2.set_yticklabels('')
plt.title('(b) wet    Fut(-) - Pres(-)', fontsize = 12)

plt.subplot(2, 4, 5)
plt.scatter(heights, q3wet, s = 6, color = '#007fff', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(q3wet[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(q3wet[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(q3wet[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(q3wet[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(q3wet[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(q3wet[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(q3wet[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(q3wet[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(q3wet[bin1]),np.mean(q3wet[bin2]),np.mean(q3wet[bin3]),np.mean(q3wet[bin4]),np.mean(q3wet[bin5]),np.mean(q3wet[bin6]),np.mean(q3wet[bin7]),np.mean(q3wet[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
#plt.yticks(np.arange(-0.0005,0.0005))
plt.xlabel('Height (m)')
plt.ylabel('(W/m^2)')
plt.title('(e) wet    Fut(+) - Fut(-)', fontsize = 12)

plt.subplot(2, 4, 6)
plt.scatter(heights, q4wet, s = 6, color = '#007fff', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(q4wet[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(q4wet[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(q4wet[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(q4wet[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(q4wet[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(q4wet[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(q4wet[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(q4wet[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(q4wet[bin1]),np.mean(q4wet[bin2]),np.mean(q4wet[bin3]),np.mean(q4wet[bin4]),np.mean(q4wet[bin5]),np.mean(q4wet[bin6]),np.mean(q4wet[bin7]),np.mean(q4wet[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
#plt.yticks(np.arange(-0.0005,0.0005))
ax6.set_yticklabels('')
plt.xlabel('Height (m)')
plt.title('(f) wet    Pres(+) - Pres(-)', fontsize = 12)

plt.subplot(2, 4, 3)
plt.scatter(heights, q1dry, s = 6, color = '#dc143c', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(q1dry[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(q1dry[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(q1dry[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(q1dry[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(q1dry[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(q1dry[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(q1dry[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(q1dry[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(q1dry[bin1]),np.mean(q1dry[bin2]),np.mean(q1dry[bin3]),np.mean(q1dry[bin4]),np.mean(q1dry[bin5]),np.mean(q1dry[bin6]),np.mean(q1dry[bin7]),np.mean(q1dry[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
#plt.yticks(np.arange(0,0.01))
ax3.set_xticklabels('')
ax3.set_yticklabels('')
plt.title('(c) dry    Fut(+) - Pres(+)', fontsize = 12)

plt.subplot(2, 4, 4)
plt.scatter(heights, q2dry, s = 6, color = '#dc143c', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(q2dry[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(q2dry[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(q2dry[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(q2dry[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(q2dry[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(q2dry[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(q2dry[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(q2dry[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(q2dry[bin1]),np.mean(q2dry[bin2]),np.mean(q2dry[bin3]),np.mean(q2dry[bin4]),np.mean(q2dry[bin5]),np.mean(q2dry[bin6]),np.mean(q2dry[bin7]),np.mean(q2dry[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
#plt.yticks(np.arange(0,0.01))
ax4.set_xticklabels('')
ax4.set_yticklabels('')
plt.title('(d) dry    Fut(-) - Pres(-)', fontsize = 12)

plt.subplot(2, 4, 7)
plt.scatter(heights, q3dry, s = 6, color = '#dc143c', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(q3dry[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(q3dry[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(q3dry[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(q3dry[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(q3dry[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(q3dry[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(q3dry[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(q3dry[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(q3dry[bin1]),np.mean(q3dry[bin2]),np.mean(q3dry[bin3]),np.mean(q3dry[bin4]),np.mean(q3dry[bin5]),np.mean(q3dry[bin6]),np.mean(q3dry[bin7]),np.mean(q3dry[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
#plt.yticks(np.arange(-0.0005,0.0005))
ax7.set_yticklabels('')
plt.xlabel('Height (m)')
plt.title('(g) dry    Fut(+) - Fut(-)', fontsize = 12)

plt.subplot(2, 4, 8)
plt.scatter(heights, q4dry, s = 6, color = '#dc143c', marker = 'o', label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(q4dry[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(q4dry[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(q4dry[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(q4dry[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(q4dry[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(q4dry[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(q4dry[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(q4dry[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(q4dry[bin1]),np.mean(q4dry[bin2]),np.mean(q4dry[bin3]),np.mean(q4dry[bin4]),np.mean(q4dry[bin5]),np.mean(q4dry[bin6]),np.mean(q4dry[bin7]),np.mean(q4dry[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
#plt.yticks(np.arange(-0.0005,0.0005))
ax8.set_yticklabels('')
plt.xlabel('Height (m)')
plt.title('(h) dry    Pres(+) - Pres(-)', fontsize = 12)

f.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.01, wspace=0.01)
plt.tight_layout()
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/T_vs_hgt_dry.png')
plt.show()

'''
heights = np.load(path+'heights_list.npy')
longman_heights = np.load(path+'longman_hgt_list.npy')
mask = np.load(path+'mask_list.npy')

#T_wet_pres = np.load(path+'t_wet_pres_list.npy')
#T_wet = np.load(path+'Twet_list.npy')
#T_dry_pres = np.load(path+'t_dry_pres_list.npy')
#T_dry = np.load(path+'Tdry_list.npy')

#T_wet = np.ma.masked_where(mask == True, T_wet)
#T_dry = np.ma.masked_where(mask == True, T_dry)

bin1 = np.where((heights > 0) & (heights <= 500))
bin2 = np.where((heights >= 500) & (heights <= 1000))
bin3 = np.where((heights >= 1000) & (heights <= 1500))
bin4 = np.where((heights >= 1500) & (heights <= 2000))
bin5 = np.where((heights >= 2000) & (heights <= 2500))
bin6 = np.where((heights >= 2500) & (heights <= 3000))
bin7 = np.where((heights >= 3000) & (heights <= 3500))
bin8 = np.where((heights >= 3500) & (heights <= 4044))

longman_bin1 = np.where((longman_heights > 0) & (longman_heights <= 500))
longman_bin2 = np.where((longman_heights >= 500) & (longman_heights <= 1000))
longman_bin3 = np.where((longman_heights >= 1000) & (longman_heights <= 1500))
longman_bin4 = np.where((longman_heights >= 1500) & (longman_heights <= 2000))
longman_bin5 = np.where((longman_heights >= 2000) & (longman_heights <= 2500))
longman_bin6 = np.where((longman_heights >= 2500) & (longman_heights <= 3000))
longman_bin7 = np.where((longman_heights >= 3000) & (longman_heights <= 3500))
longman_bin8 = np.where((longman_heights >= 3500) & (longman_heights <= 4044))


# Plot SATs as a function of height for WRF and OBS

f = plt.figure(figsize=(10,8), dpi = 150)
ax = f.add_subplot(211)
ax2 = f.add_subplot(212)

plt.subplot(2, 1, 1)
plt.scatter(longman_heights, T_wet, s = 6, color = '#007fff', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.scatter(heights, T_wet_pres, s = 6, color = '#dc143c', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.plot(np.mean(longman_heights[longman_bin1]), np.mean(T_wet[longman_bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin2]), np.mean(T_wet[longman_bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin3]), np.mean(T_wet[longman_bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin4]), np.mean(T_wet[longman_bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin5]), np.mean(T_wet[longman_bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin6]), np.mean(T_wet[longman_bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin7]), np.mean(T_wet[longman_bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin8]), np.mean(T_wet[longman_bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(longman_heights[longman_bin1]), np.mean(longman_heights[longman_bin2]), np.mean(longman_heights[longman_bin3]), np.mean(longman_heights[longman_bin4]), np.mean(longman_heights[longman_bin5]), np.mean(longman_heights[longman_bin6]), np.mean(longman_heights[longman_bin7]), np.mean(longman_heights[longman_bin8])]
temps_new = [np.mean(T_wet[longman_bin1]),np.mean(T_wet[longman_bin2]),np.mean(T_wet[longman_bin3]),np.mean(T_wet[longman_bin4]),np.mean(T_wet[longman_bin5]),np.mean(T_wet[longman_bin6]),np.mean(T_wet[longman_bin7]),np.mean(T_wet[longman_bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.plot(np.mean(heights[bin1]), np.mean(T_wet_pres[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(T_wet_pres[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(T_wet_pres[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(T_wet_pres[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(T_wet_pres[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(T_wet_pres[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(T_wet_pres[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(T_wet_pres[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(T_wet_pres[bin1]),np.mean(T_wet_pres[bin2]),np.mean(T_wet_pres[bin3]),np.mean(T_wet_pres[bin4]),np.mean(T_wet_pres[bin5]),np.mean(T_wet_pres[bin6]),np.mean(T_wet_pres[bin7]),np.mean(T_wet_pres[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.arange(275,300.5,5))
ax.set_xticklabels('')
plt.ylabel('(K)')
plt.title('(a) wet', fontsize = 12)

plt.subplot(2, 1, 2)
plt.subplot(2, 1, 2)
plt.scatter(longman_heights, T_dry, s = 6, color = '#007fff', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.scatter(heights, T_dry_pres, s = 6, color = '#dc143c', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(T_dry_pres[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(T_dry_pres[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(T_dry_pres[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(T_dry_pres[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(T_dry_pres[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(T_dry_pres[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(T_dry_pres[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(T_dry_pres[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(T_dry_pres[bin1]),np.mean(T_dry_pres[bin2]),np.mean(T_dry_pres[bin3]),np.mean(T_dry_pres[bin4]),np.mean(T_dry_pres[bin5]),np.mean(T_dry_pres[bin6]),np.mean(T_dry_pres[bin7]),np.mean(T_dry_pres[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.plot(np.mean(longman_heights[longman_bin1]), np.mean(T_dry[longman_bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin2]), np.mean(T_dry[longman_bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin3]), np.mean(T_dry[longman_bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin4]), np.mean(T_dry[longman_bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin5]), np.mean(T_dry[longman_bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin6]), np.mean(T_dry[longman_bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin7]), np.mean(T_dry[longman_bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin8]), np.mean(T_dry[longman_bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(longman_heights[longman_bin1]), np.mean(longman_heights[longman_bin2]), np.mean(longman_heights[longman_bin3]), np.mean(longman_heights[longman_bin4]), np.mean(longman_heights[longman_bin5]), np.mean(longman_heights[longman_bin6]), np.mean(longman_heights[longman_bin7]), np.mean(longman_heights[longman_bin8])]
temps_new = [np.mean(T_dry[longman_bin1]),np.mean(T_dry[longman_bin2]),np.mean(T_dry[longman_bin3]),np.mean(T_dry[longman_bin4]),np.mean(T_dry[longman_bin5]),np.mean(T_dry[longman_bin6]),np.mean(T_dry[longman_bin7]),np.mean(T_dry[longman_bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.arange(275,300.5,5))
plt.xlabel('Height (m)')
plt.ylabel('(K)')
plt.title('(b) dry', fontsize = 12)


f.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.01, wspace=0.01)
plt.tight_layout()
plt.show()
'''
'''
atlas_wet90 = np.load(path+'atlas_wet90_list.npy')
atlas_wet99 = np.load(path+'atlas_wet99_list.npy')
atlas_dry90 = np.load(path+'atlas_dry90_list.npy')
atlas_dry99 = np.load(path+'atlas_dry99_list.npy')

atlas_wet90 = np.ma.masked_where(mask == True, atlas_wet90)
atlas_wet99 = np.ma.masked_where(mask == True, atlas_wet99)
atlas_dry90 = np.ma.masked_where(mask == True, atlas_dry90)
atlas_dry99 = np.ma.masked_where(mask == True, atlas_dry99)

wrf_wet90_pres = np.load(path+'wrf_wet90_pres_list.npy')
wrf_wet99_pres = np.load(path+'wrf_wet99_pres_list.npy')
wrf_dry90_pres = np.load(path+'wrf_dry90_pres_list.npy')
wrf_dry99_pres = np.load(path+'wrf_dry99_pres_list.npy')

# Plot extreme temps as a function of height for WRF and OBS

f = plt.figure(figsize=(10,8), dpi = 150)
ax = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

plt.subplot(2, 2, 1)
plt.scatter(longman_heights, atlas_wet90, s = 6, color = '#007fff', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.scatter(heights, wrf_wet90_pres, s = 6, color = '#dc143c', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.plot(np.mean(longman_heights[longman_bin1]), np.mean(atlas_wet90[longman_bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin2]), np.mean(atlas_wet90[longman_bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin3]), np.mean(atlas_wet90[longman_bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin4]), np.mean(atlas_wet90[longman_bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin5]), np.mean(atlas_wet90[longman_bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin6]), np.mean(atlas_wet90[longman_bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin7]), np.mean(atlas_wet90[longman_bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin8]), np.mean(atlas_wet90[longman_bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(longman_heights[longman_bin1]), np.mean(longman_heights[longman_bin2]), np.mean(longman_heights[longman_bin3]), np.mean(longman_heights[longman_bin4]), np.mean(longman_heights[longman_bin5]), np.mean(longman_heights[longman_bin6]), np.mean(longman_heights[longman_bin7]), np.mean(longman_heights[longman_bin8])]
temps_new = [np.mean(atlas_wet90[longman_bin1]),np.mean(atlas_wet90[longman_bin2]),np.mean(atlas_wet90[longman_bin3]),np.mean(atlas_wet90[longman_bin4]),np.mean(atlas_wet90[longman_bin5]),np.mean(atlas_wet90[longman_bin6]),np.mean(atlas_wet90[longman_bin7]),np.mean(atlas_wet90[longman_bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.plot(np.mean(heights[bin1]), np.mean(wrf_wet90_pres[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(wrf_wet90_pres[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(wrf_wet90_pres[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(wrf_wet90_pres[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(wrf_wet90_pres[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(wrf_wet90_pres[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(wrf_wet90_pres[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(wrf_wet90_pres[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(wrf_wet90_pres[bin1]),np.mean(wrf_wet90_pres[bin2]),np.mean(wrf_wet90_pres[bin3]),np.mean(wrf_wet90_pres[bin4]),np.mean(wrf_wet90_pres[bin5]),np.mean(wrf_wet90_pres[bin6]),np.mean(wrf_wet90_pres[bin7]),np.mean(wrf_wet90_pres[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.arange(280,310.5,5))
ax.set_xticklabels('')
plt.ylabel('(K)')
plt.title('(a) wet        T90', fontsize = 12)

plt.subplot(2, 2, 2)
plt.scatter(longman_heights, atlas_dry90, s = 6, color = '#007fff', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.scatter(heights, wrf_dry90_pres, s = 6, color = '#dc143c', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(wrf_dry90_pres[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(wrf_dry90_pres[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(wrf_dry90_pres[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(wrf_dry90_pres[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(wrf_dry90_pres[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(wrf_dry90_pres[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(wrf_dry90_pres[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(wrf_dry90_pres[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(wrf_dry90_pres[bin1]),np.mean(wrf_dry90_pres[bin2]),np.mean(wrf_dry90_pres[bin3]),np.mean(wrf_dry90_pres[bin4]),np.mean(wrf_dry90_pres[bin5]),np.mean(wrf_dry90_pres[bin6]),np.mean(wrf_dry90_pres[bin7]),np.mean(wrf_dry90_pres[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.plot(np.mean(longman_heights[longman_bin1]), np.mean(atlas_dry90[longman_bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin2]), np.mean(atlas_dry90[longman_bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin3]), np.mean(atlas_dry90[longman_bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin4]), np.mean(atlas_dry90[longman_bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin5]), np.mean(atlas_dry90[longman_bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin6]), np.mean(atlas_dry90[longman_bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin7]), np.mean(atlas_dry90[longman_bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin8]), np.mean(atlas_dry90[longman_bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(longman_heights[longman_bin1]), np.mean(longman_heights[longman_bin2]), np.mean(longman_heights[longman_bin3]), np.mean(longman_heights[longman_bin4]), np.mean(longman_heights[longman_bin5]), np.mean(longman_heights[longman_bin6]), np.mean(longman_heights[longman_bin7]), np.mean(longman_heights[longman_bin8])]
temps_new = [np.mean(atlas_dry90[longman_bin1]),np.mean(atlas_dry90[longman_bin2]),np.mean(atlas_dry90[longman_bin3]),np.mean(atlas_dry90[longman_bin4]),np.mean(atlas_dry90[longman_bin5]),np.mean(atlas_dry90[longman_bin6]),np.mean(atlas_dry90[longman_bin7]),np.mean(atlas_dry90[longman_bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.arange(280,310.5,5))
ax2.set_xticklabels('')
ax2.set_yticklabels('')
#plt.xlabel('Height (m)')
plt.ylabel('')
plt.title('(b) dry        T90', fontsize = 12)


plt.subplot(2, 2, 3)
plt.scatter(longman_heights, atlas_wet99, s = 6, color = '#007fff', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.scatter(heights, wrf_wet99_pres, s = 6, color = '#dc143c', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.plot(np.mean(longman_heights[longman_bin1]), np.mean(atlas_wet99[longman_bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin2]), np.mean(atlas_wet99[longman_bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin3]), np.mean(atlas_wet99[longman_bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin4]), np.mean(atlas_wet99[longman_bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin5]), np.mean(atlas_wet99[longman_bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin6]), np.mean(atlas_wet99[longman_bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin7]), np.mean(atlas_wet99[longman_bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin8]), np.mean(atlas_wet99[longman_bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(longman_heights[longman_bin1]), np.mean(longman_heights[longman_bin2]), np.mean(longman_heights[longman_bin3]), np.mean(longman_heights[longman_bin4]), np.mean(longman_heights[longman_bin5]), np.mean(longman_heights[longman_bin6]), np.mean(longman_heights[longman_bin7]), np.mean(longman_heights[longman_bin8])]
temps_new = [np.mean(atlas_wet99[longman_bin1]),np.mean(atlas_wet99[longman_bin2]),np.mean(atlas_wet99[longman_bin3]),np.mean(atlas_wet99[longman_bin4]),np.mean(atlas_wet99[longman_bin5]),np.mean(atlas_wet99[longman_bin6]),np.mean(atlas_wet99[longman_bin7]),np.mean(atlas_wet99[longman_bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.plot(np.mean(heights[bin1]), np.mean(wrf_wet99_pres[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(wrf_wet99_pres[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(wrf_wet99_pres[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(wrf_wet99_pres[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(wrf_wet99_pres[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(wrf_wet99_pres[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(wrf_wet99_pres[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(wrf_wet99_pres[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(wrf_wet99_pres[bin1]),np.mean(wrf_wet99_pres[bin2]),np.mean(wrf_wet99_pres[bin3]),np.mean(wrf_wet99_pres[bin4]),np.mean(wrf_wet99_pres[bin5]),np.mean(wrf_wet99_pres[bin6]),np.mean(wrf_wet99_pres[bin7]),np.mean(wrf_wet99_pres[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.arange(280,310.5,5))
plt.xlabel('Height (m)')
plt.ylabel('(K)')
plt.title('(c) wet        T99', fontsize = 12)

plt.subplot(2, 2, 4)
plt.scatter(longman_heights, atlas_dry99, s = 6, color = '#007fff', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.scatter(heights, wrf_dry99_pres, s = 6, color = '#dc143c', marker = 'o', alpha = 0.5, label = 'scatter plt')
plt.plot(np.mean(heights[bin1]), np.mean(wrf_dry99_pres[bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin2]), np.mean(wrf_dry99_pres[bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin3]), np.mean(wrf_dry99_pres[bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin4]), np.mean(wrf_dry99_pres[bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin5]), np.mean(wrf_dry99_pres[bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin6]), np.mean(wrf_dry99_pres[bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin7]), np.mean(wrf_dry99_pres[bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(heights[bin8]), np.mean(wrf_dry99_pres[bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(heights[bin1]), np.mean(heights[bin2]), np.mean(heights[bin3]), np.mean(heights[bin4]), np.mean(heights[bin5]), np.mean(heights[bin6]), np.mean(heights[bin7]), np.mean(heights[bin8])]
temps_new = [np.mean(wrf_dry99_pres[bin1]),np.mean(wrf_dry99_pres[bin2]),np.mean(wrf_dry99_pres[bin3]),np.mean(wrf_dry99_pres[bin4]),np.mean(wrf_dry99_pres[bin5]),np.mean(wrf_dry99_pres[bin6]),np.mean(wrf_dry99_pres[bin7]),np.mean(wrf_dry99_pres[bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.plot(np.mean(longman_heights[longman_bin1]), np.mean(atlas_dry99[longman_bin1]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin2]), np.mean(atlas_dry99[longman_bin2]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin3]), np.mean(atlas_dry99[longman_bin3]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin4]), np.mean(atlas_dry99[longman_bin4]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin5]), np.mean(atlas_dry99[longman_bin5]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin6]), np.mean(atlas_dry99[longman_bin6]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin7]), np.mean(atlas_dry99[longman_bin7]), color='black', marker='x', markersize = 4)
plt.plot(np.mean(longman_heights[longman_bin8]), np.mean(atlas_dry99[longman_bin8]), color='black', marker='x', markersize = 4)
heights_new = [np.mean(longman_heights[longman_bin1]), np.mean(longman_heights[longman_bin2]), np.mean(longman_heights[longman_bin3]), np.mean(longman_heights[longman_bin4]), np.mean(longman_heights[longman_bin5]), np.mean(longman_heights[longman_bin6]), np.mean(longman_heights[longman_bin7]), np.mean(longman_heights[longman_bin8])]
temps_new = [np.mean(atlas_dry99[longman_bin1]),np.mean(atlas_dry99[longman_bin2]),np.mean(atlas_dry99[longman_bin3]),np.mean(atlas_dry99[longman_bin4]),np.mean(atlas_dry99[longman_bin5]),np.mean(atlas_dry99[longman_bin6]),np.mean(atlas_dry99[longman_bin7]),np.mean(atlas_dry99[longman_bin8])]
plt.plot(heights_new, temps_new, c = 'black', linewidth = 1)
plt.yticks(np.arange(280,310.5,5))
ax4.set_yticklabels('')
plt.xlabel('Height (m)')
plt.ylabel('')
plt.title('(d) dry       T99', fontsize = 12)

f.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.01, wspace=0.01)
plt.tight_layout()
plt.show()
'''