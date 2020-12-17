# Script to plot extreme precip histograms

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
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

# Plot histograms for four stations
lihue_present = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_daily_present.npy')
honolulu_present = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_daily_present.npy')
bigbog_present = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_daily_present.npy')
hilo_present = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_daily_present.npy')

lihue_future = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_daily_future.npy')
honolulu_future = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_daily_future.npy')
bigbog_future = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_daily_future.npy')
hilo_future = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_daily_future.npy')

# Wet season
lihue1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue1-5_wet.npy')
lihue2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue6-10_wet.npy')
lihue3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue11-15_wet.npy')
lihue4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_wet.npy')

honolulu1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu1-5_wet.npy')
honolulu2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu6-10_wet.npy')
honolulu3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu11-15_wet.npy')
honolulu4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_wet.npy')

bigbog1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog1-5_wet.npy')
bigbog2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog6-10_wet.npy')
bigbog3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog11-15_wet.npy')
bigbog4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_wet.npy')

hilo1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo1-5_wet.npy')
hilo2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo6-10_wet.npy')
hilo3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo11-15_wet.npy')
hilo4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_wet.npy')

# Dry season
lihue1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue1-5_dry.npy')
lihue2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue6-10_dry.npy')
lihue3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue11-15_dry.npy')
lihue4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_dry.npy')

honolulu1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu1-5_dry.npy')
honolulu2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu6-10_dry.npy')
honolulu3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu11-15_dry.npy')
honolulu4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_dry.npy')

bigbog1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog1-5_dry.npy')
bigbog2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog6-10_dry.npy')
bigbog3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog11-15_dry.npy')
bigbog4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_dry.npy')

hilo1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo1-5_dry.npy')
hilo2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo6-10_dry.npy')
hilo3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo11-15_dry.npy')
hilo4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_dry.npy')

# Plot PDFs for 90th and 99th percentiles by station
lihue1_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue1-5_90_wet.npy')
lihue2_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue6-10_90_wet.npy')
lihue3_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue11-15_90_wet.npy')
lihue4_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_90_wet.npy')

lihue1_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue1-5_99_wet.npy')
lihue2_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue6-10_99_wet.npy')
lihue3_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue11-15_99_wet.npy')
lihue4_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_99_wet.npy')

honolulu1_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu1-5_90_wet.npy')
honolulu2_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu6-10_90_wet.npy')
honolulu3_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu11-15_90_wet.npy')
honolulu4_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_90_wet.npy')

honolulu1_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu1-5_99_wet.npy')
honolulu2_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu6-10_99_wet.npy')
honolulu3_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu11-15_99_wet.npy')
honolulu4_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_99_wet.npy')

bigbog1_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog1-5_90_wet.npy')
bigbog2_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog6-10_90_wet.npy')
bigbog3_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog11-15_90_wet.npy')
bigbog4_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_90_wet.npy')

bigbog1_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog1-5_99_wet.npy')
bigbog2_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog6-10_99_wet.npy')
bigbog3_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog11-15_99_wet.npy')
bigbog4_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_99_wet.npy')

hilo1_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo1-5_90_wet.npy')
hilo2_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo6-10_90_wet.npy')
hilo3_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo11-15_90_wet.npy')
hilo4_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_90_wet.npy')

hilo1_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo1-5_99_wet.npy')
hilo2_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo6-10_99_wet.npy')
hilo3_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo11-15_99_wet.npy')
hilo4_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_99_wet.npy')

lihue1_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue1-5_90_dry.npy')
lihue2_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue6-10_90_dry.npy')
lihue3_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue11-15_90_dry.npy')
lihue4_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_90_dry.npy')

lihue1_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue1-5_99_dry.npy')
lihue2_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue6-10_99_dry.npy')
lihue3_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue11-15_99_dry.npy')
lihue4_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_99_dry.npy')

honolulu1_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu1-5_90_dry.npy')
honolulu2_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu6-10_90_dry.npy')
honolulu3_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu11-15_90_dry.npy')
honolulu4_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_90_dry.npy')

honolulu1_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu1-5_99_dry.npy')
honolulu2_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu6-10_99_dry.npy')
honolulu3_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu11-15_99_dry.npy')
honolulu4_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_99_dry.npy')

bigbog1_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog1-5_90_dry.npy')
bigbog2_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog6-10_90_dry.npy')
bigbog3_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog11-15_90_dry.npy')
bigbog4_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_90_dry.npy')

bigbog1_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog1-5_99_dry.npy')
bigbog2_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog6-10_99_dry.npy')
bigbog3_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog11-15_99_dry.npy')
bigbog4_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_99_dry.npy')

hilo1_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo1-5_90_dry.npy')
hilo2_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo6-10_90_dry.npy')
hilo3_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo11-15_90_dry.npy')
hilo4_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_90_dry.npy')

hilo1_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo1-5_99_dry.npy')
hilo2_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo6-10_99_dry.npy')
hilo3_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo11-15_99_dry.npy')
hilo4_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_99_dry.npy')

# Rainfall Atlas station data
lihue_wet_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_wet_90_atlas.npy')
lihue_wet_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_wet_99_atlas.npy')

honolulu_wet_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_wet_90_atlas.npy')
honolulu_wet_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_wet_99_atlas.npy')

bigbog_wet_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_wet_90_atlas.npy')
bigbog_wet_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_wet_99_atlas.npy')

hilo_wet_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_wet_90_atlas.npy')
hilo_wet_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_wet_99_atlas.npy')

lihue_dry_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_dry_90_atlas.npy')
lihue_dry_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_dry_99_atlas.npy')

honolulu_dry_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_dry_90_atlas.npy')
honolulu_dry_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_dry_99_atlas.npy')

bigbog_dry_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_dry_90_atlas.npy')
bigbog_dry_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_dry_99_atlas.npy')

hilo_dry_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_dry_90_atlas.npy')
hilo_dry_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_dry_99_atlas.npy')

'''
# Histogram number of days > 50mm for present and future
f = plt.figure(figsize=(10,8), dpi = 80)
ax = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

plt.subplot(2, 2, 1)
plt.hist([lihue_present, lihue_future], stacked=True, normed=True)
plt.ylabel('Normalized # of Days')
plt.title('Lihue', fontsize = 10)

plt.subplot(2, 2, 2)
plt.hist([honolulu_present, honolulu_future], stacked=True, normed=True, label = ['Present','Future'])
plt.legend()
plt.title('Honolulu', fontsize = 10)

plt.subplot(2, 2, 3)
plt.hist([bigbog_present, bigbog_future], stacked=True, normed=True)
plt.ylabel('Normalized # of Days')
plt.xlabel('Daily Rainfall (mm)')
plt.title('Big Bog', fontsize = 10)

plt.subplot(2, 2, 4)
plt.hist([hilo_present, hilo_future], stacked=True, normed=True)
plt.ylabel('Normalized # of Days')
plt.xlabel('Daily Rainfall (mm)')
plt.title('Hilo', fontsize = 10)

f.set_constrained_layout_pads(w_pad=0.125, h_pad=0.125, hspace=0.9, wspace=0.5)
plt.tight_layout()
#f.suptitle('')
plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/extreme_histogram_present.png')
plt.show()

# Plot histograms for wet season
f = plt.figure(figsize=(10,8), dpi = 80)
ax = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

names = ['Present, PDO(-)', 'Present, PDO(+)', 'Future, PDO(-)', 'Future, PDO(+)']

plt.subplot(2, 2, 1)
plt.hist([lihue1_wet, lihue2_wet, lihue3_wet, lihue4_wet], stacked=True, normed=True)
plt.ylabel('Normalized # of Days')
plt.title('Lihue', fontsize = 10)

plt.subplot(2, 2, 2)
plt.hist([honolulu1_wet, honolulu2_wet, honolulu3_wet, honolulu4_wet], stacked=True, normed=True, label = names)
plt.legend()
plt.title('Honolulu', fontsize = 10)

plt.subplot(2, 2, 3)
plt.hist([bigbog1_wet, bigbog2_wet, bigbog3_wet, bigbog4_wet], stacked=True, normed=True)
plt.xlabel('Daily Rainfall (mm)')
plt.ylabel('Normalized # of Days')
plt.title('Big Bog', fontsize = 10)

plt.subplot(2, 2, 4)
plt.hist([hilo1_wet, hilo2_wet, hilo3_wet, hilo4_wet], stacked=True, normed=True)
plt.xlabel('Daily Rainfall (mm)')
plt.title('Hilo', fontsize = 10)

f.set_constrained_layout_pads(w_pad=0.125, h_pad=0.125, hspace=0.9, wspace=0.5)
plt.tight_layout()
#f.suptitle('')
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/.png')
plt.show()

# Plot histograms for dry season
f = plt.figure(figsize=(10,8), dpi = 80)
ax = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

names = ['Present, PDO(-)', 'Present, PDO(+)', 'Future, PDO(-)', 'Future, PDO(+)']

plt.subplot(2, 2, 1)
plt.hist([lihue1_dry, lihue2_dry, lihue3_dry, lihue4_dry], stacked=True, normed=True)
plt.ylabel('Normalized # of Days')
plt.title('Lihue', fontsize = 10)

plt.subplot(2, 2, 2)
plt.hist([honolulu1_dry, honolulu2_dry, honolulu3_dry, honolulu4_dry], stacked=True, normed=True, label = names)
plt.legend()
plt.title('Honolulu', fontsize = 10)

plt.subplot(2, 2, 3)
plt.hist([bigbog1_dry, bigbog2_dry, bigbog3_dry, bigbog4_dry], stacked=True, normed=True)
plt.xlabel('Daily Rainfall (mm)')
plt.ylabel('Normalized # of Days')
plt.title('Big Bog', fontsize = 10)

plt.subplot(2, 2, 4)
plt.hist([hilo1_dry, hilo2_dry, hilo3_dry, hilo4_dry], stacked=True, normed=True)
plt.xlabel('Daily Rainfall (mm)')
plt.title('Hilo', fontsize = 10)

f.set_constrained_layout_pads(w_pad=0.125, h_pad=0.125, hspace=0.9, wspace=0.5)
plt.tight_layout()
#f.suptitle('')
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/.png')
plt.show()
'''
'''
# Plot 90th percentile density plot for wet season
import seaborn as sns

f = plt.figure(figsize=(12, 6), dpi = 80)
ax = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

plt.subplot(2, 2, 1)
sns.distplot(lihue1_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue2_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue3_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue4_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue_wet_90_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax.set_ylim(0, 0.055)
ax.set_xlim(0,375)
plt.ylabel('Density')
plt.title('Lihue', fontsize = 10)

plt.subplot(2, 2, 2)
sns.distplot(honolulu1_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Present, PDO-')
sns.distplot(honolulu2_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Present, PDO+')
sns.distplot(honolulu3_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Future, PDO-')
sns.distplot(honolulu4_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Future, PDO+')
sns.distplot(honolulu_wet_90_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Observations')
ax2.set_ylim(0, 0.055)
ax2.set_xlim(0,375)
plt.legend()
plt.title('Honolulu', fontsize = 10)

plt.subplot(2, 2, 3)
sns.distplot(bigbog1_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog2_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog3_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog4_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog_wet_90_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax3.set_ylim(0, 0.025)
ax3.set_xlim(0,850)
plt.xlabel('Daily Rainfall (mm)')
plt.ylabel('Density')
plt.title('Big Bog', fontsize = 10)

plt.subplot(2, 2, 4)
sns.distplot(hilo1_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo2_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo3_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo4_90_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo_wet_90_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax4.set_ylim(0, 0.025)
ax4.set_xlim(0,850)
plt.xlabel('Daily Rainfall (mm)')
plt.title('Hilo', fontsize = 10)

f.set_constrained_layout_pads(w_pad=0.125, h_pad=0.125, hspace=0.9, wspace=0.5)
plt.tight_layout()
#f.suptitle('90th Percentile Daily Rainfall')
plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/stations_pdf_90th_wet.png')
plt.show()

# Plot 90th percentile density plot for dry season
f = plt.figure(figsize=(12, 6), dpi = 80)
ax = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

plt.subplot(2, 2, 1)
sns.distplot(lihue1_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue2_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue3_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue4_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue_dry_90_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax.set_ylim(0, 0.2)
ax.set_xlim(0,500)
plt.ylabel('Density')
plt.title('Lihue', fontsize = 10)

plt.subplot(2, 2, 2)
sns.distplot(honolulu1_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Present, PDO-')
sns.distplot(honolulu2_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Present, PDO+')
sns.distplot(honolulu3_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Future, PDO-')
sns.distplot(honolulu4_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Future, PDO+')
sns.distplot(honolulu_dry_90_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Observations')
ax2.set_ylim(0, 0.2)
ax2.set_xlim(0,300)
plt.legend()
plt.title('Honolulu', fontsize = 10)

plt.subplot(2, 2, 3)
sns.distplot(bigbog1_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog2_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog3_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog4_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog_dry_90_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax3.set_ylim(0, 0.035)
ax3.set_xlim(0,650)
plt.xlabel('Daily Rainfall (mm)')
plt.ylabel('Density')
plt.title('Big Bog', fontsize = 10)

plt.subplot(2, 2, 4)
sns.distplot(hilo1_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo2_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo3_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo4_90_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo_dry_90_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax4.set_ylim(0, 0.035)
ax4.set_xlim(0,650)
plt.xlabel('Daily Rainfall (mm)')
plt.title('Hilo', fontsize = 10)

f.set_constrained_layout_pads(w_pad=0.125, h_pad=0.125, hspace=0.9, wspace=0.5)
plt.tight_layout()
#f.suptitle('90th Percentile Daily Rainfall')
plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/stations_pdf_90th_dry.png')
plt.show()

# Plot 99th percentile density plot for wet season
import seaborn as sns

f = plt.figure(figsize=(12, 6), dpi = 80)
ax = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

plt.subplot(2, 2, 1)
sns.distplot(lihue1_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue2_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue3_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue4_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue_wet_99_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax.set_ylim(0, 0.03)
ax.set_xlim(0,350)
plt.ylabel('Density')
plt.title('Lihue', fontsize = 10)

plt.subplot(2, 2, 2)
sns.distplot(honolulu1_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Present, PDO-')
sns.distplot(honolulu2_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Present, PDO+')
sns.distplot(honolulu3_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Future, PDO-')
sns.distplot(honolulu4_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Future, PDO+')
sns.distplot(honolulu_wet_99_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Observations')
ax2.set_ylim(0, 0.03)
ax2.set_xlim(0,350)
plt.legend()
plt.title('Honolulu', fontsize = 10)

plt.subplot(2, 2, 3)
sns.distplot(bigbog1_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog2_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog3_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog4_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog_wet_99_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax3.set_ylim(0, 0.010)
ax3.set_xlim(0,1000)
plt.xlabel('Daily Rainfall (mm)')
plt.ylabel('Density')
plt.title('Big Bog', fontsize = 10)

plt.subplot(2, 2, 4)
sns.distplot(hilo1_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo2_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo3_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo4_99_wet, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo_wet_99_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax4.set_ylim(0, 0.010)
ax4.set_xlim(0,1000)
plt.xlabel('Daily Rainfall (mm)')
plt.title('Hilo', fontsize = 10)

f.set_constrained_layout_pads(w_pad=0.125, h_pad=0.125, hspace=0.9, wspace=0.5)
plt.tight_layout()
#f.suptitle('99th Percentile Daily Rainfall')
plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/stations_pdf_99th_wet.png')
plt.show()

# Plot 99th percentile density plot for dry season
f = plt.figure(figsize=(12, 6), dpi = 80)
ax = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

plt.subplot(2, 2, 1)
sns.distplot(lihue1_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue2_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue3_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue4_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(lihue_dry_99_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax.set_ylim(0, 0.045)
ax.set_xlim(0,550)
plt.ylabel('Density')
plt.title('Lihue', fontsize = 10)

plt.subplot(2, 2, 2)
sns.distplot(honolulu1_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Present, PDO-')
sns.distplot(honolulu2_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Present, PDO+')
sns.distplot(honolulu3_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Future, PDO-')
sns.distplot(honolulu4_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Future, PDO+')
sns.distplot(honolulu_dry_99_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1}, label = 'Observations')
ax2.set_ylim(0, 0.045)
ax2.set_xlim(0,550)
plt.legend()
plt.title('Honolulu', fontsize = 10)

plt.subplot(2, 2, 3)
sns.distplot(bigbog1_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog2_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog3_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog4_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(bigbog_dry_99_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax3.set_ylim(0, 0.018)
ax3.set_xlim(0,710)
plt.xlabel('Daily Rainfall (mm)')
plt.ylabel('Density')
plt.title('Big Bog', fontsize = 10)

plt.subplot(2, 2, 4)
sns.distplot(hilo1_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo2_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo3_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo4_99_dry, hist = False, kde = True,kde_kws = {'linewidth': 1})
sns.distplot(hilo_dry_99_atlas, hist = False, kde = True,kde_kws = {'linewidth': 1})
ax4.set_ylim(0, 0.018)
ax4.set_xlim(0,710)
plt.xlabel('Daily Rainfall (mm)')
plt.title('Hilo', fontsize = 10)

f.set_constrained_layout_pads(w_pad=0.125, h_pad=0.125, hspace=0.9, wspace=0.5)
plt.tight_layout()
#f.suptitle('99th Percentile Daily Rainfall')
plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/stations_pdf_99th_dry.png')
plt.show()
'''

# Plot 90th percentile box plots for wet season
import seaborn as sns

f = plt.figure(figsize=(10, 8), dpi = 150)
ax = f.add_subplot(421)
ax2 = f.add_subplot(422)
ax3 = f.add_subplot(423)
ax4 = f.add_subplot(424)
ax5 = f.add_subplot(425)
ax6 = f.add_subplot(426)
ax7 = f.add_subplot(427)
ax8 = f.add_subplot(428)

names = ['Fut(+)','Fut(-)','Pres(+)','Pres(-)', 'Obs']

plt.subplot(4, 2, 1)
box = plt.boxplot([lihue4_90_wet, lihue3_90_wet, lihue2_90_wet, lihue1_90_wet, lihue_wet_90_atlas], patch_artist = True, vert = 0, labels = names)
plt.xlim(0,300)
ax.set_xticklabels(labels = [''])
left_aligned = "(a) wet - RF90"
right_aligned = "Lihue"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)
colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 3)
box = plt.boxplot([honolulu4_90_wet, honolulu3_90_wet, honolulu2_90_wet, honolulu1_90_wet, honolulu_wet_90_atlas],patch_artist = True, vert = 0, labels = names)
plt.xlim(0,300)
ax3.set_xticklabels(labels = [''])
plt.xlabel('')
left_aligned = "(c) wet - RF90"
right_aligned = "Honolulu"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 5)
box = plt.boxplot([bigbog4_90_wet, bigbog3_90_wet, bigbog2_90_wet, bigbog1_90_wet, bigbog_wet_90_atlas],patch_artist = True, vert = 0, labels = names)
plt.xlim(0,300)
ax5.set_xticklabels(labels = [''])
left_aligned = "(e) wet - RF90"
right_aligned = "Big Bog"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 7)
box = plt.boxplot([hilo4_90_wet, hilo3_90_wet, hilo2_90_wet, hilo1_90_wet, hilo_wet_90_atlas],patch_artist = True, vert = 0, labels = names)
plt.xlim(0,300)
plt.xlabel('(mm day$^{-1}$)')
left_aligned = "(g) wet - RF90"
right_aligned = "Hilo"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 2)
box = plt.boxplot([lihue4_90_dry, lihue3_90_dry, lihue2_90_dry, lihue1_90_dry, lihue_dry_90_atlas],patch_artist = True, vert = 0, labels = ['','','','',''])
plt.xlim(0,300)
ax2.set_xticklabels(labels = [''])
left_aligned = "(b) dry - RF90"
right_aligned = "Lihue"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 4)
box = plt.boxplot([honolulu4_90_dry, honolulu3_90_dry, honolulu2_90_dry, honolulu1_90_dry, honolulu_dry_90_atlas],patch_artist = True, vert = 0, labels = ['','','','',''])
plt.xlim(0,300)
ax4.set_xticklabels(labels = [''])
plt.xlabel('')
left_aligned = "(d) dry - RF90"
right_aligned = "Honolulu"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 6)
box = plt.boxplot([bigbog4_90_dry, bigbog3_90_dry, bigbog2_90_dry, bigbog1_90_dry, bigbog_dry_90_atlas], patch_artist = True, vert = 0, labels = ['','','','',''])
plt.xlim(0,300)
ax6.set_xticklabels(labels = [''])
left_aligned = "(f) dry - RF90"
right_aligned = "Big Bog"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 8)
box = plt.boxplot([hilo4_90_dry, hilo3_90_dry, hilo2_90_dry, hilo1_90_dry, hilo_dry_90_atlas],patch_artist = True, vert = 0, labels = ['','','','',''])
plt.xlim(0,300)
plt.xlabel('(mm day$^{-1}$)')
left_aligned = "(h) dry - RF90"
right_aligned = "Hilo"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

f.set_constrained_layout_pads(w_pad=0.125, h_pad=0.125, hspace=0.9, wspace=0.25)
plt.tight_layout()
plt.show()

# Plot 99th percentile box plots for wet season
import seaborn as sns

f = plt.figure(figsize=(10, 8), dpi = 150)
ax = f.add_subplot(421)
ax2 = f.add_subplot(422)
ax3 = f.add_subplot(423)
ax4 = f.add_subplot(424)
ax5 = f.add_subplot(425)
ax6 = f.add_subplot(426)
ax7 = f.add_subplot(427)
ax8 = f.add_subplot(428)

plt.subplot(4, 2, 1)
box = plt.boxplot([lihue4_99_wet, lihue3_99_wet, lihue2_99_wet, lihue1_99_wet, lihue_wet_99_atlas],patch_artist = True, vert = 0, labels = names)
plt.xlim(0,600)
ax.set_xticklabels(labels = [''])
#plt.title('(a) wet - RF99            Lihue', fontsize = 10)
left_aligned = "(a) wet - RF99"
right_aligned = "Lihue"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 3)
box = plt.boxplot([honolulu4_99_wet, honolulu3_99_wet, honolulu2_99_wet, honolulu1_99_wet, honolulu_wet_99_atlas],patch_artist = True, vert = 0, labels = names)
plt.xlim(0,600)
ax3.set_xticklabels(labels = [''])
plt.xlabel('')
left_aligned = "(c) wet - RF99"
right_aligned = "Honolulu"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 5)
box = plt.boxplot([bigbog4_99_wet, bigbog3_99_wet, bigbog2_99_wet, bigbog1_99_wet, bigbog_wet_99_atlas],patch_artist = True, vert = 0, labels = names)
plt.xlim(0,600)
ax5.set_xticklabels(labels = [''])
left_aligned = "(e) wet - RF99"
right_aligned = "Big Bog"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 7)
box = plt.boxplot([hilo4_99_wet, hilo3_99_wet, hilo2_99_wet, hilo1_99_wet, hilo_wet_99_atlas],patch_artist = True, vert = 0, labels = names)
plt.xlim(0,600)
plt.xlabel('(mm day$^{-1}$)')
left_aligned = "(g) wet - RF99"
right_aligned = "Hilo"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 2)
box = plt.boxplot([lihue4_99_dry, lihue3_99_dry, lihue2_99_dry, lihue1_99_dry, lihue_dry_99_atlas],patch_artist = True, vert = 0, labels = ['','','','',''])
plt.xlim(0,600)
ax2.set_xticklabels(labels = [''])
left_aligned = "(b) dry - RF99"
right_aligned = "Lihue"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 4)
box = plt.boxplot([honolulu4_99_dry, honolulu3_99_dry, honolulu2_99_dry, honolulu1_99_dry, honolulu_dry_99_atlas],patch_artist = True, vert = 0, labels = ['','','','',''])
plt.xlim(0,600)
ax4.set_xticklabels(labels = [''])
plt.xlabel('')
left_aligned = "(d) dry - RF99"
right_aligned = "Honolulu"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 6)
box = plt.boxplot([bigbog4_99_dry, bigbog3_99_dry, bigbog2_99_dry, bigbog1_99_dry, bigbog_dry_99_atlas],patch_artist = True, vert = 0, labels = ['','','','',''])
plt.xlim(0,600)
ax6.set_xticklabels(labels = [''])
left_aligned = "(f) dry - RF99"
right_aligned = "Big Bog"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.subplot(4, 2, 8)
box = plt.boxplot([hilo4_99_dry, hilo3_99_dry, hilo2_99_dry, hilo1_99_dry, hilo_dry_99_atlas],patch_artist = True, vert = 0, labels = ['','','','',''])
plt.xlim(0,600)
plt.xlabel('(mm day$^{-1}$)')
left_aligned = "(h) dry - RF99"
right_aligned = "Hilo"
title = f"{left_aligned:<25}{right_aligned:>25}"
plt.title(title, fontsize = 12)

colors = ['lightgreen','lightpink','lightblue','tan','lightyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

f.set_constrained_layout_pads(w_pad=0.125, h_pad=0.125, hspace=0.9, wspace=0.25)
plt.tight_layout()
plt.show()