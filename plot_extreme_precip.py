# Plot extreme precip
# KMF 5/6/20

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

path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'
# Load in extreme precip data

# 90th percentile daily rainfall extremes
P90_1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf1-5_90th.npy')
P90_2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf6-10_90th.npy')
P90_3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf11-15_90th.npy')
P90_4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf16-20_90th.npy')

present_90th = (P90_1 + P90_2) / 2

# 99th percentile daily rainfall extremes
P99_1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf1-5_99th.npy')
P99_2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf6-10_99th.npy')
P99_3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf11-15_99th.npy')
P99_4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf16-20_99th.npy')

present_99th = (P99_1 + P99_2) / 2

# Wet season
P90_1wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf1-5_90th_wet.npy')
P90_2wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf6-10_90th_wet.npy')
P90_3wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf11-15_90th_wet.npy')
P90_4wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf16-20_90th_wet.npy')

present_90th_wet = (P90_1wet + P90_2wet) / 2

pvals_difP901_wet = np.load(path+'p_vals_dif_P901_wet.npy')
pvals_difP902_wet = np.load(path+'p_vals_dif_P902_wet.npy')
pvals_difP903_wet = np.load(path+'p_vals_dif_P903_wet.npy')
pvals_difP904_wet = np.load(path+'p_vals_dif_P904_wet.npy')

# 99th percentile daily rainfall extremes
P99_1wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf1-5_99th_wet.npy')
P99_2wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf6-10_99th_wet.npy')
P99_3wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf11-15_99th_wet.npy')
P99_4wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf16-20_99th_wet.npy')

present_99th_wet = (P99_1wet + P99_2wet) / 2

pvals_difP991_wet = np.load(path+'p_vals_dif_P991_wet.npy')
pvals_difP992_wet = np.load(path+'p_vals_dif_P992_wet.npy')
pvals_difP993_wet = np.load(path+'p_vals_dif_P993_wet.npy')
pvals_difP994_wet = np.load(path+'p_vals_dif_P994_wet.npy')

# Dry season
P90_1dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf1-5_90th_dry.npy')
P90_2dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf6-10_90th_dry.npy')
P90_3dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf11-15_90th_dry.npy')
P90_4dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf16-20_90th_dry.npy')

present_90th_dry = (P90_1dry + P90_2dry) / 2

pvals_difP901_dry = np.load(path+'p_vals_dif_P901_dry.npy')
pvals_difP902_dry = np.load(path+'p_vals_dif_P902_dry.npy')
pvals_difP903_dry = np.load(path+'p_vals_dif_P903_dry.npy')
pvals_difP904_dry = np.load(path+'p_vals_dif_P904_dry.npy')

# 99th percentile daily rainfall extremes
P99_1dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf1-5_99th_dry.npy')
P99_2dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf6-10_99th_dry.npy')
P99_3dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf11-15_99th_dry.npy')
P99_4dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wrf16-20_99th_dry.npy')

present_99th_dry = (P99_1dry + P99_2dry) / 2

pvals_difP991_dry = np.load(path+'p_vals_dif_P991_dry.npy')
pvals_difP992_dry = np.load(path+'p_vals_dif_P992_dry.npy')
pvals_difP993_dry = np.load(path+'p_vals_dif_P993_dry.npy')
pvals_difP994_dry = np.load(path+'p_vals_dif_P994_dry.npy')

# Rainfall Atlas 90th and 99th percentiles
RainAtlas_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_90th_wet.npy')
RainAtlas_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_99th_wet.npy')

RainAtlas_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_90th_dry.npy')
RainAtlas_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_99th_dry.npy')

# Total Rainfall Days (Rainfall Atlas)
annual_rain_days_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_RainAtlas.npy')
wet_rain_days_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_atlas.npy')
dry_rain_days_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_atlas.npy')

# Total Rainfall Days WRF
annual_raindays1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_WRF1-5.npy')
annual_raindays2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_WRF6-10.npy')
annual_raindays3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_WRF11-15.npy')
annual_raindays4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_rain_days_WRF16-20.npy')

annual_raindays_present_wrf = (annual_raindays1 + annual_raindays2) / 2

# Wet/dry season total rain days WRF
wet_raindays1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_WRF1-5.npy')
wet_raindays2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_WRF6-10.npy')
wet_raindays3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_WRF11-15.npy')
wet_raindays4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_rain_days_WRF16-20.npy')

wet_raindays_present_wrf = (wet_raindays1 + wet_raindays2) / 2

dry_raindays1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_WRF1-5.npy')
dry_raindays2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_WRF6-10.npy')
dry_raindays3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_WRF11-15.npy')
dry_raindays4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_rain_days_WRF16-20.npy')

dry_raindays_present_wrf = (dry_raindays1 + dry_raindays2) / 2

pvals_RD1_wet = np.load(path+'p_vals_dif_RD1_wet.npy')
pvals_RD2_wet = np.load(path+'p_vals_dif_RD2_wet.npy')
pvals_RD3_wet = np.load(path+'p_vals_dif_RD3_wet.npy')
pvals_RD4_wet = np.load(path+'p_vals_dif_RD4_wet.npy')

pvals_RD1_dry = np.load(path+'p_vals_dif_RD1_dry.npy')
pvals_RD2_dry = np.load(path+'p_vals_dif_RD2_dry.npy')
pvals_RD3_dry = np.load(path+'p_vals_dif_RD3_dry.npy')
pvals_RD4_dry = np.load(path+'p_vals_dif_RD4_dry.npy')

# Annual consecutive dry days Rainfall Atlas
annual_CDD_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/annual_CDDs_1990-2014_RainAtlas.npy')
wet_CDD_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/wet_CDDs_1990-2014_RainAtlas.npy')
dry_CDD_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/dry_CDDs_1990-2014_RainAtlas.npy')

# Annual consecutive dry days WRF
annual_CDD_wrf1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/record_CDD_WRF1-5.npy')
annual_CDD_wrf2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/record_CDD_WRF6-10.npy')
annual_CDD_wrf3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/record_CDD_WRF11-15.npy')
annual_CDD_wrf4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/record_CDD_WRF16-20.npy')

annual_CDD_present = (annual_CDD_wrf1 + annual_CDD_wrf2) / 2

cdd_wet1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_wet_WRF1-5.npy')
cdd_wet2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_wet_WRF6-10.npy')
cdd_wet3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_wet_WRF11-15.npy')
cdd_wet4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_wet_WRF16-20.npy')

cdd_wet_present = (cdd_wet1 + cdd_wet2) / 2

cdd_dry1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_dry_WRF1-5.npy')
cdd_dry2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_dry_WRF6-10.npy')
cdd_dry3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_dry_WRF11-15.npy')
cdd_dry4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/CDD_dry_WRF16-20.npy')

cdd_dry_present = (cdd_dry1 + cdd_dry2) / 2

pvals_CDD1_wet = np.load(path+'p_vals_dif_CDD1_wet.npy')
pvals_CDD2_wet = np.load(path+'p_vals_dif_CDD2_wet.npy')
pvals_CDD3_wet = np.load(path+'p_vals_dif_CDD3_wet.npy')
pvals_CDD4_wet = np.load(path+'p_vals_dif_CDD4_wet.npy')

pvals_CDD1_dry = np.load(path+'p_vals_dif_CDD1_dry.npy')
pvals_CDD2_dry = np.load(path+'p_vals_dif_CDD2_dry.npy')
pvals_CDD3_dry = np.load(path+'p_vals_dif_CDD3_dry.npy')
pvals_CDD4_dry = np.load(path+'p_vals_dif_CDD4_dry.npy')

# Differences
dif1_90 = P90_4 - P90_2
dif2_90 = P90_3 - P90_1
dif3_90 = P90_4 - P90_3
dif4_90 = P90_2 - P90_1

dif1_99 = P99_4 - P99_2
dif2_99 = P99_3 - P99_1
dif3_99 = P99_4 - P99_3
dif4_99 = P99_2 - P99_1

###################################################################################################################
path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'
no_pvals = np.load(path+'no_border.npy')

###################################################################################################################
# Load info for plotting 
path2 = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'
ncfile = Dataset(path2+'WRF1/wrfout_d02_monthly_mean_1998_11.nc')

# Get landmask
dataDIR = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/WRF1/'
filename = 'wrfout_d02_monthly_mean_1997_01.nc'
DS = xarray.open_dataset(dataDIR+filename)
landmask = DS.LANDMASK[0,:,:]
height = DS.HGT[0,:,:]
#X_land = np.ma.masked_where(landmask == 0, X)

# Extract variables
p = getvar(ncfile, "pressure")
p = p*100 # Convert to pascal units
z = getvar(ncfile, "z", units="dm")

# Define levels
plev = np.arange(100*100,1025*100,25*100) # In pascal units
plev = plev[::-1]

# Interpolate variables to any pressure level (converts from hybrid to p coords)
ht_plev = interplevel(z, p, plev)

# Get lat/lon coordinates
lats, lons = latlon_coords(ht_plev)

# Get map projection information
cart_proj = get_cartopy(ht_plev)
crs = crs.PlateCarree()

def plot_background(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    return ax

###########################################################################################################################
# Fix colorbar so white is always on zero
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
###########################################################################################################################

# Colorbar labels
mm_day = '(mm/day)'
mm = '(mm)'

# Color schemes
wet_dry = plt.cm.BrBG
wet_dry_r = plt.cm.BrBG_r
rainbow = plt.cm.gist_rainbow_r
red_blue = plt.cm.RdBu_r
cool = plt.cm.cool

x = lons[0,:]
y = lats[:,0]

# Plot changes in 90th percentile rainfall for wet and dry season
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, P90_4wet - P90_2wet), vmin = -20, vmax = 25, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP901_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[0].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet        Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot2 = axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, P90_3wet - P90_1wet), vmin = -20, vmax = 25, levels = np.arange(-20,25.5,5), cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP902_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[2].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet        Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = '')

subplot3 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, P90_4dry - P90_2dry), vmin = -20, vmax = 25, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP901_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[1].contourf(lons, lats, pvals_difP903_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry        Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[1].set_xticklabels(labels = '')

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, P90_3dry - P90_1dry), vmin = -20, vmax = 25, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP902_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[3].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry        Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = '')
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

subplot5 = axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0, P90_4wet - P90_3wet), vmin = -20, vmax = 25, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP903_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[4].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet       Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, P90_4dry - P90_3dry), vmin = -20, vmax = 25, levels = np.arange(-20,25.5,5), cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP903_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[5].contourf(lons, lats, pvals_difP902_dry, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry        Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

subplot7 = axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0, P90_2wet - P90_1wet), vmin = -20, vmax = 25, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP904_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[6].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet         Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])

subplot8 = axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, P90_2dry - P90_1dry), vmin = -20, vmax = 25, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP904_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[7].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry          Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(-20, 25, 5), orientation='vertical', shrink = 0.40, pad = 0.0)
cbar.set_label('(mm day$^{-1}$)')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()


# Plot wet and dry season changes in 99th percentile
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, P99_4wet - P99_2wet), vmin = -100, vmax = 80, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP991_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[0].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet        Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot2 = axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, P99_3wet - P99_1wet), vmin = -100, vmax = 80, levels = np.arange(-100,80.5,10), cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP992_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[2].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet        Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = '')

subplot3 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, P99_4dry - P99_2dry), vmin = -100, vmax = 80, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP991_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[1].contourf(lons, lats, pvals_difP903_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry        Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[1].set_xticklabels(labels = '')

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, P99_3dry - P99_1dry), vmin = -100, vmax = 80, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP992_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[3].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry        Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = '')
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

subplot5 = axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0, P99_4wet - P99_3wet), vmin = -100, vmax = 80, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP993_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[4].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet       Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, P99_4dry - P99_3dry), vmin = -100, vmax = 80, levels = np.arange(-100,80.5,10), cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP993_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[5].contourf(lons, lats, pvals_difP902_dry, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry        Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

subplot7 = axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0, P99_2wet - P99_1wet), vmin = -100, vmax = 80, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_difP994_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[6].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet         Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])

subplot8 = axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, P99_2dry - P99_1dry), vmin = -100, vmax = 80, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_difP994_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[7].contourf(lons, lats, pvals_difP991_wet, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry          Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(-100, 80, 20), orientation='vertical', shrink = 0.40, pad = 0.0)
cbar.set_label('(mm day$^{-1}$)')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()


# Plot present-day 90th and 99th percentiles for wet and dry season

# Get lat/lon information for observations
from mpl_toolkits.basemap import interp

path = '/network/rit/lab/elisontimmlab_rit/DATA/HI_DAILY_PRECIP/'
ncref = 'Hawaii_Precipitation_UH_7.5sec_dTS1990.nc'
with Dataset(path+ncref, mode='r') as fh:
    lon = fh.variables['longitude'][:]
    lat = fh.variables['latitude'][:]
    rain = fh.variables['precipitation'][0,:,:]
lons_sub, lats_sub = np.meshgrid(lon[::7], lat[::6])
coarse = interp(rain, lon, lat, lons_sub, lats_sub, order=1)

fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10, 10), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, RainAtlas_90_wet), vmin = 0, vmax = 560, cmap = rainbow, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet - RF90      Obs', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot2 = axlist[2].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, RainAtlas_99_wet), vmin = 0, vmax = 560, levels = np.arange(0,560.5,25), cmap = rainbow, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet - RF99      Obs', fontsize = 10)
axlist[2].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = [''])
axlist[2].set_xlabel('')
axlist[2].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot3 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, present_90th_wet), vmin = 0, vmax = 560, cmap = rainbow, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) wet - RF90      WRF', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(x.min()+0.7, x.max(),3), decimals = 1))

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, present_99th_wet), vmin = 0, vmax = 560, cmap = rainbow, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) wet - RF99     WRF', fontsize = 10)
axlist[3].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[3].set_xticks(ticks = np.around(np.arange(x.min()+0.7, x.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = [''])
axlist[3].set_xlabel('')

subplot5 = axlist[4].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, RainAtlas_90_dry), vmin = 0, vmax = 560, cmap = rainbow, transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) dry - RF90    Obs', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[4].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot6 = axlist[6].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, RainAtlas_99_dry), vmin = 0, vmax = 560, levels = np.arange(0,560.5,25), cmap = rainbow, transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) dry - RF99    Obs', fontsize = 10)
axlist[6].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])
axlist[6].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot7 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, present_90th_dry), vmin = 0, vmax = 560, cmap = rainbow, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry - RF90    WRF', fontsize = 10)
axlist[5].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[5].set_xticks(ticks = np.around(np.arange(x.min()+0.7, x.max(),3), decimals = 1))

subplot8 = axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, present_99th_dry), vmin = 0, vmax = 560, cmap = rainbow, transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry - RF99    WRF', fontsize = 10)
axlist[7].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[7].set_xticks(ticks = np.around(np.arange(x.min()+0.7, x.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(0, 560.5, 50), orientation='vertical', shrink = 0.30, pad = 0.0)
cbar.set_label('(mm day$^{-1}$)')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()


# Plot consecutive dry days for observations and present-day WRF sims
fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), dpi = 150, sharex = 'col', sharey = 'row', constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, annual_CDD_atlas), vmin = 0, vmax = 120, cmap = rainbow, transform = crs)
axlist[0].set_title('(a) annual - Obs', fontsize = 12)
axlist[0].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, annual_CDD_present), vmin = 0, vmax = 120, levels = np.arange(0,120.5,5), cmap = rainbow, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) annual - WRF', fontsize = 12)
axlist[1].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot3 = axlist[2].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, wet_CDD_atlas), vmin = 0, vmax = 120, cmap = rainbow, transform = crs)
axlist[2].set_title('(c) wet - Obs', fontsize = 12)
axlist[2].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, cdd_wet_present), vmin = 0, vmax = 120, cmap = rainbow, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) wet - WRF', fontsize = 12)
axlist[3].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[3].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot5 = axlist[4].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, dry_CDD_atlas), vmin = 0, vmax = 120, cmap = rainbow, transform = crs)
#axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) dry - Obs', fontsize = 12)
axlist[4].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[4].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, cdd_dry_present), vmin = 0, vmax = 120, cmap = rainbow, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry - WRF', fontsize = 12)
axlist[5].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[5].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[5].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(0, 120.5, 10), orientation='vertical', shrink = 0.30, pad = 0.0)
cbar.set_label('Days')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()


# Plot differences in Wet season CDDs for WRF data
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, (cdd_wet4 - cdd_wet2)), vmin = -3, vmax = 8, norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_CDD1_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet    Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, (cdd_dry4 - cdd_dry2)), vmin = -3, vmax = 8, levels = np.arange(-3,8,.5), norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_CDD1_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry     Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot3 = axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, (cdd_wet3 - cdd_wet1)), vmin = -3, vmax = 8, norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_CDD2_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet     Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[2].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticklabels(labels = [''])

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, (cdd_dry3 - cdd_dry1)), vmin = -3, vmax = 8, norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_CDD2_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry     Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[3].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = [''])

subplot5 = axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0, (cdd_wet4 - cdd_wet3)), vmin = -3, vmax = 8, norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_CDD3_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet     Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[4].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, (cdd_dry4 - cdd_dry3)), vmin = -3, vmax = 8, levels = np.arange(-3,8,.5), norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_CDD3_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry     Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[5].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot7 = axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0, (cdd_wet2 - cdd_wet1)), vmin = -3, vmax = 8, norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_CDD4_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet      Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[6].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])

subplot8 = axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, (cdd_dry2 - cdd_dry1)), vmin = -3, vmax = 8, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_CDD4_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry      Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])
axlist[7].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(-3, 8, 1), orientation='vertical', shrink = 0.30, pad = 0.0)
cbar.set_label('Days')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()


# Plot total rain days for observations and present-day WRF sims
fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(8, 10), dpi = 150, sharex = 'col', sharey = 'row', constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, annual_rain_days_atlas), vmin = 0, vmax = 360, cmap = rainbow, transform = crs)
axlist[0].set_title('(a) annual - Obs', fontsize = 12)
axlist[0].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, annual_raindays_present_wrf), vmin = 0, vmax = 360, levels = np.arange(0,360.5,20), cmap = rainbow, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) annual - WRF', fontsize = 12)
axlist[1].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot3 = axlist[2].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, wet_rain_days_atlas), vmin = 0, vmax = 360, cmap = rainbow, transform = crs)
axlist[2].set_title('(c) wet - Obs', fontsize = 12)
axlist[2].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, wet_raindays_present_wrf), vmin = 0, vmax = 360, cmap = rainbow, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) wet - WRF', fontsize = 12)
axlist[3].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[3].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))

subplot5 = axlist[4].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, dry_rain_days_atlas), vmin = 0, vmax = 360, cmap = rainbow, transform = crs)
#axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) dry - Obs', fontsize = 12)
axlist[4].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])
axlist[4].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, dry_raindays_present_wrf), vmin = 0, vmax = 360, cmap = rainbow, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry - WRF', fontsize = 12)
axlist[5].set_yticks(ticks = np.around(np.arange(y.min()+1.4, y.max(),2), decimals = 1))
axlist[5].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(0, 360.5, 40), orientation='vertical', shrink = 0.30, pad = 0.0)
cbar.set_label('Days')

fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.05, wspace=0.0)
plt.show()


# Plot differences in wet season rain days for WRF output
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, (wet_raindays4 - wet_raindays2)), vmin = -16, vmax = 16, norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_RD1_wet), hatches = ["......"], colors = 'none', alpha = 0, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet     Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, (dry_raindays4 - dry_raindays2)), vmin = -16, vmax = 16, levels = np.arange(-16,16.5,2), norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_RD1_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry     Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[1].set_ylabel('')
axlist[1].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))

subplot3 = axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, (wet_raindays3 - wet_raindays1)), vmin = -16, vmax = 16, norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_RD2_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet     Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = [''])
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, (dry_raindays3 - dry_raindays1)), vmin = -16, vmax = 16, levels = np.arange(-16,16.5,2), norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_RD2_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry     Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = [''])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

subplot5 = axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0, (wet_raindays4 - wet_raindays3)), vmin = -16, vmax = 16, norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_RD3_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet     Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = [''])
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, (dry_raindays4 - dry_raindays3)), vmin = -16, vmax = 16, levels = np.arange(-16,16.5,2), norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_RD3_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry     Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = [''])
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[5].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[5].set_ylabel('')

subplot7 = axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0, (wet_raindays2 - wet_raindays1)), vmin = -16, vmax = 16, norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_RD4_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet     Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[6].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])

subplot8 = axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, (dry_raindays2 - dry_raindays1)), vmin = -16, vmax = 16, norm = MidpointNormalize(midpoint = 0), cmap = red_blue, transform = crs)
axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_RD4_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry     Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[7].set_xticks(ticks = np.around(np.arange(x.min()+1.7, x.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(-16, 16.5, 4), orientation='vertical', shrink = 0.30, pad = 0.0)
cbar.set_label('Days')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()

