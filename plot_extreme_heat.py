# Plot extreme heat analysis

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

# 90th and 99th percentile temps - Rainfall Atlas 
atlas_annual_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_90th_temps_annual.npy')
atlas_annual_90 = (atlas_annual_90 + 459.67) * (5/9)
atlas_annual_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_99th_temps_annual.npy')
atlas_annual_99 = (atlas_annual_99 + 459.67) * (5/9)

atlas_wet_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_90th_temps_wet.npy')
atlas_wet_90 = (atlas_wet_90 + 459.67) * (5/9)
atlas_wet_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_99th_temps_wet.npy')
atlas_wet_99 = (atlas_wet_99 + 459.67) * (5/9)

atlas_dry_90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_90th_temps_dry.npy')
atlas_dry_90 = (atlas_dry_90 + 459.67) * (5/9)
atlas_dry_99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RainfallAtlas_99th_temps_dry.npy')
atlas_dry_99 = (atlas_dry_99 + 459.67) * (5/9)

# Annual 90th and 99th percentiles
wrf1_ann_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF1-5_90th_temps_annual.npy')
wrf1_ann_T90 = (wrf1_ann_T90 + 459.67) * (5/9)
wrf1_ann_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF1-5_99th_temps_annual.npy')
wrf1_ann_T99 = (wrf1_ann_T99 + 459.67) * (5/9)

wrf2_ann_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF6-10_90th_temps_annual.npy')
wrf2_ann_T90 = (wrf2_ann_T90 + 459.67) * (5/9)
wrf2_ann_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF6-10_99th_temps_annual.npy')
wrf2_ann_T99 = (wrf2_ann_T99 + 459.67) * (5/9)

wrf_ann_present_90 = (wrf1_ann_T90 + wrf2_ann_T90) / 2
wrf_ann_present_99 = (wrf1_ann_T99 + wrf2_ann_T99) / 2

wrf3_ann_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF11-15_90th_temps_annual.npy')
wrf3_ann_T90 = (wrf3_ann_T90 + 459.67) * (5/9)
wrf3_ann_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF11-15_99th_temps_annual.npy')
wrf3_ann_T99 = (wrf3_ann_T99 + 459.67) * (5/9)

wrf4_ann_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF16-20_90th_temps_annual.npy')
wrf4_ann_T90 = (wrf4_ann_T90 + 459.67) * (5/9)
wrf4_ann_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF16-20_99th_temps_annual.npy')
wrf4_ann_T99 = (wrf4_ann_T99 + 459.67) * (5/9)

# Wet season 90th and 99th percentiles
wrf1_wet_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF1-5_90th_temps_wet.npy')
wrf1_wet_T90 = (wrf1_wet_T90 + 459.67) * (5/9)
wrf1_wet_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF1-5_99th_temps_wet.npy')
wrf1_wet_T99 = (wrf1_wet_T99 + 459.67) * (5/9)

wrf2_wet_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF6-10_90th_temps_wet.npy')
wrf2_wet_T90 = (wrf2_wet_T90 + 459.67) * (5/9)
wrf2_wet_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF6-10_99th_temps_wet.npy')
wrf2_wet_T99 = (wrf2_wet_T99 + 459.67) * (5/9)

wrf_wet_present_90 = (wrf1_wet_T90 + wrf2_wet_T90) / 2
wrf_wet_present_99 = (wrf1_wet_T99 + wrf2_wet_T99) / 2

wrf3_wet_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF11-15_90th_temps_wet.npy')
wrf3_wet_T90 = (wrf3_wet_T90 + 459.67) * (5/9)
wrf3_wet_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF11-15_99th_temps_wet.npy')
wrf3_wet_T99 = (wrf3_wet_T99 + 459.67) * (5/9)

wrf4_wet_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF16-20_90th_temps_wet.npy')
wrf4_wet_T90 = (wrf4_wet_T90 + 459.67) * (5/9)
wrf4_wet_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF16-20_99th_temps_wet.npy')
wrf4_wet_T99 = (wrf4_wet_T99 + 459.67) * (5/9)

pvals_T901_wet = np.load(path+'p_vals_dif_T901_wet.npy')
pvals_T902_wet = np.load(path+'p_vals_dif_T902_wet.npy')
pvals_T903_wet = np.load(path+'p_vals_dif_T903_wet.npy')
pvals_T904_wet = np.load(path+'p_vals_dif_T904_wet.npy')

pvals_T991_wet = np.load(path+'p_vals_dif_T991_wet.npy')
pvals_T992_wet = np.load(path+'p_vals_dif_T992_wet.npy')
pvals_T993_wet = np.load(path+'p_vals_dif_T993_wet.npy')
pvals_T994_wet = np.load(path+'p_vals_dif_T994_wet.npy')

# Dry season 90th and 99th percentiles
wrf1_dry_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF1-5_90th_temps_dry.npy')
wrf1_dry_T90 = (wrf1_dry_T90 + 459.67) * (5/9)
wrf1_dry_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF1-5_99th_temps_dry.npy')
wrf1_dry_T99 = (wrf1_dry_T99 + 459.67) * (5/9)

wrf2_dry_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF6-10_90th_temps_dry.npy')
wrf2_dry_T90 = (wrf2_dry_T90 + 459.67) * (5/9)
wrf2_dry_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF6-10_99th_temps_dry.npy')
wrf2_dry_T99 = (wrf2_dry_T99 + 459.67) * (5/9)

wrf_dry_present_90 = (wrf1_dry_T90 + wrf2_dry_T90) / 2
wrf_dry_present_99 = (wrf1_dry_T99 + wrf2_dry_T99) / 2

wrf3_dry_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF11-15_90th_temps_dry.npy')
wrf3_dry_T90 = (wrf3_dry_T90 + 459.67) * (5/9)
wrf3_dry_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF11-15_99th_temps_dry.npy')
wrf3_dry_T99 = (wrf3_dry_T99 + 459.67) * (5/9)

wrf4_dry_T90 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF16-20_90th_temps_dry.npy')
wrf4_dry_T90 = (wrf4_dry_T90 + 459.67) * (5/9)
wrf4_dry_T99 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/WRF16-20_99th_temps_dry.npy')
wrf4_dry_T99 = (wrf4_dry_T99 + 459.67) * (5/9)

pvals_T901_dry = np.load(path+'p_vals_dif_T901_dry.npy')
pvals_T902_dry = np.load(path+'p_vals_dif_T902_dry.npy')
pvals_T903_dry = np.load(path+'p_vals_dif_T903_dry.npy')
pvals_T904_dry = np.load(path+'p_vals_dif_T904_dry.npy')

pvals_T991_dry = np.load(path+'p_vals_dif_T991_dry.npy')
pvals_T992_dry = np.load(path+'p_vals_dif_T992_dry.npy')
pvals_T993_dry = np.load(path+'p_vals_dif_T993_dry.npy')
pvals_T994_dry = np.load(path+'p_vals_dif_T994_dry.npy')


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
Temp_F = '(K)'

# Color schemes
wet_dry = plt.cm.BrBG
wet_dry_r = plt.cm.BrBG_r
rainbow = plt.cm.gist_rainbow_r
red_blue = plt.cm.RdBu_r
cool = plt.cm.cool
#autumn = plt.cm.autumn_r
autumn = plt.cm.afmhot_r

x = lons[0,:]
y = lats[:,0]

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


# Plot 90th and 99th percentile temps for observations and present-day WRF sims
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10, 10), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, atlas_wet_90), vmin = 279, vmax = 312, cmap = autumn, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet - T90      Obs', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf_wet_present_90), vmin = 279, levels = np.arange(279,312,2), vmax = 312, cmap = autumn, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) wet - T90      WRF', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot3 = axlist[2].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, atlas_wet_99), vmin = 279, vmax = 312, cmap = autumn, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet - T99       Obs', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = [''])
#colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf_wet_present_99), vmin = 279, vmax = 312, cmap = autumn, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) wet - T99        WRF', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = [''])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot5 = axlist[4].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, atlas_dry_90), vmin = 279, vmax = 312, cmap = autumn, transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) dry - T90      Obs', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar5 = fig.colorbar(subplot5, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf_dry_present_90), vmin = 279, vmax = 312, cmap = autumn, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry - T90      WRF', fontsize = 10)
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar6 = fig.colorbar(subplot6, ax=axlist[5], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot7 = axlist[6].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, atlas_dry_99), vmin = 279, vmax = 312, cmap = autumn, transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) dry - T99       Obs', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
#colorbar7 = fig.colorbar(subplot7, ax=axlist[6], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot8 = axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf_dry_present_99), vmin = 279, vmax = 312, cmap = autumn, transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry - T99        WRF', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar8 = fig.colorbar(subplot8, ax=axlist[7], orientation='vertical', shrink = 0.70, pad = 0.0)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(279,312,4), orientation='vertical', shrink = 0.30, pad = 0.0)
cbar.set_label('(K)')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()

# Plot differences for wet season 90th percentile temps
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10, 10), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf4_wet_T90 - wrf2_wet_T90), vmin = -0.4, vmax = 1.5, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet      Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf4_dry_T90 - wrf2_dry_T90), vmin = -0.4, vmax = 1.5, levels = np.arange(-0.4, 1.5, 0.1), cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry      Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot3 = axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf3_wet_T90 - wrf1_wet_T90), vmin = -0.4, vmax = 1.5, cmap = red_blue, norm = MidpointNormalize(midpoint = 0),  transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet       Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = [''])
#colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf3_dry_T90 - wrf1_dry_T90), vmin = -0.4, vmax = 1.5, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry       Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = [''])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot5 = axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf4_wet_T90 - wrf3_wet_T90), vmin = -0.4, vmax = 1.5, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet        Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar5 = fig.colorbar(subplot5, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf4_dry_T90 - wrf3_dry_T90), vmin = -0.4, vmax = 1.5, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry         Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar6 = fig.colorbar(subplot6, ax=axlist[5], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot7 = axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf2_wet_T90 - wrf1_wet_T90), vmin = -0.4, vmax = 1.5, cmap = red_blue, norm = MidpointNormalize(midpoint = 0),  transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet         Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
#colorbar7 = fig.colorbar(subplot7, ax=axlist[6], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot8 = axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf2_dry_T90 - wrf1_dry_T90), vmin = -0.4, vmax = 1.5, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry           Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
#colorbar8 = fig.colorbar(subplot8, ax=axlist[7], orientation='vertical', shrink = 0.70, pad = 0.0)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(-0.4, 1.5, 0.2), orientation='vertical', shrink = 0.30, pad = 0.0)
cbar.set_label('(K)')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()


# Plot differences for wet season 99th percentile temps
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10, 10), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf4_wet_T99 - wrf2_wet_T99), vmin = -1.0, vmax = 2.0, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet      Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf4_dry_T99 - wrf2_dry_T99), vmin = -1.0, vmax = 2.0, levels = np.arange(-1.0,2.0,0.1), cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry      Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot3 = axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf3_wet_T99 - wrf1_wet_T99), vmin = -1.0, vmax = 2.0, cmap = red_blue, norm = MidpointNormalize(midpoint = 0),  transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet       Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = [''])
#colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf3_dry_T99 - wrf1_dry_T99), vmin = -1.0, vmax = 2.0, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry       Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = [''])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot5 = axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf4_wet_T99 - wrf3_wet_T99), vmin = -1.0, vmax = 2.0, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet       Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar5 = fig.colorbar(subplot5, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf4_dry_T99 - wrf3_dry_T99), vmin = -1.0, vmax = 2.0, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry       Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar6 = fig.colorbar(subplot6, ax=axlist[5], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot7 = axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf2_wet_T99 - wrf1_wet_T99), vmin = -1.0, vmax = 2.0, cmap = red_blue, norm = MidpointNormalize(midpoint = 0),  transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet       Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
#colorbar7 = fig.colorbar(subplot7, ax=axlist[6], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot8 = axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, wrf2_dry_T99 - wrf1_dry_T99), vmin = -1.0, vmax = 2.0, cmap = red_blue, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry       Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
#colorbar8 = fig.colorbar(subplot8, ax=axlist[7], orientation='vertical', shrink = 0.70, pad = 0.0)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(-1.0, 2.0, .2), orientation='vertical', shrink = 0.30, pad = 0.0)
cbar.set_label('(K)')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()

