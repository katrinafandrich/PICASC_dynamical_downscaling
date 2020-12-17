# Script to plot SSTs and differences
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
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

# Load in temperature data

# Wet season
temps_wet1 = np.load(path+'temps_wet_wrf1_5.npy')
temps_wet2 = np.load(path+'temps_wet_wrf6_10.npy')
temps_wet3 = np.load(path+'temps_wet_wrf11_15.npy')
temps_wet4 = np.load(path+'temps_wet_wrf16_20.npy')
temps_neutral_wet = np.load(path+'temps_wet_neutral1-5.npy')

temps_wet_present = (temps_wet1 + temps_wet2 + temps_neutral_wet) / 3

# Dry season
temps_dry1 = np.load(path+'temps_dry_wrf1_5.npy')
temps_dry2 = np.load(path+'temps_dry_wrf6_10.npy')
temps_dry3 = np.load(path+'temps_dry_wrf11_15.npy')
temps_dry4 = np.load(path+'temps_dry_wrf16_20.npy')
temps_neutral_dry = np.load(path+'temps_dry_neutral1-5.npy')

temps_dry_present = (temps_dry1 + temps_dry2 + temps_neutral_dry) / 3

# TSK - surface skin temp (because SSTs are missing)
TSK_wet1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/TSK_wet_1-5.npy')
TSK_wet2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/TSK_wet_6-10.npy')
TSK_wet_present = (TSK_wet1 + TSK_wet2) / 2

TSK_dry1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/TSK_dry_1-5.npy')
TSK_dry2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/TSK_dry_6-10.npy')
TSK_dry_present = (TSK_dry1 + TSK_dry2) / 2

# NOAA SST climatology
noaa_SST_mask = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/noaa_hires_sst_mask.npy')

noaa_wet_sst = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/noaa_wet_sst_1982-2010_2.npy')
noaa_dry_sst = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/noaa_dry_sst_1982-2010.npy')

#Ryan Longman 2m temp observations
#temps_ann_obs_K = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/obs_2mtemps_ann.npy')
temps_wet_obs_K = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/obs_2mtemps_wet.npy')
temps_dry_obs_K = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/obs_2mtemps_dry.npy')

# 2m Specific humidity
Q2_wet1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/Q2_wet_WRF1_5.npy')
Q2_wet2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/Q2_wet_WRF6_10.npy')
Q2_wet3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/Q2_wet_WRF11_15.npy')
Q2_wet4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/Q2_wet_WRF16_20.npy')

Q2_dry1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/Q2_dry_WRF1_5.npy')
Q2_dry2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/Q2_dry_WRF6_10.npy')
Q2_dry3 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/Q2_dry_WRF11_15.npy')
Q2_dry4 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/Q2_dry_WRF16_20.npy')

dif_q1_wet = Q2_wet4 - Q2_wet2
dif_q2_wet = Q2_wet3 - Q2_wet1
dif_q3_wet = Q2_wet4 - Q2_wet3
dif_q4_wet = Q2_wet2 - Q2_wet1

dif_q1_dry = Q2_dry4 - Q2_dry2
dif_q2_dry = Q2_dry3 - Q2_dry1
dif_q3_dry = Q2_dry4 - Q2_dry3
dif_q4_dry = Q2_dry2 - Q2_dry1

# Differences
dif_T1_wet = temps_wet4 - temps_wet2
dif_T2_wet = temps_wet3 - temps_wet1
dif_T3_wet = temps_wet4 - temps_wet3
dif_T4_wet = temps_wet2 - temps_wet1

dif_T1_dry = temps_dry4 - temps_dry2
dif_T2_dry = temps_dry3 - temps_dry1
dif_T3_dry = temps_dry4 - temps_dry3
dif_T4_dry = temps_dry2 - temps_dry1

# P-values
pvals_dif_T1_wet = np.load(path+'p_vals_dif_T1_wet.npy')
pvals_dif_T2_wet = np.load(path+'p_vals_dif_T2_wet.npy')
pvals_dif_T3_wet = np.load(path+'p_vals_dif_T3_wet.npy')
pvals_dif_T4_wet = np.load(path+'p_vals_dif_T4_wet.npy')

pvals_dif_T1_dry = np.load(path+'p_vals_dif_T1_dry.npy')
pvals_dif_T2_dry = np.load(path+'p_vals_dif_T2_dry.npy')
pvals_dif_T3_dry = np.load(path+'p_vals_dif_T3_dry.npy')
pvals_dif_T4_dry = np.load(path+'p_vals_dif_T4_dry.npy')

###################################################################################################################
###################################################################################################################

# Load info for plotting 
path2 = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'
ncfile = Dataset(path2+'WRF1/wrfout_d02_monthly_mean_1998_11.nc')

# Extract variables
p = getvar(ncfile, "pressure")
p = p*100 # Convert to pascal units
z = getvar(ncfile, "z", units="dm")

# Define levels
plev = np.arange(100*100,1025*100,25*100) # In pascal units
plev = plev[::-1]

# Interpolate variables to any pressure level (converts from hybrid to p coords)
ht_plev = interplevel(z, p, plev)

# Get landmask
dataDIR = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/WRF1/'
filename = 'wrfout_d02_monthly_mean_1997_01.nc'
DS = xarray.open_dataset(dataDIR+filename)

landmask = DS.LANDMASK[0,:,:]
#T_land = np.ma.masked_where(landmask == 0, )

# Wet
SST_wet1 = np.ma.masked_where(landmask == 1, dif_T1_wet)
pval_SST_wet1 = np.ma.masked_where(landmask == 1, pvals_dif_T1_wet)
SST_wet2 = np.ma.masked_where(landmask == 1, dif_T2_wet)
pval_SST_wet2 = np.ma.masked_where(landmask == 1, pvals_dif_T2_wet)
SST_wet3 = np.ma.masked_where(landmask == 1, dif_T3_wet)
pval_SST3 = np.ma.masked_where(landmask == 1, pvals_dif_T3_wet)
SST_wet4 = np.ma.masked_where(landmask == 1, dif_T4_wet)
pval_SST4 = np.ma.masked_where(landmask == 1, pvals_dif_T4_wet)

# Dry
SST_dry1 = np.ma.masked_where(landmask == 1, dif_T1_dry)
pval_SST_dry1 = np.ma.masked_where(landmask == 1, pvals_dif_T1_dry)
SST_dry2 = np.ma.masked_where(landmask == 1, dif_T2_dry)
pval_SST_dry2 = np.ma.masked_where(landmask == 1, pvals_dif_T2_dry)
SST_dry3 = np.ma.masked_where(landmask == 1, dif_T3_dry)
pval_SST3 = np.ma.masked_where(landmask == 1, pvals_dif_T3_dry)
SST_dry4 = np.ma.masked_where(landmask == 1, dif_T4_dry)
pval_SST4 = np.ma.masked_where(landmask == 1, pvals_dif_T4_dry)

height = DS.HGT[0,:,:]

# Get lat/lon coordinates
lats, lons = latlon_coords(ht_plev)

# Get map projection information
cart_proj = get_cartopy(ht_plev)
crs = crs.PlateCarree()

# Choose p-values to plot
no_pvals = np.load(path+'no_border.npy')

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
T = '(K)'

# Color schemes
hot = plt.cm.YlOrRd
cool = plt.cm.YlGnBu_r
red_blue = plt.cm.RdBu_r
red_yl_bu = plt.cm.RdYlBu_r
spectral = plt.cm.Spectral_r
autumn = plt.cm.afmhot_r

x = lons[0,:]
y = lats[:,0]

'''
# Plot wet and dry season SST differences
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10, 10), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

vmin_hot = SST_wet1.min()
vmax_hot = SST_dry2.max()
ticks_hot = np.arange(vmin_hot, vmax_hot, 0.04)

vmin_cool = SST_wet3.min()
vmax_cool = SST_wet4.max()
ticks_cool = np.arange(vmin_cool, vmax_cool, 0.04)

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 1, SST_wet1), vmin = vmin_hot, vmax = vmax_hot, cmap = hot, transform = crs)
axlist[0].contourf(lons, lats, pval_SST_wet1, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[0].set_title('(a) wet      Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_xticklabels(labels = [''])
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 1, SST_dry1), cmap = hot, vmin = vmin_hot, vmax = vmax_hot, transform = crs)
axlist[1].contourf(lons, lats, pval_SST_dry1, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_title('(b) dry      Fut(+) - Pres(+)', fontsize = 10)

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 1, SST_wet2), vmin = vmin_hot, vmax = vmax_hot, cmap = hot, transform = crs)
axlist[2].contourf(lons, lats, pval_SST_wet2, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[2].set_title('(c) wet      Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 1, SST_dry2), cmap = hot, vmin = vmin_hot, vmax = vmax_hot, transform = crs)
axlist[3].contourf(lons, lats, pval_SST_dry2, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = [''])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_title('(d) dry      Fut(-) - Pres(-)', fontsize = 10)


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.525, 0.02, 0.35])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(np.around(vmin_hot,2), np.around(vmax_hot,2), 0.04), orientation='vertical', shrink = 0.60, pad = 0.0)
cbar.set_label(T)

subplot5 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == 1, SST_wet3), cmap = cool, vmin = vmin_cool, vmax = vmax_cool, transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet      Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))

subplot6 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == 1, SST_dry3), cmap = cool, vmin = vmin_cool, vmax = vmax_cool, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry      Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))

subplot7 = axlist[6].pcolor(lons, lats, np.ma.masked_where(landmask == 1, SST_wet4), cmap = cool, vmin = vmin_cool, vmax = vmax_cool, transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet      Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'], size = 'small')

subplot8 = axlist[7].pcolor(lons, lats, np.ma.masked_where(landmask == 1, SST_dry4), cmap = cool, vmin = vmin_cool, vmax = vmax_cool, transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry      Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'], size = 'small')
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.125, 0.02, 0.35])
cbar = fig.colorbar(subplot6, cax=cbar_ax, ticks = np.arange(np.around(vmin_cool,2), np.around(vmax_cool,2), 0.04), orientation='vertical', shrink = 0.60, pad = 0.0)
cbar.set_label(T)

fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.02, wspace=0.0)
plt.show()


# Plot wet and dry season 2m temp difs over land
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10,10), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

ticks_all = np.arange(dif_T3_wet.min(), dif_T1_wet.max(), 0.25)
vmin_all = dif_T3_wet.min()
vmax_all = dif_T1_wet.max()

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_T1_wet), vmin = vmin_all, vmax = vmax_all, norm = MidpointNormalize(midpoint = 0), cmap = spectral, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[0].set_title('(a) wet      Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_xticklabels(labels = [''])

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_T1_dry), cmap = spectral, vmin = vmin_all, vmax = vmax_all, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[1].set_title('(b) dry      Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[1].set_xticklabels(labels = [''])

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_T2_wet), cmap = spectral, vmin = vmin_all, vmax = vmax_all, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[2].set_title('(c) wet      Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = [''])

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_T2_dry), cmap = spectral, vmin = vmin_all, vmax = vmax_all, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[3].set_title('(d) dry      Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = [''])

subplot5 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_T3_wet), cmap = spectral, vmin = vmin_all, vmax = vmax_all, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet      Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = [''])

subplot6 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_T3_dry), cmap = spectral, vmin = vmin_all, vmax = vmax_all, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry      Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = [''])
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

subplot7 = axlist[6].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_T4_wet), cmap = spectral, vmin = vmin_all, vmax = vmax_all, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet      Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'], size = 'small')

subplot8 = axlist[7].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_T4_dry), cmap = spectral, vmin = vmin_all, vmax = vmax_all, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry      Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'], size = 'small')
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot6, cax=cbar_ax, ticks = np.arange(np.around(vmin_all,2), np.around(vmax_all,2), 0.15), orientation='vertical', shrink = 0.60, pad = 0.0)
cbar.set_label(T)

fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)
plt.show()


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


# Plot 2m temperatures for observations and present-day WRF simulations
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, temps_wet_obs_K), vmin = 272, vmax = 306, cmap = autumn, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet           Observations', fontsize = 12)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, temps_wet_present), vmin = 272, vmax = 306, levels = np.arange(272,306.2,2), cmap = autumn, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) wet           WRF', fontsize = 12)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))

# Add markers for stations
hilo = axlist[1].plot(-155.05, 19.72, color='black', marker='o', markersize = 2.5)
bigbog = axlist[1].plot(-156.092, 20.73, color='black', marker='o', markersize = 2.5)
honolulu = axlist[1].plot(-157.94, 21.32, color='black', marker='o', markersize = 2.5)
lihue = axlist[1].plot(-159.34, 21.98, color='black', marker='o', markersize = 2.5)

# Add labels to markers
axlist[1].annotate(' Hilo', xy = (-155.05,19.72), size = 9)
axlist[1].annotate('  Big Bog', xy = (-156.092,20.73), size = 9)
axlist[1].annotate('    Honolulu', xy = (-159.0,21.0), size = 9)
axlist[1].annotate(' Lihue', xy = (-159.34,21.98), size = 9)

subplot3 = axlist[2].contourf(lons_sub, lats_sub, np.ma.masked_where(coarse.mask[:] == True, temps_dry_obs_K), vmin = 272, vmax = 306, cmap = autumn, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) dry            Observations', fontsize = 12)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, temps_dry_present), vmin = 272, vmax = 306, cmap = autumn, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry             WRF', fontsize = 12)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(272, 306.2, 4), orientation='vertical', shrink = 0.30, pad = 0.0)
cbar.set_label(T)

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()


#Plot SSTs for NOAA observations and compare to present-day WRF simulations
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

path1 = '/network/rit/lab/elisontimmlab_rit/kf835882/python/noaa_hires_sst/clim_mean_1982_2010/'
infile1 = 'sst.day.mean.ltm.1982-2010.nc'

mask = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/noaa_sst_mask2.npy')

subplot1 = axlist[0].contourf(lons, lats, np.ma.masked_where(mask==True, noaa_wet_sst + 273.15), vmin = 296, vmax = 300, cmap = autumn, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet           Observations', fontsize = 12)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 1,TSK_wet_present), vmin = 296, vmax = 300, levels = np.arange(296, 300.2, .2), cmap = autumn, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) wet           WRF', fontsize = 12)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))

subplot3 = axlist[2].contourf(lons, lats, np.ma.masked_where(mask == True, noaa_dry_sst + 273.15), vmin = 296, vmax = 300, cmap = autumn, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) dry            Observations', fontsize = 12)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 1, TSK_dry_present), vmin = 296, vmax = 300, cmap = autumn, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry             WRF', fontsize = 12)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(296, 300.2, .4), orientation='vertical', shrink = 0.30, pad = 0.0)
cbar.set_label(T)

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()
'''

# Plot wet and dry season 2m specific humidity difs over land
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10,10), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)


subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_q1_wet*1000), vmin = -0.400, vmax = 0.9, norm = MidpointNormalize(midpoint = 0), cmap = spectral, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[0].set_title('(a) wet      Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_xticklabels(labels = [''])
#colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar1.set_label(mm_day, size = 'medium')

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_q1_dry*1000), vmin = -0.400, vmax = 0.9, cmap = spectral, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[1].set_title('(b) dry      Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[1].set_xticklabels(labels = [''])
#colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar2.set_label(mm_day, size = 'medium')

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_q2_wet*1000), vmin = -0.400, vmax = 0.9, cmap = spectral, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[2].set_title('(c) wet      Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = [''])
#colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar3.set_label(mm_day, size = 'medium')

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_q2_dry*1000), vmin = -0.400, vmax = 0.9, cmap = spectral, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0.0, transform = crs)
axlist[3].set_title('(d) dry      Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = [''])
#colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar4.set_label(mm_day, size = 'medium')

subplot5 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_q3_wet*1000), vmin = -0.400, vmax = 0.9, cmap = spectral, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet      Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = [''])
#colorbar5 = fig.colorbar(subplot5, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar5.set_label(mm_day, size = 'medium')

subplot6 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_q3_dry*1000), vmin = -0.400, vmax = 0.9, cmap = spectral, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry      Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = [''])
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar6 = fig.colorbar(subplot6, ax=axlist[5], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar6.set_label(mm_day, size = 'medium')

subplot7 = axlist[6].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_q4_wet*1000), vmin = -0.400, vmax = 0.9, cmap = spectral, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet      Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'], size = 'small')
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'], size = 'small')
#colorbar7 = fig.colorbar(subplot7, ax=axlist[6], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar7.set_label(mm_day, size = 'medium')

subplot8 = axlist[7].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_q4_dry*1000), vmin = -0.400, vmax = 0.9, cmap = spectral, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry      Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'], size = 'small')
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar8 = fig.colorbar(subplot8, ax=axlist[7], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar8.set_label(mm_day, size = 'medium')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(-0.400, 0.9, .1), orientation='vertical', shrink = 0.60, pad = 0.0)
cbar.set_label('(g kg$^{-1}$)')

fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)
plt.show()