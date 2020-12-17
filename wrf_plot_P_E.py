# Script for plotting rainfall maps

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

# Wet season
rain_wet1 = np.load(path+'rain_wet_wrf1_5.npy')
E_wet1 = np.load(path+'E_wet1_5.npy')
P_wet1 = rain_wet1 / 181 # Seasonal rainfall in mm/day
PE_wet1 = P_wet1 - E_wet1

rain_wet2 = np.load(path+'rain_wet_wrf6_10.npy')
E_wet2 = np.load(path+'E_wet6_10.npy')
P_wet2 = rain_wet2 / 181
PE_wet2 = P_wet2 - E_wet2

rain_wet3 = np.load(path+'rain_wet_wrf11_15.npy')
E_wet3 = np.load(path+'E_wet11_15.npy')
P_wet3 = rain_wet3 / 181
PE_wet3 = P_wet3 - E_wet3

rain_wet4 = np.load(path+'rain_wet_wrf16_20.npy')
E_wet4 = np.load(path+'E_wet16_20.npy')
P_wet4 = rain_wet4 / 181
PE_wet4 = P_wet4 - E_wet4

pvals_E1_wet = np.load(path+'p_vals_dif_E1_wet.npy')
pvals_E2_wet = np.load(path+'p_vals_dif_E2_wet.npy')
pvals_E3_wet = np.load(path+'p_vals_dif_E3_wet.npy')
pvals_E4_wet = np.load(path+'p_vals_dif_E4_wet.npy')

pvals_PE1_wet = np.load(path+'p_vals_dif_PE1_wet.npy')
pvals_PE2_wet = np.load(path+'p_vals_dif_PE2_wet.npy')
pvals_PE3_wet = np.load(path+'p_vals_dif_PE3_wet.npy')
pvals_PE4_wet = np.load(path+'p_vals_dif_PE4_wet.npy')

neut_rain_wet_pres = np.load(path+'neutral_rain_wet_wrf1_5.npy')
neut_rain_wet_fut = np.load(path+'neutral_rain_wet_wrf6_10.npy')
neutral_rain_wet = (neut_rain_wet_pres + neut_rain_wet_fut) / 2

# Dry season
rain_dry1 = np.load(path+'rain_dry_wrf1_5.npy')
E_dry1 = np.load(path+'E_dry1_5.npy')
P_dry1 = rain_dry1 / 184
PE_dry1 = P_dry1 - E_dry1

rain_dry2 = np.load(path+'rain_dry_wrf6_10.npy')
E_dry2 = np.load(path+'E_dry6_10.npy')
P_dry2 = rain_dry2 / 184
PE_dry2 = P_dry2 - E_dry2

rain_dry3 = np.load(path+'rain_dry_wrf11_15.npy')
E_dry3 = np.load(path+'E_dry11_15.npy')
P_dry3 = rain_dry3 / 184
PE_dry3 = P_dry3 - E_dry3

rain_dry4 = np.load(path+'rain_dry_wrf16_20.npy')
E_dry4 = np.load(path+'E_dry16_20.npy')
P_dry4 = rain_dry4 / 184
PE_dry4 = P_dry4 - E_dry4

pvals_E1_dry = np.load(path+'p_vals_dif_E1_dry.npy')
pvals_E2_dry = np.load(path+'p_vals_dif_E2_dry.npy')
pvals_E3_dry = np.load(path+'p_vals_dif_E3_dry.npy')
pvals_E4_dry = np.load(path+'p_vals_dif_E4_dry.npy')

pvals_PE1_dry = np.load(path+'p_vals_dif_PE1_dry.npy')
pvals_PE2_dry = np.load(path+'p_vals_dif_PE2_dry.npy')
pvals_PE3_dry = np.load(path+'p_vals_dif_PE3_dry.npy')
pvals_PE4_dry = np.load(path+'p_vals_dif_PE4_dry.npy')

neut_rain_dry_pres = np.load(path+'neutral_rain_dry_wrf1_5.npy')
neut_rain_dry_fut = np.load(path+'neutral_rain_dry_wrf6_10.npy')
neutral_rain_dry = (neut_rain_dry_pres + neut_rain_dry_fut) / 2
present_rain_wet = np.load(path+'present-day_rain_wet.npy')
present_rain_dry = np.load(path+'present-day_rain_dry.npy')
RainfallAtlas_wet = np.load(path+'RainfallAtlas_wet.npy')
RainfallAtlas_dry = np.load(path+'RainfallAtlas_dry.npy')

# Differences

# Wet season
dif_P1_wet = rain_wet4 - rain_wet2
pvals_dif_P1_wet = np.load(path+'p_vals_dif_P1_wet.npy')

dif_P2_wet = rain_wet3 - rain_wet1
pvals_dif_P2_wet = np.load(path+'p_vals_dif_P2_wet.npy')

dif_P3_wet = rain_wet4 - rain_wet3
pvals_dif_P3_wet = np.load(path+'p_vals_dif_P3_wet.npy')

dif_P4_wet = rain_wet2 - rain_wet1
pvals_dif_P4_wet = np.load(path+'p_vals_dif_P4_wet.npy')

dif_PE1_wet = PE_wet4 - PE_wet2
dif_PE2_wet = PE_wet3 - PE_wet1
dif_PE3_wet = PE_wet4 - PE_wet3
dif_PE4_wet = PE_wet2 - PE_wet1

# Dry season
dif_P1_dry = rain_dry4 - rain_dry2
pvals_dif_P1_dry = np.load(path+'p_vals_dif_P1_dry.npy')

dif_P2_dry = rain_dry3 - rain_dry1
pvals_dif_P2_dry = np.load(path+'p_vals_dif_P2_dry.npy')

dif_P3_dry = rain_dry4 - rain_dry3
pvals_dif_P3_dry = np.load(path+'p_vals_dif_P3_dry.npy')

dif_P4_dry = rain_dry2 - rain_dry1
pvals_dif_P4_dry = np.load(path+'p_vals_dif_P4_dry.npy')

dif_PE1_dry = PE_dry4 - PE_dry2
dif_PE2_dry = PE_dry3 - PE_dry1
dif_PE3_dry = PE_dry4 - PE_dry3
dif_PE4_dry = PE_dry2 - PE_dry1

###################################################################################################################

no_pvals = np.load(path+'no_border.npy')

###################################################################################################################
# Load info for plotting 
path2 = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'
ncfile = Dataset(path2+'WRF1/wrfout_d02_monthly_mean_1998_11.nc')

# Get landmask and height
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
mm_day = '(mm day$^{-1}$)'
mm = '(mm)'

# Color schemes
wet_dry = plt.cm.BrBG
wet_dry_r = plt.cm.BrBG_r
rainbow = plt.cm.gist_rainbow_r
hot = plt.cm.YlOrRd

x = lons[0,:]
y = lats[:,0]

'''
# Plot wet season rainfall differences
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_P1_wet/181), vmin = -1.0, vmax = 5.0, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_dif_P1_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet     Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar1 = fig.colorbar(subplot1, ax=axlist[0],orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar1.set_label(mm_day, size = 'small')

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_P1_dry/184), vmin = -1.5, vmax = 2.5, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_dif_P1_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry    Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar2 = fig.colorbar(subplot2, ax=axlist[1], ticks = np.arange(-1.5,2.5,1), orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar2.set_label(mm_day, size = 'small')

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_P2_wet/181), vmin = -0.5, vmax = 2.5, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_dif_P2_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet    Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels('')
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar3.set_label(mm_day, size = 'small')

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_P2_dry/184), vmin = -3.0, vmax = 2.0, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_dif_P2_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry    Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels('')
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar4.set_label(mm_day, size = 'small')

subplot5 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_P3_wet/181), vmin = -1.0, vmax = 4.0, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_dif_P3_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet    Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_xticklabels('')
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar5 = fig.colorbar(subplot5, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar5.set_label(mm_day, size = 'small')

subplot6 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_P3_dry/184), vmin = -3.0, vmax = 1.1, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_dif_P3_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry    Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar6 = fig.colorbar(subplot6, ax=axlist[5], ticks = np.arange(-3.0,1.1,1), orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar6.set_label(mm_day, size = 'small')

subplot7 = axlist[6].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_P4_wet/181), vmin = -1.0, vmax = 1.1, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_dif_P4_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet    Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar7 = fig.colorbar(subplot7, ax=axlist[6], ticks = np.arange(-1.0,1.1,0.5), orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar7.set_label(mm_day, size = 'small')

subplot8 = axlist[7].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_P4_dry/184), vmin = -4.0, vmax = 0.6, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_dif_P4_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry    Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar8 = fig.colorbar(subplot8, ax=axlist[7], ticks = np.arange(-4.0,0.6,1), orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar8.set_label(mm_day, size = 'small')

fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.6, 0.725, 0.01, 0.15])
cbar1 = fig.colorbar(subplot1, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.725, 0.01, 0.15])
cbar2 = fig.colorbar(subplot2, cax=cbar_ax2, ticks = np.arange(-1.5,2.5,1), orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.6, 0.525, 0.01, 0.15])
cbar3 = fig.colorbar(subplot3, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.525, 0.01, 0.15])
cbar4 = fig.colorbar(subplot4, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label(mm_day, size = 'small')
cbar4.ax.tick_params(labelsize = 'small')

cbar_ax5 = fig.add_axes([0.6, 0.325, 0.01, 0.15])
cbar5 = fig.colorbar(subplot5, cax=cbar_ax5, orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar5.set_label(mm_day, size = 'small')
cbar5.ax.tick_params(labelsize = 'small')

cbar_ax6 = fig.add_axes([0.8, 0.325, 0.01, 0.15])
cbar6 = fig.colorbar(subplot6, cax=cbar_ax6, ticks = np.arange(-3.0,1.1,1), orientation='vertical', shrink = 0.50, pad = 0.0)
cbar6.set_label(mm_day, size = 'small')
cbar6.ax.tick_params(labelsize = 'small')

cbar_ax7 = fig.add_axes([0.6, 0.125, 0.01, 0.15])
cbar7 = fig.colorbar(subplot7, cax=cbar_ax7, ticks = np.arange(-1.0,1.1,0.5), orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar7.set_label(mm_day, size = 'small')
cbar7.ax.tick_params(labelsize = 'small')

cbar_ax8 = fig.add_axes([0.8, 0.125, 0.01, 0.15])
cbar8 = fig.colorbar(subplot8, cax=cbar_ax8, ticks = np.arange(-4.0,0.6,1), orientation='vertical', shrink = 0.50, pad = 0.0)
cbar8.set_label(mm_day, size = 'small')
cbar8.ax.tick_params(labelsize = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()
'''
'''
# Plot wet season evaporation
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex = 'col', sharey = 'row', dpi = 85, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, E_wet1), cmap = hot, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('$E_{Pres(-)}$', fontsize = 14)
axlist[0].set_yticks(ticks = np.around(np.arange(y.min()+0.9, y.max(),2), decimals = 1))
axlist[0].set_ylabel('Latitude')
colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, E_wet2), cmap = hot, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('$E_{Pres(+)}$', fontsize = 14)
colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar2.set_label(mm_day, size = 'small')

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, E_wet3), cmap = hot, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('$E_{Fut(-)}$', fontsize = 14)
axlist[2].set_yticks(np.around(np.arange(y.min()+0.9, y.max(),2), decimals = 1))
axlist[2].set_ylabel('Latitude')
axlist[2].set_xticks(ticks = np.around(np.arange(x.min()+1.2, x.max(),2), decimals = 1))
axlist[2].set_xlabel('Longitude')
colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, E_wet4), cmap = hot, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('$E_{Fut(+)}$', fontsize = 14)
axlist[3].set_xticks(ticks = np.around(np.arange(x.min()+1.2, x.max(),2), decimals = 1))
axlist[3].set_xlabel('Longitude')
colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar4.set_label(mm_day, size = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
fig.suptitle('Wet season', fontsize = 16)
plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/E_wet.png')
plt.show()

# Plot dry season evaporation
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex = 'col', sharey = 'row', dpi = 85, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, E_dry1), cmap = hot, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('$E_{Pres(-)}$', fontsize = 14)
axlist[0].set_yticks(ticks = np.around(np.arange(y.min()+0.9, y.max(),2), decimals = 1))
colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, E_dry2), cmap = hot, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('$E_{Pres(+)}$', fontsize = 14)
colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar2.set_label(mm_day, size = 'small')

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, E_dry3), cmap = hot, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('$E_{Fut(-)}$', fontsize = 14)
axlist[2].set_yticks(np.around(np.arange(y.min()+0.9, y.max(),2), decimals = 1))
axlist[2].set_xticks(ticks = np.around(np.arange(x.min()+1.2, x.max(),2), decimals = 1))
colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, E_dry4), cmap = hot, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('$E_{Fut(+)}$', fontsize = 14)
axlist[3].set_xticks(ticks = np.around(np.arange(x.min()+1.2, x.max(),2), decimals = 1))
colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar4.set_label(mm_day, size = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
fig.suptitle('Dry season', fontsize = 16)
plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/E_dry.png')
plt.show()


# Plot wet season evaporation differences
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)


subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, (E_wet4 - E_wet2)), cmap = wet_dry_r, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet       Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, (E_dry4 - E_dry2)), cmap = wet_dry_r, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry         Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar2.set_label('(mm day$^{-1}$)', size = 'small')

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, (E_wet3 - E_wet1)), cmap = wet_dry_r, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet         Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_ylabel('Latitude', size = 'small')
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = [''])
axlist[2].set_xlabel('', size = 'small')
colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, (E_dry3 - E_dry1)), cmap = wet_dry_r, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry         Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = [''])
axlist[3].set_xlabel('', size = 'small')
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar4.set_label('(mm day$^{-1}$)', size = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()

# Plot dry season evaporation differences
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, (E_wet4 - E_wet3)), cmap = wet_dry_r, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_E3_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(e) wet        Fut(+) - Fut(-)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, (E_dry4 - E_dry3)), cmap = wet_dry_r, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_E3_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(f) dry       Fut(+) - Fut(-)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar2.set_label('(mm day$^{-1}$)', size = 'small')

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, (E_wet2 - E_wet1)), cmap = wet_dry_r, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_E4_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(g) wet        Pres(+) - Pres(-) ', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, (E_dry2 - E_dry1)), cmap = wet_dry_r, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_E4_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(h) dry         Pres(+) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar4.set_label('(mm day$^{-1}$)', size = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()


# Plot P-E for wet season
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize = (10, 6), sharex = 'col', sharey = 'row', dpi = 85 , constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, PE_wet1), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('$P-E_{Pres(-)}$', fontsize = 14)
axlist[0].set_yticks(ticks = np.around(np.arange(y.min()+0.9, y.max(),2), decimals = 1))
colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, PE_wet2), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('$P-E_{Pres(+)}$', fontsize = 14)
colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar2.set_label(mm_day, size = 'small')

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, PE_wet3), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('$P-E_{Fut(-)}$', fontsize = 14)
axlist[2].set_yticks(np.around(np.arange(y.min()+0.9, y.max(),2), decimals = 1))
axlist[2].set_xticks(ticks = np.around(np.arange(x.min()+1.2, x.max(),2), decimals = 1))
colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, PE_wet4), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('$P-E_{Fut(+)}$', fontsize = 14)
axlist[3].set_xticks(ticks = np.around(np.arange(x.min()+1.2, x.max(),2), decimals = 1))
colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar4.set_label(mm_day, size = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
fig.suptitle('Wet season', fontsize = 16)
plt.show()

# Plot P-E for dry season
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize = (10, 6), sharex = 'col', sharey = 'row', dpi = 85 , constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, PE_dry1), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('$P-E_{Pres(-)}$', fontsize = 14)
axlist[0].set_yticks(ticks = np.around(np.arange(y.min()+0.9, y.max(),2), decimals = 1))
colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, PE_dry2), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('$P-E_{Pres(+)}$', fontsize = 14)
colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar2.set_label(mm_day, size = 'small')

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, PE_dry3), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('$P-E_{Fut(-)}$', fontsize = 14)
axlist[2].set_yticks(np.around(np.arange(y.min()+0.9, y.max(),2), decimals = 1))
axlist[2].set_xticks(ticks = np.around(np.arange(x.min()+1.2, x.max(),2), decimals = 1))
colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, PE_dry4), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('$P-E_{Fut(+)}$', fontsize = 14)
axlist[3].set_xticks(ticks = np.around(np.arange(x.min()+1.2, x.max(),2), decimals = 1))
colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar4.set_label(mm_day, size = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
fig.suptitle('Dry season', fontsize = 16)
plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/P-E_dry.png')
plt.show()
'''

# Plot P-E difs for wet season
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize = (10,8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_PE1_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_PE1_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet      Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar1.set_label('', size = 'small')

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_PE1_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_PE1_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry       Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar2.set_label('(mm day$^{-1}$)', size = 'small')

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_PE2_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_PE2_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet       Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_PE2_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_PE2_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry       Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar4.set_label('(mm day$^{-1}$)', size = 'small')

subplot5 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_PE3_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == 0,pvals_PE3_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(d) wet       Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar5 = fig.colorbar(subplot5, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot6 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_PE3_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_PE3_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(e) dry       Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar6 = fig.colorbar(subplot6, ax=axlist[5], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar6.set_label('(mm day$^{-1}$)', size = 'small')

subplot7 = axlist[6].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_PE4_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_PE4_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(f) wet       Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar7 = fig.colorbar(subplot7, ax=axlist[6], orientation='vertical', shrink = 0.70, pad = 0.0)

subplot8 = axlist[7].pcolor(lons, lats, np.ma.masked_where(landmask == 0, dif_PE4_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == 0, pvals_PE4_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(g) dry       Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar8 = fig.colorbar(subplot8, ax=axlist[7], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar8.set_label('(mm day$^{-1}$)', size = 'small')

fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.6, 0.725, 0.01, 0.15])
cbar1 = fig.colorbar(subplot1, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.725, 0.01, 0.15])
cbar2 = fig.colorbar(subplot2, cax=cbar_ax2, ticks = np.arange(-1.5,2.5,1), orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.6, 0.525, 0.01, 0.15])
cbar3 = fig.colorbar(subplot3, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.525, 0.01, 0.15])
cbar4 = fig.colorbar(subplot4, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label(mm_day, size = 'small')
cbar4.ax.tick_params(labelsize = 'small')

cbar_ax5 = fig.add_axes([0.6, 0.325, 0.01, 0.15])
cbar5 = fig.colorbar(subplot5, cax=cbar_ax5, orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar5.set_label(mm_day, size = 'small')
cbar5.ax.tick_params(labelsize = 'small')

cbar_ax6 = fig.add_axes([0.8, 0.325, 0.01, 0.15])
cbar6 = fig.colorbar(subplot6, cax=cbar_ax6, ticks = np.arange(-3.0,1.1,1), orientation='vertical', shrink = 0.50, pad = 0.0)
cbar6.set_label(mm_day, size = 'small')
cbar6.ax.tick_params(labelsize = 'small')

cbar_ax7 = fig.add_axes([0.6, 0.125, 0.01, 0.15])
cbar7 = fig.colorbar(subplot7, cax=cbar_ax7, ticks = np.arange(-1.0,1.1,0.5), orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar7.set_label(mm_day, size = 'small')
cbar7.ax.tick_params(labelsize = 'small')

cbar_ax8 = fig.add_axes([0.8, 0.125, 0.01, 0.15])
cbar8 = fig.colorbar(subplot8, cax=cbar_ax8, ticks = np.arange(-4.0,0.6,1), orientation='vertical', shrink = 0.50, pad = 0.0)
cbar8.set_label(mm_day, size = 'small')
cbar8.ax.tick_params(labelsize = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()

'''
# Plot present day rainfall for wet and dry season (compare to Rainfall Atlas)
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == 0, present_rain_wet), vmin = 0, vmax = 9000, cmap = rainbow, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) wet           WRF', fontsize = 12)
axlist[1].set_yticks(ticks = np.around(np.arange(y.min()+0.9, y.max(),2), decimals = 1))

subplot2 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == 0, present_rain_dry), vmin = 0, vmax = 9000, levels = np.arange(0,9000,500), cmap = rainbow, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry           WRF', fontsize = 12)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(y.min()+0.9, y.max(),2), decimals = 1))

# Get lat/lon info for RainfallAtlas maps
dpath = 'http://apdrc.soest.hawaii.edu:80/dods/public_data/Model_output/statistical_downscale_Hawaii/Scenario_maps/climatology/250m/1978-2007/HawaiianIslands_mm_wet'
ref = xarray.open_dataset(dpath)

lat = ref.lat
lon = ref.lon

y2 = lat
x2 = lon

subplot3 = axlist[0].contourf(lon, lat, RainfallAtlas_wet, vmin = 0, vmax = 9000, levels = np.arange(0,9000,500), cmap = rainbow, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet           Observations', fontsize = 12)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot4 = axlist[2].contourf(lon, lat, RainfallAtlas_dry, vmin = 0, vmax = 9000, levels = np.arange(0,9000,500), cmap = rainbow, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) dry           Observations', fontsize = 12)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])

# Add markers for stations
hilo = axlist[1].plot(-155.05, 19.72, color='black', marker='o', markersize = 2.5)
bigbog = axlist[1].plot(-156.092, 20.73, color='black', marker='o', markersize = 2.5)
honolulu = axlist[1].plot(-157.94, 21.32, color='black', marker='o', markersize = 2.5)
lihue = axlist[1].plot(-159.34, 21.98, color='black', marker='o', markersize = 2.5)

# Add labels to markers
axlist[1].annotate(' Hilo', xy = (-155.05,19.72), size = 8)
axlist[1].annotate('  Big Bog', xy = (-156.092,20.73), size = 8)
axlist[1].annotate('Honolulu', xy = (-158.9,21.0), size = 8)
axlist[1].annotate(' Lihue', xy = (-159.34,21.98), size = 8)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(0, 9000, 1000), orientation='vertical', shrink = 0.50, pad = 0.0)
cbar.set_label('(mm)', size = 'medium')
cbar.ax.tick_params(labelsize = 'medium')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.0, wspace=0.0)
plt.show()
'''
'''
# Plot the inner domain with topography
fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

ocn = np.ma.masked_where(landmask == 1, height)
axarr.pcolormesh(lons,lats, ocn, color = 'lightblue')

hgt = axarr.contourf(lons, lats, np.ma.masked_where(landmask == 0, height), transform = crs, cmap = 'terrain', levels = np.arange(0, 4500, 250))
axarr.contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)

colorbar = fig.colorbar(hgt, ticks = np.arange(0, 4500, 250), orientation='horizontal', shrink = 0.60, pad = 0.0)
colorbar.set_label('Height (m)', size = 'small')
colorbar.ax.tick_params(labelsize = 8, labelrotation = 45)

axarr.set_yticks(ticks = np.around(np.arange(lats.min()+0.4, lats.max(),1), decimals = 1))
axarr.set_yticklabels(labels = ['19$^\circ$N','20$^\circ$N','21$^\circ$N','22$^\circ$N','23$^\circ$N'], fontsize = 10)
axarr.set_ylabel('Latitude', size = 'small')
axarr.set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),1), decimals = 1))
axarr.set_xticklabels(labels = ['160$^\circ$W','159$^\circ$W','158$^\circ$W','157$^\circ$W','156$^\circ$W','155$^\circ$W','154$^\circ$W'], fontsize = 10)
axarr.set_xlabel('Longitude', size = 'small')

# Add markers for stations
hilo = axarr.plot(-155.05, 19.72, color='black', marker='o', markersize = 4)
bigbog = axarr.plot(-156.092, 20.73, color='black', marker='o', markersize = 4)
honolulu = axarr.plot(-157.94, 21.32, color='black', marker='o', markersize = 4)
lihue = axarr.plot(-159.34, 21.98, color='black', marker='o', markersize = 4)

# Add labels to markers
axarr.annotate(' Hilo', xy = (-155.05,19.72), size = 12)
axarr.annotate('   Big Bog', xy = (-156.092,20.73), size = 12)
axarr.annotate(' Honolulu', xy = (-158.7,21.0), size = 12)
axarr.annotate(' Lihue', xy = (-159.34,21.98), size = 12)

plt.show()
'''
'''
# Plot future rainfall changes as percentages
percent1_wet = (dif_P1_wet / present_rain_wet) * 100
percent2_wet = (dif_P2_wet / present_rain_wet) * 100
percent3_wet = (dif_P3_wet / neutral_rain_wet) * 100
percent4_wet = (dif_P4_wet / neutral_rain_wet) * 100

percent1_dry = (dif_P1_dry / present_rain_dry) * 100
percent2_dry = (dif_P2_dry / present_rain_dry) * 100
percent3_dry = (dif_P3_dry / neutral_rain_dry) * 100
percent4_dry = (dif_P4_dry / neutral_rain_dry) * 100

fig, axarr = plt.subplots(nrows=4, ncols=2, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == 0, percent1_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet       Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == 0, percent1_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry       Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == 0, percent2_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet       Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == 0, percent2_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry       Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

subplot5 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == 0, percent3_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet       Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot6 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == 0, percent3_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry       Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))

subplot7 = axlist[6].pcolor(lons, lats, np.ma.masked_where(landmask == 0, percent4_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet       Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot8 = axlist[7].pcolor(lons, lats, np.ma.masked_where(landmask == 0, percent4_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry       Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))

fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.6, 0.725, 0.01, 0.15])
cbar1 = fig.colorbar(subplot1, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.725, 0.01, 0.15])
cbar2 = fig.colorbar(subplot2, cax=cbar_ax2, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label('(%)', size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.6, 0.525, 0.01, 0.15])
cbar3 = fig.colorbar(subplot3, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.525, 0.01, 0.15])
cbar4 = fig.colorbar(subplot4, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label('(%)', size = 'small')
cbar4.ax.tick_params(labelsize = 'small')

cbar_ax5 = fig.add_axes([0.6, 0.325, 0.01, 0.15])
cbar5 = fig.colorbar(subplot5, cax=cbar_ax5, orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar5.set_label(mm_day, size = 'small')
cbar5.ax.tick_params(labelsize = 'small')

cbar_ax6 = fig.add_axes([0.8, 0.325, 0.01, 0.15])
cbar6 = fig.colorbar(subplot6, cax=cbar_ax6, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar6.set_label('(%)', size = 'small')
cbar6.ax.tick_params(labelsize = 'small')

cbar_ax7 = fig.add_axes([0.6, 0.125, 0.01, 0.15])
cbar7 = fig.colorbar(subplot7, cax=cbar_ax7, orientation='vertical', shrink = 0.50, pad = 0.0)
#cbar7.set_label(mm_day, size = 'small')
cbar7.ax.tick_params(labelsize = 'small')

cbar_ax8 = fig.add_axes([0.8, 0.125, 0.01, 0.15])
cbar8 = fig.colorbar(subplot8, cax=cbar_ax8, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar8.set_label('(%)', size = 'small')
cbar8.ax.tick_params(labelsize = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()
'''

'''
# Plot Abby's rainfall/PDO composites and compare to present PDO+/- WRF sims
# Rainfall is in inches per season/ convert to mm
# We divide the PDO composites by a reference state to reveal the signal from the PDO
frazier_mask = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/frazier_mask.npy')
neg_PDO_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/frazier_neg_PDO_wet.npy')
pos_PDO_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/frazier_pos_PDO_wet.npy')
neg_PDO_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/frazier_neg_PDO_dry.npy')
pos_PDO_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/frazier_pos_PDO_dry.npy')

path3 = '/network/rit/lab/elisontimmlab_rit/DATA/RainfallAtlasHawaii/PDO_Composites_TIF/derived_results_with_QGIS/'
infile3 = 'PDO_RF_Composite_Inches_WarmPhase_Nov_Apr.nc'

d = Dataset(path3+infile3, mode = 'r')
lats2 = d.variables['lat'][:]
lons2 = d.variables['lon'][:]

fig, axarr = plt.subplots(nrows=4, ncols=2, figsize = (10, 10), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons2, lats2, np.ma.masked_where(frazier_mask == True, ((neg_PDO_wet*25.4)/RainfallAtlas_wet)), vmin = 0.4, vmax = 1.8, cmap = wet_dry, norm = MidpointNormalize(midpoint = 1), transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet/PDO(-)      Obs', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
#axlist[0].set_ylabel('Latitude', size = 'medium')
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.8, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == False, (rain_wet1/present_rain_wet)), vmin = 0.4, vmax = 1.8, levels = np.arange(0.4,1.8,.1), cmap = wet_dry, norm = MidpointNormalize(midpoint = 1), transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) wet/PDO(-)   WRF', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.8, lons.max(),3), decimals = 1))

subplot3 = axlist[2].contourf(lons2, lats2, np.ma.masked_where(frazier_mask == True, ((pos_PDO_wet*25.4)/RainfallAtlas_wet)), vmin = 0.4, vmax = 1.8, cmap = wet_dry,norm = MidpointNormalize(midpoint = 1), transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet/PDO(+)       Obs', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.8, lons.max(),3), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == False, (rain_wet2/present_rain_wet)), vmin = 0.4, vmax = 1.8, cmap = wet_dry, norm = MidpointNormalize(midpoint = 1), transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) wet/PDO(+)    WRF', fontsize = 10)
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.8, lons.max(),3), decimals = 1))

subplot5 = axlist[4].contourf(lons2, lats2, np.ma.masked_where(frazier_mask == True, ((neg_PDO_dry*25.4)/RainfallAtlas_dry)), vmin = 0.4, vmax = 1.8, cmap = wet_dry,norm = MidpointNormalize(midpoint = 1), transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) dry/PDO(-)       Obs', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
#axlist[4].set_ylabel('Latitude', size = 'medium')
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
#axlist[4].set_xlabel('Longitude', size = 'medium')
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == False, (rain_dry1/present_rain_dry)), vmin = 0.4, vmax = 1.8, cmap = wet_dry, norm = MidpointNormalize(midpoint = 1), transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry/PDO(-)    WRF', fontsize = 10)
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
#axlist[5].set_ylabel('Latitude', size = 'medium')
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
#axlist[5].set_xlabel('Longitude', size = 'medium')
axlist[5].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot7 = axlist[6].contourf(lons2, lats2, np.ma.masked_where(frazier_mask == True, ((pos_PDO_dry*25.4)/RainfallAtlas_dry)), vmin = 0.4, vmax = 1.8, cmap = wet_dry,norm = MidpointNormalize(midpoint = 1), transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) dry/PDO(+)       Obs', fontsize = 10)
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
#axlist[6].set_xlabel('Longitude', size = 'medium')
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot8 = axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == False, (rain_dry2/present_rain_dry)), vmin = 0.4, vmax = 1.8, cmap = wet_dry, norm = MidpointNormalize(midpoint = 1), transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry/PDO(+)    WRF', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
#axlist[7].set_xlabel('Longitude', size = 'medium')
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(0.4, 1.8, .1), orientation='vertical', shrink = 0.50, pad = 0.0)
cbar.set_label('', size = 'medium')
cbar.ax.tick_params(labelsize = 'medium')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()

# Plot rainfall/PDO composites for obs and present-day WRF output

fig, axarr = plt.subplots(nrows=4, ncols=2, figsize = (10, 10), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons2, lats2, np.ma.masked_where(frazier_mask == True, neg_PDO_wet*25.4), vmin = 0, vmax = 8500, cmap = rainbow, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet/PDO(-)      Obs', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.8, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == False, rain_wet1), vmin = 0, vmax = 8500, levels = np.arange(0,8500,250), cmap = rainbow, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) wet/PDO(-)   WRF', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.8, lons.max(),3), decimals = 1))

subplot3 = axlist[2].contourf(lons2, lats2, np.ma.masked_where(frazier_mask == True, pos_PDO_wet*25.4), vmin = 0, vmax = 8500, cmap = rainbow, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet/PDO(+)       Obs', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.8, lons.max(),3), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == False, rain_wet2), vmin = 0, vmax = 8500, cmap = rainbow, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) wet/PDO(+)    WRF', fontsize = 10)
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.8, lons.max(),3), decimals = 1))

subplot5 = axlist[4].contourf(lons2, lats2, np.ma.masked_where(frazier_mask == True, neg_PDO_dry*25.4), vmin = 0, vmax = 8500, cmap = rainbow, transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) dry/PDO(-)       Obs', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot6 = axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == False, rain_dry1), vmin = 0, vmax = 8500, cmap = rainbow, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry/PDO(-)    WRF', fontsize = 10)
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[5].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot7 = axlist[6].contourf(lons2, lats2, np.ma.masked_where(frazier_mask == True, pos_PDO_dry*25.4), vmin = 0, vmax = 8500, cmap = rainbow, transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) dry/PDO(+)       Obs', fontsize = 10)
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot8 = axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == False, rain_dry2), vmin = 0, vmax = 8500, cmap = rainbow, transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry/PDO(+)    WRF', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.1, lats.max(),2), decimals = 1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(0, 8500, 500), orientation='vertical', shrink = 0.50, pad = 0.0)
cbar.set_label('(mm)', size = 'medium')
cbar.ax.tick_params(labelsize = 'medium')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()
'''

