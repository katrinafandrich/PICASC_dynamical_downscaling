from sympy import symbols
import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim

path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'

# Wet season rainfall
rain_wet1 = np.load(path+'rain_wet_wrf1_5.npy')
E_wet1 = np.load(path+'E_wet1_5.npy')
P_wet1 = rain_wet1 / 181 # Seasonal rainfall in mm/day
PE_wet1 = P_wet1 - E_wet1

rain_wet2 = np.load(path+'rain_wet_wrf6_10.npy')
E_wet2 = np.load(path+'E_wet6_10.npy')
P_wet2 = rain_wet2 / 181
PE_wet2 = P_wet2 - E_wet2

PE_pres_wet = (PE_wet1 + PE_wet2) / 2

rain_wet3 = np.load(path+'rain_wet_wrf11_15.npy')
E_wet3 = np.load(path+'E_wet11_15.npy')
P_wet3 = rain_wet3 / 181
PE_wet3 = P_wet3 - E_wet3

rain_wet4 = np.load(path+'rain_wet_wrf16_20.npy')
E_wet4 = np.load(path+'E_wet16_20.npy')
P_wet4 = rain_wet4 / 181
PE_wet4 = P_wet4 - E_wet4

# Dry season
rain_dry1 = np.load(path+'rain_dry_wrf1_5.npy')
E_dry1 = np.load(path+'E_dry1_5.npy')
P_dry1 = rain_dry1 / 184
PE_dry1 = P_dry1 - E_dry1

rain_dry2 = np.load(path+'rain_dry_wrf6_10.npy')
E_dry2 = np.load(path+'E_dry6_10.npy')
P_dry2 = rain_dry2 / 184
PE_dry2 = P_dry2 - E_dry2

PE_pres_dry = (PE_dry1 + PE_dry2) / 2

rain_dry3 = np.load(path+'rain_dry_wrf11_15.npy')
E_dry3 = np.load(path+'E_dry11_15.npy')
P_dry3 = rain_dry3 / 184
PE_dry3 = P_dry3 - E_dry3

rain_dry4 = np.load(path+'rain_dry_wrf16_20.npy')
E_dry4 = np.load(path+'E_dry16_20.npy')
P_dry4 = rain_dry4 / 184
PE_dry4 = P_dry4 - E_dry4

# Wet season
mfc_wet1 = np.load(path+'mfc_wet_wrf1_5.npy')
u1 = np.load(path+'u_wet_wrf1_5.npy')
v1 = np.load(path+'v_wet_wrf1_5.npy')
#LH1 = np.load(path+'evap_wet1_5.npy')
pvals_mfc1_wet = np.load(path+'p_vals_dif_mfc1_wet.npy')

mfc_wet2 = np.load(path+'mfc_wet_wrf6_10.npy')
u2=np.load(path+'u_wet_wrf6_10.npy')
v2=np.load(path+'v_wet_wrf6_10.npy')
#LH2 = np.load(path+'evap_wet6_10.npy')
pvals_mfc2_wet = np.load(path+'p_vals_dif_mfc2_wet.npy')

mfc_wet3 = np.load(path+'mfc_wet_wrf11_15.npy')
u3=np.load(path+'u_wet_wrf11_15.npy')
v3=np.load(path+'v_wet_wrf11_15.npy')
#LH3 = np.load(path+'evap_wet11_15.npy')
pvals_mfc3_wet = np.load(path+'p_vals_dif_mfc3_wet.npy')

mfc_wet4 = np.load(path+'mfc_wet_wrf16_20.npy')
u4=np.load(path+'u_wet_wrf16_20.npy')
v4=np.load(path+'v_wet_wrf16_20.npy')
#LH4 = np.load(path+'evap_wet16_20.npy')
pvals_mfc4_wet = np.load(path+'p_vals_dif_mfc4_wet.npy')

###################################################################################################################

# Dry season
mfc_dry1 = np.load(path+'mfc_dry_wrf1_5.npy')
u1_dry = np.load(path+'u_dry_wrf1_5.npy')
v1_dry = np.load(path+'v_dry_wrf1_5.npy')
pvals_mfc1_dry = np.load(path+'p_vals_dif_mfc1_dry.npy')

mfc_dry2 = np.load(path+'mfc_dry_wrf6_10.npy')
u2_dry = np.load(path+'u_dry_wrf6_10.npy')
v2_dry = np.load(path+'v_dry_wrf6_10.npy')
pvals_mfc2_dry = np.load(path+'p_vals_dif_mfc2_dry.npy')

mfc_dry3 = np.load(path+'mfc_dry_wrf11_15.npy')
u3_dry = np.load(path+'u_dry_wrf11_15.npy')
v3_dry = np.load(path+'v_dry_wrf11_15.npy')
pvals_mfc3_dry = np.load(path+'p_vals_dif_mfc3_dry.npy')

mfc_dry4 = np.load(path+'mfc_dry_wrf16_20.npy')
u4_dry = np.load(path+'u_dry_wrf16_20.npy')
v4_dry = np.load(path+'v_dry_wrf16_20.npy')
pvals_mfc4_dry = np.load(path+'p_vals_dif_mfc4_dry.npy')

# MFC data
mfc_wet_pres = (mfc_wet1 + mfc_wet2) / 2
u_pres = (u1 + u2) / 2
v_pres = (v1 + v2) / 2

mfc_dry_pres = (mfc_dry1 + mfc_dry2) / 2
u_dry_pres = (u1_dry + u2_dry) / 2
v_dry_pres = (v1_dry + v2_dry) / 2

# Differences

mfc_dif1_wet = mfc_wet4 - mfc_wet2
mfc_dif2_wet = mfc_wet3 - mfc_wet1
mfc_dif3_wet = mfc_wet4 - mfc_wet3
mfc_dif4_wet = mfc_wet2 - mfc_wet1

mfc_dif1_dry = mfc_dry4 - mfc_dry2
mfc_dif2_dry = mfc_dry3 - mfc_dry1
mfc_dif3_dry = mfc_dry4 - mfc_dry3
mfc_dif4_dry = mfc_dry2 - mfc_dry1

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

# Use this to get rid of the white border
no_border = np.load(path+'no_border.npy')

# Get landmask
dataDIR = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/WRF1/'
filename = 'wrfout_d02_monthly_mean_1997_01.nc'
DS = xarray.open_dataset(dataDIR+filename)

landmask = DS.LANDMASK[0,:,:]

# Get lat/lon coordinates
lats, lons = latlon_coords(ht_plev)

# Get map projection information
cart_proj = get_cartopy(ht_plev)
crs = crs.PlateCarree()

no_pvals = np.load(path+'no_border.npy')

# Download/add coastlines
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
mfc_SI = '(10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
mm_day = '(mm day$^{-1}$)'

# Color schemes
wet_dry = plt.cm.BrBG

# Add wind vectors if necessary
step = 9

# Plot present-day wet and dry season MFC and wind vectors
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].pcolor(lons, lats, PE_pres_wet, vmin = -15, vmax = 45, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[0].quiver(lons.values[0::step,::step],lats.values[::step, ::step],u_pres[::step,::step],v_pres[::step,::step],scale=1E7,transform=crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet (P - E)       Present-day', fontsize = 12)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar1.set_label(mm_day, size = 'medium')

subplot2 = axlist[1].pcolor(lons, lats, mfc_wet_pres*-10, vmin = -40, vmax = 60, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].quiver(lons.values[0::step,::step],lats.values[::step, ::step],u_pres[::step,::step],v_pres[::step,::step],scale=1E7,transform=crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) wet (MFC)       Present-day', fontsize = 12)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar2.set_label(mm_day, size = 'medium')

subplot3 = axlist[2].pcolor(lons, lats, PE_pres_dry, vmin = -15, vmax = 45, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
#axlist[2].quiver(lons.values[0::step,::step],lats.values[::step, ::step],u_dry_pres[::step,::step],v_dry_pres[::step,::step],scale=1E7,transform=crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) dry (P - E)      Present-day', fontsize = 12)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar3.set_label(mm_day, size = 'medium')

subplot4 = axlist[3].pcolor(lons, lats, mfc_dry_pres*-10, vmin = -40, vmax = 60, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].quiver(lons.values[0::step,::step],lats.values[::step, ::step],u_dry_pres[::step,::step],v_dry_pres[::step,::step],scale=1E7,transform=crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry (MFC)       Present-day', fontsize = 12)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar4.set_label(mm_day, size = 'medium')

fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.815, 0.15, 0.015, 0.7])
cbar1 = fig.colorbar(subplot2, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.455, 0.15, 0.015, 0.7])
cbar2 = fig.colorbar(subplot1, cax=cbar_ax2, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.05, wspace=0.0)
plt.show()

'''
# Plot wet and dry season MFC difs with statistical gridpoints
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif1_wet*-10), vmin = -4.0, vmax = 5.0, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].contourf(lons, lats, np.ma.masked_where(landmask == False, pvals_mfc1_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[0].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[0].set_title('(a) wet       Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar1 = fig.colorbar(subplot1, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar1.set_label(mm_day, size = 'medium')

subplot2 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif1_dry*-10), vmin = -4.0, vmax = 5.1, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].contourf(lons, lats, np.ma.masked_where(landmask == False, pvals_mfc1_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[1].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[1].set_title('(b) dry       Fut(+) - Pres(+)', fontsize = 10)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar2 = fig.colorbar(subplot2, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar2.set_label(mm_day, size = 'medium')

subplot3 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif2_wet*-10), vmin = -4.0, vmax = 5.1, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].contourf(lons, lats, np.ma.masked_where(landmask == False, pvals_mfc2_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[2].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[2].set_title('(c) wet       Fut(-) - Pres(-)', fontsize = 10)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar3 = fig.colorbar(subplot3, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar3.set_label(mm_day, size = 'medium')

subplot4 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif2_dry*-10), vmin = -4.0, vmax = 5.1, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].contourf(lons, lats, np.ma.masked_where(landmask == False, pvals_mfc2_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[3].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[3].set_title('(d) dry       Fut(-) - Pres(-)', fontsize = 10)
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar4 = fig.colorbar(subplot4, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar4.set_label(mm_day, size = 'medium')

subplot5 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif3_wet*-10), vmin = -4.0, vmax = 5.1, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].contourf(lons, lats, np.ma.masked_where(landmask == False,pvals_mfc3_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[4].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[4].set_title('(e) wet      Fut(+) - Fut(-)', fontsize = 10)
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar5 = fig.colorbar(subplot5, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar5.set_label(mm_day, size = 'medium')

subplot6 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif3_dry*-10), vmin = -4.0, vmax = 5.1, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].contourf(lons, lats, np.ma.masked_where(landmask == False, pvals_mfc3_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[5].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[5].set_title('(f) dry       Fut(+) - Fut(-)', fontsize = 10)
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#colorbar6 = fig.colorbar(subplot6, ax=axlist[5], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar6.set_label(mm_day, size = 'medium')

subplot7 = axlist[6].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif4_wet*-10), vmin = -4.0, vmax = 5.1, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[6].contourf(lons, lats, np.ma.masked_where(landmask == False, pvals_mfc4_wet), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[6].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[6].set_title('(g) wet      Pres(+) - Pres(-)', fontsize = 10)
axlist[6].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[6].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[6].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[6].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar7 = fig.colorbar(subplot7, ax=axlist[6], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar7.set_label(mm_day, size = 'medium')

subplot8 = axlist[7].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif4_dry*-10), vmin = -4.0, vmax = 5.1, cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[7].contourf(lons, lats, np.ma.masked_where(landmask == False, pvals_mfc4_dry), hatches = ["....."], colors = 'none', alpha = 0, transform = crs)
axlist[7].contourf(lons, lats, no_pvals, hatches = [""], colors = 'none', alpha = 0, transform = crs)
axlist[7].set_title('(h) dry      Pres(+) - Pres(-)', fontsize = 10)
axlist[7].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[7].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[7].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar8 = fig.colorbar(subplot8, ax=axlist[7], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar8.set_label(mm_day, size = 'medium')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot2, cax=cbar_ax, ticks = np.arange(-4, 5.1, 1), orientation='vertical', shrink = 0.70, pad = 0.0)
cbar.set_label(mm_day, size = 'medium')
cbar.ax.tick_params(labelsize = 'medium')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()
'''