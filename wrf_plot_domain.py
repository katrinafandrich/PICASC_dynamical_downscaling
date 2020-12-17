import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import netCDF4 as nc
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
import wrf
from mpl_toolkits.basemap import Basemap, maskoceans
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER 
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim

def get_plot_element(infile):
    rootgroup = nc.Dataset(infile, 'r')
    p = wrf.getvar(rootgroup, 'GREENFRAC')
    cart_proj = wrf.get_cartopy(p)
    xlim = wrf.cartopy_xlim(p)
    ylim = wrf.cartopy_ylim(p)
    rootgroup.close()
    return cart_proj, xlim, ylim
 
path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/model_domain_info'

infile_d01 = path+'/geo_em.d01.nc'
cart_proj, xlim_d01, ylim_d01 = get_plot_element(infile_d01)
 
infile_d02 = path+'/geo_em.d02.nc'
cart_proj, xlim_d02, ylim_d02 = get_plot_element(infile_d02)

# Extract other variables we need
domain1 = nc.Dataset(infile_d01, 'r')
hgt = wrf.getvar(domain1, 'HGT_M')
lat = wrf.getvar(domain1, 'XLAT_M')
lon = wrf.getvar(domain1, 'XLONG_M')
mask = wrf.getvar(domain1, 'LANDMASK')
 
# Plot domain and boxes

fig = plt.figure(figsize=(10,8), dpi = 100)
ax = plt.axes(projection=cart_proj)

# d01
#ax.set_xlim([xlim_d01[0]-(xlim_d01[1]-xlim_d01[0])/15, xlim_d01[1]+(xlim_d01[1]-xlim_d01[0])/15])
#ax.set_ylim([ylim_d01[0]-(ylim_d01[1]-ylim_d01[0])/15, ylim_d01[1]+(ylim_d01[1]-ylim_d01[0])/15])

ax.set_xlim([xlim_d01[0], xlim_d01[1]])
ax.set_ylim([ylim_d01[0], ylim_d01[1]])

# d01 box
ax.add_patch(mpl.patches.Rectangle((xlim_d01[0], ylim_d01[0]), xlim_d01[1]-xlim_d01[0], ylim_d01[1]-ylim_d01[0],
             fill=None, lw=2, edgecolor='blue', zorder=10))
ax.text(xlim_d01[0]+(xlim_d01[1]-xlim_d01[0])*0.05, ylim_d01[0]+(ylim_d01[1]-ylim_d01[0])*0.9, 'D01',
        size=15, color='blue', zorder=10)

# d02 box
ax.add_patch(mpl.patches.Rectangle((xlim_d02[0], ylim_d02[0]), xlim_d02[1]-xlim_d02[0], ylim_d02[1]-ylim_d02[0],
             fill=None, lw=2, edgecolor='black', zorder=10))
ax.text(xlim_d02[0]+(xlim_d02[1]-xlim_d02[0])*0.05, ylim_d02[0]+(ylim_d02[1]-ylim_d02[0])*1.1, 'D02',
        size=15, color='red', zorder=10)
ax.set_title('WRF nested-domain', size=16)

# Add coastlines
states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_shp')
ax.add_feature(states, linewidth=0.3)
ax.coastlines('50m', linewidth=0.5)
crs = crs.PlateCarree()

# Add topography and color ocean blue
ocn = np.ma.masked_where(mask == 1, hgt)
plt.pcolormesh(lon, lat, ocn, color = 'lightblue', transform = crs)
plt.contourf(lon, lat, np.ma.masked_where(mask == 0, hgt), cmap = 'terrain', transform = crs)

plt.show()