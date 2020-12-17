# Plot MFC decomp terms
# KMF 3/23/20

import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim

# Load in wet season MFC decomp terms
term1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1_wet.npy')
term1a = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1a_wet.npy')
term1b = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1b_wet.npy')
term1c = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1c_wet.npy')
q_div = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/q_div_LHS_wet.npy')

LHS1 = q_div - term1
RHS1 = term1a + term1b + term1c

term2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2_wet.npy')
term2a = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2a_wet.npy')
term2b = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2b_wet.npy')
term2c = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2c_wet.npy')
v_div = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/v_div_LHS_wet.npy')

mfc_orig = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_orig_wet.npy')

LHS2 = v_div - term2
RHS2 = term2a + term2b + term2c

###########################################################################################################
###########################################################################################################
'''
# Load in dry season MFC decomp terms
term1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1_dry.npy')
term1a = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1a_dry.npy')
term1b = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1b_dry.npy')
term1c = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1c_dry.npy')
q_div = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/q_div_LHS_dry.npy')

LHS1 = q_div - term1
RHS1 = term1a + term1b + term1c

term2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2_dry.npy')
term2a = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2a_dry.npy')
term2b = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2b_dry.npy')
term2c = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2c_dry.npy')
v_div = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/v_div_LHS_dry.npy')

LHS2 = v_div - term2
RHS2 = term2a + term2b + term2c
'''
###########################################################################################################
###########################################################################################################

# Data we want to plot
data = LHS2

# Apply gaussian filter
from scipy.ndimage.filters import gaussian_filter
blurred = gaussian_filter(data, sigma=3)
blurred = blurred*100 

###########################################################################################################
###########################################################################################################

# Load in data/info for plotting
path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'
ncfile = Dataset(path+'WRF1/wrfout_d02_monthly_mean_1998_11.nc')

# Extract variables
p = getvar(ncfile, "pressure")
p = p*100 # Convert to pascal units
z = getvar(ncfile, "z", units="dm")

# Define levels
plev = np.arange(100*100,1025*100,25*100) # In pascal units
plev = plev[::-1]

# Interpolate variables to any pressure level (converts from hybrid to p coords)
ht_plev = interplevel(z, p, plev)

# Get the lat/lon coordinates
LAT = getvar(ncfile, 'XLAT')
LONG = getvar(ncfile, 'XLONG')

# Get the map projection information
cart_proj = get_cartopy(ht_plev)
crssupport=crs.PlateCarree()

# Create the figure
fig = plt.figure(figsize=(12,9))
ax = plt.axes(projection=cart_proj)

# Download and add coastlines
states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_shp')
ax.add_feature(states, linewidth=0.5)
ax.coastlines('50m', linewidth=0.8)

# Plot
plt.pcolormesh(LONG, LAT, blurred, vmin = , vmax = , transform = crssupport, cmap=plt.cm.BrBG_r)
plt.colorbar(label='(10$^{-4}$ kg m$^{-2}$ s$^{-1}$)')
plt.title("", fontsize=16)
plt.show()
