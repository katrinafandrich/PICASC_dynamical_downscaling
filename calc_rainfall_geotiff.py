# Code to plot rainfall differences from Abbys geotiff files
# Units are inches

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, shiftgrid, cm

path = '/network/rit/lab/elisontimmlab_rit/DATA/RainfallAtlasHawaii/PDO_Composites_TIF/derived_results_with_QGIS/'
infile = 'PDO_RF_Composite_Inches_WarmPhase_Nov_Apr.nc'

d = Dataset(path+infile, mode = 'r')
lats = d.variables['lat'][:]
lons = d.variables['lon'][:]
rain = d.variables['Band1'][:]

mask = rain.mask[:]

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/frazier_pos_PDO_wet.npy', rain.data[:])
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/frazier_mask.npy', rain.mask[:])

