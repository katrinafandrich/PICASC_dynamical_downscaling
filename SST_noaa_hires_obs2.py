# 9/24/20
# Author: KMF
# Read in SSTs from NOAA satellite observations and calculate seasonal means

from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import interp

path1 = '/network/rit/lab/elisontimmlab_rit/kf835882/python/noaa_hires_sst/clim_mean_1982_2010/'
infile1 = 'sst.day.mean.ltm.1982-2010.nc'

d = Dataset(path2+infile2, mode = 'r')
lon = d.variables["lon"][:]
lat = d.variables["lat"][:]

# Loop through and calculate average SSTs for wet and dry season
sst_wet = np.zeros(shape=[181,])
sst_dry = np.zeros(shape=[184,])

wet_days = 0
dry_days = 0

for n in range(0,365):
    print(n, wet_days, dry_days)
    sst = d.variables["sst"][n,:,:]

    if n >= 0 and n <= 119:
        sst_wet[wet_days] = tavg_K
        wet_days = wet_days + 1
    if n > 119 and n < 304:
        sst_dry[dry_days] = tavg_K
        dry_days = dry_days + 1
    if n >= 304:
        sst_wet[wet_days] = tavg_K
        wet_days = wet_days + 1

avg_sst_wet = np.mean(sst_wet, axis = 0)
avg_sst_dry = np.mean(sst_dry, axis = 0)

wet_sst, lon = shiftgrid(185, avg_sst_wet, lon, start = False)
m = Basemap(projection='merc', llcrnrlat=18.56, urcrnrlat=22.86, llcrnrlon=-160.69, urcrnrlon=-153.77, resolution='f')
wet_sst = m.transform_scalar(wet_sst, lon, lat, 360, 240)

#dry_sst, lon = shiftgrid(185, avg_sst_dry, lon, start = False)
#m = Basemap(projection='merc', llcrnrlat=18.56, urcrnrlat=22.86, llcrnrlon=-160.69, urcrnrlon=-153.77, resolution='f')
#dry_sst = m.transform_scalar(dry_sst, lon, lat, 360, 240)

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/.npy', wet_sst)
#np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/.npy', dry_sst)




    

  