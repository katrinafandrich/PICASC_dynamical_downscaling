# Create netcdf files for PICASC
# KMF 11/19/20

import netCDF4 as nc
import numpy as np
from netCDF4 import Dataset
import xarray

#path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/model_domain_info/'

ds = nc.Dataset('WRF2026-2035_dry_PDOpos.nc', 'r+', format='NETCDF4')

print(ds.dimensions)

path1 = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'

# Load in variables to store in nc file
rain_wet1 = np.load(path1+'rain_dry_wrf16_20.npy')
temp_wet1 = np.load(path1+'temps_dry_wrf16_20.npy')
RF90_wet1 = np.load(path1+'wrf16-20_90th_dry.npy')
RF99_wet1 = np.load(path1+'wrf16-20_99th_dry.npy')
T90_wet1 = np.load(path1+'WRF16-20_90th_temps_dry.npy')
T90_wet1 = (T90_wet1 + 459.67) * (5/9)
T99_wet1 = np.load(path1+'WRF16-20_99th_temps_dry.npy')
T99_wet1 = (T99_wet1 + 459.67) * (5/9)
mfc_wet1 = np.load(path1+'mfc_dry_wrf16_20.npy')
mfc_wet1 = mfc_wet1*-10

# Append new variables
rain = ds.createVariable('PR', 'f4', ('Time', 'south_north', 'west_east'))
rain.FieldType = '104'
rain.MemoryOrder = 'XY'
rain.units = 'mm'
rain.description = 'Rainfall'

temp = ds.createVariable('TAS', 'f4', ('Time','south_north', 'west_east'))
temp.FieldType = '104'
temp.MemoryOrder = 'XY'
temp.units = 'K'
temp.description = '2M air temperature'

RF90 = ds.createVariable('PR90', 'f4', ('Time','south_north', 'west_east'))
RF90.FieldType = '104'
RF90.MemoryOrder = 'XY'
RF90.units = 'mm/day'
RF90.description = '90th percentile rainfall'

RF99 = ds.createVariable('PR99', 'f4', ('Time','south_north', 'west_east'))
RF99.FieldType = '104'
RF99.MemoryOrder = 'XY'
RF99.units = 'mm/day'
RF99.description = '99th percentile rainfall'

T90 = ds.createVariable('TAS90', 'f4', ('Time','south_north', 'west_east'))
T90.FieldType = '104'
T90.MemoryOrder = 'XY'
T90.units = 'K'
T90.description = '90th percentile 2M air temperature'

T99 = ds.createVariable('TAS99', 'f4', ('Time','south_north', 'west_east'))
T99.FieldType = '104'
T99.MemoryOrder = 'XY'
T99.units = 'K'
T99.description = '99th percentile 2M air temperature'

MFC = ds.createVariable('VIMFC', 'f4', ('Time','south_north', 'west_east'))
MFC.FieldType = '104'
MFC.MemoryOrder = 'XY'
MFC.units = 'mm/day'
MFC.description = 'Vertically integrated moisture flux convergence'

# Assign values to variables
rain[0, :, :] = rain_wet1
temp[0, :, :] = temp_wet1
RF90[0, :, :] = RF90_wet1
RF99[0, :, :] = RF99_wet1
T90[0, :, :] = T90_wet1
T99[0, :, :] = T99_wet1
MFC[0, :, :] = mfc_wet1

ds.close()