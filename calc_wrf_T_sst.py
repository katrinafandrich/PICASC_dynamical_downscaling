# Script to calculate dry season T2m temps and plot
# Plot differences between groups of 5 WRF runs

import numpy as np
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

##############################################################################
##############################################################################
# Get the lat and lon dimension information from input netcdf file:
reference_file="/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/WRF1/" + \
    "wrfout_d02_monthly_mean_2005_12.nc"
# NOTE: check the names in the attributes for your variable:
ncref=xarray.open_dataset(reference_file)
NLAT=ncref['XLAT'].values[0,:,:]
NLON=ncref['XLONG'].values[0,:,:]
##############################################################################
##############################################################################

path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'

WRF_LIST1 = ['WRF1','WRF2','WRF3','WRF4','WRF5'] # historical (1996-2005), negative PDO
WRF_LIST2 = ['WRF6','WRF7','WRF8','WRF9','WRF10'] # historical, positive PDO
WRF_LIST3 = ['WRF11','WRF12','WRF13','WRF14','WRF15'] # rcp85 (2004-2013), negative PDO
WRF_LIST4 = ['WRF16','WRF17','WRF18','WRF19','WRF20'] # rcp85, positive PDO

HIST_Years = ["1996","1997","1998","1999","2000","2001","2002","2003","2004","2005"]
RCP85_Years = ['2026','2027','2028','2029','2030','2031','2032','2033','2034','2035']

DRYMONTHS = ["05","06","07","08","09","10"] # Dry season months
WETMONTHS = ["11","12","01","02","03","04"] # Wet season months

# Calculates dry season temps
dsum=0.0
n=0
WRF_LIST = WRF_LIST2
YEARS = HIST_Years
for run in WRF_LIST:
    for year in YEARS:
        for mon in DRYMONTHS:
            print(run, mon, year)
            DPATH='/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'
            file=str(run)+"/wrfout_d02_monthly_mean_"+str(year)+"_"+str(mon)+".nc"
            DS_temp = xarray.open_dataset(DPATH+file)
            #temp1 = DS_temp.T2
            temp1 = DS_temp.TSK
            temp_numpy = temp1.values[0,:,:]
            dsum=dsum+temp_numpy
            n=n+1
            DS_temp.close()

average1=dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/TSK_dry_6-10.npy', average1)

# Calculates wet season temps
dsum=0.0
n=0
WRF_LIST = WRF_LIST2
YEARS = HIST_Years
for run in WRF_LIST:
    for year in YEARS:
        for mon in WETMONTHS:
            if mon == '01' and year != '2005':
                oldyear = YEARS.index(year)
                year = YEARS[oldyear+1]
            elif mon == '02':
                oldyear = YEARS.index(year)
                year = YEARS[oldyear]
            elif mon == '03':
                oldyear = YEARS.index(year)
                year = YEARS[oldyear]
            elif mon == '04':
                oldyear = YEARS.index(year)
                year = YEARS[oldyear]
            elif mon == '11' and year == '2005':
                break
            else:
                oldyear = YEARS.index(year)
                year = YEARS[oldyear]
            print(run, mon, year)
            DPATH='/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'
            file=run+"/wrfout_d02_monthly_mean_"+year+"_"+mon+".nc"
            DS_temp = xarray.open_dataset(DPATH+file)
            #temp1 = DS_temp.T2
            temp1 = DS_temp.TSK
            temp_numpy = temp1.values[0,:,:]
            dsum=dsum+temp_numpy
            n=n+1
            DS_temp.close()

average2=dsum/n

# Save arrays in files
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/TSK_wet_6-10.npy', average2)

print('DONE')