import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim


WRF_LIST1 = ['WRF1','WRF2','WRF3','WRF4','WRF5']
WRF_LIST2 = ['WRF6','WRF7','WRF8','WRF9','WRF10']
WRF_LIST3 = ['WRF11','WRF12','WRF13','WRF14','WRF15']
WRF_LIST4 = ['WRF16','WRF17','WRF18','WRF19','WRF20']

WET_MONTHS = ["11","12","01","02","03","04"] # Wet season
DRY_MONTHS = ['05','06','07','08','09','10'] # Dry season

HIST_Years = ["1997","1998","1999","2000","2001","2002","2003","2004","2005"]
FUT_Years = ['2027','2028','2029','2030','2031','2032','2033','2034','2035']

cloud_frac = 0
YEARS = HIST_Years
n = 0
for run in WRF_LIST1:
    for year in YEARS:
        for mon in WET_MONTHS:
            
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
            
            print(n,run, mon, year)

            path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'
            DS = xarray.open_dataset(path+run+"/wrfout_d02_monthly_mean_"+year+"_"+mon+".nc")

            # Extract variables to calc cloud frac
            SW = DS.SWDOWN[0,:,:]
            LW = DS.GLW[0,:,:]
            total = SW + LW
            cloud_frac = cloud_frac + total

            n = n + 1

cloud_frac_avg = cloud_frac/n
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/SW_LW_wet_WRF1_5.npy', cloud_frac_avg)

