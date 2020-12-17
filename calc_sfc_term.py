import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature

path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'

WRF_LIST = ['WRF15']

HIST_Years = ["1997","1998","1999","2000","2001","2002","2003","2004","2005"]
RCP85_Years = ["2027","2028","2029","2030","2031","2032","2033","2034","2035"]

WET_MONTHS = ["11","12","01","02","03","04"] # Wet season months
DRY_MONTHS = ["05","06","07","08","09","10"]

YEARS = RCP85_Years
MONTHS = WET_MONTHS

# Part 3: Calculate service term (Seager and Henderson 2010)
dsum = 0
n = 0
for run in WRF_LIST:
    for year in YEARS:
        for mon in MONTHS:

            if mon == '01' and year != '2035':
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
            elif mon == '11' and year == '2035':
                break
            else:
                oldyear = YEARS.index(year)
                year = YEARS[oldyear]
            
            print(run, mon, year)
            ncfile = xarray.open_dataset(path+run+"/wrfout_d02_monthly_mean_"+year+"_"+mon+".nc")

            # Extract variables
            PSFC = ncfile.PSFC[0,:,:]
            Q2 = ncfile.Q2[0,:,:]
            U10 = ncfile.U10[0,:,:]
            V10 = ncfile.V10[0,:,:]

            dp_dy, dp_dx = np.gradient(PSFC, 2000)
            sfc_term_u = (-1/9.8)*Q2*U10*dp_dx
            sfc_term_v = (-1/9.8)*Q2*V10*dp_dy

            sfc_term = sfc_term_u + sfc_term_v

            dsum = dsum + sfc_term
            n = n + 1

avg_sfc_term = dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/sfc_term_wet_'+run+'.npy', avg_sfc_term)

sfc_term = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/sfc_term_wet_'+run+'.npy')

#####################################################################################################################################


