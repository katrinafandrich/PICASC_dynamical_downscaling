# Step 1 of Moisture Flux Convergence decomposition
# Calculate reference state for q, u, and v
# Calculate future q, u, and v
# Calculate perturbation from reference state dq, du, and dv (dq = q_fut - q_ref)

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

#MONTHS = ["11","12","01","02","03","04"] # Wet season
MONTHS = ['05','06','07','08','09','10'] # Dry season

HIST_Years = ["1997","1998","1999","2000","2001","2002","2003","2004","2005"]
FUT_Years =  ['2027','2028','2029','2030','2031','2032','2033','2034','2035']

# Calculate reference state for wet season
usum=0
vsum=0
qsum=0
n=0
YEARS = HIST_Years
for run in WRF_LIST1:
    for year in YEARS:
        for mon in MONTHS:
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

            path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'
            ncfile = Dataset(path+run+"/wrfout_d02_monthly_mean_"+year+"_"+mon+".nc")
            ncfile2 = xarray.open_dataset(path+run+"/wrfout_d02_monthly_mean_"+year+"_"+mon+".nc")

            # Extract the pressure, geopotential height, moisture, and wind variables
            p = getvar(ncfile, "pressure")
            p = p*100 # Convert to pascal units
            z = getvar(ncfile, "z", units="dm")
            u = getvar(ncfile, "ua", units="kt")
            u = u*.514 # Convert to m/s
            v = getvar(ncfile, "va", units="kt")
            v = v*.514 # Convert to m/s
            qvapor = ncfile2.QVAPOR
            Q = 1000*(qvapor/(qvapor+1)) # Specific humidity, q (units?)

            # Define levels
            plev = np.arange(100*100,1025*100,10*100) # In pascal units
            plev = plev[::-1]

            # Interpolate variables to any pressure level (converts from hybrid to p coords)
            ht_plev = interplevel(z, p, plev)
            u_plev = interplevel(u, p, plev)
            v_plev = interplevel(v, p, plev) 
            q_plev = interplevel(Q, p, plev)
            
            qsum = qsum + q_plev
            usum = usum + u_plev
            vsum = vsum + v_plev
            n = n + 1

            ncfile.close()
            ncfile2.close()


q_avg = qsum/n
u_avg = usum/n
v_avg = vsum/n

q_bar = np.zeros((len(plev),240,360))
u_bar = np.zeros((len(plev),240,360))
v_bar = np.zeros((len(plev),240,360))
    
q_bar = q_avg
u_bar = u_avg
v_bar = v_avg

# Save reference states as numpy array   
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_dry1_5.npy', q_bar)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_dry1_5.npy', u_bar)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_dry1_5.npy', v_bar)

#np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_dry.npy', q_bar)
#np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_dry.npy', u_bar)
#np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_dry.npy', v_bar)


