# This is the original MFC code

import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim

def total_mfc(u,v,q,p,dx=2000):
    u_flux,v_flux=0,0
    for height in range(1,43):
        u_flux=u_flux+((u[height,:,:]*q[height,:,:])/9.8)*(p[height-1]-p[height])
        u_flux = u_flux.values[:]
        u_NaN = np.isnan(u_flux)
        u_flux[u_NaN] = 0

        v_flux=v_flux+((v[height,:,:]*q[height,:,:])/9.8)*(p[height-1]-p[height])
        v_flux = v_flux.values[:]
        v_NaN = np.isnan(v_flux)
        v_flux[v_NaN] = 0
    
    du_dy,du_dx=np.gradient(u_flux,dx)
    dv_dy,dv_dx=np.gradient(v_flux,dx)
    
    return (u_flux,v_flux,du_dx,dv_dy)

path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'

WRF_LIST1 = ['WRF1','WRF2','WRF3','WRF4','WRF5']
WRF_LIST2 = ['WRF6','WRF7','WRF8','WRF9','WRF10'] 
WRF_LIST3 = ['WRF11','WRF12','WRF13','WRF14','WRF15']
WRF_LIST4 = ['WRF16','WRF17','WRF18','WRF19','WRF20']

WRF_LIST5 = ['WRF1','WRF2','WRF3','WRF4','WRF5','WRF6','WRF7','WRF8','WRF9','WRF10']
WRF_LIST6 = ['WRF11','WRF12','WRF13','WRF14','WRF15','WRF16','WRF17','WRF18','WRF19','WRF20']

HIST_Years = ["1997","1998","1999","2000","2001","2002","2003","2004","2005"]
RCP85_Years = ["2027","2028","2029","2030","2031","2032","2033","2034","2035"]

WET_MONTHS = ["11","12","01","02","03","04"] # Wet season months
DRY_MONTHS = ["05","06","07","08","09","10"]

# Loop through simulations and calc mfc
WRF_LIST = WRF_LIST4
YEARS = RCP85_Years
usum=0.0
vsum=0.0
dsum=0.0
n=0
for run in WRF_LIST:
    for year in YEARS:
        for mon in DRY_MONTHS:
            
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

            ncfile = Dataset(path+run+"/wrfout_d02_monthly_mean_"+year+"_"+mon+".nc")
            ncfile2 = xarray.open_dataset(path+run+"/wrfout_d02_monthly_mean_"+year+"_"+mon+".nc")
  
            # Extract the pressure, geopotential height, moisture, and wind variables
            p = getvar(ncfile, "pressure")
            p = p*100 # Convert to Pa units (kg/m*s^2)
            z = getvar(ncfile, "z", units="dm")
            u = getvar(ncfile, "ua", units="kt")
            u = u*.514 # Convert to m/s
            v = getvar(ncfile, "va", units="kt")
            v = v*.514 # Convert to m/s
            qvapor = ncfile2.QVAPOR # w.v. mixing ratio (units: kg/kg)
            q = 1000*(qvapor/(qvapor+1)) # Specific humidity, q (units: g/kg)
            rho_w = 1000000 # density of water (g/m^3)

            # Define levels
            plev = np.arange(100*100,1025*100,10*100) # In pascal units
            plev = plev[::-1]

            # Interpolate variables to any pressure level (converts from hybrid to p coords)
            ht_plev = interplevel(z, p, plev)
            u_plev = interplevel(u, p, plev)
            v_plev = interplevel(v, p, plev) 
            q_plev = interplevel(q, p, plev)

            mfu,mfv,mfcu,mfcv = total_mfc(u=u_plev,v=v_plev,q=q_plev,p=plev)
            usum=usum+mfu
            vsum=vsum+mfv
            dsum=dsum+(mfcu+mfcv)
            n=n+1

            ncfile.close()
            ncfile2.close()

average=dsum/n
u_average=usum/n
v_average=vsum/n

# Convert to mm/day
average = (average/rho_w)*86400*100 

# Apply Gaussian filter for smoothing
from scipy.ndimage.filters import gaussian_filter
blurred = gaussian_filter(average, sigma=3)

print('Saving data to npy file...')

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_dry_wrf16_20.npy', blurred)
#np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/u_wet_wrf16_20.npy', u_average)
#np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/v_wet_wrf16_20.npy', v_average)

print("DONE")
