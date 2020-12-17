# Steps 2 and 3 of Moisture Flux Convergence decomposition
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

WRF_LIST1 = ['WRF11','WRF12','WRF13','WRF14','WRF15','WRF16','WRF17','WRF18','WRF19','WRF20']
WRF_LIST2 = ['WRF11','WRF12','WRF13','WRF14','WRF15']
WRF_LIST3 = ['WRF16','WRF17','WRF18','WRF19','WRF20']
#MONTHS = ["11","12","01","02","03","04"] # Wet season
MONTHS = ['05','06','07','08','09','10']  # Dry season
RCP85_Years = ["2027","2028","2029","2030","2031","2032","2033","2034","2035"]


# Calculate future state for q, u, and v from future WRF sims
YEARS = RCP85_Years
for run in WRF_LIST:
    usum=0
    vsum=0
    qsum=0
    n=0
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

    q_fut = np.zeros((len(plev),240,360))
    u_fut = np.zeros((len(plev),240,360))
    v_fut = np.zeros((len(plev),240,360))
    
    q_fut = q_avg
    u_fut = u_avg
    v_fut = v_avg

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qfut_wet_'+run+'.npy', q_fut)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ufut_wet_'+run+'.npy', u_fut)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vfut_wet_'+run+'.npy', v_fut)

'''
# Calculate future state for q, u, and v from future WRF sims
YEARS = RCP85_Years
for run in WRF_LIST:
    usum=0
    vsum=0
    qsum=0
    n=0
    for year in YEARS:
        for mon in MONTHS:
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

    q_fut = np.zeros((len(plev),240,360))
    u_fut = np.zeros((len(plev),240,360))
    v_fut = np.zeros((len(plev),240,360))
    
    q_fut = q_avg
    u_fut = u_avg
    v_fut = v_avg

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qfut_dry_'+run+'.npy', q_fut)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ufut_dry_'+run+'.npy', u_fut)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vfut_dry_'+run+'.npy', v_fut)
'''
# Step 3: calculate the perturbations to q, u, and v
# Load in q_bar and q_fut

q_bar = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_dry.npy')
u_bar = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_dry.npy')
v_bar = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_dry.npy')

for run in WRF_LIST:
    q = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qfut_dry_'+run+'.npy')
    u = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ufut_dry_'+run+'.npy')
    v = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vfut_dry_'+run+'.npy')

    q_pert = q - q_bar
    u_pert = u - u_bar
    v_pert = v - v_bar

    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/q_pert_dry_'+run+'.npy', q_pert)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/u_pert_dry_'+run+'.npy', u_pert)
    np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/v_pert_dry_'+run+'.npy', v_pert)