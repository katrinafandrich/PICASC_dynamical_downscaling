import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature

'''
# Load in reference state
#q_bar_wet1_10 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_wet.npy')
#u_bar_wet_1_10 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_wet.npy')
#v_bar_wet1_10 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_wet.npy')

q_bar_wet1_5 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_wet1_5.npy')
u_bar_wet1_5 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_wet1_5.npy')
v_bar_wet1_5 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_wet1_5.npy')

q_bar_wet6_10 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_wet6_10.npy')
u_bar_wet6_10 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_wet6_10.npy')
v_bar_wet6_10 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_wet6_10.npy')

q_bar_wet11_15 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_wet11_15.npy')
u_bar_wet11_15 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_wet11_15.npy')
v_bar_wet11_15 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_wet11_15.npy')

q_bar_wet16_20 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_wet16_20.npy')
u_bar_wet16_20 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_wet16_20.npy')
v_bar_wet16_20 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_wet16_20.npy')

q_bar_dry1_5 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_dry1_5.npy')
u_bar_dry1_5 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_dry1_5.npy')
v_bar_dry1_5 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_dry1_5.npy')

q_bar_dry6_10 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_dry6_10.npy')
u_bar_dry6_10 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_dry6_10.npy')
v_bar_dry6_10 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_dry6_10.npy')

q_bar_dry11_15 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_dry11_15.npy')
u_bar_dry11_15 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_dry11_15.npy')
v_bar_dry11_15 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_dry11_15.npy')

q_bar_dry16_20 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_dry16_20.npy')
u_bar_dry16_20 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_dry16_20.npy')
v_bar_dry16_20 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_dry16_20.npy')

# Functions to calculate MFC and decomp
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
    
    return (du_dx+dv_dy)

def mfc_decomp1a(u,v,q,p,dx=2000):
    u_flux,v_flux=0,0
    for height in range(1,43):
        dq_dy, dq_dx = np.gradient(q[height],dx)

        u_flux=u_flux+((u[height,:,:]*dq_dx)/9.8)*(p[height-1]-p[height])
        u_flux = u_flux.values[:]
        u_NaN = np.isnan(u_flux)
        u_flux[u_NaN] = 0

        v_flux=v_flux+((v[height,:,:]*dq_dy)/9.8)*(p[height-1]-p[height])
        v_flux = v_flux.values[:]
        v_NaN = np.isnan(v_flux)
        v_flux[v_NaN] = 0

    return (u_flux+v_flux)

def mfc_decomp1b(u,v,q,p,dx=2000):
    u_flux,v_flux=0,0
    for height in range(1,43):
        dq_dy, dq_dx = np.gradient(q[height],dx)

        u_flux=u_flux+((u[height,:,:]*dq_dx)/9.8)*(p[height-1]-p[height])
        u_NaN = np.isnan(u_flux)
        u_flux[u_NaN] = 0

        v_flux=v_flux+((v[height,:,:]*dq_dy)/9.8)*(p[height-1]-p[height])
        v_NaN = np.isnan(v_flux)
        v_flux[v_NaN] = 0

    return (u_flux+v_flux)

def mfc_decomp2a(u,v,q,p,dx=2000):
    u_flux,v_flux=0,0
    for height in range(1,43):
        du_dy, du_dx = np.gradient(u[height,:,:], dx)

        u_flux=u_flux+((du_dx*q[height,:,:])/9.8)*(p[height-1]-p[height])
        u_flux = u_flux.values[:]
        u_NaN = np.isnan(u_flux)
        u_flux[u_NaN] = 0

        dv_dy, dv_dx = np.gradient(v[height,:,:], dx)

        v_flux=v_flux+((dv_dy*q[height,:,:])/9.8)*(p[height-1]-p[height])
        v_flux = v_flux.values[:]
        v_NaN = np.isnan(v_flux)
        v_flux[v_NaN] = 0

    return (u_flux+v_flux)

def mfc_decomp2b(u,v,q,p,dx=2000):
    u_flux,v_flux=0,0
    for height in range(1,43):
        du_dy, du_dx = np.gradient(u[height,:,:], dx)

        u_flux=u_flux+((du_dx*q[height,:,:])/9.8)*(p[height-1]-p[height])
        u_NaN = np.isnan(u_flux)
        u_flux[u_NaN] = 0

        dv_dy, dv_dx = np.gradient(v[height,:,:], dx)

        v_flux=v_flux+((dv_dy*q[height,:,:])/9.8)*(p[height-1]-p[height])
        v_NaN = np.isnan(v_flux)
        v_flux[v_NaN] = 0

    return (u_flux+v_flux)

##############################################################################################
##############################################################################################
path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'

# Loop through simulations
WRF_LIST = ['WRF6','WRF7','WRF8','WRF9','WRF10']
#YEARS =  ['2027','2028','2029','2030','2031','2032','2033','2034','2035']
YEARS = ['1997','1998','1999','2000','2001','2002','2003','2004','2005']
MONTHS = ['05','06','07','08','09','10']
#MONTHS = ['11','12','01','02','03','04']

# Choose the reference state
q_bar = q_bar_dry1_5
u_bar = u_bar_dry1_5
v_bar = v_bar_dry1_5

mfc = 0
LHS1 = 0
term1 = 0
term1a = 0
term1b = 0
term1c = 0
LHS2 = 0
term2 = 0
term2a = 0
term2b = 0
term2c = 0
sfc_term = 0
n=0
for run in WRF_LIST:
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
            q = 1000*(qvapor/(qvapor+1)) # Specific humidity, q (units: kg/kg)
            rho_w = 1000000 # Density of water (g/m^3)

            # Define levels
            plev = np.arange(100*100,1025*100,10*100) # In pascal units
            plev = plev[::-1]

            # Interpolate variables to any pressure level (converts from hybrid to p coords)
            ht_plev = interplevel(z, p, plev)
            u_plev = interplevel(u, p, plev)
            v_plev = interplevel(v, p, plev) 
            q_plev = interplevel(q, p, plev)

            # Calculate perturbation terms
            q_fut = q_plev
            q_pert = q_fut - q_bar
            u_fut = u_plev
            u_pert = u_fut - u_bar
            v_fut = v_plev
            v_pert = v_fut - v_bar

            # Calculate individual terms
            mfc = mfc + total_mfc(u=u_fut,v=v_fut,q=q_fut,p=plev)

            LHS1 = LHS1 + mfc_decomp1a(u=u_fut, v=v_fut, q=q_fut, p=plev)
            term1 = term1 + mfc_decomp1b(u=u_bar, v=v_bar, q=q_bar, p=plev)
            term1a = term1a + mfc_decomp1a(u=u_pert, v=v_pert, q=q_bar, p=plev)
            term1b = term1b + mfc_decomp1b(u=u_bar, v=v_bar, q=q_pert, p=plev)
            term1c = term1c + mfc_decomp1a(u=u_pert, v=v_pert, q=q_pert, p=plev)

            LHS2 = LHS2 + mfc_decomp2a(u=u_fut, v=v_fut, q=q_fut, p=plev)
            term2 = term2 + mfc_decomp2b(u=u_bar, v=v_bar, q=q_bar, p=plev)
            term2a = term2a + mfc_decomp2b(u=u_pert, v=v_pert, q=q_bar, p=plev)
            term2b = term2b + mfc_decomp2a(u=u_bar, v=v_bar, q=q_pert, p=plev)
            term2c = term2c + mfc_decomp2a(u=u_pert, v=v_pert, q=q_pert, p=plev)

            # Extract variables for surface term
            PSFC = ncfile2.PSFC[0,:,:]
            Q2 = ncfile2.Q2[0,:,:]
            U10 = ncfile2.U10[0,:,:]
            V10 = ncfile2.V10[0,:,:]

            dp_dy, dp_dx = np.gradient(PSFC, 2000)
            sfc_term_u = (1/9.8)*Q2*U10*dp_dx
            sfc_term_v = (1/9.8)*Q2*V10*dp_dy

            sfc = sfc_term_u + sfc_term_v

            sfc_term = sfc_term + sfc

            n=n+1

            ncfile.close()
            ncfile2.close()

# Calculate average for each term
mfc = mfc / n

LHS1 = LHS1 / n
LHS2 = LHS2 / n

term1 = term1 / n
term1a = term1a / n
term1b = term1b / n
term2 = term2 / n
term2a = term2a / n
term2b = term2b / n
sfc_term = sfc_term / n

term1c = term1c / n
term2c = term2c / n

LHS_sum = LHS1 + LHS2
RHS_sum = term1 + term1a + term1b + term2 + term2a + term2b + sfc_term
ref_state = term1 + term2
v_terms = term1a + term2a
q_terms = term1b + term2b
nonlinear = term1c + term2c

# Convert each term to mm/day
mfc = (mfc/rho_w)*86400*100
LHS_sum = (LHS_sum/rho_w)*86400*100
RHS_sum = (RHS_sum/rho_w)*86400*100
ref_state = (ref_state/rho_w)*86400*100
v_terms = (v_terms/rho_w)*86400*100
q_terms = (q_terms/rho_w)*86400*100
nonlinear = (nonlinear/rho_w)*86400*100

# Apply Gaussian filter for smoothing
from scipy.ndimage.filters import gaussian_filter
mfc_smooth = gaussian_filter(mfc, sigma=3)
LHS_smooth = gaussian_filter(LHS_sum, sigma=3)
RHS_smooth = gaussian_filter(RHS_sum, sigma=3)
v_smooth = gaussian_filter(v_terms, sigma=3)
q_smooth = gaussian_filter(q_terms, sigma=3)
ref_smooth = gaussian_filter(ref_state, sigma=3)
nonlinear = gaussian_filter(nonlinear, sigma=3)

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_dry_dif4.npy', mfc_smooth)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/LHS_dry_dif4.npy', LHS_smooth)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RHS_dry_dif4.npy', RHS_smooth)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ref_state_dry_dif4.npy', ref_smooth)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vterms_dry_dif4.npy', v_smooth)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qterms_dry_dif4.npy', q_smooth)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/nonlinear_dry_dif4.npy', nonlinear)
'''

#############################################################################################################

# Load in original MFC and mfc difs
path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'

mfc_wet1 = np.load(path+'mfc_wet_wrf1_5.npy')
mfc_wet2 = np.load(path+'mfc_wet_wrf6_10.npy')
mfc_wet3 = np.load(path+'mfc_wet_wrf11_15.npy')
mfc_wet4 = np.load(path+'mfc_wet_wrf16_20.npy')

mfc_dry1 = np.load(path+'mfc_dry_wrf1_5.npy')
mfc_dry2 = np.load(path+'mfc_dry_wrf6_10.npy')
mfc_dry3 = np.load(path+'mfc_dry_wrf11_15.npy')
mfc_dry4 = np.load(path+'mfc_dry_wrf16_20.npy')

mfc_dif1_wet = mfc_wet4 - mfc_wet2
mfc_dif2_wet = mfc_wet3 - mfc_wet1
mfc_dif3_wet = mfc_wet4 - mfc_wet3
mfc_dif4_wet = mfc_wet2 - mfc_wet1

mfc_dif1_dry = mfc_dry4 - mfc_dry2
mfc_dif2_dry = mfc_dry3 - mfc_dry1
mfc_dif3_dry = mfc_dry4 - mfc_dry3
mfc_dif4_dry = mfc_dry2 - mfc_dry1

mfc1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_wet_dif1.npy')
LHS1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/LHS_wet_dif1.npy')
RHS1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RHS_wet_dif1.npy')
ref1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ref_state_wet_dif1.npy')
vterms1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vterms_wet_dif1.npy')
qterms1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qterms_wet_dif1.npy')
nonlinear1_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/nonlinear_wet_dif1.npy')

mfc1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_dry_dif1.npy')
LHS1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/LHS_dry_dif1.npy')
RHS1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RHS_dry_dif1.npy')
ref1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ref_state_dry_dif1.npy')
vterms1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vterms_dry_dif1.npy')
qterms1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qterms_dry_dif1.npy')
nonlinear1_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/nonlinear_dry_dif1.npy')

mfc2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_wet_dif2.npy')
LHS2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/LHS_wet_dif2.npy')
RHS2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RHS_wet_dif2.npy')
ref2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ref_state_wet_dif2.npy')
vterms2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vterms_wet_dif2.npy')
qterms2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qterms_wet_dif2.npy')
nonlinear2_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/nonlinear_wet_dif2.npy')

mfc2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_dry_dif2.npy')
LHS2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/LHS_dry_dif2.npy')
RHS2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RHS_dry_dif2.npy')
ref2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ref_state_dry_dif2.npy')
vterms2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vterms_dry_dif2.npy')
qterms2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qterms_dry_dif2.npy')
nonlinear2_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/nonlinear_dry_dif2.npy')

mfc3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_wet_dif3.npy')
LHS3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/LHS_wet_dif3.npy')
RHS3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RHS_wet_dif3.npy')
ref3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ref_state_wet_dif3.npy')
vterms3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vterms_wet_dif3.npy')
qterms3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qterms_wet_dif3.npy')
nonlinear3_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/nonlinear_wet_dif3.npy')

mfc3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_dry_dif3.npy')
LHS3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/LHS_dry_dif3.npy')
RHS3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RHS_dry_dif3.npy')
ref3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ref_state_dry_dif3.npy')
vterms3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vterms_dry_dif3.npy')
qterms3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qterms_dry_dif3.npy')
nonlinear3_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/nonlinear_dry_dif3.npy')

mfc4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_wet_dif4.npy')
LHS4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/LHS_wet_dif4.npy')
RHS4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RHS_wet_dif4.npy')
ref4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ref_state_wet_dif4.npy')
vterms4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vterms_wet_dif4.npy')
qterms4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qterms_wet_dif4.npy')
nonlinear4_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/nonlinear_wet_dif4.npy')

mfc4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_dry_dif4.npy')
LHS4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/LHS_dry_dif4.npy')
RHS4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/RHS_dry_dif4.npy')
ref4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ref_state_dry_dif4.npy')
vterms4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vterms_dry_dif4.npy')
qterms4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qterms_dry_dif4.npy')
nonlinear4_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/nonlinear_dry_dif4.npy')

# Load info for plotting 
path2 = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'
ncfile = Dataset(path2+'WRF1/wrfout_d02_monthly_mean_1998_11.nc')

# Extract variables
p = getvar(ncfile, "pressure")
p = p*100 # Convert to pascal units
z = getvar(ncfile, "z", units="dm")

# Define levels
plev = np.arange(100*100,1025*100,25*100) # In pascal units
plev = plev[::-1]

# Interpolate variables to any pressure level (converts from hybrid to p coords)
ht_plev = interplevel(z, p, plev)

# Get landmask
dataDIR = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/WRF1/'
filename = 'wrfout_d02_monthly_mean_1997_01.nc'
DS = xarray.open_dataset(dataDIR+filename)

landmask = DS.LANDMASK[0,:,:]

# Get lat/lon coordinates
lats, lons = latlon_coords(ht_plev)

# Get map projection information
cart_proj = get_cartopy(ht_plev)
crs = crs.PlateCarree()

# Download/add coastlines
def plot_background(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    return ax

###########################################################################################################################
# Fix colorbar so white is always on zero
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
###########################################################################################################################

# Colorbar labels
mm_day = '(mm day$^{-1}$)'

# Color schemes
wet_dry = plt.cm.BrBG

# Plot present-day wet and dry season MFC and wind vectors
fig, axarr = plt.subplots(nrows=3, ncols=2, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot2 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif1_wet*-10), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].set_title('(a) wet   Fut(+) - Pres(+)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar2 = fig.colorbar(subplot2, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar2.set_label(mm_day, size = 'medium')

subplot4 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == False, vterms1_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].set_title("(b) changes due to v'", fontsize = 10)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[1].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar4 = fig.colorbar(subplot4, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar4.set_label(mm_day, size = 'medium')

subplot6 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == False, qterms1_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].set_title("(d) changes due to q'", fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
#colorbar6 = fig.colorbar(subplot6, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar6.set_label(mm_day, size = 'medium')

subplot7 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == False, nonlinear1_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].set_title("(f) nonlinear term", fontsize = 10)
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[5].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar7 = fig.colorbar(subplot7, ax=axlist[5], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar7.set_label(mm_day, size = 'medium')

subplot8 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == False, (vterms1_wet+qterms1_wet+nonlinear1_wet)), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].set_title('(c) sum of terms', fontsize = 10)
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar8 = fig.colorbar(subplot8, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar8.set_label(mm_day, size = 'medium')

subplot9 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == False, (mfc_dif1_wet*-10)-(vterms1_wet+qterms1_wet+nonlinear1_wet)), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].set_title('(e) residual', fontsize = 10)
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
#colorbar9 = fig.colorbar(subplot9, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar9.set_label(mm_day, size = 'medium')

fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.55, 0.7, 0.01, 0.2])
cbar1 = fig.colorbar(subplot2, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.7, 0.01, 0.2])
cbar2 = fig.colorbar(subplot4, cax=cbar_ax2, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.55, 0.4, 0.01, 0.2])
cbar3 = fig.colorbar(subplot8, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.4, 0.01, 0.2])
cbar4 = fig.colorbar(subplot6, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label(mm_day, size = 'small')
cbar4.ax.tick_params(labelsize = 'small')

cbar_ax5 = fig.add_axes([0.55, 0.115, 0.01, 0.2])
cbar5 = fig.colorbar(subplot9, cax=cbar_ax5, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar5.set_label(mm_day, size = 'small')
cbar5.ax.tick_params(labelsize = 'small')

cbar_ax6 = fig.add_axes([0.8, 0.115, 0.01, 0.2])
cbar6 = fig.colorbar(subplot7, cax=cbar_ax6, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar6.set_label(mm_day, size = 'small')
cbar6.ax.tick_params(labelsize = 'small')


fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.05, wspace=0.0)
plt.show()

'''
fig, axarr = plt.subplots(nrows=3, ncols=2, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot2 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif1_dry*-10), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].set_title('(a) dry   Fut(+) - Pres(+)', fontsize = 8)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = [''])
colorbar2 = fig.colorbar(subplot2, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar2.set_label(mm_day, size = 'medium')

subplot4 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == False, vterms1_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].set_title('(b) term 1', fontsize = 8)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[1].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = [''])
colorbar4 = fig.colorbar(subplot4, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar4.set_label(mm_day, size = 'medium')

subplot6 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == False, qterms1_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].set_title('(c) term 2', fontsize = 8)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_yticklabels(labels = [''])
colorbar6 = fig.colorbar(subplot6, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar6.set_label(mm_day, size = 'medium')

subplot7 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == False, nonlinear1_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].set_title('(d) term 3', fontsize = 8)
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[5].set_yticklabels(labels = [''])
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
colorbar7 = fig.colorbar(subplot7, ax=axlist[5], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar7.set_label(mm_day, size = 'medium')

subplot8 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == False, (vterms1_dry+qterms1_dry+nonlinear1_dry)), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].set_title('(d) sum of terms', fontsize = 8)
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
colorbar8 = fig.colorbar(subplot8, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar8.set_label(mm_day, size = 'medium')

subplot9 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == False, (mfc_dif1_dry*-10)-(vterms1_dry+qterms1_dry+nonlinear1_dry)), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].set_title('(d) residual', fontsize = 8)
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
colorbar9 = fig.colorbar(subplot9, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar9.set_label(mm_day, size = 'medium')


fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.8, 0.725, 0.01, 0.15])
cbar1 = fig.colorbar(subplot2, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.525, 0.01, 0.15])
cbar2 = fig.colorbar(subplot4, cax=cbar_ax2, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.8, 0.325, 0.01, 0.15])
cbar3 = fig.colorbar(subplot6, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.125, 0.01, 0.15])
cbar4 = fig.colorbar(subplot7, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label(mm_day, size = 'small')
cbar4.ax.tick_params(labelsize = 'small')


fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.05, wspace=0.0)
plt.show()


fig, axarr = plt.subplots(nrows=3, ncols=2, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot2 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif2_wet*-10), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].set_title('(a) wet   Fut(-) - Pres(-)', fontsize = 8)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = [''])
colorbar2 = fig.colorbar(subplot2, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar2.set_label(mm_day, size = 'medium')

subplot4 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == False, vterms2_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].set_title('(b) term 1', fontsize = 8)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[1].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = [''])
colorbar4 = fig.colorbar(subplot4, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar4.set_label(mm_day, size = 'medium')

subplot6 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == False, qterms2_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].set_title('(c) term 2', fontsize = 8)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_yticklabels(labels = [''])
colorbar6 = fig.colorbar(subplot6, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar6.set_label(mm_day, size = 'medium')

subplot7 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == False, nonlinear2_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].set_title('(d) term 3', fontsize = 8)
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[5].set_yticklabels(labels = [''])
colorbar7 = fig.colorbar(subplot7, ax=axlist[5], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar7.set_label(mm_day, size = 'medium')

subplot8 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == False, (vterms2_wet+qterms2_wet+nonlinear2_wet)), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].set_title('(d) sum of terms', fontsize = 8)
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
colorbar8 = fig.colorbar(subplot8, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar8.set_label(mm_day, size = 'medium')

subplot9 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == False, (mfc_dif2_wet*-10)-(vterms2_wet+qterms2_wet+nonlinear2_wet)), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].set_title('(d) residual', fontsize = 8)
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
colorbar9 = fig.colorbar(subplot9, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar9.set_label(mm_day, size = 'medium')


fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.8, 0.725, 0.01, 0.15])
cbar1 = fig.colorbar(subplot2, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.525, 0.01, 0.15])
cbar2 = fig.colorbar(subplot4, cax=cbar_ax2, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.8, 0.325, 0.01, 0.15])
cbar3 = fig.colorbar(subplot6, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.125, 0.01, 0.15])
cbar4 = fig.colorbar(subplot7, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label(mm_day, size = 'small')
cbar4.ax.tick_params(labelsize = 'small')


fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.05, wspace=0.0)
plt.show()

fig, axarr = plt.subplots(nrows=3, ncols=2, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot2 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif2_dry*-10), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].set_title('(a) dry   Fut(-) - Pres(-)', fontsize = 8)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = [''])
colorbar2 = fig.colorbar(subplot2, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar2.set_label(mm_day, size = 'medium')

subplot4 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == False, vterms2_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].set_title('(b) term 1', fontsize = 8)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[1].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = [''])
colorbar4 = fig.colorbar(subplot4, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar4.set_label(mm_day, size = 'medium')

subplot6 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == False, qterms2_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].set_title('(c) term 2', fontsize = 8)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_yticklabels(labels = [''])
colorbar6 = fig.colorbar(subplot6, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar6.set_label(mm_day, size = 'medium')

subplot7 = axlist[5].pcolor(lons, lats, np.ma.masked_where(landmask == False, nonlinear2_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[5].set_title('(d) term 3', fontsize = 8)
axlist[5].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[5].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[5].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[5].set_yticklabels(labels = [''])
colorbar7 = fig.colorbar(subplot7, ax=axlist[5], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar7.set_label(mm_day, size = 'medium')

subplot8 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == False, (vterms2_dry+qterms2_dry+nonlinear2_dry)), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].set_title('(d) sum of terms', fontsize = 8)
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
colorbar8 = fig.colorbar(subplot8, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar8.set_label(mm_day, size = 'medium')

subplot9 = axlist[4].pcolor(lons, lats, np.ma.masked_where(landmask == False, (mfc_dif2_dry*-10)-(vterms2_dry+qterms2_dry+nonlinear2_dry)), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[4].set_title('(d) residual', fontsize = 8)
axlist[4].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[4].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[4].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[4].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
colorbar9 = fig.colorbar(subplot9, ax=axlist[4], orientation='vertical', shrink = 0.70, pad = 0.0)
colorbar9.set_label(mm_day, size = 'medium')


fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.8, 0.725, 0.01, 0.15])
cbar1 = fig.colorbar(subplot2, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.525, 0.01, 0.15])
cbar2 = fig.colorbar(subplot4, cax=cbar_ax2, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.8, 0.325, 0.01, 0.15])
cbar3 = fig.colorbar(subplot6, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.125, 0.01, 0.15])
cbar4 = fig.colorbar(subplot7, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label(mm_day, size = 'small')
cbar4.ax.tick_params(labelsize = 'small')


fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.05, wspace=0.0)
plt.show()


fig, axarr = plt.subplots(nrows=4, ncols=1, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot2 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif3_wet*-10), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].set_title('(a) total change   wet   Fut(+) - Fut(-)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = [''])
#colorbar2 = fig.colorbar(subplot2, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar2.set_label(mm_day, size = 'medium')

subplot4 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == False, vterms3_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].set_title('(b) changes due to v', fontsize = 10)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[1].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = [''])
#colorbar4 = fig.colorbar(subplot4, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar4.set_label(mm_day, size = 'medium')

subplot6 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == False, qterms3_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].set_title('(c) changes due to q', fontsize = 10)
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = [''])
#colorbar6 = fig.colorbar(subplot6, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar6.set_label(mm_day, size = 'medium')

subplot7 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == False, nonlinear3_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].set_title('(d) nonlinear term', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_yticklabels(labels = [''])
#colorbar7 = fig.colorbar(subplot7, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar7.set_label(mm_day, size = 'medium')

fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.8, 0.725, 0.01, 0.15])
cbar1 = fig.colorbar(subplot2, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.525, 0.01, 0.15])
cbar2 = fig.colorbar(subplot4, cax=cbar_ax2, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.8, 0.325, 0.01, 0.15])
cbar3 = fig.colorbar(subplot6, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.125, 0.01, 0.15])
cbar4 = fig.colorbar(subplot7, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label(mm_day, size = 'small')
cbar4.ax.tick_params(labelsize = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.05, wspace=0.0)
plt.show()


fig, axarr = plt.subplots(nrows=4, ncols=1, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot2 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif3_dry*-10), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].set_title('(a) total change   dry   Fut(+) - Fut(-)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = [''])
#colorbar2 = fig.colorbar(subplot2, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar2.set_label(mm_day, size = 'medium')

subplot4 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == False, vterms3_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].set_title('(b) changes due to v', fontsize = 10)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[1].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = [''])
#colorbar4 = fig.colorbar(subplot4, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar4.set_label(mm_day, size = 'medium')

subplot6 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == False, qterms3_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].set_title('(c) changes due to q', fontsize = 10)
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = [''])
#colorbar6 = fig.colorbar(subplot6, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar6.set_label(mm_day, size = 'medium')

subplot7 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == False, nonlinear3_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].set_title('(d) nonlinear term', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_yticklabels(labels = [''])
#colorbar7 = fig.colorbar(subplot7, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar7.set_label(mm_day, size = 'medium')

fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.8, 0.725, 0.01, 0.15])
cbar1 = fig.colorbar(subplot2, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.525, 0.01, 0.15])
cbar2 = fig.colorbar(subplot4, cax=cbar_ax2, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.8, 0.325, 0.01, 0.15])
cbar3 = fig.colorbar(subplot6, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.125, 0.01, 0.15])
cbar4 = fig.colorbar(subplot7, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label(mm_day, size = 'small')
cbar4.ax.tick_params(labelsize = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.05, wspace=0.0)
plt.show()

fig, axarr = plt.subplots(nrows=4, ncols=1, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot2 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif4_wet*-10), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].set_title('(a) total change   wet   Pres(+) - Pres(-)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = [''])
#colorbar2 = fig.colorbar(subplot2, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar2.set_label(mm_day, size = 'medium')

subplot4 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == False, vterms4_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].set_title('(b) changes due to v', fontsize = 10)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[1].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = [''])
#colorbar4 = fig.colorbar(subplot4, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar4.set_label(mm_day, size = 'medium')

subplot6 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == False, qterms4_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].set_title('(c) changes due to q', fontsize = 10)
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = [''])
#colorbar6 = fig.colorbar(subplot6, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar6.set_label(mm_day, size = 'medium')

subplot7 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == False, nonlinear4_wet), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].set_title('(d) nonlinear term', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_yticklabels(labels = [''])
#colorbar7 = fig.colorbar(subplot7, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar7.set_label(mm_day, size = 'medium')

fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.8, 0.725, 0.01, 0.15])
cbar1 = fig.colorbar(subplot2, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.525, 0.01, 0.15])
cbar2 = fig.colorbar(subplot4, cax=cbar_ax2, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.8, 0.325, 0.01, 0.15])
cbar3 = fig.colorbar(subplot6, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.125, 0.01, 0.15])
cbar4 = fig.colorbar(subplot7, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label(mm_day, size = 'small')
cbar4.ax.tick_params(labelsize = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.05, wspace=0.0)
plt.show()


fig, axarr = plt.subplots(nrows=4, ncols=1, figsize = (10, 8), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot2 = axlist[0].pcolor(lons, lats, np.ma.masked_where(landmask == False, mfc_dif4_dry*-10), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[0].set_title('(a) total change   dry   Pres(+) - Pres(-)', fontsize = 10)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = [''])
#colorbar2 = fig.colorbar(subplot2, ax=axlist[0], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar2.set_label(mm_day, size = 'medium')

subplot4 = axlist[1].pcolor(lons, lats, np.ma.masked_where(landmask == False, vterms4_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[1].set_title('(b) changes due to v', fontsize = 10)
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[1].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_yticklabels(labels = [''])
#colorbar4 = fig.colorbar(subplot4, ax=axlist[1], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar4.set_label(mm_day, size = 'medium')

subplot6 = axlist[2].pcolor(lons, lats, np.ma.masked_where(landmask == False, qterms4_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[2].set_title('(c) changes due to q', fontsize = 10)
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
#axlist[2].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = [''])
#colorbar6 = fig.colorbar(subplot6, ax=axlist[2], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar6.set_label(mm_day, size = 'medium')

subplot7 = axlist[3].pcolor(lons, lats, np.ma.masked_where(landmask == False, nonlinear4_dry), cmap = wet_dry, norm = MidpointNormalize(midpoint = 0), transform = crs)
axlist[3].set_title('(d) nonlinear term', fontsize = 10)
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+0.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['160$^\circ$W','157$^\circ$W','154$^\circ$W'])
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_yticklabels(labels = [''])
#colorbar7 = fig.colorbar(subplot7, ax=axlist[3], orientation='vertical', shrink = 0.70, pad = 0.0)
#colorbar7.set_label(mm_day, size = 'medium')

fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.8, 0.725, 0.01, 0.15])
cbar1 = fig.colorbar(subplot2, cax=cbar_ax1, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar1.set_label(mm_day, size = 'small')
cbar1.ax.tick_params(labelsize = 'small')

cbar_ax2 = fig.add_axes([0.8, 0.525, 0.01, 0.15])
cbar2 = fig.colorbar(subplot4, cax=cbar_ax2, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar2.set_label(mm_day, size = 'small')
cbar2.ax.tick_params(labelsize = 'small')

cbar_ax3 = fig.add_axes([0.8, 0.325, 0.01, 0.15])
cbar3 = fig.colorbar(subplot6, cax=cbar_ax3, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar3.set_label(mm_day, size = 'small')
cbar3.ax.tick_params(labelsize = 'small')

cbar_ax4 = fig.add_axes([0.8, 0.125, 0.01, 0.15])
cbar4 = fig.colorbar(subplot7, cax=cbar_ax4, orientation='vertical', shrink = 0.50, pad = 0.0)
cbar4.set_label(mm_day, size = 'small')
cbar4.ax.tick_params(labelsize = 'small')

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.05, wspace=0.0)
plt.show()
'''



