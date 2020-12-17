import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature

# Load in reference state
q_bar = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_wet.npy')
u_bar = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_wet.npy')
v_bar = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_wet.npy')

# Define pressure levels
plev = np.arange(100*100,1025*100,10*100) # In pascal units
plev = plev[::-1]
    
# Calculate Term 1 (RHS): Unperturbed moisture divergence in the present state
def mfc_decomp1(u,v,q,p,dx=2000):
    u_flux,v_flux=0,0
    for height in range(1,43):
        dq_dy, dq_dx = np.gradient(q[height],dx)

        u_flux=u_flux+((u[height,:,:]*dq_dx)/9.8)*(p[height-1]-p[height])
        u_NaN = np.isnan(u_flux)
        u_flux[u_NaN] = 0

        v_flux=v_flux+((v[height,:,:]*dq_dy)/9.8)*(p[height-1]-p[height])
        v_NaN = np.isnan(v_flux)
        v_flux[v_NaN] = 0

    return (u_flux,v_flux)

mfcu,mfcv = mfc_decomp1(u=u_bar, v=v_bar, q=q_bar, p=plev)

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1_wet.npy', (mfcu+mfcv))
term1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1_wet.npy')
##############################################################################################################
##############################################################################################################

# Calculate Term 1a (RHS): Changes induced by wind acting on unperturbed moisture gradient
path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'
WRF_LIST = ['WRF15']

dsum = 0
n = 0
for run in WRF_LIST:
    u_pert = np.load(path+'u_pert_wet_'+run+'.npy')
    v_pert = np.load(path+'v_pert_wet_'+run+'.npy')
    mfcu,mfcv = mfc_decomp1(u=u_pert, v=v_pert, q=q_bar, p=plev)
    dsum = dsum + (mfcu+mfcv)
    n = n + 1

average_1a = dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1a_wet.npy', average_1a)
term1a = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1a_wet.npy')
##############################################################################################################
##############################################################################################################

# Calculate Term 1b (RHS): Changes in moisture gradient acting on unperturbed wind field
dsum = 0
n = 0
for run in WRF_LIST:
    q_pert = np.load(path+'q_pert_wet_'+run+'.npy')
    mfcu,mfcv = mfc_decomp1(u=u_bar, v=v_bar, q=q_pert, p=plev)
    dsum = dsum + (mfcu+mfcv)
    n = n + 1

average_1b = dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1b_wet.npy', average_1b)
term1b = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1b_wet.npy')
##############################################################################################################
##############################################################################################################

# Calculate Term 1c (RHS): Non-linear term due to changes in moisture gradient and wind field
dsum = 0
n = 0
for run in WRF_LIST:
    q_pert = np.load(path+'q_pert_wet_'+run+'.npy')
    u_pert = np.load(path+'u_pert_wet_'+run+'.npy')
    v_pert = np.load(path+'v_pert_wet_'+run+'.npy')
    mfcu,mfcv = mfc_decomp1(u=u_pert, v=v_pert, q=q_pert, p=plev)
    dsum = dsum + (mfcu+mfcv)
    n = n + 1

average_1c = dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1c_wet.npy', average_1c)
term1c = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term1c_wet.npy')
##############################################################################################################
##############################################################################################################

# Check by calculating LHS and subtracting out present state (term 1)
dsum = 0
n = 0
for run in WRF_LIST:
    q_fut = np.load(path+'qfut_wet_'+run+'.npy')
    u_fut = np.load(path+'ufut_wet_'+run+'.npy')
    v_fut = np.load(path+'vfut_wet_'+run+'.npy')
    mfcu,mfcv = mfc_decomp1(u=u_fut, v=v_fut, q=q_fut, p=plev)
    dsum = dsum + (mfcu+mfcv)
    n = n + 1

average_qdiv = dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/q_div_LHS_wet.npy', average_qdiv)
LHS1 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/q_div_LHS_wet.npy')

##############################################################################################################
##############################################################################################################
# Calculate Term 2: Unperturbed wind divergence in the present state

def mfc_decomp2(u,v,q,p,dx=2000):
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

    return (u_flux,v_flux)

mfcu,mfcv = mfc_decomp2(u=u_bar, v=v_bar, q=q_bar, p=plev)

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2_wet.npy', (mfcu+mfcv))
term2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2_wet.npy')
##############################################################################################################
##############################################################################################################

# Calculate Term 2a: Changes in wind divergence acting on unperturbed moisture content
path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'
WRF_LIST = ['WRF15']

dsum = 0
n = 0
for run in WRF_LIST:
    u_pert = np.load(path+'u_pert_wet_'+run+'.npy')
    v_pert = np.load(path+'v_pert_wet_'+run+'.npy')
    mfcu,mfcv = mfc_decomp2(u=u_pert, v=v_pert, q=q_bar, p=plev)
    dsum = dsum + (mfcu+mfcv)
    n = n + 1

average_2a = dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2a_wet.npy', average_2a)
term2a = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2a_wet.npy')
##############################################################################################################
##############################################################################################################

# Calculate Term 2b: Changes in moisture acting on unperturbed wind divergent wind field
dsum = 0
n = 0
for run in WRF_LIST:
    q_pert = np.load(path+'q_pert_wet_'+run+'.npy')
    mfcu,mfcv = mfc_decomp2(u=u_bar, v=v_bar, q=q_pert, p=plev)
    dsum = dsum + (mfcu+mfcv)
    n = n + 1

average_2b = dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2b_wet.npy', average_2b)
term2b = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2b_wet.npy')
##############################################################################################################
##############################################################################################################

# Calculate term 2c: Non-linear term due to changes in both moisture and divergent wind field
dsum = 0
n = 0
for run in WRF_LIST:
    q_pert = np.load(path+'q_pert_wet_'+run+'.npy')
    u_pert = np.load(path+'u_pert_wet_'+run+'.npy')
    v_pert = np.load(path+'v_pert_wet_'+run+'.npy')
    mfcu,mfcv = mfc_decomp2(u=u_pert, v=v_pert, q=q_pert, p=plev)
    dsum = dsum + (mfcu+mfcv)
    n = n + 1

average_2c = dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2c_wet.npy', average_2c)
term2c = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/mfc_term2c_wet.npy')
##############################################################################################################
##############################################################################################################

# Check by calculating LHS and subtracting out present state (term 2)
dsum = 0
n = 0
for run in WRF_LIST:
    q_fut = np.load(path+'qfut_wet_'+run+'.npy')
    u_fut = np.load(path+'ufut_wet_'+run+'.npy')
    v_fut = np.load(path+'vfut_wet_'+run+'.npy')
    mfcu,mfcv = mfc_decomp2(u=u_fut, v=v_fut, q=q_fut, p=plev)
    dsum = dsum + (mfcu+mfcv)
    n = n + 1

average_vdiv = dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/v_div_LHS_wet.npy', average_vdiv)
LHS2 = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/v_div_LHS_wet.npy')

##############################################################################################################
##############################################################################################################
# Part 3: Calculate service term (Seager and Henderson 2010)
dsum = 0
n = 0
for run in WRF_LIST:
    sfc_term = np.load(path+'sfc_term_wet_'+run+'.npy')
    dsum = dsum + sfc_term
    n = n + 1

avg_sfc_term = dsum/n

np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/sfc_term_wet.npy', avg_sfc_term)
sfc_term = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/sfc_term_wet.npy')




