# Calculate terms for Part 2 of moisture flux decomposition
# See notes for more details

import xarray
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Load in reference state
q_bar = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/qbar_wet.npy')
u_bar = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/ubar_wet.npy')
v_bar = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/vbar_wet.npy')

# Define pressure levels
plev = np.arange(100*100,1025*100,10*100) # In pascal units
plev = plev[::-1]

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
##############################################################################################################
##############################################################################################################

# Calculate Term 2a: Changes in wind divergence acting on unperturbed moisture content
path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'
WRF_LIST = ['WRF11','WRF12','WRF13','WRF14','WRF15', 'WRF16','WRF17','WRF18','WRF19','WRF20']

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