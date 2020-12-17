
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
import xarray
from netCDF4 import Dataset
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt
from scipy.stats import norm
import cartopy.feature as cfeature

# Load rainfall data

path = '/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'

rain_wet1 = np.load(path+'rain_wet_wrf1_5.npy')
rain_wet2 = np.load(path+'rain_wet_wrf6_10.npy')
rain_wet3 = np.load(path+'rain_wet_wrf11_15.npy')
rain_wet4 = np.load(path+'rain_wet_wrf16_20.npy')

rain_dry1 = np.load(path+'rain_dry_wrf1_5.npy')
rain_dry2 = np.load(path+'rain_dry_wrf6_10.npy')
rain_dry3 = np.load(path+'rain_dry_wrf11_15.npy')
rain_dry4 = np.load(path+'rain_dry_wrf16_20.npy')

# Create 3D array to work with
all_rain_wet = np.zeros((4,240,360))
all_rain_wet[0,:,:] = rain_wet1
all_rain_wet[1,:,:] = rain_wet2
all_rain_wet[2,:,:] = rain_wet3
all_rain_wet[3,:,:] = rain_wet4

all_rain_dry = np.zeros((4,240,360))
all_rain_dry[0,:,:] = rain_dry1
all_rain_dry[1,:,:] = rain_dry2
all_rain_dry[2,:,:] = rain_dry3
all_rain_dry[3,:,:] = rain_dry4

# Loop through grid points and do ANOVA test
p_values = np.zeros((240,360))
sig_p = np.zeros((240,360))
for k in range(1,239):
    for l in range(1,359):
        d = all_rain_dry[:,k,l]
        scenario = [1,1,2,2]
        pdo = [1,2,1,2]

        # Calculate degrees of freedom
        df_pdo = 1                      # (number of PDO phases) - 1
        df_scen = 1                     # (number of scenarios) - 1
        df_both= df_pdo*df_scen
        df_total = 19                   # (number of measurements) - 1
        df_w = 17                       # (df_total) - (df_pdo + df_scen)

        # Calculate mean
        grand_mean = d.mean()

        # Start with calculating Sum of Squares for the first factor (PDO-/+)
        pdo_neg = [d[0], d[2]]
        pdo_pos = [d[1], d[3]]
        mean_neg_ssq = (np.mean(pdo_neg) - grand_mean)**2
        mean_pos_ssq = (np.mean(pdo_pos) - grand_mean)**2
        ssq_pdo = mean_neg_ssq + mean_pos_ssq

        # Calculate Sum of Squares for second factor (Present/Future)
        pres = [d[0], d[1]]
        fut = [d[2], d[3]]
        mean_pres_ssq = (np.mean(pres) - grand_mean)**2
        mean_fut_ssq = (np.mean(fut) - grand_mean)**2
        ssq_scen = mean_pres_ssq + mean_fut_ssq

        # Total Sum of Squares
        ssq_t = sum((d - grand_mean)**2)

        # Sum of Squares within (error/residual)
        ssq_w_pdo = ssq_t - ssq_pdo
        ssq_w_scen = ssq_t - ssq_scen
        ssq_w = sum((pdo_neg - np.mean(pdo_neg))**2) +sum((pdo_pos - np.mean(pdo_pos))**2)

        # Sum of Squares interaction
        ssq_both = ssq_t - ssq_pdo - ssq_scen - ssq_w

        # Calculate mean square within
        ms_pdo = ssq_pdo / df_pdo
        ms_scen = ssq_scen / df_scen
        ms_both = ssq_both / df_both
        msw = ssq_w / df_w

        # Compute F-statistic
        f_pdo = ms_pdo / msw
        f_scen = ms_scen / msw
        f_both = ms_both / msw

        # Calculate p-values
        p_pdo = stats.f.sf(f_pdo, df_pdo, df_w)
        p_scen = stats.f.sf(f_scen, df_scen, df_w)
        p_both = stats.f.sf(f_both, df_both, df_w)

        p_values[k,l] = p_scen   
        if (p_values[k,l] <= 0.05):
            sig_p[k,l] = p_values[k,l]
        #elif (p_values[k,l] >= 0.95):
        #    sig_p[k,l] = p_values[k,l]
        else:
            sig_p[k,l] = np.nan

#np.save(path+'pvals_RF_wet_pdo.npy', sig_p)
#np.save(path+'pvals_RF_wet_scen.npy', sig_p)
#np.save(path+'pvals_RF_dry_pdo.npy', sig_p)
#np.save(path+'pvals_RF_dry_scen.npy', sig_p)

pvals_wet_pdo = np.load(path+'pvals_RF_wet_pdo.npy')
pvals_wet_scen = np.load(path+'pvals_RF_wet_scen.npy')
pvals_dry_pdo = np.load(path+'pvals_RF_dry_pdo.npy')
pvals_dry_scen = np.load(path+'pvals_RF_dry_scen.npy')

'''
# Print results in a data table
results = {'sum_sq':[ssq_pdo, ssq_scen, ssq_both, ssq_w],
           'df':[df_pdo, df_scen, df_both, df_w],
           'F':[f_pdo, f_scen, f_both, 'NaN'],
            'PR(>F)':[p_pdo, p_scen, p_both, 'NaN']}
columns=['sum_sq', 'df', 'F', 'PR(>F)']

aov_table1 = pd.DataFrame(results, columns=columns,
                          index=['pdo', 'scen', 
                          'pdo:scen', 'Residual'])

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov

def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov


eta_squared(aov_table1)
omega_squared(aov_table1)
print(aov_table1)
'''

################################################################################
################################################################################

# Load in data for plotting
path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'
ncfile = Dataset(path+'WRF1/wrfout_d02_monthly_mean_1998_11.nc')

# Extract variables
p = getvar(ncfile, "pressure")
p = p*100 # Convert to pascal units
z = getvar(ncfile, "z", units="dm")

# Define levels
plev = np.arange(100*100,1025*100,25*100) # In pascal units
plev = plev[::-1]

# Interpolate variables to any pressure level (converts from hybrid to p coords)
ht_plev = interplevel(z, p, plev)
height = ncfile.variables['HGT'][0,:,:]
mask = ncfile.variables['LANDMASK'][0,:,:]

# Get lat/lon coordinates
lats, lons = latlon_coords(ht_plev)

# Get map projection information
cart_proj = get_cartopy(ht_plev)
crs=crs.PlateCarree()

def plot_background(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    return ax

# Create figure
fig = plt.figure(figsize=(12,9))
ax = plt.axes(projection=cart_proj)

# Download and add coastlines
states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_shp')
ax.add_feature(states, linewidth=0.5)
ax.coastlines('50m', linewidth=0.8)

# Plot and shade regions that are statistically significant
#plt.contourf(lons, lats, np.ma.masked_where(mask == False, height), alpha = 0.3, cmap = 'terrain', transform = crs)
#plt.contourf(lons, lats, np.ma.masked_where(mask == False, sig_p), hatches = ["....."], colors = 'none', alpha = 0.0, transform = crs)
#plt.show()


fig, axarr = plt.subplots(nrows=2, ncols=2, figsize = (10, 10), sharex = 'col', sharey = 'row', dpi = 150, constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

subplot1 = axlist[0].contourf(lons, lats, np.ma.masked_where(mask == False, height), alpha = 0.3, cmap = 'terrain', transform = crs)
axlist[0].contourf(lons, lats, np.ma.masked_where(mask == False, pvals_wet_pdo), hatches = ["....."], colors = 'none', alpha = 0.0, transform = crs)
axlist[0].set_title('(a) wet                     PDO ', fontsize = 12)
axlist[0].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[0].set_xticks(ticks = np.around(np.arange(lons.min()+1.7, lons.max(),3), decimals = 1))
axlist[0].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])

subplot2 = axlist[1].contourf(lons, lats, np.ma.masked_where(mask == False, height), alpha = 0.3, cmap = 'terrain', transform = crs)
axlist[1].contourf(lons, lats, np.ma.masked_where(mask == False, pvals_wet_scen), hatches = ["....."], colors = 'none', alpha = 0.0, transform = crs)
axlist[1].set_title('(b) wet                     GHG emissions', fontsize = 12)
axlist[1].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[1].set_xticks(ticks = np.around(np.arange(lons.min()+1.7, lons.max(),3), decimals = 1))

subplot3 = axlist[2].contourf(lons, lats, np.ma.masked_where(mask == False, height), alpha = 0.3, cmap = 'terrain', transform = crs)
axlist[2].contourf(lons, lats, np.ma.masked_where(mask == False, pvals_dry_pdo), hatches = ["....."], colors = 'none', alpha = 0.0, transform = crs)
axlist[2].set_title('(c) dry                     PDO ', fontsize = 12)
axlist[2].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[2].set_yticklabels(labels = ['20$^\circ$N','22$^\circ$N'])
axlist[2].set_xticks(ticks = np.around(np.arange(lons.min()+1.7, lons.max(),3), decimals = 1))
axlist[2].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])

subplot4 = axlist[3].contourf(lons, lats, np.ma.masked_where(mask == False, height), alpha = 0.3, cmap = 'terrain', transform = crs)
axlist[3].contourf(lons, lats, np.ma.masked_where(mask == False, pvals_dry_scen), hatches = ["....."], colors = 'none', alpha = 0.0, transform = crs)
axlist[3].set_title('(d) dry                     GHG emissions', fontsize = 12)
axlist[3].set_yticks(ticks = np.around(np.arange(lats.min()+1.4, lats.max(),2), decimals = 1))
axlist[3].set_xticks(ticks = np.around(np.arange(lons.min()+1.7, lons.max(),3), decimals = 1))
axlist[3].set_xticklabels(labels = ['159$^\circ$W','156$^\circ$W'])

fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, hspace=0.025, wspace=0.0)
plt.show()







