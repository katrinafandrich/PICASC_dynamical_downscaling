# Conduct a two-way ANOVA using python
# Plot interaction plots
# 2/17/20

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
import xarray
from netCDF4 import Dataset
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# Make interaction plots; 1 for Present 2 for Future; 1 for PDO- and 2 for PDO+

d = all_rain_wet[:,152,143]
scenario = [1,1,2,2]
pdo = [1,2,1,2] 
fig = interaction_plot(scenario, pdo, d, colors =['blue','red'], markers = ['D','^'], ms=10)
plt.xticks([1,2], ['Present','Future'], size = 16)
plt.legend(labels = ['PDO(-)','PDO(+)'], fontsize = 14)
plt.title('(a) wet            Honolulu', fontsize = 16)
plt.xlabel('', fontsize = 12)
plt.yticks(size = 16)
plt.ylabel('Rainfall (mm)', fontsize = 16)

plt.show()
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/figs/anova_honolulu_wet.png', dpi=150)

d = all_rain_dry[:,152,143]
scenario = [1,1,2,2]
pdo = [1,2,1,2] 
fig = interaction_plot(scenario, pdo, d, colors =['blue','red'], markers = ['D','^'], ms=10)
plt.xticks([1,2], ['Present','Future'], size = 16)
plt.legend(labels = ['PDO(-)','PDO(+)'], fontsize = 14)
plt.title('(b) dry            Honolulu', fontsize = 16)
plt.xlabel('', fontsize = 12)
plt.yticks(size = 16)
plt.ylabel('', fontsize = 12)

plt.show()
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/figs/anova_honolulu_dry.png', dpi=150)

d = all_rain_wet[:,189,70]
scenario = [1,1,2,2]
pdo = [1,2,1,2] 
fig = interaction_plot(scenario, pdo, d, colors =['blue','red'], markers = ['D','^'], ms=10)
plt.xticks([1,2], ['Present','Future'], size = 16)
plt.legend(labels = ['PDO(-)','PDO(+)'], fontsize = 14)
plt.title('(c) wet            Lihue', fontsize = 16)
plt.xlabel('', fontsize = 12)
plt.yticks(size = 16)
plt.ylabel('Rainfall (mm)', fontsize = 16)

plt.show()
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/figs/anova_lihue_wet.png', dpi=150)

d = all_rain_dry[:,189,70]
scenario = [1,1,2,2]
pdo = [1,2,1,2] 
fig = interaction_plot(scenario, pdo, d, colors =['blue','red'], markers = ['D','^'], ms=10)
plt.xticks([1,2], ['Present','Future'], size = 16)
plt.legend(labels = ['PDO(-)','PDO(+)'], fontsize = 14)
plt.title('(d) dry            Lihue', fontsize = 16)
plt.xlabel('', fontsize = 12)
plt.ylabel('', fontsize = 12)
plt.yticks(size = 16)

plt.show()
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/figs/anova_lihue_dry.png', dpi=150)

d = all_rain_wet[:,119,239]
scenario = [1,1,2,2]
pdo = [1,2,1,2] 
fig = interaction_plot(scenario, pdo, d, colors =['blue','red'], markers = ['D','^'], ms=10)
plt.xticks([1,2], ['Present','Future'], size = 16)
plt.legend(labels = ['PDO(-)','PDO(+)'], fontsize = 14)
plt.title('(e) wet            Big Bog', fontsize = 16)
plt.xlabel('', fontsize = 12)
plt.yticks(size = 16)
plt.ylabel('Rainfall (mm)', fontsize = 16)

plt.show()
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/figs/anova_bigbog_wet.png', dpi=150)

d = all_rain_dry[:,119,239]
scenario = [1,1,2,2]
pdo = [1,2,1,2] 
fig = interaction_plot(scenario, pdo, d, colors =['blue','red'], markers = ['D','^'], ms=10)
plt.xticks([1,2], ['Present','Future'], size = 16)
plt.legend(labels = ['PDO(-)','PDO(+)'], fontsize = 14)
plt.title('(f) dry            Big Bog', fontsize = 16)
plt.xlabel('', fontsize = 12)
plt.ylabel('', fontsize = 12)
plt.yticks(size = 16)

plt.show()
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/figs/anova_bigbog_dry.png', dpi=150)

d = all_rain_wet[:,64,293]
scenario = [1,1,2,2]
pdo = [1,2,1,2] 
fig = interaction_plot(scenario, pdo, d, colors =['blue','red'], markers = ['D','^'], ms=10)
plt.xticks([1,2], ['Present','Future'], size = 16)
plt.legend(labels = ['PDO(-)','PDO(+)'], fontsize = 14)
plt.title('(g) wet            Hilo', fontsize = 16)
plt.xlabel('GHG emissions level', fontsize = 16)
plt.ylabel('Rainfall (mm)', fontsize = 16)
plt.yticks(size = 16)

plt.show()
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/figs/anova_hilo_wet.png', dpi=150)

d = all_rain_dry[:,64,293]
scenario = [1,1,2,2]
pdo = [1,2,1,2] 
fig = interaction_plot(scenario, pdo, d, colors =['blue','red'], markers = ['D','^'], ms=10)
plt.xticks([1,2], ['Present','Future'], size = 16)
plt.legend(labels = ['PDO(-)','PDO(+)'], fontsize = 14)
plt.title('(h) dry           Hilo', fontsize = 16)
plt.xlabel('GHG emissions level', fontsize = 16)
plt.ylabel('', fontsize = 16)
plt.yticks(size = 16)

plt.show()
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/figs/anova_hilo_dry.png', dpi=150)







