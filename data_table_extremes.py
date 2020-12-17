# Code to calculate and print out statistics from box and whisker plots to make data table

import numpy as np
import matplotlib.pyplot as plt 

# 90th and 99th percentiles by season and station
lihue1_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue1-5_90_wet.npy')
lihue2_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue6-10_90_wet.npy')
lihue3_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue11-15_90_wet.npy')
lihue4_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_90_wet.npy')

lihue1_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue1-5_99_wet.npy')
lihue2_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue6-10_99_wet.npy')
lihue3_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue11-15_99_wet.npy')
lihue4_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_99_wet.npy')

honolulu1_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu1-5_90_wet.npy')
honolulu2_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu6-10_90_wet.npy')
honolulu3_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu11-15_90_wet.npy')
honolulu4_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_90_wet.npy')

honolulu1_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu1-5_99_wet.npy')
honolulu2_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu6-10_99_wet.npy')
honolulu3_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu11-15_99_wet.npy')
honolulu4_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_99_wet.npy')

bigbog1_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog1-5_90_wet.npy')
bigbog2_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog6-10_90_wet.npy')
bigbog3_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog11-15_90_wet.npy')
bigbog4_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_90_wet.npy')

bigbog1_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog1-5_99_wet.npy')
bigbog2_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog6-10_99_wet.npy')
bigbog3_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog11-15_99_wet.npy')
bigbog4_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_99_wet.npy')

hilo1_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo1-5_90_wet.npy')
hilo2_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo6-10_90_wet.npy')
hilo3_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo11-15_90_wet.npy')
hilo4_90_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_90_wet.npy')

hilo1_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo1-5_99_wet.npy')
hilo2_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo6-10_99_wet.npy')
hilo3_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo11-15_99_wet.npy')
hilo4_99_wet = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_99_wet.npy')

lihue1_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue1-5_90_dry.npy')
lihue2_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue6-10_90_dry.npy')
lihue3_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue11-15_90_dry.npy')
lihue4_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_90_dry.npy')

lihue1_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue1-5_99_dry.npy')
lihue2_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue6-10_99_dry.npy')
lihue3_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue11-15_99_dry.npy')
lihue4_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue16-20_99_dry.npy')

honolulu1_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu1-5_90_dry.npy')
honolulu2_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu6-10_90_dry.npy')
honolulu3_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu11-15_90_dry.npy')
honolulu4_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_90_dry.npy')

honolulu1_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu1-5_99_dry.npy')
honolulu2_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu6-10_99_dry.npy')
honolulu3_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu11-15_99_dry.npy')
honolulu4_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu16-20_99_dry.npy')

bigbog1_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog1-5_90_dry.npy')
bigbog2_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog6-10_90_dry.npy')
bigbog3_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog11-15_90_dry.npy')
bigbog4_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_90_dry.npy')

bigbog1_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog1-5_99_dry.npy')
bigbog2_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog6-10_99_dry.npy')
bigbog3_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog11-15_99_dry.npy')
bigbog4_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog16-20_99_dry.npy')

hilo1_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo1-5_90_dry.npy')
hilo2_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo6-10_90_dry.npy')
hilo3_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo11-15_90_dry.npy')
hilo4_90_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_90_dry.npy')

hilo1_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo1-5_99_dry.npy')
hilo2_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo6-10_99_dry.npy')
hilo3_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo11-15_99_dry.npy')
hilo4_99_dry = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo16-20_99_dry.npy')

# Rainfall Atlas station data
lihue_wet_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_wet_90_atlas.npy')
lihue_wet_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_wet_99_atlas.npy')

honolulu_wet_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_wet_90_atlas.npy')
honolulu_wet_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_wet_99_atlas.npy')

bigbog_wet_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_wet_90_atlas.npy')
bigbog_wet_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_wet_99_atlas.npy')

hilo_wet_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_wet_90_atlas.npy')
hilo_wet_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_wet_99_atlas.npy')

lihue_dry_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_dry_90_atlas.npy')
lihue_dry_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_dry_99_atlas.npy')

honolulu_dry_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_dry_90_atlas.npy')
honolulu_dry_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_dry_99_atlas.npy')

bigbog_dry_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_dry_90_atlas.npy')
bigbog_dry_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_dry_99_atlas.npy')

hilo_dry_90_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_dry_90_atlas.npy')
hilo_dry_99_atlas = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_dry_99_atlas.npy')

# Choose data to plot

name = hilo4_90_dry          

boxplot = plt.boxplot(name)

 # Grab the relevant Line2D instances from the boxplot dictionary
iqr = boxplot['boxes'][0]
caps = boxplot['caps']
med = boxplot['medians'][0]
fly = boxplot['fliers'][0]

# The x position of the median line
xpos = med.get_xdata()

# The median is the y-position of the median line
median = med.get_ydata()[1]

# The 25th and 75th percentiles are found from the
# top and bottom (max and min) of the box
pc25 = iqr.get_ydata().min()
pc75 = iqr.get_ydata().max()

# The caps give the vertical position of the ends of the whiskers
capbottom = caps[0].get_ydata()[0]
captop = caps[1].get_ydata()[0]

# Get number of outliers and max outlier
outliers = fly.get_ydata()[:]
num_outliers = len(outliers)
max_outlier = outliers.max()

# Print statistics
print(name)

print('Median = {:6.3g}'.format(median))
    
print('25th percentile = {:6.3g}'.format(pc25))

print('75th percentile = {:6.3g}'.format(pc75))

print('Bottom cap = {:6.3g}'.format(capbottom))

print('Top cap = {:6.3g}'.format(captop))

print('# of Outliers = {:6.3g}'.format(num_outliers))

print('Highest Outlier = {:6.3g}'.format(max_outlier))

