# Plot rainfall for 6 stations chosen
# Make time series of monthly rainfall throughout entire study period
# Plot annual rainfall cycle for each station
# Include mean and standard deviation

import xarray
import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import cftime as cft
import seaborn
from statistics import mean, stdev

'''
# Get lat and lon info from netcdf file:
reference_file="/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/WRF1/" + \
    "wrfout_d02_monthly_mean_2005_12.nc"

BUCKET=100

ncref=xarray.open_dataset(reference_file)
NLAT=ncref['XLAT'].shape[1]
NLON=ncref['XLONG'].shape[2]

WRF_LIST1 = ['WRF1','WRF2','WRF3','WRF4','WRF5','WRF6','WRF7','WRF8','WRF9','WRF10'] # Present day runs
WRF_LIST2 = ['WRF1','WRF2','WRF3','WRF4','WRF5'] # Can be used for neutral phase

months=[1,2,3,4,5,6,7,8,9,10,11,12]

# Reference year
cref_date="1979-1-1 00:00:00"
ref_date=cft.datetime(1979,1,1,00,00,00)

years=[1996,1997,1998,1999,2000,2001,2002,2003,2004,2005] # Wet season 2005/2006 incomplete
filename_yrs='1997-2005'

# Create arrays to store monthly rainfall
precip0 = np.zeros(shape=[100])
precip1 = np.zeros(shape=[100])
precip2 = np.zeros(shape=[100])
precip3 = np.zeros(shape=[100])
precip4 = np.zeros(shape=[100])
precip5 = np.zeros(shape=[100])
precip6 = np.zeros(shape=[100])
precip7 = np.zeros(shape=[100])
precip8 = np.zeros(shape=[100])
precip9 = np.zeros(shape=[100])
precip10 = np.zeros(shape=[100])
precip11 = np.zeros(shape=[100])

list = WRF_LIST1
n=0
for run in list:
    for startyear in years:
        for month in months:
            DPATH='/network/rit/lab/elisontimmlab_rit/DATA/WRF/hires/'+run+'/hourly/'

            startdate=cft.DatetimeNoLeap(startyear,month,2,00,00,00)
            if startyear == 2005 and month == 12:
                enddate=cft.DatetimeNoLeap(startyear,12,31,00,00,00)
            elif month == 12:
                enddate=cft.DatetimeNoLeap(startyear+1,1,2,00,00,00)
            else:
                enddate=cft.DatetimeNoLeap(startyear,month+1,2,00,00,00)

            time_delta=enddate-startdate
            stime1=cft.datetime.strftime(startdate,"%Y-%m-%d_%H:%M:%S")
            file1="surface_d02_"+stime1
            stime2=cft.datetime.strftime(enddate,"%Y-%m-%d_%H:%M:%S")
            file2="surface_d02_"+stime2
            info="Total precipitation accumulation from " \
            +startdate.strftime('%Y-%m-%d %H:%M:%S') \
            +" to "+enddate.strftime('%Y-%m-%d %H:%M:%S')
            print (info)

            nc1=xarray.open_dataset(DPATH+str(startdate.year)+'/'+file1)
            nc2=xarray.open_dataset(DPATH+str(enddate.year)+'/'+file2)

            i_rainnc1=nc1['I_RAINNC'].values[0,64, 293]
            rainnc1=nc1['RAINNC'].values[0,64, 293]
            accum1=i_rainnc1*BUCKET+rainnc1
            i_rainnc2=nc2['I_RAINNC'].values[0,64, 293]
            rainnc2=nc2['RAINNC'].values[0,64, 293]
            accum2=i_rainnc2*BUCKET+rainnc2

            print(accum2-accum1)

            nc1.close()
            nc2.close()
            if month == months[0]:
                precip0[n]=accum2-accum1
            elif month == months[1]:
                precip1[n]=accum2-accum1
            elif month == months[2]:
                precip2[n]=accum2-accum1
            elif month == months[3]:
                precip3[n]=accum2-accum1
            elif month == months[4]:
                precip4[n]=accum2-accum1
            elif month == months[5]:
                precip5[n]=accum2-accum1
            elif month == months[6]:
                precip6[n]=accum2-accum1
            elif month == months[7]:
                precip7[n]=accum2-accum1
            elif month == months[8]:
                precip8[n]=accum2-accum1
            elif month == months[9]:
                precip9[n]=accum2-accum1
            elif month == months[10]:
                precip10[n]=accum2-accum1
            else:
                precip11[n]=accum2-accum1
        n=n+1

Jan1 = np.mean(precip0[:])
Feb1 = np.mean(precip1[:])
Mar1 = np.mean(precip2[:])
Apr1 = np.mean(precip3[:])
May1 = np.mean(precip4[:])
Jun1 = np.mean(precip5[:])
Jul1 = np.mean(precip6[:])
Aug1 = np.mean(precip7[:])
Sep1 = np.mean(precip8[:])
Oct1 = np.mean(precip9[:])
Nov1 = np.mean(precip10[:])
Dec1 = np.mean(precip11[:])

# Do this again for neutral phase

neut_precip0 = np.zeros(shape=[50])
neut_precip1 = np.zeros(shape=[50])
neut_precip2 = np.zeros(shape=[50])
neut_precip3 = np.zeros(shape=[50])
neut_precip4 = np.zeros(shape=[50])
neut_precip5 = np.zeros(shape=[50])
neut_precip6 = np.zeros(shape=[50])
neut_precip7 = np.zeros(shape=[50])
neut_precip8 = np.zeros(shape=[50])
neut_precip9 = np.zeros(shape=[50])
neut_precip10 = np.zeros(shape=[50])
neut_precip11 = np.zeros(shape=[50])

list = WRF_LIST2
n=0
for run in list:
    for startyear in years:
        for month in months:
            DPATH = '/data/elisontimm_scr/DATA/WRF_PDO_neutral/'+run+'/hourly/'

            startdate=cft.DatetimeNoLeap(startyear,month,2,00,00,00)
            if startyear == 2005 and month == 12:
                enddate=cft.DatetimeNoLeap(startyear,12,31,00,00,00)
            elif month == 12:
                enddate=cft.DatetimeNoLeap(startyear+1,1,2,00,00,00)
            else:
                enddate=cft.DatetimeNoLeap(startyear,month+1,2,00,00,00)

            time_delta=enddate-startdate
            stime1=cft.datetime.strftime(startdate,"%Y-%m-%d_%H_%M_%S")
            file1="surface_d02_"+stime1
            stime2=cft.datetime.strftime(enddate,"%Y-%m-%d_%H_%M_%S")
            file2="surface_d02_"+stime2
            info="Total precipitation accumulation from " \
            +startdate.strftime('%Y-%m-%d_%H_%M_%S') \
            +" to "+enddate.strftime('%Y-%m-%d_%H_%M_%S')
            print (info)

            nc1=xarray.open_dataset(DPATH+str(startdate.year)+'/'+file1)
            nc2=xarray.open_dataset(DPATH+str(enddate.year)+'/'+file2)

            i_rainnc1=nc1['I_RAINNC'].values[0,64, 293]
            rainnc1=nc1['RAINNC'].values[0,64, 293]
            accum1=i_rainnc1*BUCKET+rainnc1
            i_rainnc2=nc2['I_RAINNC'].values[0,64, 293]
            rainnc2=nc2['RAINNC'].values[0,64, 293]
            accum2=i_rainnc2*BUCKET+rainnc2

            print(accum2-accum1)

            nc1.close()
            nc2.close()

            if month == months[0]:
                neut_precip0[n]=accum2-accum1
            elif month == months[1]:
                neut_precip1[n]=accum2-accum1
            elif month == months[2]:
                neut_precip2[n]=accum2-accum1
            elif month == months[3]:
                neut_precip3[n]=accum2-accum1
            elif month == months[4]:
                neut_precip4[n]=accum2-accum1
            elif month == months[5]:
                neut_precip5[n]=accum2-accum1
            elif month == months[6]:
                neut_precip6[n]=accum2-accum1
            elif month == months[7]:
                neut_precip7[n]=accum2-accum1
            elif month == months[8]:
                neut_precip8[n]=accum2-accum1
            elif month == months[9]:
                neut_precip9[n]=accum2-accum1
            elif month == months[10]:
                neut_precip10[n]=accum2-accum1
            else:
                neut_precip11[n]=accum2-accum1
        n=n+1


Jan2 = np.mean(neut_precip0[:])
Feb2 = np.mean(neut_precip1[:])
Mar2 = np.mean(neut_precip2[:])
Apr2 = np.mean(neut_precip3[:])
May2 = np.mean(neut_precip4[:])
Jun2 = np.mean(neut_precip5[:])
Jul2 = np.mean(neut_precip6[:])
Aug2 = np.mean(neut_precip7[:])
Sep2 = np.mean(neut_precip8[:])
Oct2 = np.mean(neut_precip9[:])
Nov2 = np.mean(neut_precip10[:])
Dec2 = np.mean(neut_precip11[:])

# Average neutral, PDO+, and PDO- together
Jan = (Jan1 + Jan2) / 2
Feb = (Feb1 + Feb2) / 2
Mar = (Mar1 + Mar2) / 2
Apr = (Apr1 + Apr2) / 2
May = (May1 + May2) / 2
Jun = (Jun1 + Jun2) / 2
Jul = (Jul1 + Jul2) / 2
Aug = (Aug1 + Aug2) / 2
Sep = (Sep1 + Sep2) / 2
Oct = (Oct1 + Oct2) / 2
Nov = (Nov1 + Nov2) / 2
Dec = (Dec1 + Dec2) / 2

station = 'hilo'

# Save as numpy arrays
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_01.npy', Jan)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_02.npy', Feb)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_03.npy', Mar)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_04.npy', Apr)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_05.npy', May)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_06.npy', Jun)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_07.npy', Jul)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_08.npy', Aug)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_09.npy', Sep)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_10.npy', Oct)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_11.npy', Nov)
np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/'+str(station)+'_12.npy', Dec)
'''

# Load in monthly rainfall by station 
lihue_pres = np.zeros(shape=[12])

lihue_pres[0] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_01.npy')
lihue_pres[1] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_02.npy')
lihue_pres[2] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_03.npy')
lihue_pres[3] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_04.npy')
lihue_pres[4] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_05.npy')
lihue_pres[5] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_06.npy')
lihue_pres[6] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_07.npy')
lihue_pres[7] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_08.npy')
lihue_pres[8] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_09.npy')
lihue_pres[9] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_10.npy')
lihue_pres[10] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_11.npy')
lihue_pres[11] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/lihue_12.npy')

hon_pres = np.zeros(shape=[12])

hon_pres[0] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_01.npy')
hon_pres[1] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_02.npy')
hon_pres[2] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_03.npy')
hon_pres[3] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_04.npy')
hon_pres[4] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_05.npy')
hon_pres[5] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_06.npy')
hon_pres[6] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_07.npy')
hon_pres[7] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_08.npy')
hon_pres[8] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_09.npy')
hon_pres[9] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_10.npy')
hon_pres[10] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_11.npy')
hon_pres[11] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/honolulu_12.npy')

bigbog_pres = np.zeros(shape=[12])

bigbog_pres[0] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_01.npy')
bigbog_pres[1] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_02.npy')
bigbog_pres[2] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_03.npy')
bigbog_pres[3] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_04.npy')
bigbog_pres[4] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_05.npy')
bigbog_pres[5] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_06.npy')
bigbog_pres[6] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_07.npy')
bigbog_pres[7] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_08.npy')
bigbog_pres[8] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_09.npy')
bigbog_pres[9] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_10.npy')
bigbog_pres[10] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_11.npy')
bigbog_pres[11] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/bigbog_12.npy')

hilo_pres = np.zeros(shape=[12])

hilo_pres[0] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_01.npy')
hilo_pres[1] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_02.npy')
hilo_pres[2] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_03.npy')
hilo_pres[3] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_04.npy')
hilo_pres[4] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_05.npy')
hilo_pres[5] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_06.npy')
hilo_pres[6] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_07.npy')
hilo_pres[7] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_08.npy')
hilo_pres[8] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_09.npy')
hilo_pres[9] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_10.npy')
hilo_pres[10] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_11.npy')
hilo_pres[11] = np.load('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/hilo_12.npy')

# Observations (Rainfall Atlas)
lihue = [101.6, 88.1, 118.4, 57.7, 64.8, 46.4, 52.4, 52.7, 66.9, 107.3, 118.3, 140.3]
honolulu = [63.7, 55.8, 50.5, 23.9, 18.2, 9.8, 13.1, 15.7, 16.8, 51.8, 65.3, 67.0]
bigbog = [801.5, 616.6, 1331.1, 989.4, 629.8, 733.9, 826.5, 761.7, 658.3, 981.5, 949.2, 919.5]
hilo = [247.5, 240.5, 350.5, 295.6, 208.6, 195.6, 281.3, 263.3, 260.0, 247.8, 387.3, 266.3]

labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

x = np.arange(len(labels))  # the label locations
width =0.3

f = plt.figure(figsize=(8,8), dpi = 150)
ax = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

plt.subplot(2, 2, 1)
plt.bar(x - width/2, lihue_pres, width, label='Present-day')
plt.bar(x + width/2, lihue, width, label='Observations')
plt.xticks(x, labels)
plt.ylabel('Rainfall (mm)')
plt.title('Lihue', fontsize = 12)
plt.ylim(0,175)

plt.subplot(2, 2, 2)
plt.bar(x - width/2, hon_pres, width, label='Present-day')
plt.bar(x + width/2, honolulu, width, label='Observations')
plt.xticks(x, labels)
plt.legend()
plt.title('Honolulu', fontsize = 12)
plt.ylim(0,175)

plt.subplot(2, 2, 3)
plt.bar(x - width/2, bigbog_pres, width, label='Present-day')
plt.bar(x + width/2, bigbog, width, label='Observations')
plt.xticks(x, labels)
plt.ylabel('Rainfall (mm)')
plt.title('Big Bog', fontsize = 12)

plt.subplot(2, 2, 4)
plt.bar(x - width/2, hilo_pres, width, label='Present-day')
plt.bar(x + width/2, hilo, width, label='Observations')
plt.xticks(x, labels)
plt.title('Hilo', fontsize = 12)

f.set_constrained_layout_pads(w_pad=0.125, h_pad=0.125, hspace=0.9, wspace=0.5)
plt.tight_layout()
#f.suptitle('Annual Rainfall Cycle by Station')
#plt.savefig('/network/rit/lab/elisontimmlab_rit/kf835882/python/figs/stations_ann_cycle.png')
plt.show()

