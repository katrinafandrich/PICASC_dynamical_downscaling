# Script to calculate evaporation from latent heat flux (W m^-2) for wet and dry season
# Convert to SI units (10^-4 kg m^-2 s^-1)
# Store as numpy arrays to be used in calculations later (P - E) = integrated MFC
# KMF 4/10/20

import xarray
import numpy as np

path = '/network/rit/lab/elisontimmlab_rit/DATA/WRF/mon/'

WRF_LIST1 = ['WRF1','WRF2','WRF3','WRF4','WRF5']
WRF_LIST2 = ['WRF6','WRF7','WRF8','WRF9','WRF10']
WRF_LIST3 = ['WRF11','WRF12','WRF13','WRF14','WRF15']
WRF_LIST4 = ['WRF16','WRF17','WRF18','WRF19','WRF20']

HIST_Years = ["1996","1997","1998","1999","2000","2001","2002","2003","2004","2005"]
RCP85_Years = ["2026","2027","2028","2029","2030","2031","2032","2033","2034","2035"]

WETMONTHS = ["11","12","01","02","03","04"]
DRYMONTHS = ["05","06","07","08","09","10"]

# Loop through simulations and calc evaporation rate in mm/day
dsum=0.0
n=0
WRF_LIST = WRF_LIST
YEARS = RCP85_Years
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

            ncfile = xarray.open_dataset(path+run+"/wrfout_d02_monthly_mean_"+year+"_"+mon+".nc")
            LH = ncfile.LH[0,:,:]

            evap = LH / (2500000*1000) # divide by L_v and rho 
            evap = evap*86400*1000 # convert to mm/day
            
            dsum = dsum + evap
            n=n+1

            ncfile.close()

average=dsum/n

#np.save('/network/rit/lab/elisontimmlab_rit/kf835882/python/DATA/np_arrays/.npy', average)

print("DONE")