
# Author: Arthur Prigent
# Email: prigent.arthur29@gmail.com

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import scipy.stats as stats
from datetime import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#import cartopy.crs as ccrs
#import cartopy
import matplotlib.patches as mpatches
now = datetime.now()
print(now)
date_time = now.strftime("%d/%m/%Y")
import matplotlib
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import datetime

import requests




def is_mam(month):
    return (month >= 3) & (month <= 5)
def is_jja(month):
    return (month >= 6) & (month <= 8)
def nandetrend(y):
    ''' Remove the linear trend from the data '''
    
    x = np.arange(0,y.shape[0],1)
    m, b, r_val, p_val, std_err = stats.linregress(x,np.array(y))
    y_detrended= np.array(y) - m*x
    return y_detrended




def load_armor3d(USERNAME,PASSWORD):

    DATASET_ID = 'dataset-armor-3d-nrt-monthly'
    time_tmp = xr.open_dataset(f'https://{USERNAME}:{PASSWORD}@nrt.cmems-du.eu/thredds/dodsC/dataset-armor-3d-nrt-monthly?time')
    lon_tmp1 = xr.open_dataset(f'https://{USERNAME}:{PASSWORD}@nrt.cmems-du.eu/thredds/dodsC/dataset-armor-3d-nrt-monthly?longitude[0:1:100]')
    lon_tmp2 = xr.open_dataset(f'https://{USERNAME}:{PASSWORD}@nrt.cmems-du.eu/thredds/dodsC/dataset-armor-3d-nrt-monthly?longitude[1200:1:1439]')
    to_tmp1 = xr.open_dataset(f'https://{USERNAME}:{PASSWORD}@nrt.cmems-du.eu/thredds/dodsC/dataset-armor-3d-nrt-monthly?to[0:1:'+
                         str(time_tmp.time.shape[0]-1)+'][0:1:18][180:1:400][0:1:100]')
    to_tmp2 = xr.open_dataset(f'https://{USERNAME}:{PASSWORD}@nrt.cmems-du.eu/thredds/dodsC/dataset-armor-3d-nrt-monthly?to[0:1:'+
                         str(time_tmp.time.shape[0]-1)+'][0:1:18][180:1:400][1200:1:1439]')
    lat_tmp = xr.open_dataset(f'https://{USERNAME}:{PASSWORD}@nrt.cmems-du.eu/thredds/dodsC/dataset-armor-3d-nrt-monthly?latitude[180:1:400]')
    depth_tmp = xr.open_dataset(f'https://{USERNAME}:{PASSWORD}@nrt.cmems-du.eu/thredds/dodsC/dataset-armor-3d-nrt-monthly?depth[0:1:18]')



    to_tmp = xr.concat([to_tmp2,to_tmp1],dim='longitude')
    lon_tmp2_new = (lon_tmp2.longitude+180)%360-180
    lon_new = xr.concat([lon_tmp2_new,lon_tmp1.longitude],dim='longitude')
    lat_new = lat_tmp.latitude[:]
    depth_new = depth_tmp.depth[:19]
    time_new = time_tmp.time[-3:]

    to_xarray = xr.Dataset({'to': (['time','depth','lat','lon'], np.array(to_tmp.to[-3:,:,:,:])),

                              },
                          coords={'time':(np.array(time_new)),
                                  'depth':(np.array(depth_new)),
                                  'lat':(np.array(lat_new)),
                                  'lon':(np.array(lon_new))})

    lat_interp = np.arange(-35,6,1)
    lon_interp = np.arange(-55,26,1)
    coarse = to_xarray.interp(lon=lon_interp, lat=lat_interp, method="nearest")
    
    return coarse



def plot_last3months(temp):
    f,ax = plt.subplots(2,3,figsize=[15,10])
    levels=np.arange(15,32,1)
    cmap = plt.cm.RdYlBu_r

    ftz= 15
    ax=ax.ravel()


    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.5,
                        wspace=0.3)

    cax0 = inset_axes(ax[3],
                       width="360%",  # width = 5% of parent_bbox width
                       height="5%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(0, -0.25, 1, 1),
                       bbox_transform=ax[3].transAxes,
                       borderpad=0,
                       )


    p0 = ax[0].contourf(temp.lon,temp.lat,temp.to[0,0,:,:],cmap=cmap,levels=levels)
    ax[0].set_xlabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[0].set_ylabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[0].tick_params(labelsize=ftz)
    cbar = plt.colorbar(p0,cax0,orientation='horizontal')
    cbar.ax.tick_params(labelsize=ftz)
    cbar.set_label(r' SST and temperature ($^{\circ}$C)', size=ftz)
    ax[0].set_title('SST in '+ str(temp.time[0].values)[:7],fontsize=ftz)


    ax[1].contourf(temp.lon,temp.lat,temp.to[1,0,:,:],cmap=cmap,levels=levels)
    ax[1].set_title('SST in '+ str(temp.time[1].values)[:7],fontsize=ftz)
    ax[1].set_xlabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[1].set_ylabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[1].tick_params(labelsize=ftz)

    ax[2].contourf(temp.lon,temp.lat,temp.to[2,0,:,:],cmap=cmap,levels=levels)
    ax[2].set_title('SST in '+ str(temp.time[2].values)[:7],fontsize=ftz)
    ax[2].set_xlabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[2].set_ylabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[2].tick_params(labelsize=ftz)

    ax[3].contourf(temp.lon,temp.lat,temp.to.mean(dim='depth')[0,:,:],cmap=cmap,levels=levels)
    ax[3].set_title('Top 100m temp in '+ str(temp.time[0].values)[:7],fontsize=ftz)
    ax[3].set_xlabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[3].set_ylabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[3].tick_params(labelsize=ftz)

    ax[4].contourf(temp.lon,temp.lat,temp.to.mean(dim='depth')[1,:,:],cmap=cmap,levels=levels)
    ax[4].set_title('Top 100m temp in '+ str(temp.time[1].values)[:7],fontsize=ftz)
    ax[4].set_xlabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[4].set_ylabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[4].tick_params(labelsize=ftz)

    ax[5].contourf(temp.lon,temp.lat,temp.to.mean(dim='depth')[2,:,:],cmap=cmap,levels=levels)
    ax[5].set_title('Top 100m temp in '+ str(temp.time[2].values)[:7],fontsize=ftz)
    ax[5].set_xlabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[5].set_ylabel('Longitude ($^{\circ}$)',fontsize=ftz)
    ax[5].tick_params(labelsize=ftz)


def prepros_forecast(var,for_file):
    Lead_max=4
    Lead=np.arange(1,Lead_max+1)

    # Open observations
    f = xr.open_dataset('./data/dataset-armor-3d-nrt-monthly_2019_2023_inter_det_ML_ABA_ATL3.nc')
    date = f.time
    last = date[-1]
    day = date[-1].dt.day; month = date[-1].dt.month; year = date[-1].dt.year
    date = f.time.sel(time=slice('2020-01-01',datetime.date(year, month, day+10)))
    obs = f.get(var).sel(time=slice('2020-01-01',datetime.date(year, month, day+10)))
    
    # Open forcasts
    da = xr.DataArray(np.arange(len(date)+Lead_max), dims=["x"])
    da_ = xr.DataArray(np.arange(Lead_max+1), dims=["x"])
    date_forcast = np.datetime64(str(date[0].dt.strftime('%Y-%m-%d').values)+'T00:00:00.000000000') + da.astype("timedelta64[M]")
    date_forecast_last = np.datetime64(str(date[len(date)-1].dt.strftime('%Y-%m-%d').values)+'T00:00:00.000000000') + da_.astype("timedelta64[M]")
    NbY = int(date_forcast.dt.year[-1]-date_forcast.dt.year[0])
    
    TS_cnn = np.zeros((5,NbY+1,len(date_forcast)))*np.nan
    for_cnn = np.zeros((5,len(date_forecast_last)))*np.nan
    for_cnn[:,0] = obs[-1:]
    for i in range(0,Lead_max,1):    # forecast lead
        for j in range(1,13,1):  # season target
            for k in range(0,5,1):
                f_ = open('./output/bn_'+str(i+1)+'month_'+str(j)+'_'+str(for_file)+'/EN'+str(k+1)+'/result.gdat','r')
                cnn = np.fromfile(f_, dtype=np.float32)[:]
                for l in range(len(cnn)):
                    TS_cnn[k,i,l*12+j-1] = (cnn[l]*np.nanstd(f.get(var),0))+np.nanmean(f.get(var),0)
        for_cnn[:,i+1] = TS_cnn[:,i,-Lead_max+i]

    OBS = xr.Dataset({"obs": (("time"), obs.data)},
          coords={"time": date})

    da = xr.DataArray(TS_cnn, dims=['member','lead','time'])
    da = da.assign_coords({'member': [1, 2, 3, 4, 5],'lead': np.arange(1,Lead_max+1),'time': np.array(date_forcast)})
    TS = xr.Dataset({"cnn": (("member","lead","time"), da.data)},
         coords={"member": [1, 2, 3, 4, 5], "lead": np.arange(1,Lead_max+1), "time": np.array(date_forcast)})

    da = xr.DataArray(for_cnn, dims=['member','time'])
    da = da.assign_coords({'member': [1, 2, 3, 4, 5],'time': np.array(date_forecast_last)})
    forecasts = xr.Dataset({"cnn": (("member","time"), da.data)},
         coords={"member": [1, 2, 3, 4, 5], "time": np.array(date_forecast_last)})

    
    return OBS,TS,forecasts


def plot_forecasts(OBS,forecasts,zone):
    date = OBS.time
    day = date[-1].dt.day; month = date[-1].dt.month; year = date[-1].dt.year
    
    plt.subplot2grid((2,3),(0,0),rowspan=1,colspan=3)
    
    plt.plot(OBS.time.sel(time=slice(date[-6].dt.strftime('%Y-%m-%d').values,datetime.date(year, month, day+10))),
             OBS.obs.sel(time=slice(date[-6].dt.strftime('%Y-%m-%d').values,datetime.date(year, month, day+10))),'k')  

    for k in range(0,5,1):
        plt.plot(forecasts.time,forecasts.cnn[k,:],'darkorange')
    plt.axhline(y=0,color='grey',linewidth=0.7,alpha=0.6,linestyle = 'dashed')
    plt.plot(forecasts.time,np.nanmean(forecasts.cnn,0),'orangered')

    model_list = ['OBS','Forecast']
    plt.xlabel('Time', fontsize=7.5)
    plt.ylabel('SST anomalies', fontsize=7.5)
    plt.title('Forecasts, 1-4 months from last observation - Region: '+str(zone), fontsize=9)
    plt.legend(model_list,loc='best', prop={'size':7.5}, ncol=5)
    plt.tick_params(labelsize=7.5,direction='in',length=3,width=0.4,color='black')
    plt.tight_layout()
    
def plot_TS(OBS,forecasts,zone):
    date = forecasts.time
    day = date[-1].dt.day; month = date[-1].dt.month; year = date[-1].dt.year
    forC = np.nanmean(forecasts.cnn,0)
    Lead_max=len(forC)
    
    plt.subplot2grid((2,3),(1,0),rowspan=1,colspan=3)
    model_list = ['OBS','Pred Lead1','Pred Lead2','Pred Lead3','Pred Lead4']
    c=np.linspace(start=1, stop=0.2, num=Lead_max)
    plt.plot(OBS.time.sel(time=slice('2020-01-01',datetime.date(year, month, day+10))),OBS.obs.sel(time=slice('2020-01-01',datetime.date(year, month, day+10))),'k')
    for j in range(0,Lead_max): 
        plt.plot(forecasts.time,forC[j,:],'orangered', alpha=c[j])
    #plt.plot(forecasts.time,forC*0,color='grey',linewidth=0.7,alpha=0.6,linestyle = 'dashed')
    plt.axhline(y=0,color='grey',linewidth=0.7,alpha=0.6,linestyle = 'dashed')
    plt.axhline(y=np.nanstd(OBS.obs.sel(time=slice('2020-01-01',datetime.date(year, month, day+10))),0),color='grey',linewidth=0.7,alpha=0.6,linestyle = ':')
    plt.axhline(y=-1*np.nanstd(OBS.obs.sel(time=slice('2020-01-01',datetime.date(year, month, day+10))),0),color='grey',linewidth=0.7,alpha=0.6,linestyle = ':')
    
    #plt.plot(forecasts.time,forC/forC*np.nanstd(OBS.obs.sel(time=slice('2020-01-01',datetime.date(year, month, day+10))),0),color='grey',linewidth=0.7,alpha=0.6,linestyle = ':')
    #plt.plot(forecasts.time,-1*forC/forC*np.nanstd(OBS.obs.sel(time=slice('2020-01-01',datetime.date(year, month, day+10))),0),color='grey',linewidth=0.7,alpha=0.6,linestyle = ':')
    plt.xlabel('Time', fontsize=7.5)
    plt.xlim([datetime.date(2020, 1, 1), datetime.date(year, month, day+10)])
    plt.ylim([-1.2,1.8])
    plt.ylabel('SST anomalies', fontsize=7.5)
    plt.title('CNN Forecasts in '+str(zone), fontsize=9)
    plt.legend(model_list,loc='best', prop={'size':7.5}, ncol=5)
    plt.tick_params(labelsize=7.5,direction='in',length=3,width=0.4,color='black')
    plt.tight_layout()


