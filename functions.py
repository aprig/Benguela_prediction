
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


