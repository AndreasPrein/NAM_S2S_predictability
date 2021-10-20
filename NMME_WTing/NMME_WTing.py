#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python


# # NMME_WTing.ipynb

# In[1]:


'''
    File name: NMME_WTing.ipynb
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 04.03.2021
    Date last modified: 04.03.2021

    ##############################################################
    Purpos:

    1) Read in NMME variables over season and period of interest

    2) Read in the ERA-Interim WTs for the region

    3) Preprocess the data for the WTing

    4) Assign WTs to each day of the forecast and save the results

'''


# In[18]:


from dateutil import rrule
import datetime
import glob
from netCDF4 import Dataset
import sys, traceback
import dateutil.parser as dparser
import string
from pdb import set_trace as stop
import numpy as np
import numpy.ma as ma
import os
from mpl_toolkits import basemap
import pickle
import subprocess
import pandas as pd
from scipy import stats
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pylab as plt
import random
import scipy.ndimage as ndimage
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import string
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
import shapely.geometry
# import descartes
import shapefile
import math
from scipy.stats.kde import gaussian_kde
from math import radians, cos, sin, asin, sqrt
from scipy import spatial
import scipy.ndimage
import matplotlib.path as mplPath
from scipy.interpolate import interp1d
import time
from math import atan2, degrees, pi
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import csv
import pygrib
from scipy import interpolate
from scipy import signal
# from netcdftime import utime
from scipy.ndimage import gaussian_filter
import scipy.ndimage.filters as filters
from calendar import monthrange

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
import h5py
import xarray as xr

def interp_weights(xy, uv,d=2):
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)


# In[134]:


########################################
#                            USER INPUT SECTION
NMME_YYYY=1990 #int(sys.argv[1])

mo = int(sys.argv[1]) # model implemented are [0,1,2,3]
iRegion = int(sys.argv[2])

# mo = 3 # model implemented are [0,1,2]
# sRegion = '1501'  # selected regions are '1501', '1502', 'HUC6-00', 'HUC6-03'

DataDirERAI='/glade/campaign/mmm/c3we/prein/Projects/Arizona_WTing/data/HandK/'
SRegionsAll = ['1501', '1502', 'HUC6-00', 'HUC6-03'] 
sRegion = SRegionsAll[iRegion]
SRegionNames = ['AZ West', 'AZ East', 'NM North', 'NM South']
iSelReg = SRegionNames[SRegionsAll.index(sRegion)]

NMME_dir = '/glade/collections/cdg/data/nmme/output1/'
IFS_dir = '/glade/campaign/mmm/c3we/ECMWF/'
# center/model name, ensemble members, file convention
NMME_models = [['NCAR/CESM1',10,'day_CESM1'], #['CCCMA','NASA-GMAO','NCAR/CESM1', 'NCEP','NOAA-GFDL','UM-RSMAS']
               ['NASA-GMAO/GEOS-5',10,'day_GEOS-5'], # maskes out below surface areas --> use 650 hPa level
               ['UM-RSMAS/CCSM4', 10, 'day_CCSM4'],
               ['CCCMA/CanCM4', 10, 'day_CanCM4'],  # only has 675 hPa data
               ['IFS', 25, 'day_CanCM4']] # only 7 month forecast but 25 members
ConstantFile = [NMME_dir+'NCAR/CESM1/19820101/day/atmos/hus/hus_day_CESM1_19820101_r4i1p1_19820100-19821231.nc4',
               NMME_dir+'NASA-GMAO/GEOS-5/19820101/day/atmos/hus/hus_day_GEOS-5_19820101_r1i1p1.nc',
               NMME_dir+'UM-RSMAS/CCSM4/20050801/day/atmos/hus/hus_day_CCSM4_20050801_r10i1p1_20050801-20060731.nc',
               NMME_dir+'CCCMA/CanCM4/19840101/day/atmos/v20181101/hus/hus_day_CanCM4_198401_r10i1p1_19840101-19841231.nc4',
               IFS_dir+'20050601/Q_GDS0_ISBL/Q_GDS0_ISBL_day_ECMWF_mem01_20050601.nc']
DataDir = [NMME_dir,
          NMME_dir,
          NMME_dir,
          NMME_dir,
          IFS_dir]
# for each variable we have the general varname, the netCDF var name, and the pressure level (-1 means 2D field), netCDF varname
ImputVars=[[['Q850','hus',0,'hus']],
            [['Q850','hus',13,'hus']], # level 13 is 650hPa; 850 hPa is level 6
            [['Q850','hus',0,'HUS']],
            [['Q850','hus',0,'hus']],
            [['Q850','Q_GDS0_ISBL',-1,'Q_GDS0_ISBL_daily']]]

SaveDir='/glade/campaign/mmm/c3we/prein/Projects/Arizona_WTing/data/NMME/'

MONTHS=[6,7,8,9,10] # [1,2,3,4,5,6,7,8,9,10,11,12]
StartMonths=[2,3,4,5,6,7,8]

dStartDay=datetime.datetime(int(1982), 1, 1,12)
dStopDay=datetime.datetime(int(2010), 12, 31,12)
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTimeMM = pd.date_range(dStartDay, end=dStopDay, freq='m')
rgiYY=np.unique(rgdTimeDD.year)
rgdTimeDD = rgdTimeDD[np.isin(rgdTimeDD.month, MONTHS)]
rgdTimeMM = rgdTimeMM[np.isin(rgdTimeMM.month, MONTHS)]


# In[135]:


# Get the ERA-Interim centroids and data
if sRegion == '1501':
    RegName = 'Arizone West'
    CentroidFile = DataDirERAI+'1501_XWT-centroids_train-1982-2018_eval-1982-2018_E13514_XWTs3_Vars-Q850_M-6-7-8-9-10.nc'
    # we get this data from previously running ~/projects/Arizona_WTing/programs/WTing/Centroids-and-Scatterplot.py
    ERA_data = '/glade/campaign/mmm/c3we/prein/Projects/Arizona_WTing/data/HandK/ERA-Interim_PRISM_data13514_1501_1982-2018_Q850_JJASO.npz'
    WTdata = DataDirERAI+'Clusters13514_1501_1982-2018_Q850_JJASO'
if sRegion == '1502':
    RegName = 'Arizone East'
    ERA_data = '/glade/campaign/mmm/c3we/prein/Projects/Arizona_WTing/data/HandK/ERA-Interim_PRISM_data13514_1502_1982-2018_Q850_JJASO.npz'
    CentroidFile = DataDirERAI+'1502_XWT-centroids_train-1982-2018_eval-1982-2018_E13514_XWTs3_Vars-Q850_M-6-7-8-9-10.nc'
    WTdata = DataDirERAI+'Clusters13514_1502_1982-2018_Q850_JJASO'
if sRegion == 'HUC6-00':
    RegName = 'New Mexico North'
    ERA_data = '/glade/campaign/mmm/c3we/prein/Projects/Arizona_WTing/data/HandK/ERA-Interim_PRISM_data13514_HUC6-00_1982-2018_Q850_JJASO.npz'
    CentroidFile = DataDirERAI+'HUC6-00_XWT-centroids_train-1982-2018_eval-1982-2018_E13514_XWTs3_Vars-Q850_M-6-7-8-9-10.nc'
    WTdata = DataDirERAI+'Clusters13514_HUC6-00_1982-2018_Q850_JJASO'
if sRegion == 'HUC6-03':
    RegName = 'New Mexico South'
    ERA_data = '/glade/campaign/mmm/c3we/prein/Projects/Arizona_WTing/data/HandK/ERA-Interim_PRISM_data13514_HUC6-03_1982-2018_Q850_JJASO.npz'
    CentroidFile = DataDirERAI+'HUC6-03_XWT-centroids_train-1982-2018_eval-1982-2018_E13514_XWTs3_Vars-Q850_M-6-7-8-9-10.nc'
    WTdata = DataDirERAI+'Clusters13514_HUC6-03_1982-2018_Q850_JJASO'
    
ncid=Dataset(CentroidFile, mode='r')
rgrLat75=np.squeeze(ncid.variables['centroids'][:])
rgrLonC=np.squeeze(ncid.variables['rlon'][:])
rgrLatC=(np.squeeze(ncid.variables['rlat'][:]))
ncid.close()

# load preprocessed ERA and PRISM data
DATA = np.load(ERA_data)
ERA_Data = DATA['DailyVarsOrig']
rgrLonT = DATA['LonWT']
rgrLatT = DATA['LatWT']
PR_record = DATA['rgrPRrecords']
TimeERA = pd.DatetimeIndex(DATA['rgdTime'])
YYYY_era = np.unique(TimeERA.year)

ERA_annual = np.reshape(ERA_Data, (len(YYYY_era),int(ERA_Data.shape[0]/len(YYYY_era)), ERA_Data.shape[1], ERA_Data.shape[2], ERA_Data.shape[3]))

# Load the Centroids
print('    Restore: '+WTdata)
with open(WTdata, 'rb') as handle:
    npzfile = pickle.load(handle)
WTclusters=npzfile['grClustersFin']['Full']
WTlat=npzfile['LatWT']#; rgrLatT1 = WTlat
WTlon=npzfile['LonWT']#; rgrLonT1 = WTlon
WTlon[WTlon<0] = WTlon[WTlon<0]+360
WTtime=npzfile['rgdTime']
SpatialSmoothing=npzfile['SpatialSmoothing']


# In[136]:


# ________________________________________________________________________
# read ERA-Interim & NMME Grid
sERAconstantFields='/glade/work/prein/reanalyses/ERA-Interim/ERA_Inerim_stationary-files_75x75.nc'
# read the ERA-Interim elevation
ncid=Dataset(sERAconstantFields, mode='r')
rgrLat75=np.squeeze(ncid.variables['latitude'][:])
rgrLon75=np.squeeze(ncid.variables['longitude'][:])
rgrHeight=(np.squeeze(ncid.variables['z'][:]))/9.81
rgrLSM=(np.squeeze(ncid.variables['lsm'][:]))
ncid.close()
rgdTimeDD_Full = pd.date_range(datetime.datetime(int(1979), 1, 1,12), end=datetime.datetime(int(2017), 12, 31,12), freq='d')

# read NMME coordinates
ncid=Dataset(ConstantFile[mo], mode='r')
if 'lon' in list(ncid.variables.keys()):
    rgrLonS=np.squeeze(ncid.variables['lon'][:])
    rgrLatS=np.squeeze(ncid.variables['lat'][:])
elif 'LON' in list(ncid.variables.keys()):
    rgrLonS=np.squeeze(ncid.variables['LON'][:])
    rgrLatS=np.squeeze(ncid.variables['LAT'][:])
elif 'g0_lon_4' in list(ncid.variables.keys()):
    rgrLonS=np.squeeze(ncid.variables['g0_lon_4'][:])
    rgrLatS=np.squeeze(ncid.variables['g0_lat_3'][:])
    
ncid.close()
rgrLonSF, rgrLatSF = np.meshgrid(rgrLonS, rgrLatS)
rgrLonSF[rgrLonSF>180] = rgrLonSF[rgrLonSF>180]-360

if NMME_models[mo][0] == 'IFS':
    rgrLatSF = rgrLatSF[::-1] # IFS lat runs from N to S

# get the region of interest
iAddCells= 4 # grid cells added to subregion
iWest=np.argmin(np.abs(rgrLonT.min() - rgrLonSF[0,:]))-iAddCells
iEast=np.argmin(np.abs(rgrLonT.max() - rgrLonSF[0,:]))+iAddCells
iNort=np.argmin(np.abs(rgrLatT.max() - rgrLatSF[:,0]))+iAddCells
iSouth=np.argmin(np.abs(rgrLatT.min() - rgrLatSF[:,0]))-iAddCells

rgrLonS=rgrLonSF[iSouth:iNort,iWest:iEast]
rgrLatS=rgrLatSF[iSouth:iNort,iWest:iEast]

# create gregridding weights
# Remap Gridsat to ERA5
points=np.array([rgrLonS.flatten(), rgrLatS.flatten()]).transpose()
vtx, wts = interp_weights(points, np.append(rgrLonT.flatten()[:,None], rgrLatT.flatten()[:,None], axis=1))


# ### Read NMME data

# In[137]:


from dateutil.relativedelta import relativedelta
from datetime import timedelta

NMMEdata = np.zeros((sum(rgdTimeDD.year == rgiYY[0]), len(rgiYY), len(StartMonths),rgrLonT.shape[0],rgrLonT.shape[1],len(ImputVars[mo]), NMME_models[mo][1])); NMMEdata[:] = np.nan
for yy in range(len(rgiYY)):
    print('    process '+str(rgiYY[yy]))
    DD_monsoonseason = pd.date_range(datetime.datetime(rgiYY[yy], MONTHS[0], 1,0), end=datetime.datetime(rgiYY[yy], MONTHS[-1], monthrange(rgiYY[yy], MONTHS[-1])[1],0), freq='d')
    for mm in range(len(StartMonths)):
        print('        month '+str(StartMonths[mm]))
        TIMEstamp = str(rgiYY[yy])+str(StartMonths[mm]).zfill(2)+'01'
        dStartACT=datetime.datetime(rgiYY[yy], StartMonths[mm], 1,0)
        if NMME_models[mo][0] in ['NCAR/CESM1','UM-RSMAS/CCSM4','CCCMA/CanCM4']:
            dStopACT = dStartACT + relativedelta(months=+12) - timedelta(days=1)
            rgdTimeACT = pd.date_range(dStartACT, end=dStopACT, freq='d')
        elif NMME_models[mo][0] == 'IFS':
            dStopACT = dStartACT + relativedelta(days=+214)
            rgdTimeACT = pd.date_range(dStartACT, end=dStopACT, freq='d')
        else:
            dStopACT = dStartACT + relativedelta(months=+9) - timedelta(days=1)
            rgdTimeACT = pd.date_range(dStartACT, end=dStopACT, freq='d')
        for va in range(len(ImputVars[mo])):
            if NMME_models[mo][0] == 'CCCMA/CanCM4':
                DirNameAct = DataDir[mo]+NMME_models[mo][0]+'/'+TIMEstamp+'/day/atmos/v20181101/'+ImputVars[mo][va][1]+'/'
            elif NMME_models[mo][0] == 'IFS':
                if (rgiYY[yy] >= 1993) & (StartMonths[mm] >= 4):
                    DirNameAct = DataDir[mo]+'/'+TIMEstamp+'/'+ImputVars[mo][va][1]+'/'
                else:
                    continue
            else:
                DirNameAct = DataDir[mo]+NMME_models[mo][0]+'/'+TIMEstamp+'/day/atmos/'+ImputVars[mo][va][1]+'/'
            
            for en in range(NMME_models[mo][1]):
                YYYYMMDD_start = str(rgdTimeACT[0].year)+str(rgdTimeACT[0].month).zfill(2)+'00' #str(rgdTimeACT[0].day).zfill(2)
                YYYYMMDD_stop = str(rgdTimeACT[-1].year)+str(rgdTimeACT[-1].month).zfill(2)+'*' #str(rgdTimeACT[-1].day).zfill(2)
                if NMME_models[mo][0] == 'CCCMA/CanCM4':
                    TIMEstamp = str(rgiYY[yy])+str(StartMonths[mm]).zfill(2)
                if NMME_models[mo][0] != 'IFS':
                    try:
                        FileName = glob.glob(DirNameAct+ImputVars[mo][va][1]+'_'+NMME_models[mo][2]+'_'+TIMEstamp+'_r'+str(en+1)+'i1p1*'+'.nc*')[0]
                    except:
                        print('            Data missing: '+DirNameAct+ImputVars[mo][va][1]+'_'+NMME_models[mo][2]+'_'+TIMEstamp+'_r'+str(en+1)+'i1p1*'+'.nc*')
                        continue
                else:
                    FileName = glob.glob(DirNameAct+ImputVars[mo][va][1]+'_day_ECMWF_mem'+str(en+1).zfill(2)+'*.nc')[0]
#                 FileName = glob.glob(DirNameAct+ImputVars[mo][va][1]+'_'+NMME_models[mo][2]+'_'+TIMEstamp+'_r'+str(en+1)+'i1p1_'+YYYYMMDD_start+'-'+YYYYMMDD_stop+'.nc*')[0]
                iTime = np.isin(rgdTimeACT.month, MONTHS) & np.isin(rgdTimeACT.year, rgiYY[yy])
                # read the data
                ncid=Dataset(FileName, mode='r')
                if ImputVars[mo][va][2] == -1:
                    # data has no level dimension (is 3D)
                    DATAact=np.squeeze(ncid.variables[ImputVars[mo][va][3]][iTime,:,:])[:,::-1,:][:,iSouth:iNort,iWest:iEast]
                else:
                    try:
                        DATAact=np.squeeze(ncid.variables[ImputVars[mo][va][3]][iTime,ImputVars[mo][va][2],iSouth:iNort,iWest:iEast])
                    except:
                        # some models do not have leap years
                        DATAact=np.squeeze(ncid.variables[ImputVars[mo][va][3]][iTime[:-1],ImputVars[mo][va][2],iSouth:iNort,iWest:iEast])
                ncid.close()
                try:
                    T0 = np.where(rgdTimeACT[0] == DD_monsoonseason)[0][0]
                except:
                    T0 = 0
                # Remap the data to the target (centroid) grid
                for tt in range(sum(iTime)):
                    valuesi=interpolate(DATAact[tt,:,:].flatten(), vtx, wts)
                    NMMEdata[T0+tt,yy,mm,:,:,va,en] = valuesi.reshape(rgrLonT.shape[0],rgrLonT.shape[1])


# ### Reduce the data arrays to valid years and valid months only

# In[138]:


# FIN_Y = ~np.isnan(NMMEdata[0,:,4,0,0,0,0])
# FIN_M = ~np.isnan(NMMEdata[0,-1,:,0,0,0,0])

# NMMEdata = NMMEdata[:,FIN_Y][:,:,FIN_M]
# rgiYY = rgiYY[FIN_Y]
# StartMonths = np.array(StartMonths)[FIN_M]

# # # ERA
# # ERA_WTfin = ERA_WTfin[FIN_Y]


# In[139]:


m = Basemap(llcrnrlon=-121,llcrnrlat=20,urcrnrlon=-62,urcrnrlat=51,
    projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

tt=100
mon=4
yy = 15
xi,yi=m(rgrLonT,rgrLatT)
m.contourf(xi,yi,NMMEdata[tt,yy,mon,:,:,0,0]*1000.,cmap='coolwarm', levels=np.linspace(0,10,11), extend='both')

m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.drawstates(linewidth=0.25)


# ### Associate Days to WTs

# In[140]:


from Functions_Extreme_WTs import PreprocessWTdata
from Functions_Extreme_WTs import EucledianDistance

# Setup clustering algorithm
ClusterMeth='HandK'  # current options are ['HandK','hdbscan']
ClusterBreakup = 0   # breakes up clusters that are unproportionally large (only for hdbscan)
RelAnnom=1           # 1 - calculates daily relative anomalies
NormalizeData='C'    # normalize variables | options are  - 'C' - climatology
                                                        # - 'D' - daily (default)
                                                        # - 'N' - none
MinDistDD=1          # minimum nr of days between XWT events
RemoveAnnualCycl=0   # remove annual cycle in varaiables with 21 day moving average filter

SHAPE = NMMEdata.shape
WT_NMME = np.zeros((SHAPE[0],SHAPE[1],SHAPE[2],SHAPE[6])); WT_NMME[:] = np.nan
for mm in range(len(StartMonths)):
    print('    month '+str(StartMonths[mm]))
    Mean = np.nanmean(NMMEdata[:,:,mm,:], axis=(0,1,5))
    Anomalies = (NMMEdata[:,:,mm,:] - Mean[None,None,:,:,None])/Mean[None,None,:,:,None]
    Normalize = [np.nanmean(Anomalies[:,:,:], axis=(0,1,2,3,5)), np.nanstd(Anomalies[:,:,:], axis=(0,1,2,3,5)),Mean]
    for yy in range(len(rgiYY)):
        print('        process '+str(rgiYY[yy]))
        for va in range(len(ImputVars[mo])):
            for en in range(NMME_models[mo][1]):
                if np.isnan(np.nanmean(NMMEdata[:,yy,mm,:,:,:,en])) == False:
                    isNAN = np.isnan(np.nanmean(NMMEdata[:,yy,mm,:,:,:,en], axis=(1,2,3)))
                    DailyVarsEvalNorm=PreprocessWTdata(NMMEdata[:,yy,mm,:,:,:,en],               # WT data [time,lat,lon,var]
                                       RelAnnom=RelAnnom,                     # calculate relative anomalies [1-yes; 0-no]
                                       SmoothSigma=0,                         # Smoothing stddev (Gaussian smoothing)
                                       RemoveAnnualCycl=RemoveAnnualCycl,             # remove annual cycle [1-yes; 0-no]
                                       NormalizeData=NormalizeData,                # normalize data [1-yes; 0-no]
                                       Normalize = Normalize)                     # predefined mean and std for normalization          
                    EucledianDist, Correlation =EucledianDistance(DailyVarsEvalNorm,
                                                                  WTclusters)
                    EucledianDist_orig=np.copy(EucledianDist)
                    EucledianDist=np.nanmin(EucledianDist,axis=1)
                    MinED =  np.nanargmin(EucledianDist_orig,axis=1).astype(float)
                    MinED[isNAN] = np.nan
                    WT_NMME[:,yy,mm,en] =MinED


# ### Save the data for external processing

# In[141]:


ERA_YYYY = np.unique(WTtime.year)
VARS_ET = '-'.join([ImputVars[mo][ii][0] for ii in range(len(ImputVars[mo]))])
PredMonths = '-'.join(np.array(MONTHS).astype('str'))
sStartMonths = '-'.join(np.array(StartMonths).astype('str'))
ERA_ETs = np.reshape(WTclusters[1], (len(YYYY_era), int(WTclusters[1].shape[0]/len(ERA_YYYY))))

SaveFile = SaveDir+NMME_models[mo][0].replace("/", "-")+'_'+VARS_ET+'_'+str(rgiYY[0])+'-'+str(rgiYY[-1])+'_'+iSelReg.replace(" ", "-")+'_ForecastMonths-'+sStartMonths+'_MonsoonMonths-'+PredMonths+'.npz'
SaveData = np.savez(SaveFile,
                   NMMEpredictors=NMMEdata,
                   TimeMonsSeason = rgdTimeACT[iTime],
                   YYYYY_NMME=rgiYY,
                   StartMonths=StartMonths,
                   rgrLatT=rgrLatT,
                   rgrLonT=rgrLonT,
                   WT_vars=VARS_ET,
                   WT_NMME=WT_NMME,
                   ERA_predictors = ERA_annual,
                   YYYY_ERA = YYYY_era,
                   ERA_WTfin=ERA_ETs)


