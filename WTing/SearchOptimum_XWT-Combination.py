#!/usr/bin/env python
'''File name: ExtremeEvent-WeatherTyping.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 16.04.2018
    Date last modified: 20.04.2018
    
    This version of the program does not use two loops to find the 
    best variables but rather test all possible combintions of up to
    4 variables.

    ##############################################################
    INPUT

    ./ExtremeEvent-WeatherTyping.py Input1 Input2

    Input1) index for the region to run the algorithm on (e.g., 0, 1, 2...)
    Input2) Name of the setup file that should executed without the .py (e.g., example_setup)


    ############################################################## 
    Purpos:
    Classifies weather patterns that cause precipitation extremes

    1) read in shape file for area under cosideration

    2) read in precipitation data from PRISM

    3) identify the N-days that had highest rainfall records

    4) read in ERA-Interim data for these days

    5) remove the 30-year mooving average from time series and
       normalize the variables

    5) run clustering algorithm on extreme WT patterns

    6) search for the extreme WT centroids in the full record


'''

import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print( "testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print( "Using:",matplotlib.get_backend())
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
import scipy
import shapefile
import matplotlib.path as mplPath
from matplotlib.patches import Polygon as Polygon2
# Cluster specific modules
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.vq import kmeans2,vq, whiten
from scipy.ndimage import gaussian_filter
# import seaborn as sns
# import metpy.calc as mpcalc
import shapefile as shp
import sys
from itertools import combinations

from Functions_Extreme_WTs import XWT
from Functions_Extreme_WTs import MRR, MRD, perkins_skill

# ###################################################
# This information comes from the setup file

print('Number of arguments: '+str(len(sys.argv))+' arguments.')
iRegion=int(sys.argv[1])
sSetupFile=str(sys.argv[2])
subprocess.call('cp '+sSetupFile+'.py'+' sSetupFile.py', shell=True)

# from HUC2_XWTs_search import rgdTime, iMonths, sPlotDir, sDataDir, Region, sSubregionPR, rgsWTvars, VarsFullName,rgsWTfolders, rgrNrOfExtremes, WT_Domains, DomDegreeAdd, Annual_Cycle, SpatialSmoothing, Metrics, Dimensions, DW_Regions
from sSetupFile import rgdTime, iMonths, sPlotDir, sDataDir, Region, sSubregionPR, rgsWTvars, VarsFullName,rgsWTfolders, rgrNrOfExtremes, WT_Domains, DomDegreeAdd, Annual_Cycle, SpatialSmoothing, Metrics, Dimensions, DW_Regions, ClusterMeth, ClusterBreakup, RelAnnom, NormalizeData, MinDistDD, RemoveAnnualCycl
subprocess.call('rm '+'sSetupFile.py', shell=True)

# '/glade/campaign/mmm/c3we/prein/Shapefiles/HUC4/NHD_H_1501_HU4_Shape/Shape/WBDHU4'

if np.max(rgrNrOfExtremes) > len(rgdTime):
    rgrNrOfExtremes = np.array(rgrNrOfExtremes)
    rgrNrOfExtremes[rgrNrOfExtremes > len(rgdTime)] = len(rgdTime)

# create all possible combinations of variables
VarIndeces=np.array(range(len(rgsWTvars)))
Combinations1=np.array(list(combinations(VarIndeces, 1)))
Combinations2=np.squeeze(np.array(list(combinations(VarIndeces, 2))))
Combinations3=np.squeeze(np.array(list(combinations(VarIndeces, 3))))
Combinations4=np.squeeze(np.array(list(combinations(VarIndeces, 4))))
Combinations=list(Combinations1)+list(Combinations2)+list(Combinations3) #+list(Combinations4)

sRegion=DW_Regions[iRegion]
# # Arizona
# sSubregionPR=sSubregionPR+'NHD_H_'+sRegion+'_HU4_Shape/Shape/WBDHU4'
# New Mexico
sSubregionPR=sSubregionPR+sRegion

print('---- Process Region '+sRegion+' ----')

# create nessesary directories
if not os.path.exists(sDataDir):
    os.makedirs(sDataDir)
if not os.path.exists(sPlotDir):
    os.makedirs(sPlotDir)
sRegion=sRegion.replace('/','-')

ss='-'
sMonths=ss.join([str(iMonths[ii]) for ii in range(len(iMonths))])
print( sMonths)

# ###################################################
# use setup to generate data
rgiYears=np.unique(rgdTime.year)
YYYY_stamp=str(rgdTime.year[0])+'-'+str(rgdTime.year[-1])
rgiSeasonWT=np.isin(rgdTime.month, iMonths)
rgdTime=rgdTime[rgiSeasonWT]

SPLIT=np.where(rgdTime.year <= rgiYears[int(len(rgiYears)/2)])[0][-1]
SkillScores_All=np.zeros((len(Combinations),len(rgrNrOfExtremes),len(WT_Domains),len(Annual_Cycle),len(SpatialSmoothing),3,len(Metrics))); SkillScores_All[:]=np.nan

### CHECK IF DATA IS ALREADY PROCESSED ###
SaveStats=sDataDir+sRegion+'-'+str(iRegion).zfill(2)+'_'+YYYY_stamp+'-'+sMonths+'.npz'
if os.path.isfile(SaveStats) == 0:
    # ###################################################
    print('    Read the PRISM grid and data')
    PR_File = sDataDir+'PRISM-PR_'+sRegion+'-'+str(iRegion).zfill(2)+'_'+YYYY_stamp+'-'+sMonths+'.npz'
    if os.path.isfile(PR_File) == 0:
        ncid=Dataset('/glade/campaign/mmm/c3we/prein/observations/PRISM/data/PR/PRISM_daily_ppt_2014.nc', mode='r') # open the netcdf file
        rgrLatPR=np.squeeze(ncid.variables['lat'][:])
        rgrLonPR=np.squeeze(ncid.variables['lon'][:])
        ncid.close()
        rgrGridCells=[(rgrLonPR.ravel()[ii],rgrLatPR.ravel()[ii]) for ii in range(len(rgrLonPR.ravel()))]
        rgrSRactP=np.zeros((rgrLonPR.shape[0]*rgrLonPR.shape[1]))
        sf = shp.Reader(sSubregionPR)
        from Functions_Extreme_WTs import read_shapefile
        df = read_shapefile(sf)
        if df.shape[0] == 1:
            for sf in range(df.shape[0]):
                ctr = df['coords'][sf]
                if len(ctr) > 10000:
                    ctr=np.array(ctr)[::100,:] # carsen the shapefile accuracy
                else:
                    ctr=np.array(ctr)
                grPRregion=mplPath.Path(ctr)
                TMP=np.array(grPRregion.contains_points(rgrGridCells))
                rgrSRactP[TMP == 1]=1
        else:
            sf = iRegion
            ctr = df['coords'][sf]
            if len(ctr) > 10000:
                ctr=np.array(ctr)[::100,:] # carsen the shapefile accuracy
            else:
                ctr=np.array(ctr)
            grPRregion=mplPath.Path(ctr)
            TMP=np.array(grPRregion.contains_points(rgrGridCells))
            rgrSRactP[TMP == 1]=1

        rgrSRactP=np.reshape(rgrSRactP, (rgrLatPR.shape[0], rgrLatPR.shape[1]))
        rgiSrPR=np.array(np.where(rgrSRactP == True))
        iLatMaxP=rgiSrPR[0,:].max()+1
        iLatMinP=rgiSrPR[0,:].min()
        iLonMaxP=rgiSrPR[1,:].max()+1
        iLonMinP=rgiSrPR[1,:].min()
        rgrPRdata=np.zeros((sum(rgiSeasonWT),iLatMaxP-iLatMinP,iLonMaxP-iLonMinP))
        jj=0

        for yy in range(len(rgiYears)):
            rgdTimeYY = pd.date_range(datetime.datetime(rgiYears[0]+yy, 1, 1,0), end=datetime.datetime(rgiYears[0]+yy, 12, 31,23), freq='d')
            rgiDD=np.where(((rgdTimeYY.year == rgiYears[0]+yy) & (np.isin(rgdTimeYY.month, iMonths))))[0]
            # rgiDD=np.where(((rgdTimeYY.year == rgiYears[0]+yy) & (rgdTimeYY.month >=iStartMon ) & (rgdTimeYY.month <= iStopMon)))[0]
            ncid=Dataset('/glade/campaign/mmm/c3we/prein/observations/PRISM/data/PR/PRISM_daily_ppt_'+str(rgiYears[0]+yy)+'.nc', mode='r')
            rgrPRdata[jj:jj+len(rgiDD),:,:]=np.squeeze(ncid.variables['PR'][rgiDD,iLatMinP:iLatMaxP,iLonMinP:iLonMaxP])
            ncid.close()
            jj=jj+len(rgiDD)
        rgrPRdata[rgrPRdata<0] = np.nan
        
        # Save data
        np.savez(PR_File, rgrPRdata=rgrPRdata, rgdTime=rgdTime, sRegion=sRegion, ctr=ctr, rgrSRactP=rgrSRactP,
                iLatMaxP=iLatMaxP, iLatMinP=iLatMinP, iLonMaxP=iLonMaxP, iLonMinP=iLonMinP)
    else:
        print('    Load pre-processed PRISM data from: '+PR_File)
        DATA = np.load(PR_File)
        rgrPRdata = DATA['rgrPRdata']
        ctr= DATA['ctr']
        rgrSRactP= DATA['rgrSRactP']
        iLatMaxP= DATA['iLatMaxP']
        iLatMinP= DATA['iLatMinP']
        iLonMaxP= DATA['iLonMaxP']
        iLonMinP= DATA['iLonMinP']
    
#     # temp plot code
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(rgdTime, np.nanmean(rgrPRdata, axis=(1,2)))
#     ax.set_xlabel('Time [days]')
#     ax.set_ylabel('catchment average precipitation [mm d$^{-1}$]')
#     plt.show()
    
#     # sorted precipitation - nr. of days that contribute 80 % of total rainfall
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     SortedPR = np.sort(np.nanmean(rgrPRdata, axis=(1,2)))
#     CumACC = np.cumsum(SortedPR)
#     # plt.plot(np.linspace(0,100,len(CumACC)), ((CumACC/np.max(CumACC))*100)[::-1])
#     plt.plot(((CumACC/np.max(CumACC))*100)[::-1])
#     ax.set_xlabel('percent of days [%]')
#     ax.set_ylabel('contribution to total precipitation [%]')
#     plt.show()
    

    print( '    Read the ERA-Interim data')
    # We read in ERA-Interim data for the largest region and cut it to fit smaller regions
    DomDelta=np.max(DomDegreeAdd)
    Wlon=ctr[:,0].min()
    Elon=ctr[:,0].max()
    Nlat=ctr[:,1].max()
    Slat=ctr[:,1].min()
    DomainWT=np.array([[Elon+DomDelta,Slat-DomDelta],
                       [Wlon-DomDelta,Slat-DomDelta],
                       [Wlon-DomDelta,Nlat+DomDelta],
                       [Elon+DomDelta,Nlat+DomDelta],
                       [Elon+DomDelta,Slat-DomDelta]])
    grWTregion=mplPath.Path(DomainWT)

    # ###################################################
    #         Read the ERA-Interim grid and data
    from Functions_Extreme_WTs import ReadERAI
    DailyVarsLargeDom=ReadERAI(grWTregion,      # shapefile with WTing region
                       rgdTime,         # time period for WTing
                       iMonths,         # list of months that should be considered
                       rgsWTfolders,    # directories containing WT files
                       rgsWTvars)       # netcdf variable names of WT variables

    # ###################################################
    print( '    Read the ERA-Interim data specific for the region')
    
    for re in range(len(WT_Domains)):
        print( '    ------')
        print( '    Domain '+WT_Domains[re])
        DeltaX=np.max(DomDegreeAdd)-DomDegreeAdd[re]
        if DeltaX != 0:
            DomainWT=np.array([[Elon+DomDegreeAdd[re],Slat-DomDegreeAdd[re]],
                       [Wlon-DomDegreeAdd[re],Slat-DomDegreeAdd[re]],
                       [Wlon-DomDegreeAdd[re],Nlat+DomDegreeAdd[re]],
                       [Elon+DomDegreeAdd[re],Nlat+DomDegreeAdd[re]],
                       [Elon+DomDegreeAdd[re],Slat-DomDegreeAdd[re]]])

            grWTregion=mplPath.Path(DomainWT)
            rgrGridCells=[(DailyVarsLargeDom[1].ravel()[ii],DailyVarsLargeDom[2].ravel()[ii]) for ii in range(len(DailyVarsLargeDom[1].ravel()))]
            rgrSRact=np.array(grWTregion.contains_points(rgrGridCells)); rgrSRact=np.reshape(rgrSRact, (DailyVarsLargeDom[1].shape[0], DailyVarsLargeDom[1].shape[1]))
            rgiSrWT=np.array(np.where(rgrSRact == True))
            iLatMax=rgiSrWT[0,:].max()
            iLatMin=rgiSrWT[0,:].min()
            iLonMax=rgiSrWT[1,:].max()
            iLonMin=rgiSrWT[1,:].min()
            DailyVars=DailyVarsLargeDom[0][:,iLatMin:iLatMax,iLonMin:iLonMax,:]
        else:
            DailyVars=DailyVarsLargeDom[0]

        # perform split sample statistic
        for ss in range(3):
            print( '    Split Sample Nr. '+str(ss+1))
            if ss == 0:
                DailyVarsTrain=DailyVars[:SPLIT,:]
                DailyVarsEval=DailyVars[-SPLIT:,:]
                Ptrain=rgrPRdata[:SPLIT]
                Peval=rgrPRdata[-SPLIT:]
                TimeTrain=rgdTime[:SPLIT]
                TimeEval=rgdTime[-SPLIT:]
            elif ss == 1:
                DailyVarsTrain=DailyVars[-SPLIT:,:]
                DailyVarsEval=DailyVars[:SPLIT,:]
                Ptrain=rgrPRdata[-SPLIT:]
                Peval=rgrPRdata[:SPLIT]
                TimeTrain=rgdTime[-SPLIT:]
                TimeEval=rgdTime[:SPLIT]
            elif ss == 2:
                DailyVarsTrain=DailyVars
                DailyVarsEval=DailyVars
                Ptrain=rgrPRdata
                Peval=rgrPRdata
                TimeTrain=rgdTime
                TimeEval=rgdTime
    
            for ne in range(len(rgrNrOfExtremes)):
                DailyVarsAct=np.copy(DailyVarsTrain)
                print( '        '+str(rgrNrOfExtremes[ne])+' EXTREMES')
                iNrOfExtremes=rgrNrOfExtremes[ne]   # we consider the N highest rainfall extremes
                rgiSRgridcells=rgrSRactP[iLatMinP:iLatMaxP,iLonMinP:iLonMaxP].astype('int')
                rgrPRrecords=Ptrain[:,(rgiSRgridcells==1)] #np.nanmean(Ptrain[:,(rgiSRgridcells==1)], axis=(1))
                rgrPReval=Peval[:,(rgiSRgridcells == 1)] #np.nanmean(Peval[:,(rgiSRgridcells == 1)], axis=(1))
                ExtrNr = rgrNrOfExtremes[ne]
                if ExtrNr > len(rgrPReval):
                    ExtrNr = len(rgrPReval)
                # Test effect of spatial smoothing
                for sm in range(len(SpatialSmoothing)):
                    # annual cycle treatment
                    for ac in range(len(Annual_Cycle)):
                        print( '            Loop over variable permutations')
                        for va1 in range(len(Combinations)): 
                            XWT_output=XWT(DailyVarsTrain[:,:,:,Combinations[va1]],
                                           DailyVarsEval[:,:,:,Combinations[va1]],
                                           rgrPRrecords,
                                           rgrPReval,
                                           TimeTrain,
                                           TimeEval,
                                           ExtrNr,
                                           SpatialSmoothing[sm],
                                           ClusterMeth=ClusterMeth,
                                           ClusterBreakup=ClusterBreakup,
                                           RelAnnom=RelAnnom,
                                           NormalizeData=NormalizeData,
                                           MinDistDD=MinDistDD,
                                           RemoveAnnualCycl=RemoveAnnualCycl)
                            if XWT_output != None:
                                SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('PSS')]=XWT_output['grPSS'] # Perkins Skill Score
                                SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('MRD')]=XWT_output['grMRD'] # Mean relative difference
                                SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('MRR')]=XWT_output['grMRR'] # Mean Rank Ratio
                                SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('APR')]=np.abs(XWT_output['APR']) # Average precision-recall score
                                SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('PEX')]=np.abs(XWT_output['PEX']) # Percent of points excluded for ED larger than the 75 percentile
                                SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('AUC')]=np.abs(XWT_output['AUC'])
                                SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('PRanom')]=np.abs(XWT_output['PRanom'])
                                SkillScores_All[va1, ne, re, ac, sm, ss, Metrics.index('InterVSIntra')]=np.abs(XWT_output['InterVSIntra'])
                                
#                                 # ### Plot the synoptic setup and the centroids
#                                 # WT histograms
#                                 rgrClustersFin = XWT_output['grClustersFin']
#                                 WTnr = rgrClustersFin[1].max()
#                                 MonthlyWTfreq = np.zeros((12,WTnr+1))
#                                 for wt in range(WTnr+1):
#                                     TimeTMP = TimeEval[rgrClustersFin[1] == wt]
#                                     for mo in range(12):
#                                         MonthlyWTfreq[mo,wt] = np.sum(TimeTMP.month == (mo+1))
#                                 plt.plot(MonthlyWTfreq); plt.show()
                                
                print( ' ')
    
    np.savez(SaveStats, 
             SkillScores_All=SkillScores_All, 
             Combinations=Combinations, 
             rgsWTvars=VarsFullName,
             rgrNrOfExtremes=rgrNrOfExtremes,
             WT_Domains=WT_Domains,
             Annual_Cycle=Annual_Cycle,
             SpatialSmoothing=SpatialSmoothing,
             Metrics=Metrics,
             Dimensions=Dimensions)
    
else:
    print('    Load: '+SaveStats)
    DATA=np.load(SaveStats)
    SkillScores_All=DATA['SkillScores_All']
    Combinations=DATA['Combinations']
    VarsFullName=DATA['rgsWTvars']
    rgrNrOfExtremes=DATA['rgrNrOfExtremes']
    WT_Domains=DATA['WT_Domains']
    Annual_Cycle=DATA['Annual_Cycle']
    SpatialSmoothing=DATA['SpatialSmoothing']
    Metrics=DATA['Metrics']
    Dimensions=DATA['Dimensions']

# Find optimum and print best setting
Metrics=list(Metrics)
Scores=[Metrics.index('PRanom'),Metrics.index('InterVSIntra')]
Mean_SS=np.nanmean(SkillScores_All[:,:,:,:,:,:,Scores], axis=(5,6)) # axis=(5,6)
iOpt=np.where(np.nanmax(Mean_SS) == Mean_SS)

print(' ')
print('====================================')
print('======    OPTIMAL SETTINGS    ======')
print('VARIABLES')
for va in range(len(Combinations[iOpt[0][0]])):
    print('    '+VarsFullName[int(Combinations[iOpt[0][0]][va])])
print('Extreme Nr     : '+str(rgrNrOfExtremes[iOpt[1][0]]))
print('Domain Size    : '+str(WT_Domains[iOpt[2][0]]))
print('Annual Cy. Rem.: '+str(Annual_Cycle[iOpt[3][0]]))
print('Smoothing      : '+str(SpatialSmoothing[iOpt[4][0]]))
print('Average Score  : '+str(np.round(np.nanmax(Mean_SS),2)))
print('====================================')
stop()

# In[80]:



# PlotFile=sRegion+'_XWT_Search-Optimum_'+YYYY_stamp+'_'+sMonths+'.pdf'
# from Functions_Extreme_WTs import SearchOptimum_XWT
# SearchOptimum_XWT(PlotFile,
#                  sPlotDir,
#                  SkillScores_All,
#                  GlobalMinimum1,
#                  GlobalMinimum2,
#                  Optimum,
#                  VariableIndices,
#                  Dimensions,
#                  Metrics,
#                  VarsFullName,
#                  ss,
#                  rgrNrOfExtremes,
#                  WT_Domains,
#                  Annual_Cycle,
#                  SpatialSmoothing)



