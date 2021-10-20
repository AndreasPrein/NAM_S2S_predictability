#!/usr/bin/env python
'''File name: Denver-Water_XWT.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 16.04.2018
    Date last modified: 16.04.2018

    ############################################################## 
    Purpos:
    Contains the setup for extreme weather typing (XWT) for
    Denver Water watersheds

'''

from pdb import set_trace as stop
import numpy as np
import os
import pandas as pd
import datetime

# ###################################################

dStartDayPR=datetime.datetime(1982, 1, 1,0)
dStopDayPR=datetime.datetime(2018, 12, 31,23)
rgdTime = pd.date_range(dStartDayPR, end=dStopDayPR, freq='d')
iMonths=[6,7,8,9,10] # [1,2,3,10,11,12] # [4,5,6,7,8,9]

# # ARIZONA
# sPlotDir='/glade/u/home/prein/projects/Arizona_WTing/plots/WT-Centroids'
# DW_Regions=['1501','1502','1503','1504','1505','1506','1507','1810'] 
# iRegion=0
# Region=DW_Regions
# sSubregionPR='/glade/campaign/mmm/c3we/prein/Shapefiles/HUC4/'

# NEW MEXICO
sPlotDir='/glade/u/home/prein/projects/Arizona_WTing/plots/WT-Centroids'
DW_Regions=['HUC6','HUC6','HUC6','HUC6','HUC6','HUC6'] 
iRegion=0
Region=DW_Regions
sSubregionPR='/glade/campaign/mmm/c3we/prein/Shapefiles/HUC6/NewMexico/'


# ---------
# Setup clustering algorithm
ClusterMeth='HandK'  # current options are ['HandK','hdbscan']
ClusterBreakup = 0     # breakes up clusters that are unproportionally large (only for hdbscan)
RelAnnom=1           # 1 - calculates daily relative anomalies
NormalizeData='C'    # normalize variables | options are  - 'C' - climatology
                                                        # - 'D' - daily (default)
                                                        # - 'N' - none
MinDistDD=1          # minimum nr of days between XWT events
RemoveAnnualCycl=0   # remove annual cycle in varaiables with 21 day moving average filter
# ---------
sDataDir='/glade/campaign/mmm/c3we/prein/Projects/Arizona_WTing/data/'+ClusterMeth+'/'


rgsWTvars= ['var151','u',   'v',     'UV',  'MFL850',  'MFL500',    'q',    'q'   , 'z', 'UV','t','t']
VarsFullName=['PSL','U850','V850','UV850',  'MFL850',  'MFL500', 'Q850', 'Q500', 'ZG500', 'UV200','T850','T500']
rgsWTfolders=['/glade/campaign/mmm/c3we/prein/ERA-Interim/PSL/fin_PSL-sfc_ERA-Interim_12-0_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/MFL850/MFL850_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/MFL500/MFL500_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/Q850/Q850_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/Q500/Q500_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/Z500/Z500_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV200/UV200_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/T850/T850_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/T500/T500_daymean_']

# rgsWTvars= ['var151','u',   'v',     'UV', 'tcw',  'MFL850',  'MFL500',    'q',    'q'   , 'z', 'UV','t','t']
# VarsFullName=['PSL','U850','V850','UV850',  'PW',  'MFL850',  'MFL500', 'Q850', 'Q500', 'ZG500', 'UV200','T850','T500']
# rgsWTfolders=['/glade/campaign/mmm/c3we/prein/ERA-Interim/PSL/fin_PSL-sfc_ERA-Interim_12-0_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/TCW/fin_TCW-sfc_ERA-Interim_12-0_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/MFL850/MFL850_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/MFL500/MFL500_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/Q850/Q850_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/Q500/Q500_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/Z500/Z500_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV200/UV200_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/T850/T850_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/T500/T500_daymean_']

# rgsWTvars= ['var151','u',   'v',     'UV', 'tcw',  'MFL850',  'MFL500',    'q',    'q'   , 'z','cape', 'UV']
# VarsFullName=['PSL','U850','V850','UV850',  'PW',  'MFL850',  'MFL500', 'Q850', 'Q500', 'ZG500','CAPE', 'UV200']
# rgsWTfolders=['/glade/campaign/mmm/c3we/prein/ERA-Interim/PSL/fin_PSL-sfc_ERA-Interim_12-0_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/TCW/fin_TCW-sfc_ERA-Interim_12-0_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/MFL850/MFL850_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/MFL500/MFL500_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/Q850/Q850_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/Q500/Q500_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/Z500/Z500_daymean_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/CAPE_ECMWF/fin_CAPE-ECMWF-sfc_ERA-Interim_12-9_',\
#               '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV200/UV200_daymean_']

rgrNrOfExtremes=[9999999] #[6,10,15,30]

WT_Domains=['S','M','L'] #,'M','L'] # ['S','M','L','XXL'] 
DomDegreeAdd=[2,5,10] #,5,10]   # [2,5,10,20] 

Annual_Cycle=['0'] # '1' means that the annual cycle gets removed before clustering; '0' nothing is done

SpatialSmoothing=[0.5] #[0,0.5,1]

Metrics=['PSS','MRD','MRR','APR','PEX','AUC','PRanom','InterVSIntra']

Dimensions=['Variables','Extreme Nr.','Domain Size','Annual Cycle','Smoothing','Split Sample','Metrics']

