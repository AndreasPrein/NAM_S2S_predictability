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

def HUC2_XWTs_apply(Season,
                    Region):

    from pdb import set_trace as stop
    import numpy as np
    import os
    import pandas as pd
    import datetime
    
    # ###################################################
    
    dStartDayPR=datetime.datetime(1982, 1, 1,0)
    dStopDayPR=datetime.datetime(2018, 12, 31,23)
    rgdTime = pd.date_range(dStartDayPR, end=dStopDayPR, freq='d')

    if Season == 'AMJJAS':
        iMonths=[4,5,6,7,8,9]
    elif Season == 'ONDJFM':
        iMonths=[1,2,3,10,11,12]
    elif Season == 'JJASOND':
        iMonths=[6,7,8,9,10,11,12]
    elif Season == 'JJASO':
        iMonths=[6,7,8,9,10]
    elif Season == 'Annual':
        iMonths=[1,2,3,4,5,6,7,8,9,10,11,12]
        
    # ---------
    # Setup clustering algorithm
    ClusterMeth='HandK'  # current options are ['HandK','hdbscan']
    ClusterBreakup = 0   # breakes up clusters that are unproportionally large (only for hdbscan)
    RelAnnom=1           # 1 - calculates daily relative anomalies
    NormalizeData='C'    # normalize variables | options are  - 'C' - climatology
                                                            # - 'D' - daily (default)
                                                            # - 'N' - none
    MinDistDD=1          # minimum nr of days between XWT events
    RemoveAnnualCycl=0   # remove annual cycle in varaiables with 21 day moving average filter
    # ---------

    # DENVER WATER REGIONS
    sPlotDir='/glade/u/home/prein/projects/Arizona_WTing/plots/WT-Centroids'
    sDataDir='/glade/campaign/mmm/c3we/prein/Projects/Arizona_WTing/data/'+ClusterMeth+'/'

#     # Arizona
#     DW_Regions=['1501','1502','1503','1504','1505','1506','1507','1810']
#     # sRegion=Regions.index(Region)
#     Region=DW_Regions[Region]
#     sSubregionPR='/glade/campaign/mmm/c3we/prein/Shapefiles/HUC4/' #+Regions[sRegion]

    # New Mexico
    DW_Regions=['HUC6-00','HUC6-01','HUC6-02','HUC6-03','HUC6-04','HUC6-05']
    Region=DW_Regions[Region]
    sSubregionPR='/glade/campaign/mmm/c3we/prein/Shapefiles/HUC6/NewMexico/'

    Metrics=['PSS','MRD','MRR','APR','PEX','AUC','PRanom','InterVSIntra']
    Dimensions=['Variables','Extreme Nr.','Domain Size','Annual Cycle','Smoothing','Split Sample','Metrics']

    if (Season == 'JJASO') & (Region == '1501'):
        VarsFullName=['Q850'] #['PSL','PW','MFL850'] #
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='M' #'M'    # ['S','M','L','XXL'] 
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #0.5 #[0,0.5,1]
    elif (Season == 'JJASO') & (Region == '1502'):
        VarsFullName=['Q850'] # ['U850','MFL850','UV200']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='M' #'L' #'M'    # ['S','M','L','XXL'] 
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #0.5 #[0,0.5,1]
    elif (Season == 'JJASO') & (Region == '1503'):
        VarsFullName=['Q850'] # ['U850','MFL850','UV200']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='M' #'L' #'M'    # ['S','M','L','XXL'] 
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #0.5 #[0,0.5,1]
    elif (Season == 'JJASO') & (Region == '1504'):
        VarsFullName=['Q850'] # ['U850','MFL850','UV200']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='M' #'L' #'M'    # ['S','M','L','XXL'] 
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #0.5 #[0,0.5,1]
    elif (Season == 'JJASO') & (Region == '1505'):
        VarsFullName=['Q850'] # ['U850','MFL850','UV200']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='M' #'L' #'M'    # ['S','M','L','XXL'] 
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #0.5 #[0,0.5,1]
    elif (Season == 'JJASO') & (Region == '1506'):
        VarsFullName=['Q850'] # ['U850','MFL850','UV200']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='M' #'L' #'M'    # ['S','M','L','XXL'] 
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #0.5 #[0,0.5,1]
    elif (Season == 'JJASO') & (Region == '1507'):
        VarsFullName=['Q850'] # ['U850','MFL850','UV200']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='M' #'L' #'M'    # ['S','M','L','XXL'] 
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #0.5 #[0,0.5,1]
    elif (Season == 'JJASO') & (Region == '1810'):
        VarsFullName=['Q850'] # ['U850','MFL850','UV200']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='M' #'L' #'M'    # ['S','M','L','XXL'] 
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #0.5 #[0,0.5,1]
    elif (Season == 'JJASO') & (Region == 'HUC6-00'):
        VarsFullName=['Q850']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='S'
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5
    elif (Season == 'JJASO') & (Region == 'HUC6-01'):
        VarsFullName=['Q850']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='S'
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5
    elif (Season == 'JJASO') & (Region == 'HUC6-02'):
        VarsFullName=['Q850']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='S'
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5
    elif (Season == 'JJASO') & (Region == 'HUC6-03'):
        VarsFullName=['Q850']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='S'
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5
    elif (Season == 'JJASO') & (Region == 'HUC6-04'):
        VarsFullName=['Q850']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='S'
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5
    elif (Season == 'JJASO') & (Region == 'HUC6-05'):
        VarsFullName=['Q850']
        rgrNrOfExtremes=99999 #3 #[6,10,15,30]
        WT_Domains='S'
        Annual_Cycle='0' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5






    # ---------------
    # Full list of available variables
    
    rgsWTvarsAll= ['var151','u',   'v',     'UV', 'tcw',  'MFL850',  'MFL500',    'q',    'q'   , 'z', 'UV','t','t']
    VarsFullNameAll=['PSL','U850','V850','UV850',  'PW',  'MFL850',  'MFL500', 'Q850', 'Q500', 'ZG500', 'UV200','T850','T500']
    rgsWTfoldersAll=['/glade/campaign/mmm/c3we/prein/ERA-Interim/PSL/fin_PSL-sfc_ERA-Interim_12-0_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/TCW/fin_TCW-sfc_ERA-Interim_12-0_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/MFL850/MFL850_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/MFL500/MFL500_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/Q850/Q850_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/Q500/Q500_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/Z500/Z500_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/UV200/UV200_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/T850/T850_daymean_',\
              '/glade/campaign/mmm/c3we/prein/ERA-Interim/T500/T500_daymean_']
    
    iSelVariables=[VarsFullNameAll.index(VarsFullName[ii]) for ii in range(len(VarsFullName))]
    rgsWTvars=np.array(rgsWTvarsAll)[np.array(iSelVariables).astype('int')]
    rgsWTfolders=np.array(rgsWTfoldersAll)[np.array(iSelVariables).astype('int')]

    DomDegreeAdd=np.array([2, 5, 10, 20])[['S','M','L','XXL'].index(WT_Domains)]

    return rgdTime, iMonths, sPlotDir, Region, sDataDir, sSubregionPR, rgsWTvars, VarsFullName,rgsWTfolders, rgrNrOfExtremes, WT_Domains, DomDegreeAdd, Annual_Cycle, SpatialSmoothing, Metrics, Dimensions, ClusterMeth, ClusterBreakup, RelAnnom, NormalizeData, MinDistDD, RemoveAnnualCycl
