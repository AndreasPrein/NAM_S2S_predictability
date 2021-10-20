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
    
    dStartDayPR=datetime.datetime(1981, 01, 01,0)
    dStopDayPR=datetime.datetime(2018, 12, 31,23)
    rgdTime = pd.date_range(dStartDayPR, end=dStopDayPR, freq='d')

    if Season == 'AMJJAS':
        iMonths=[4,5,6,7,8,9]
    elif Season == 'ONDJFM':
        iMonths=[1,2,3,10,11,12]
    elif Season == 'Annual':
        iMonths=[1,2,3,4,5,6,7,8,9,10,11,12]

    # DENVER WATER REGIONS
    sPlotDir='/glade/u/home/prein/papers/Extreme-WTs-US/plots/WT-Centroids/best_less_extreme_days/'# +str(iNrOfExtremes)+'_Events/'
    sDataDir='/glade/work/prein/papers/Extreme-WTs/data/best/'
    DW_Regions=['01/WBDHU2','02/WBDHU2','03/WBDHU2','04/WBDHU2','05/WBDHU2','06/WBDHU2','07/WBDHU2','08/WBDHU2','09/WBDHU2',\
                '10/WBDHU2','11/WBDHU2','12/WBDHU2','13/WBDHU2','14/WBDHU2','15/WBDHU2','16/WBDHU2','17/WBDHU2','18/WBDHU2']
    # sRegion=Regions.index(Region)
    # Region=Regions[sRegion]
    sSubregionPR='/glade/u/home/prein/ShapeFiles/US_HUC2-Regions/' #+Regions[sRegion]

    Metrics=['PSS','MRD','MRR']
    Dimensions=['Variables','Extreme Nr.','Domain Size','Annual Cycle','Smoothing','Split Sample','Metrics']

    if (Season == 'Annual') & (Region == '01/WBDHU2'):
        VarsFullName=['Q850','UV850']
        rgrNrOfExtremes=15 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '02/WBDHU2'):
        VarsFullName=['U850','V850','ZG500','UV200','FLX850']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='M'    # ['S','M','L','XXL'] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '03/WBDHU2'):
        VarsFullName=['PSL','U850','UV850']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='M'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=1 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '04/WBDHU2'):
        VarsFullName=['PSL','U850','ZG500']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='S'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '05/WBDHU2'):
        VarsFullName=['PW','U850','UV850','MFL500']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=1 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '06/WBDHU2'):
        VarsFullName=['FLX850','UV850','Q500','UV200']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '07/WBDHU2'):
        VarsFullName=['ZG500','UV850','U850','UV200','CAPE']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=1 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '08/WBDHU2'):
        VarsFullName=['MFL850','PW','U850','Q850','UV200','CAPE','V850']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='S'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=1 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '09/WBDHU2'):
        VarsFullName=['ZG500','Q850','V850','PSL','Q500','FLX850']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='S'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '10/WBDHU2'):
        VarsFullName=['ZG500','UV850','PSL','CAPE']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '11/WBDHU2'):
        VarsFullName=['FLX850','UV850','U850','Q500','CAPE','V850','PW','ZG500']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '12/WBDHU2'):
        VarsFullName=['PSL','U850','MFL850','Q500','PW','UV200','MFL500','V850']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='S'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=1 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '13/WBDHU2'):
        VarsFullName=['Q500','Q850']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='S'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=1 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '14/WBDHU2'):
        VarsFullName=['MFL500','PW','MFL850','ZG500','U850']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='M'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '15/WBDHU2'):
        VarsFullName=['MFL850','Q850','MFL500','V850','UV200']
        rgrNrOfExtremes=15 #[6,10,15,30]
        WT_Domains='M'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=1 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '16/WBDHU2'):
        VarsFullName=['FLX850','PW','V850']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '17/WBDHU2'):
        VarsFullName=['PW','UV850','PSL','ZG500','Q850']
        rgrNrOfExtremes=30 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #[0,0.5,1]
    if (Season == 'Annual') & (Region == '18/WBDHU2'):
        VarsFullName=['PW','FLX850']
        rgrNrOfExtremes=15 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] # grid cells added [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=1 #[0,0.5,1]




    # ---------------
    # Full list of available variables
    VarsFullNameAll=['PSL','U850','V850','UV850',  'PW',  'FLX850', 'Q850', 'Q500', 'ZG500','CAPE', 'UV200',  'MFL850',  'MFL500']
    rgsWTvarsAll   =['var151','u',   'v',     'UV', 'tcw',  'FLX',    'q',    'q'   , 'z','cape', 'UV',  'MFL850',  'MFL500']
    rgsWTfoldersAll=['/glade/scratch/prein/ERA-Interim/PSL/fin_PSL-sfc_ERA-Interim_12-0_',\
                     '/glade/scratch/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
                     '/glade/scratch/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
                     '/glade/scratch/prein/ERA-Interim/UV850/fin_UV850-sfc_ERA-Interim__',\
                     '/glade/scratch/prein/ERA-Interim/TCW/fin_TCW-sfc_ERA-Interim_12-0_',\
                     '/glade/scratch/prein/ERA-Interim/UV850xTCW/fin_FLX-pl_ERA-Interim_',\
                     '/glade/scratch/prein/ERA-Interim/Q850/Q850_daymean_',\
                     '/glade/scratch/prein/ERA-Interim/Q500/Q500_daymean_',\
                     '/glade/scratch/prein/ERA-Interim/Z500/Z500_daymean_',\
                     '/glade/scratch/prein/ERA-Interim/CAPE_ECMWF/fin_CAPE-ECMWF-sfc_ERA-Interim_12-9_',\
                     '/glade/scratch/prein/ERA-Interim/UV200/UV200_daymean_',\
                     '/glade/campaign/mmm/c3we/prein/ERA-Interim/MFL850/MFL850_daymean_',\
                     '/glade/campaign/mmm/c3we/prein/ERA-Interim/MFL500/MFL500_daymean_']
    iSelVariables=[VarsFullNameAll.index(VarsFullName[ii]) for ii in range(len(VarsFullName))]
    rgsWTvars=np.array(rgsWTvarsAll)[np.array(iSelVariables).astype('int')]
    rgsWTfolders=np.array(rgsWTfoldersAll)[np.array(iSelVariables).astype('int')]

    DomDegreeAdd=np.array([2,   5,  10, 20])[['S','M','L','XXL'].index(WT_Domains)]


    return rgdTime, iMonths, sPlotDir, sDataDir, sSubregionPR, rgsWTvars, VarsFullName,rgsWTfolders, rgrNrOfExtremes, WT_Domains, DomDegreeAdd, Annual_Cycle, SpatialSmoothing, Metrics, Dimensions
