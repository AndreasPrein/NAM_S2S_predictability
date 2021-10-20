#!/usr/bin/env python
'''File name: Denver-Water_XWT_CESM.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 16.04.2018
    Date last modified: 16.04.2018

    ############################################################## 
    Purpos:
    Contains the setup for extreme weather typing (XWT) for
    Denver Water watersheds applied to CESM

'''


def Denver_Water_XWT_CESM(Season,
                          Region):  

    from pdb import set_trace as stop
    import numpy as np
    import os
    import pandas as pd
    import datetime
    
    # ###################################################
    
    dStartDayPR=datetime.datetime(1920, 01, 01,0)
    dStopDayPR=datetime.datetime(2100, 12, 31,23)
    rgdTime = pd.date_range(dStartDayPR, end=dStopDayPR, freq='d')
    # iMonths=[4,5,6,7,8,9] # [1,2,3,10,11,12] # [4,5,6,7,8,9]
    
    if Season == 'AMJJAS':
        iMonths=[4,5,6,7,8,9]
    elif Season == 'ONDJFM':
        iMonths=[1,2,3,10,11,12]
    
    # DENVER WATER REGIONS
    sPlotDir='/glade/u/home/prein/projects/2019_Denver-Water_Extreme-WTs/plots/LENS/'# +str(iNrOfExtremes)+'_Events/'
    sDataDir='/glade/u/home/prein/projects/2019_Denver-Water_Extreme-WTs/data/LENS/'
    # DW_Regions=['DenverWater_CollectionSystem',\
    #             'East-Continental-Divide',\
    #             'West-Continental-Divide']
    sSubregionPR='/glade/u/home/prein/projects/2019_Denver-Water_Extreme-WTs/Shapefiles/For_Python/'+Region

    LENSmembers=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','101','102','103','104','105']
    s20Cname='b.e11.B20TRC5CNBDRD.f09_g16.'
    s21Cname='b.e11.BRCP85C5CNBDRD.f09_g16.'

    Metrics=['PSS','MRD','MRR']
    Dimensions=['Variables','Extreme Nr.','Domain Size','Annual Cycle','Smoothing','Split Sample','Metrics']
    
    # LENS AVAILABLE VARIABLE LISTING
    # rgsWTvarsLENS= ['PSL','U850','V850','U850',  'TMQ',     'TMQ', 'Q850', 'Q500',  'Z500'] # name in the file!
    # VarsFullName=  ['PSL','U850','V850','UV850',  'TMQ',  'FLX850', 'Q850', 'Q500', 'Z500']
    # rgsWTfoldersLENS=['/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/PSL/',\
    #                   '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/U850/',\
    #                   '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/V850/',\
    #                   '/glade/scratch/prein/LENS/UV850/',\
    #                   '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/TMQ/',\
    #                   '/glade/scratch/prein/LENS/FLX850/',\
    #                   '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Q850/',\
    #                   '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Q500/',\
    #                   '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Z500/']
    
    
    # LENS VARIABLE SELECTION
    if (Season == 'AMJJAS') & (Region == 'DenverWater_CollectionSystem'):
        sClusterSave='/glade/u/home/prein/projects/2019_Denver-Water_Extreme-WTs/programs/Extreme-WTs/data/Clusters15_DenverWater_CollectionSystem_1981-2018_ZG500-PSL-FLX850-U850-UV850'+'_'+Season
        rgsWTvarsLENS=   ['Z500','PSL','TMQ'   ,'U850','U850'] # name in the file!
        VarsFullNameLENS=['Z500','PSL','FLX850','U850','UV850']
        rgsWTfoldersLENS=['/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Z500/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/PSL/',\
                          '/glade/scratch/prein/LENS/FLX850/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/U850/',\
                          '/glade/scratch/prein/LENS/UV850/']
        rgrNrOfExtremes=15 #[6,10,15,30]
        WT_Domains='M'    # ['S','M','L','XXL'] 
        DomDegreeAdd=5   # [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #[0,0.5,1]
    if (Season == 'ONDJFM') & (Region == 'DenverWater_CollectionSystem'):
        sClusterSave='/glade/u/home/prein/projects/2019_Denver-Water_Extreme-WTs/programs/Extreme-WTs/data/Clusters6_DenverWater_CollectionSystem_1981-2018_ZG500-PW-PSL_'+Season
        rgsWTvarsLENS=   ['Z500','TMQ','PSL'] # name in the file!
        VarsFullNameLENS=['Z500','TMQ','PSL']
        rgsWTfoldersLENS=['/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Z500/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/TMQ/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/PSL/']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] 
        DomDegreeAdd=10   # [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=1 #[0,0.5,1]
    if (Season == 'AMJJAS') & (Region == 'West-Continental-Divide'):
        sClusterSave='/glade/u/home/prein/projects/2019_Denver-Water_Extreme-WTs/programs/Extreme-WTs/data/Clusters6_West-Continental-Divide_1981-2018_Q850-ZG500-UV850-Q500-PSL_'+Season
        rgsWTvarsLENS=   ['Q850','Z500','U850','Q500','PSL'] # name in the file!
        VarsFullNameLENS=['Q850','Z500','UV850','Q500','PSL']
        rgsWTfoldersLENS=['/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Q850/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Z500/',\
                          '/glade/scratch/prein/LENS/UV850/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Q500/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/PSL/']
        rgrNrOfExtremes=6 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] 
        DomDegreeAdd=10   # [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #[0,0.5,1]
    if (Season == 'ONDJFM') & (Region == 'West-Continental-Divide'):
        sClusterSave='/glade/u/home/prein/projects/2019_Denver-Water_Extreme-WTs/programs/Extreme-WTs/data/Clusters10_West-Continental-Divide_1981-2018_PW-Q500-ZG500-Q850-UV850_'+Season
        rgsWTvarsLENS=   ['TMQ','Q500','Z500','Q850','U850'] # name in the file!
        VarsFullNameLENS=['TMQ','Q500','Z500','Q850','UV850']
        rgsWTfoldersLENS=['/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/TMQ/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Q500/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Z500/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Q850/',\
                          '/glade/scratch/prein/LENS/UV850/']
        rgrNrOfExtremes=10 #[6,10,15,30]
        WT_Domains='M'    # ['S','M','L','XXL'] 
        DomDegreeAdd=5   # [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=1 #[0,0.5,1]
    if (Season == 'AMJJAS') & (Region == 'East-Continental-Divide'):
        sClusterSave='/glade/u/home/prein/projects/2019_Denver-Water_Extreme-WTs/programs/Extreme-WTs/data/Clusters15_East-Continental-Divide_1981-2018_ZG500-PSL-Q500-Q850-UV850_'+Season
        rgsWTvarsLENS=   ['Z500','PSL','Q500','Q850','U850'] # name in the file!
        VarsFullNameLENS=['Z500','PSL','Q500','Q850','UV850']
        rgsWTfoldersLENS=['/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Z500/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/PSL/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Q500/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Q850/',\
                          '/glade/scratch/prein/LENS/UV850/']
        rgrNrOfExtremes=15 #[6,10,15,30]
        WT_Domains='M'    # ['S','M','L','XXL'] 
        DomDegreeAdd=5   # [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0.5 #[0,0.5,1]
    if (Season == 'ONDJFM') & (Region == 'East-Continental-Divide'):
        sClusterSave='/glade/u/home/prein/projects/2019_Denver-Water_Extreme-WTs/programs/Extreme-WTs/data/Clusters10_East-Continental-Divide_1981-2018_ZG500-PSL-PW-FLX850_'+Season
        rgsWTvarsLENS=   ['Z500','PSL','TMQ','TMQ'] # name in the file!
        VarsFullNameLENS=['Z500','PSL','TMQ','FLX850']
        rgsWTfoldersLENS=['/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/Z500/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/PSL/',\
                          '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/TMQ/',\
                          '/glade/scratch/prein/LENS/FLX850/']
        rgrNrOfExtremes=10 #[6,10,15,30]
        WT_Domains='L'    # ['S','M','L','XXL'] 
        DomDegreeAdd=10   # [2,   5,  10, 20] 
        Annual_Cycle='1' # '1' means that the annual cycle gets removed before clustering; '0' nothing is done
        SpatialSmoothing=0 #[0,0.5,1]

    return rgdTime, iMonths, sPlotDir, sDataDir, Region, sSubregionPR, rgsWTvarsLENS, VarsFullNameLENS,rgsWTfoldersLENS, rgrNrOfExtremes, WT_Domains, DomDegreeAdd, Annual_Cycle, SpatialSmoothing, Metrics, Dimensions, sClusterSave, LENSmembers, s20Cname, s21Cname
