import math
import pandas as pd
import numpy as np
import networkx as nx
import re
import matplotlib.pyplot as plt
import scipy
from scipy.special import comb
import itertools
from os import listdir
from os.path import isfile, join

#mypath = "VSDATA_2017"
mypath = "C:\Work Temporary Storage\SCATSRawData\VSDATA_2018"

SLDS = pd.read_csv('ScatLinkDataSmall.csv')
#ODF = pd.read_csv('ODFinalSmallDataset1.csv')
AllFrom = np.array(SLDS.From)
AllTo = np.array(SLDS.TO)
i = 0
dataframes = ''
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
numFiles = len(filenames)
Rows = len(SLDS)
timeintervals = 96
YhatFinal = np.zeros((timeintervals,numFiles,Rows))
Header = []
i = 0
date_list = []
for f in filenames[-27::]:
    print(f)
    y = f[7:11]
    m = f[11:13]
    d = f[13:15]
    date = str(d) + '/' + str(m) + '/' + str(y)
    date_list.append(date)
    #print(date_list)
    dataframes = pd.read_csv(join(mypath,f))

    for linkCount in range(0,Rows):
        indexLinkAll = linkCount
        LinkToSearch = int(SLDS.SCATtoSearch[indexLinkAll])
        SensorNumbersL = list((SLDS.Lane1[indexLinkAll],SLDS.Lane2[indexLinkAll],SLDS.Lane3[indexLinkAll],SLDS.Lane4[indexLinkAll]))
        ActiveSensor = list(filter(None,SensorNumbersL)) 
        #ActiveSensorIndex = np.nonzero(np.array(SensorNumbersL))
        #print(linkCount)
        #print(ActiveSensor)
        
        for tiCount in range(1,timeintervals):
            NumVehicales = 0
            num = '%02d'%tiCount
            Vs = 'V'+str(num)
            for SensorCount in range(0,len(ActiveSensor)):
                CurDetector = int(ActiveSensor[SensorCount])
                NVList = dataframes[(dataframes.NB_SCATS_SITE == LinkToSearch) & (dataframes.NB_DETECTOR == CurDetector)][Vs]# 32 36 48 70 & dataframes.V47 == change V47 for other time
                
                ##       interval 32 means 7:45 to 8:00 am
                if NVList.empty ==True or math.isnan(NVList) ==True:
                    NumVehicales = 0
                elif int(NVList) > 0:
                    NumVehicales = NumVehicales + int(NVList)
                else:
                    NumVehicales = 0
            #print(NumVehicales)
            YhatFinal[tiCount,i,linkCount] = NumVehicales
          
    i = i + 1
        
for linkCount in range(0,Rows):
    CurLinkSource = SLDS.From[linkCount]
    CurLinkSink = SLDS.TO[linkCount]
    print(str(CurLinkSource)+'    '+ str(CurLinkSink))
    TH =  str(CurLinkSource)+'-'+ str(CurLinkSink)
    Header.append(TH)
#dataframes[(df.NB_SCATS_SITE == 4604)]
#dataframes[(df.NB_SCATS_SITE == 4604) & (df.NB_DETECTOR == 1)]

header = Header
for tiCount in range(1,timeintervals):
    M = (tiCount * 15 )%60
    H = (tiCount * 15) //60
    timeM = '%02d'%M
    timeH = '%02d'%H
    curName = 'LinkTrainData_' + str(timeH)+ str(timeM) +str(filenames[0][6:11])+ '.csv'
    YhatdataFrame = pd.DataFrame(YhatFinal[tiCount])
    YhatdataFrame.index = pd.to_datetime(date_list,format='%d/%m/%Y')  
    YhatdataFrame.to_csv(curName,header=Header)
