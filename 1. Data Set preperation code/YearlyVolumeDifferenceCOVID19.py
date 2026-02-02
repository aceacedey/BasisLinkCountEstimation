from sklearn import svm, preprocessing
import scipy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score,mean_squared_error
from sklearn.linear_model import LinearRegression
from math import sqrt
#import statsmodels.api as sm
from scipy.stats.stats import pearsonr as pr
from sklearn import ensemble
import os,glob,io,holidays
from scipy.optimize import minimize
from sklearn.cross_decomposition import CCA
import matplotlib
import itertools
from os import listdir
from os.path import isfile, join

AU_holidays = holidays.Australia()
plt.rcParams.update({'font.size': 15})


def Plot(Real,Title,XL,YL,Links,Xticks):
    
    NoSubplot = len(Real)
    ax=plt.subplot()
    for i in range (0,NoSubplot):
        TotalX = len(Real[i])
        linkname = Links[i]
        x=np.arange(TotalX)
        
        y = Real[i]
        lab = 'Link id -' + str(linkname)
        ax.plot(x,y,marker=".",ls='-',label=lab)
        #ax.plot(x,yr,c='b',marker="v",ls=' ',label='Link Count (Observed)')
        #plt.legend(loc='upper left')
    plt.title(Title)
    plt.xlabel(str(XL)) # (Link Id: 2920-2919)')
    plt.ylabel(str(YL))
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xticks(x, Xticks, rotation=90)
    plt.show()

def RPlot(X,Y,method,df):
    
    wkday = np.asarray(df.index.dayofweek < 5)
    index_holidays = df.loc[df.index.isin(AU_holidays)]
    holi = np.asarray(index_holidays)
    print(wkday)
    print(holi)
    wkendSat = np.asarray(df.index.dayofweek == 5)
    wkendSun = np.asarray(df.index.dayofweek == 6)
    #print(wkday)
    inx = wkday.astype(int)
    #print(inx)
    #index = np.where(np.array(timeDiffInt) == 0, 0, 1)
    x = X
    y = Y
    #colors = ['blue','green','red','yellow']
    #levels = [0, 1, 2,3]
    #cmap, norm = matplotlib.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')
    fig, ax = plt.subplots()
    #ax.scatter(X[],Y,c=inx,marker="o",cmap=cmap,norm=norm,label=colors)
    #ax.plot(x,yr,c='b',marker="v",ls=' ',label='')
    #plt.legend(loc='upper left')
    #ax.hold(True)
    ax.scatter(X[wkday], Y[wkday], label='Weekdays', c='g')
    # scatter warning points in red (c='r')
    ax.scatter(X[wkendSat], Y[wkendSat], label='Saturdays', c='b')
    ax.scatter(X[wkendSun], Y[wkendSun], label='Sundays', c='r')
    if holi.size > 0:
        ax.scatter(X[holi], Y[holi], label='Holidays', c='y')
    Title = str(method)
    
    plt.title(Title)
    plt.legend(loc='upper left')
    plt.xlabel('Link Count (Observed)') # (Link Id: 2920-2919)')
    plt.ylabel('Link Count (Predicted)')
    
    res = sm.OLS(Y,sm.add_constant(X)).fit()
    X_plot = np.linspace(0,np.amax(X) + 10,10)
    plt.plot(X_plot, X_plot*res.params[1] + res.params[0])
    plt.grid(True)
    plt.show()

def SPlot(X,Y,Title,XL,YL,XTick):
    
    y = Y
    x = X
    plt.plot(x,y,c='g',marker="o",ls=' ')
    #plt.plot(x,yr,c='b',marker="v",ls=' ',label='Link Count (Observed)')
    plt.title(str(Title))
    plt.xlabel(str(XL))
    plt.ylabel(str(YL))
    #plt.yticks(Y,np.round(Y))
    labels = XTick
    for label, xt, yt in zip(labels, x, y):
        plt.annotate(
            label,
            xy=(xt, yt), xytext=(-10, 10),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.1),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()


def Stats(Pred,Real):
    yS = Pred
    xR = Real
    cc = np.corrcoef(yS,xR)
    results = sm.OLS(yS,sm.add_constant(xR)).fit()
    #print (results.summary())
    rms = sqrt(mean_squared_error(yS, xR))
    mape = np.mean(np.abs((xR - yS) /xR)) * 100
    return rms,mape

def corCCA(X,Y):
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    CCA(copy=True, max_iter=500, n_components=1, scale=True, tol=1e-10)
    X_c, Y_c = cca.transform(X, Y)
    return X_c,Y_c

def objective(x):
    
    alphaTemp = np.array(x)
    xt = np.matmul(X,alphaTemp.T) ### xt = np.matmul(X,alpha0.T)    xt = xt.reshape(len(xt),1)           TX = np.hstack((X,xt))
    xt = xt.reshape(len(xt),1)
    yt = np.matmul(Y,alphaTemp.T)
    yt = yt.reshape(len(yt),1)
    TX = np.hstack((X,xt))
    TY = np.hstack((Y,yt))
    l1 = np.matmul(a_p.T,TX)[0]
    l2 = np.matmul(b_p.T,TY)[0]
    CurrentCorr = pr(l1,l2)[0] ########## 
    #print(CurrentCorr)
    return abs(TargetCorr - CurrentCorr)

def constraint1(x):
    return 1 - (x[0] + x[1] + x[2])    ## always put > sign

def IndtialAlpha(x,y,z,k):
    a = 1 /(np.exp(-abs(x-k)) +  np.exp(-abs(y-k)) + np.exp(-abs(z-k))  )
    return a * np.exp(-abs(x-k)), a * np.exp(-abs(y-k)), a * np.exp(-abs(z-k))

def CalculateCount(X,Y,alpha0):
    

    #print(np.shape(X))
    # optimize
    b = (0.0,1.0)
    bnds = (b, b, b)
    con1 = {'type': 'ineq', 'fun': constraint1} 

    solution = minimize(objective,alpha0,method='SLSQP',bounds=bnds,constraints=con1,options={'ftol':1e-12})
    x = solution.x
    predX = np.matmul(X,x.T)
    #print(Stats(predX,RealX))
    #RPlot(predX,RealX,method)
    predY = np.matmul(Y,x.T)
    #print(Stats(predY,RealY))
    mpe = Stats(predX,RealX)[1]
    return predX,predY,mpe

corrAr = []
corrAr2 = []

##date_index16 = pd.date_range('1/1/2016', periods=366, freq='D')
##date_index17 = pd.date_range('1/1/2017', periods=364, freq='D')
##date_index18 = pd.date_range('1/1/2018', periods=365, freq='D')
##date_index19 = pd.date_range('1/1/2019', periods=364, freq='D')
##date_index20 = pd.date_range('1/1/2020', periods=46, freq='D')

import ast 
indf = pd.read_csv('inLinksN.csv')

AU_holidays = holidays.Australia()

ErrorX = []
mypath16 = r"C:\Users\deys\OneDrive - The University of Melbourne\Ph D\Codes\Missing Link paper code\Link Dataset\More Link data\2016 Link data new"
mypath17 = r"C:\Users\deys\OneDrive - The University of Melbourne\Ph D\Codes\Missing Link paper code\Link Dataset\More Link data\2017 Link data"
mypath18 = r"C:\Users\deys\OneDrive - The University of Melbourne\Ph D\Codes\Missing Link paper code\Link Dataset\More Link data\2018 Link data"
mypath19 = r"C:\Users\deys\OneDrive - The University of Melbourne\Ph D\Codes\Missing Link paper code\Link Dataset\More Link data\2019 Link data"
mypath20 = r"C:\Users\deys\OneDrive - The University of Melbourne\Ph D\Codes\Missing Link paper code\Link Dataset\More Link data\2020 Link data"

numLinks = len(indf)
i = 0
dataframes = ''
filenames16 = [f16 for f16 in listdir(mypath16) if isfile(join(mypath16, f16))]
filenames17 = [f17 for f17 in listdir(mypath17) if isfile(join(mypath17, f17))]
filenames18 = [f18 for f18 in listdir(mypath18) if isfile(join(mypath18, f18))]
filenames19 = [f19 for f19 in listdir(mypath19) if isfile(join(mypath19, f19))]
filenames20 = [f20 for f20 in listdir(mypath20) if isfile(join(mypath20, f20))]

timeSlice = len(filenames16)
#ErrorXAlllinks = np.zeros((numLinks,timeSlice))
#NoDataXAlllinks = np.zeros((numLinks,timeSlice),dtype='object')
ColTimes = []
for tiCount in range(0,timeSlice): ## Create time line cols
    M = (tiCount * 15 )%60
    H = (tiCount * 15) //60
    timeM = '%02d'%M
    timeH = '%02d'%H
    temp = str(timeH)+ ':' + str(timeM) + ':00' 
    ColTimes.append(temp)
    
errdfEq = pd.DataFrame(index=indf.out1.values,columns=ColTimes)
errdfBi = pd.DataFrame(index=indf.out1.values,columns=ColTimes)

nodatadf  = pd.DataFrame(index=indf.out1.values,columns=ColTimes)
corrdf = pd.DataFrame(index=indf.out1.values,columns=ColTimes)
x_tod = []
y20_tod = []
y19_tod = []
y18_tod = []
y20mean = []
y19mean = []
y18mean = []

y20std = []
y19std = []
y18std = []

##vol19 = pd.DataFrame()
##vol19 = pd.DataFrame()
##vol19 = pd.DataFrame()

for inxf, f in enumerate(filenames20):
    print(f)
    #print(inxf)
    cur16 = filenames16[inxf]
    cur17 = filenames17[inxf]
    cur18 = filenames18[inxf]
    cur19 = filenames19[inxf]
    cur20 = filenames20[inxf]
    
    x16 = pd.read_csv(join(mypath16,cur16))
    x17 = pd.read_csv(join(mypath17,cur17))

    x18 = pd.read_csv(join(mypath18,cur18))
    x18.rename(columns={'Unnamed: 0':'Date'},inplace=True)
    x18.index = pd.to_datetime(x18.Date)# ('Date',drop=True,inplace=True)
    x18wk = x18[x18.index.dayofweek <5] ## only weekdays
    ## remove year using split ## '-'.join(s.split('-')[1::] ##x18.Date.str.split('-',expand=True).agg('-'.join, axis=1)
    x18.index = x18.Date.str.split('-',expand=True)[[2,1]].agg('-'.join, axis=1) ## split the date as string, take months and day and then combine 
    x18wk.index = x18wk.Date.str.split('-',expand=True)[[2,1]].agg('-'.join, axis=1) ## split the date as string, take months and day and then combine 

    #######################
    #x18.set_index('Date',drop=True,inplace=True)
    
    x19 = pd.read_csv(join(mypath19,cur19))
    x19.rename(columns={'Unnamed: 0':'Date'},inplace=True)
    x19.index = pd.to_datetime(x19.Date)
    x19wk = x19[x19.index.dayofweek <5]
    x19.index = x19.Date.str.split('-',expand=True)[[2,1]].agg('-'.join, axis=1)
    x19wk.index = x19wk.Date.str.split('-',expand=True)[[2,1]].agg('-'.join, axis=1) ## split the date as string, take months and day and then combine 

    #x19.set_index('Date',drop=True,inplace=True)

    x20 = pd.read_csv(join(mypath20,cur20))
    x20.rename(columns={'Unnamed: 0':'Date'},inplace=True)
    x20.index = pd.to_datetime(x20.Date)
    x20wk = x20[x20.index.dayofweek <5]
    x20.index = x20.Date.str.split('-',expand=True)[[2,1]].agg('-'.join, axis=1)
    x20wk.index = x20wk.Date.str.split('-',expand=True)[[2,1]].agg('-'.join, axis=1) ## split the date as string, take months and day and then combine 

    #x20.set_index('Date',drop=True,inplace=True)

    links = x20.columns.values
    x20 = x20.drop('Date',axis=1).sum(axis=1)
    x19 = x19.drop('Date',axis=1).sum(axis=1)
    x18 = x18.drop('Date',axis=1).sum(axis=1)

    x20wk = x20wk.drop('Date',axis=1).sum(axis=1)
    x19wk = x19wk.drop('Date',axis=1).sum(axis=1)
    x18wk = x18wk.drop('Date',axis=1).sum(axis=1)
    if inxf == 0:
        vol19 = x19
        vol20 = x20
        vol18 = x18
        
        vol19wk = x19wk
        vol20wk = x20wk
        vol18wk = x18wk
    else:
        vol18 = vol18+x18
        vol19 = vol19+x19
        vol20 = vol20+x20

        vol18wk = vol18wk+x18wk
        vol19wk = vol19wk+x19wk
        vol20wk = vol20wk+x20wk


if 1 == 1:
    x_date = np.arange(len(vol20.index.values[0:-1]))
    x_ticks = vol20.index.values[0:-1]
    y18_date = vol18.loc[vol20.index[0:-1]]
    y19_date = vol19.loc[vol20.index[0:-1]]
    y20_date = vol20.values[0:-1] ## 29th days in feb

    plt.plot(x_date,y18_date,c='b', marker='*',ls='-', label='Link Count for year 2018')
    plt.plot(x_date,y19_date,c='g', marker='*',ls='-', label='Link Count for year 2019')
    plt.plot(x_date,y20_date,c='r', marker='*',ls='-',label='Link Count for year 2020')

    plt.title("Total link counts for three different calender years in CBD for two months")
    plt.xlabel("Date")
    plt.ylabel("Number of counted cars")
    plt.xticks(x_date,x_ticks,rotation='vertical')
    #labels = XTick
    plt.legend()
    plt.show()

if 1 == 1:
    comin = vol20wk.index.intersection(vol19wk.index)
    comin1 = comin.intersection(vol18wk.index)
    
    x_date = np.arange(len(comin1))
    x_ticks = comin1.to_list() #vol20wk.index.values[0:-1]
    y18_date = vol18wk.loc[comin1]
    y19_date = vol19wk.loc[comin1]
    y20_date = vol20wk.loc[comin1] ## 29th days in feb

    plt.plot(x_date,y18_date,c='b', marker='*',ls='-', label='Link Count for year 2018')
    plt.plot(x_date,y19_date,c='g', marker='*',ls='-', label='Link Count for year 2019')
    plt.plot(x_date,y20_date,c='r', marker='*',ls='-',label='Link Count for year 2020')

    plt.title("Total link counts for three different calender years in CBD of two months (Weekdays only)")
    plt.xlabel("Date")
    plt.ylabel("Number of counted cars")
    plt.xticks(x_date,x_ticks,rotation='vertical')
    #labels = XTick
    plt.legend()
    plt.show()


##baydf = pd.read_csv('LinkidParkingbayid.csv')
##
##ploc = pd.read_csv('PSensor.csv')
##if 1 == 1:
##    podf =pd.read_csv("10 02 00 00_ParkingData.txt", sep='\t',header=None)
##    podf.columns = podf.iloc[-1].values
##    podf.drop(podf.index.values[-1],axis=0,inplace=True)
##    #podf[podf[podf.columns.values[0]]]
##    podf = podf.rename(columns = {podf.columns.values[0]:'time'})
##    podf.time = pd.to_timedelta(podf.time)
##
##    
##v = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 3, 4, 3, 4, 3, 4, 5, 5, 5])
##np.where(v[:-1] != v[1:])[0] ## find where changes happened
##
##l = ast.literal_eval(baydf.st_marker_id.values[0]) ## for one link, we get list of bay ids associated with that link
##ioLinkid = ['4517-2925','2917-2925','2926-2925','2925-2924']


    #vol.append()
    
    #nxt17 = filenames17[inxf+1]
    #nxt16 = filenames16[inxf+1]
    #x1 = pd.read_csv(join(mypath16,f))
    #x2 = pd.read_csv(join(mypath17,cur17))
    
    #y1 = pd.read_csv(join(mypath16,nxt16))
    #y2 = pd.read_csv(join(mypath17,nxt17))


    #dfTraindataX = x2#pd.concat([x1,x2])

    
##    x19['Date'] = date_index19
##    x19.index = date_index19
##
    
##    x20['Date'] = date_index20
##    x20.index = date_index20
##    
##
##    #dfTraindataY = y2 #pd.concat([y1,y2])
##
##    
##    x18['Date'] = date_index18
##    x18.index = date_index18
##    
##    x19f = x19.iloc[31:55]
##    x20f = x20.iloc[-24::]
##    x19f = x19f.loc[:, (x19f != 0).any(axis=0)] ### delete all zero columns
##    x20f = x20f.loc[:, (x20f != 0).any(axis=0)]
    
##        dfYW = dfYW.loc[:, (dfYW != 0).any(axis=0)] ### delete all zero columns
##    if ColTimes[inxf] =="09:00": 
##        plt.errorbar(np.arange(len(links)), x20f.mean(axis=0), x20f.std(axis=0), color='r',linestyle='None', marker='*')
##        plt.errorbar(np.arange(len(links)) - 0.1, x19f.mean(axis=0), x19f.std(axis=0), color='g',linestyle='None', marker='*')
##        plt.xticks(np.arange(len(links)),links,rotation='vertical')
##        plt.show()
##    if ColTimes[inxf] =="09:00":
##        #plt.plot(np.arange(len(x20f.index.values)),y18_tod,c='b',ls='-', label='Link Count on 04/02/2018')
##        plt.plot(np.arange(len(x20f.index.values)),x19f.mean(axis=1),c='g',ls='-', label='February 2019')
##        plt.plot(np.arange(len(x20f.index.values)),x20f.mean(axis=1),c='r',ls='-',label='February 2020')
##
##        plt.title("Average link count varries on the month of February in 2019 and 2020")
##        plt.xlabel("Day of the month")
##        plt.ylabel("Number of counted cars")
##        plt.xticks(np.arange(len(x20f.index.values)),x19f.index.values,rotation='vertical')
##        #labels = XTick
##        plt.legend()
##        plt.show()

##    if ColTimes[inxf] =="09:00": 
##        plt.errorbar(np.arange(len(links)), x20.iloc[21::].mean(axis=0), x20.iloc[31::].std(axis=0), color='r',linestyle='None', marker='*')
##        plt.errorbar(np.arange(len(links)) - 0.1, x19.iloc[31:60].mean(axis=0), x19.iloc[31:60].std(axis=0), color='g',linestyle='None', marker='*')
##        plt.xticks(np.arange(len(links)),links,rotation='vertical')
##        plt.show()
##    
##    x_tod.append(inxf)
##    y18_tod.append(x18['2925-2924'].iloc[59])
##    y19_tod.append(x19['2925-2924'].iloc[59])
##    y20_tod.append(x20['2925-2924'].iloc[-2])
##    ## means and std of all links
##
##    
##    y18mean.append(x19['2925-2924'].iloc[31:60].mean()) 
##    y19mean.append(x19['2925-2924'].iloc[31:60].mean()) 
##    y20mean.append(x20['2925-2924'].iloc[21::].mean())
##
##    y18std.append(x19['2925-2924'].iloc[31:60].std()) 
##    y19std.append(x19['2925-2924'].iloc[31:60].std()) 
##    y20std.append(x20['2925-2924'].iloc[21::].std())
##    
##        ## x19['2925-2924'].iloc[31:60].mean()
##    
####    y18mean.append(x18.iloc[34].values[1::].mean()) ## means and std of all links
####    y19mean.append(x19.iloc[34].values[1::].mean()) 
####    y20mean.append(x20.iloc[34].values[1::].mean())
##
##
##if 1 == 1:
##    
##    
##    plt.plot(x_tod,y18_tod,c='b',ls='-', label='Link Count on 28/02/2018')
##    plt.plot(x_tod,y19_tod,c='g',ls='-', label='Link Count on 28/02/2019')
##    plt.plot(x_tod,y20_tod,c='r',ls='-',label='Link Count on 28/02/2020')
##
##    plt.title("Link count at link id: 2925-2924")
##    plt.xlabel("Time of the day")
##    plt.ylabel("Number of counted cars")
##    plt.xticks(x_tod,ColTimes,rotation='vertical')
##    #labels = XTick
##    plt.legend()
##    plt.show()
##
#### plot error bar for all links 

