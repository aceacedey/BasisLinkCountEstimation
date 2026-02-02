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
import statsmodels.api as sm
from scipy.stats.stats import pearsonr as pr
from sklearn import ensemble
import os,glob,io,holidays
from scipy.optimize import minimize
from sklearn.cross_decomposition import CCA
import matplotlib
import itertools
from os import listdir
from os.path import isfile, join
import seaborn as sns

AU_holidays = holidays.Australia()
plt.rcParams.update({'font.size': 28})


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
    #plt.grid(True)
    plt.xticks(x, Xticks, rotation=90)
    plt.show()

def RPlot(X,Y,method,df,mpe):
    
    wkday = np.asarray(df.index.dayofweek < 5)
    index_holidays = df.loc[df.index.isin(AU_holidays)]
    holi = np.asarray(index_holidays)
    #print(wkday)
    #print(holi)
    wkendSat = np.asarray(df.index.dayofweek >= 5)
    #wkendSun = np.asarray(df.index.dayofweek == 6)
    #print(wkday)
    inx = wkday.astype(int)
    #print(inx)
    #index = np.where(np.array(timeDiffInt) == 0, 0, 1)
    x = X
    y = Y
    fig, ax = plt.subplots()

    ax.scatter(X[wkday], Y[wkday], label='Weekdays', c='b',marker='*')
    # scatter warning points in red (c='r')
    ax.scatter(X[wkendSat], Y[wkendSat], label='Weekends', c='r',marker='^')
    if holi.size > 0:
    #ax.scatter(X[wkendSun], Y[wkendSun], label='Sundays', c='r')
        ax.scatter(X[holi], Y[holi], label='National holidays', c='y')
    Title = str(method)
    
    plt.title(Title)
    plt.legend(loc='upper left')
    plt.xlabel('Link Count (Observed)') # (Link Id: 2920-2919)')
    mpe = round(mpe,2)
    yl = 'Link Count (Predicted) with MAPE = ' + str(mpe)
    plt.ylabel(yl)

    res = sm.OLS(Y,sm.add_constant(X)).fit()
    X_plot = np.linspace(0,np.amax(X) + 10,10)
    #plt.plot(X_plot, X_plot*res.params[1] + res.params[0])
    plt.plot(X_plot, X_plot)
    #plt.grid(True)
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
    #plt.grid(True)
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
    CCA(copy=True, max_iter=5000, n_components=1, scale=True, tol=1e-10)
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

def CalculateCount(X,Y,alpha0,RealX):

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
    rmse = np.sqrt(np.mean((RealX-predX)**2))
    return predX,predY,rmse,mpe

def objective_newCCA(x):
    
    alphaTemp = np.array(x)
    xt = np.matmul(X,alphaTemp.T) ### xt = np.matmul(X,alpha0.T)    xt = xt.reshape(len(xt),1)           TX = np.hstack((X,xt))
    xt = xt.reshape(len(xt),1)
    yt = np.matmul(Y,alphaTemp.T)
    yt = yt.reshape(len(yt),1)
    TX = np.hstack((X,xt))
    TY = np.hstack((Y,yt))

    
    cca = CCA(n_components=1)
    cca.fit(X, TX)
    #print(cca.x_weights_)
    #print(cca.y_weights_)
    U0, V0 = cca.transform(X, TX)
    cor_1 = pr(U0.T[0],V0.T[0])[0] #
    a_ = cca.x_weights_
    b_ = cca.y_weights_
    
    T_a = np.dot(Y,a_)
    T_b = np.dot(TY,b_)
    
    cor_2 = pr(T_a.T[0],T_b.T[0])[0]
    return abs(cor_1 - cor_2)
    


corrAr = []
corrAr2 = []

date_index = pd.date_range('1/1/2018', periods=365, freq='D')
#AU_holidays = holidays.Australia()

indf = pd.read_csv('inLinksN.csv')

ErrorX = []
#mypath16 = "2016 Link data"
mypath17 = r"LC data\2018 Link data"
numLinks = len(indf)
i = 0
dataframes = ''
#filenames16 = [f16 for f16 in listdir(mypath16) if isfile(join(mypath16, f16))]
filenames17 = [f17 for f17 in listdir(mypath17) if isfile(join(mypath17, f17))]

timeSlice = len(filenames17)
#ErrorXAlllinks = np.zeros((numLinks,timeSlice))
#NoDataXAlllinks = np.zeros((numLinks,timeSlice),dtype='object')
ColTimes = []
for tiCount in range(1,timeSlice): ## Create time line cols
    M = (tiCount * 15 )%60
    H = (tiCount * 15) //60
    timeM = '%02d'%M
    timeH = '%02d'%H
    temp = str(timeH)+ ':' + str(timeM)
    ColTimes.append(temp)
    
errdfEq = pd.DataFrame(index=indf.out1.values,columns=ColTimes)
#errdfBi = pd.DataFrame(index=indf.out1.values,columns=ColTimes)
#nodatadf  = pd.DataFrame(index=indf.out1.values,columns=ColTimes)
#corrdf = pd.DataFrame(index=indf.out1.values,columns=ColTimes)

n = list(indf.out1.values)
c = list(np.arange(0,7)) 
mind = pd.MultiIndex.from_product([n, c],
                           names=['link', 'dow'])

errdfBi= pd.DataFrame(index=mind,columns=ColTimes)
errdfBir= pd.DataFrame(index=mind,columns=ColTimes)
nodatadf = pd.DataFrame(index=mind,columns=ColTimes)
corrdf = pd.DataFrame(index=mind,columns=ColTimes)
rawdf = pd.DataFrame()#columns=['Rmean','Pmean','Rstd','Pstd'])
AllRealX = []
AllPredX = []
real_pred_df = pd.DataFrame(columns=['Link','RealX','PredX','dow','tslice'])

for inxf, f in enumerate(filenames17[:-1]):
    print(f)

    nxt16 = filenames17[inxf+1]
    x1 = pd.read_csv(join(mypath17,f))
    y1 = pd.read_csv(join(mypath17,nxt16))

    TraindataX = x1#pd.concat([x1,x2])
    TraindataX.rename(columns={'Unnamed: 0':'Date'},inplace=True)
    TraindataX.index = pd.to_datetime(x1['Date'], format='%Y-%m-%d') #date_index
    
    TraindataY = y1 #pd.concat([y1,y2])
    TraindataY.rename(columns={'Unnamed: 0':'Date'},inplace=True)
    TraindataY.index = pd.to_datetime(x1['Date'], format='%Y-%m-%d') #date_index
    
    for dow in np.arange(0,2):
    ####### this loop is day of week ###     
        ## only select same days
        if dow == 1:
            dfTraindataX = TraindataX[TraindataX.index.dayofweek >= 5]
            dfTraindataY = TraindataY[TraindataY.index.dayofweek >= 5]
        else:
            dfTraindataX = TraindataX[TraindataX.index.dayofweek < 5]
            dfTraindataY = TraindataY[TraindataY.index.dayofweek < 5]
        ## only select same days
        numberOfData = []
        rmseX = []
        rmseY = []
        mapeX = []
        mapeY = []
        LinkName = []
        NodeID = []
        for numDataset in range(0,len(indf)):
            A = indf.in1[numDataset]
            B = indf.in2[numDataset]
            C = indf.in3[numDataset]
            O = indf.out1[numDataset]
            dfXW = pd.DataFrame([dfTraindataX[A],dfTraindataX[B],dfTraindataX[C],dfTraindataX[O]]).transpose()
            dfYW = pd.DataFrame([dfTraindataY[A],dfTraindataY[B],dfTraindataY[C],dfTraindataY[O]]).transpose()
        
            
            dfXW = dfXW.loc[:, (dfXW != 0).any(axis=0)] ### delete all zero columns
            dfYW = dfYW.loc[:, (dfYW != 0).any(axis=0)] ### delete all zero columns
            if len(dfXW.columns) == 4 and len(dfYW.columns) == 4:
                
                dfXW = dfXW[(dfXW != 0).all(1)] ## Ignore zero values in any rows
                dfYW = dfYW[(dfYW != 0).all(1)] ## Ignore zero values in any rows
                
                if (len(dfXW.columns) == 4 and len(dfYW.columns) == 4):
                    
                    commonIndex = dfXW.index.intersection(dfYW.index)
                    dfYW = dfYW.loc[commonIndex]
                    dfXW = dfXW.loc[commonIndex]
                    
                    if (len(dfXW.index) > 2 and len(dfYW.index) > 2 and len(dfXW) > 20 and len(dfXW) > 20):
                        Y4 = np.array((dfYW[A],dfYW[B],dfYW[C],dfYW[O]))
                        X4 = np.array((dfXW[A],dfXW[B],dfXW[C],dfXW[O]))

                        X = X4.T[:,0:3]  ## for 5 days, later have to do for weekdays only
                        Y = Y4.T[:,0:3]

                        RealX = X4.T[:,-1]
                        RealY = Y4.T[:,-1]

                        
                        alpha0 = np.array(IndtialAlpha(indf.x[numDataset],indf.y[numDataset],indf.z[numDataset],indf.k[numDataset]))
                        #print(O)

                        
                        cca = CCA(n_components=1)
                        cca.fit(X, Y)
                        CCA(copy=True, max_iter=5000, n_components=1, scale=True, tol=1e-10)
                        a_p, b_p = cca.transform(X, Y)
                        T_a = np.dot(a_p.T,X)
                        T_b = np.dot(b_p.T,Y)
                        #print(T_a[0])
                        #print(a_p)
                        TargetCorr = pr(T_a[0],T_b[0])[0]
                        #TargetCorr = pr(a_p.T[0],b_p.T[0])[0]
                        #print(TargetCorr)
                        #curCorr = pr(RealX,RealY)[0]
                        #corrdf.loc[corrdf.index == O,ColTimes[inxf]] = curCorr
                        
                        #print(methodx)
                        #methody = "Predicted data without training at 8:15 AM at Link ID: " + str(O)
                        results = CalculateCount(X,Y,alpha0,RealX)
                        #resEq = CalculateCount(X,Y,np.array([.5,.5,.5]))
                        XYCorr = pr(RealX,RealY)[0]
                        mpe = results[-1]
                        rmse = results[-2]
                        #print(mpe)
                        #mpeEq = resEq[-1]
                        predX = results[0]
                        predY = results[1]
                        index_holidays = dfXW.loc[dfXW.index.isin(AU_holidays)]
                        holi = np.asarray(index_holidays)
                        methodx = "Prediction in the year 2016 - Link id: " + str(O) + " at: " + str(f[14:16]) + ':' + str(f[16:18])
                        #if O == '2903-2902' and f =='LinkTrainData_0800_2017.csv': #  LinkTrainData_0800_2017.csv
                        #if mpe < 10:
                            
                            #RPlot(RealX,predX,methodx,dfXW,mpe)
                            #RPlot(RealY,predY,methody,dfXW)


                        
                        if ((mpe < 100) and (rmse < 100)):
                        #if (curCorr > 0.0):
                            #real_pred_df = real_pred_df.append({'RealX':RealX,'PredX':predX,'Link':O,'dow':dow,'tslice': str(f[14:16]) + ':' + str(f[16:18]),'CorrCoeff':XYCorr},ignore_index=True)
                            real_pred_df = real_pred_df.append({'RealX':RealX,'PredX':predX,'Link':O,'dow':dow,'tslice': str(f[14:16]) + ':' + str(f[16:18]),'CorrCoeff':XYCorr},ignore_index=True)
                        
                            nOd = np.shape(X)[0]
                            xi = (O,dow)
                            errdfBi.iloc[errdfBi.index.tolist().index(xi)][ColTimes[inxf]] = mpe
                            errdfBir.iloc[errdfBir.index.tolist().index(xi)][ColTimes[inxf]] = rmse#mpe
                            #errdfEq.loc[errdfEq.index == O,ColTimes[inxf]] = mpeEq
                                #ErrorXAlllinks[len(NodeID),inxf] = Stats(predX,RealX)[1]
                                #NoDataXAlllinks[len(NodeID),inxf] = nOd
                            if dow == 0:
                                nodatadf.iloc[nodatadf.index.tolist().index(xi)][ColTimes[inxf]] =  nOd/(52*5)
                            else:
                                nodatadf.iloc[nodatadf.index.tolist().index(xi)][ColTimes[inxf]] =  nOd/(52*2)
                            rawdf = rawdf.append({'Rmean':np.mean(RealX),'Pmean':np.mean(predX),
                                                  'Rstd':np.std(RealX),'Pstd':np.std(predX),'Link':O,'dow':dow},ignore_index=True)

                            #NodeID.append(O[0:4])
                            #AllRealX = AllRealX + RealX.tolist()
                            #AllPredX = AllPredX + predX.tolist()
                            curCorr = pr(RealX,RealY)[0]
                            corrdf.iloc[corrdf.index.tolist().index(xi)][ColTimes[inxf]] = TargetCorr#curCorr
        
if 1 == 1:
    erdfBi = errdfBi[(errdfBi.isnull() == False).all(1)]
    erdfBir = errdfBir[(errdfBir.isnull() == False).all(1)]
    
    #comIndx = errdfBi.index.intersection(crdf.index) 
    crdf = corrdf.fillna(0).drop_duplicates()
    dcidf = nodatadf.fillna(0).drop_duplicates()
    
    erdfBir.to_csv('rmseBi18y.csv')
    erdfBi.to_csv('erdfBi18y.csv')
    crdf.to_csv('crdf18y.csv')
    rawdf.to_csv('rawdata18y.csv')
##    #erdf.dropna(axis=1)
    dcidf.to_csv('noData18y.csv')
##    
    rawdf.plot.scatter(x='Rmean',y='Pmean')
    plt.show()
##    plt.scatter(AllRealX,AllPredX)
##    plt.show()
##    rvp = pd.DataFrame({'Real':AllRealX,'Pred':AllPredX})
##    rvp.to_csv('RvP16d.csv')
real_pred_df.explode(['RealX','PredX']).to_csv('real_pred_df18_brp.csv')
