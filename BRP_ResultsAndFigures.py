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
from seriate import seriate
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
hours_str = ['00:00','01:00', '02:00', '03:00', '04:00', '05:00','06:00', '07:00', '08:00', '09:00','10:00', '11:00','12:00', '13:00', '14:00', '15:00','16:00',
            '17:00', '18:00', '19:00','20:00',  '21:00', '22:00', '23:00', '24:00']

def error(yS,xR):
    rms = sqrt(mean_squared_error(yS, xR))
    mape = np.mean(np.abs((xR - yS) /xR)) * 100
    return rms,mape

def mapePlot_MRP(df):
#if 1 == 1:
    #plt.rcParams.update({'font.size': 20})
    name =[x for x in globals() if globals()[x] is df][0]
    yr = str(20) + name[-2] + name[-1]
    if name[1] == 'm':
        errtype = "MAPE"
    else:
        errtype = "RMSE"
    plt.rcParams.update({'font.size': 18})
    edata = df.drop(['link','dow'],axis=1)
    te = edata.astype(float)
    #ie = te.mean(axis=1).values.argsort()
    #te = te.iloc[ie]

    grid_kws = {"height_ratios": (.9, .01), "hspace": .04}

    #f, (ax, cbar_ax) = plt.subplots(2,gridspec_kw=grid_kws)

    #ax = sns.heatmap(te, cmap="RdYlGn_r", ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})  # cmap="YlGnBu",PiYG_r,Reds
    #ax.set_yticks(ccrdf.index.values)
    #fig,ax = plt.figure()
    ax = sns.heatmap(te, cmap="RdYlGn_r", vmin=0, vmax=100)
    #numy = np.arange(0,len(edata.index.values))
    idxs = df.link.to_list()
    numy = np.unique(idxs,return_index=True)[1]#.tolist()
    numy = numy + .5
    ax.set_yticks(numy)
    ylabels = np.unique(idxs,return_index=True)[0].tolist() #df.apply(lambda x: str(x.link)+' '+str(x.dow),axis=1).to_list()
    ax.set_yticklabels(ylabels,rotation=0)
    
    ax.set_ylabel("Link id")
    ax.set_title(str(errtype) + " heatmap for every time slices in year " + str(yr))
    #ax.set_xlabel("Time of a day")
    ax2 = ax.twinx()
    numy2 = np.arange(0,len(df.dow.to_list())) + 0.5
    
    ax2.set_ylabel('Weekend indicator', color='blue')
    ax2.set_yticks(numy2)
    ax2.set_yticklabels(df.dow.to_list())
    for t in ax2.get_yticklabels():
        txt = t.get_text()
        #print(t.get_text())
        if int(txt) == 0:
            t.set_color('black')
        else:
            t.set_color('blue')
            
    list_ylab = [i if i ==1 else ' ' for i in df.dow.to_list()]
    ax2.set_yticklabels(list_ylab)        
    plt.setp(ax2.get_yticklabels(), rotation=90, fontsize='small')
    plt.tight_layout()
    plt.show()
    
##    ax2.set_ylabel('Day of a week', color='blue')
##    ax2.set_yticks(numy2)
##    ax2.set_yticklabels(df.dow.to_list())
##    plt.setp(ax2.get_yticklabels(), rotation=90, fontsize='small')
##    plt.show()

def rmsePlot(rmsedf16):
    rdata = rmsedf16.drop(['link'],axis=1)
    te = rdata.astype(float)
    #ie = te.mean(axis=1).values.argsort()
    #te = te.iloc[ie]

    grid_kws = {"height_ratios": (.9, .02), "hspace": .4}

    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)

    ax = sns.heatmap(te, cmap="RdYlGn", ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})  # cmap="YlGnBu",PiYG_r,Reds
    #ax.set_yticks(ccrdf.index.values)
    numy = np.arange(0,len(rdata.index.values))
    numy = numy + .5
    ax.set_yticks(numy)
    idxs = rmsedf16.link.to_list()#df.apply(lambda x: str(x.link)+' '+str(x.dow),axis=1).to_list()
    ax.set_yticklabels(idxs)
    #ax.set_title("RMSE error heatmap for each outgoing link for every time slices in year 2019")
    ax.set_xlabel("Time of a day")
    plt.show()
    return te

def Geh_Heatmap(df):
    hours_str = ['00:00','01:00', '02:00', '03:00', '04:00', '05:00','06:00', '07:00', '08:00', '09:00','10:00', '11:00','12:00', '13:00', '14:00', '15:00','16:00',
            '17:00', '18:00', '19:00','20:00',  '21:00', '22:00', '23:00', '24:00']
    plt.rcParams.update({'font.size': 20})
    name =[x for x in globals() if globals()[x] is df][0]
    yr = str(20) + name[-2] + name[-1]
    #if name[0] == 'm':
    #    errtype = "MAPE"
    #else:
    errtype = "GEH"
    plt.rcParams.update({'font.size': 20})
    edata = df.drop(['Link','dow'],axis=1)
    te = edata.astype(float)
    #ie = te.mean(axis=1).values.argsort()
    #te = te.iloc[ie]

    grid_kws = {"height_ratios": (.9, .01), "hspace": .4}

    #f, (ax, cbar_ax) = plt.subplots(2,gridspec_kw=grid_kws)

    #ax = sns.heatmap(te, cmap="RdYlGn_r", ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})  # cmap="YlGnBu",PiYG_r,Reds
    #ax.set_yticks(ccrdf.index.values)
    #fig,ax = plt.figure()
    ax = sns.heatmap(te, cmap="RdYlGn_r", vmin=0, vmax=15)
    #numy = np.arange(0,len(edata.index.values))
    ax.set_xticks(np.arange(0,25))
    ax.set_xticklabels(hours_str)
    ax.set_xlabel('Hours of a day (HH:MM)')
    idxs = df.Link.to_list()
    numy = np.unique(idxs,return_index=True)[1]#.tolist()
    numy = numy + .5
    ax.set_yticks(numy)
    ylabels = np.unique(idxs,return_index=True)[0].tolist() #df.apply(lambda x: str(x.link)+' '+str(x.dow),axis=1).to_list()
    ax.set_yticklabels(ylabels,rotation=0)
    
    ax.set_ylabel("Link id")
    ax.set_title(str(errtype) + " heatmap for hourly estimation in year " + str(yr))
    #ax.set_xlabel("Time of a day")
    ax2 = ax.twinx()
    numy2 = np.arange(0,len(df.dow.to_list())) + 0.5
    
    ax2.set_ylabel('Weekend indicator', color='blue')
    ax2.set_yticks(numy2)
    

    ax2.set_yticklabels(df.dow.to_list())
    for t in ax2.get_yticklabels():
        txt = t.get_text()
        #print(t.get_text())
        if int(txt) == 0:
            t.set_color('black')
            #t.set_ylticklabels([])
        else:
            t.set_color('blue')

    list_ylab = [i if i ==1 else ' ' for i in df.dow.to_list()]
    ax2.set_yticklabels(list_ylab)
    plt.setp(ax2.get_yticklabels(), rotation=90, fontsize='small')
    plt.tight_layout()
    plt.show()

AU_holidays = holidays.Australia()
plt.rcParams.update({'font.size': 16})
def GEH(M,C):
    return np.sqrt(2 * (M-C)**2/(M+C))




#all_geh = pd.concat([geh_mrp_all,geh_brp_all],axis=1)
#print(all_geh.round(2).dropna('-').to_latex())
#print(all_geh.round(2).fillna('-').to_latex())

if True:
    bmpe19 = pd.read_csv('erdfBi19y.csv')
    bmpe18 = pd.read_csv('erdfBi18y.csv')
    bmpe17 = pd.read_csv('erdfBi17y.csv')
    bmpe16 = pd.read_csv('erdfBi16y.csv')

    brmse19 = pd.read_csv('rmseBi19y.csv')
    brmse18 = pd.read_csv('rmseBi18y.csv')
    brmse17 = pd.read_csv('rmseBi17y.csv')
    brmse16 = pd.read_csv('rmseBi16y.csv')




    bmpeL = [bmpe19, bmpe18, bmpe17, bmpe16]
    brmseL = [brmse19, brmse18, brmse17, brmse16]

    #bgehL = [] #geh_df16m_avg.reset_index().groupby(['Link']).agg(lambda x: x.unique().mean()).drop('dow',axis=1).mean(axis=1)
#uLinks = list(mpe19['link'].unique()) + list(mpe18['link'].unique()) + list(mpe17['link'].unique()) + list(mpe16['link'].unique())
#uLinks = list(np.unique(uLinks))

#create brp plots
#erdfBi = bmpe19.groupby(['link']).agg(lambda x: x.unique().mean()) ### Very important   
#rmsedf = rmsedf16.groupby(['link']).agg(lambda x: x.unique().mean()) ### Very important
#if True:
#    linkmstd = pd.DataFrame()
#    linkmstd = bmpe19.groupby(['link']).agg(lambda x: x.unique().mean()).drop('dow',axis=1).mean(axis=1)
    
if True:
    plt.rcParams.update({'font.size': 20})
    Links = bmpe19[['link','dow']].values.tolist() + bmpe18[['link','dow']].values.tolist() + bmpe17[['link','dow']].values.tolist() + bmpe16[['link','dow']].values.tolist() 
    uLinks = [list(x) for x in set(tuple(x) for x in Links)] #list(k for k,_ in itertools.groupby(Links))
    linkmstd = pd.DataFrame()
    posxticks =  np.arange(0,len(uLinks))
    xticks = []

    #axm = plt.subplot()
    factor = 0.2

    yearIndex = [2016,2017,2018,2019]
    for xxr,link in enumerate(uLinks):
        xticks.append(str(link[0]) + ' ' + str(link[1]))
        for i,tempdf in enumerate(bmpeL):
            year = 2019 - i
            tempdfM = bmpeL[i]
            tempdfR = brmseL[i]
            tempM = tempdfM[(tempdfM.link == link[0]) & (tempdfM.dow == link[1])].drop(['link','dow'],axis=1)
            tempR = tempdfR[(tempdfR.link == link[0]) & (tempdfR.dow == link[1])].drop(['link','dow'],axis=1)
            if len(tempM) > 0 and len(tempR) > 0:
                linkmstd = linkmstd.append({'link':link[0],'dow':int(link[1]),'year':int(year),'meanMAPE':tempM.mean(axis=1).values[0],'stdMAPE':tempM.std(axis=1).values[0],
                             'meanRMSE':tempR.mean(axis=1).values[0],'stdRMSE':tempR.std(axis=1).values[0]},ignore_index=True)
                lab = 'MAPE for ' + str(year)#2016'

                         
    linkmstd['dow'] =linkmstd.dow.astype(int)
    linkmstd['year'] =linkmstd.year.astype(int)
    linkmstdfb = linkmstd.set_index(['link','dow','year']).round(2)
    linkmstdfb.to_csv('linkmstdf_BRP.csv')

if True:
    plt.rcParams.update({'font.size': 22})
    clr = ['k','b','g','r']
    mkr = ['s','^','o','*']
    ####### Plot error bar RMSE ######
    axr = plt.subplot()
    factor = 0.2

    #ldf = linkmstd.set_index(['link','dow']).round(2) ## changed on 2023
    ldf = linkmstd.set_index(['link','year']).round(2).groupby(['link','year']).mean()
    
    ls = ldf.sort_values(by='meanRMSE',ascending=True)
    ldfs = ldf.reset_index().set_index('link')
    L = linkmstd.groupby('link').mean().sort_values(by='meanRMSE',ascending=True).index.tolist() ####

    xxr = np.arange(0,len(L))
    for idx,item in enumerate(L):
        try:
            tdf = ldfs.loc[item].sort_values(by='meanRMSE',ascending=True)#.sort_index()
            up =  -len(tdf)/10
            low =  len(tdf)/10
            xvalue = idx + np.arange(up,low,factor)
            for i in range(len(tdf)):
                
                axr.errorbar(xvalue[i],tdf.meanRMSE.values[i],tdf.stdRMSE.values[i],capsize=2,linestyle='None',lw=2,
                             c=clr[yearIndex.index(tdf.year.values[i])],marker=mkr[yearIndex.index(tdf.year.values[i])])
        except:
            tdf = ldfs.loc[item]
            #up =  -len(tdf)/20
            #low =  len(tdf)/20
            axr.errorbar(idx,tdf.meanRMSE,tdf.stdRMSE,capsize=2,linestyle='None',lw=2,
                             c=clr[yearIndex.index(int(tdf.year))],marker=mkr[yearIndex.index(int(tdf.year))])

            
    plt.ylabel("Mean and standard deviation of RMSE")
    #plt.legend(loc='upper left')
    plt.xticks(xxr,L,rotation=90)
    plt.xlabel('Link id') 
    plt.ylim(0,100)
    plt.grid(True)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='k', lw=2,marker='s'),
                    Line2D([0], [0], color='b', lw=2,marker='^'),
                    Line2D([0], [0], color='g', lw=2,marker='o'),
                    Line2D([0], [0], color='r', lw=2,marker='*')]
    axr.legend(custom_lines, ['2016', '2017', '2018','2019'])
    plt.show()

if True:
    plt.rcParams.update({'font.size': 22})
    clr = ['k','b','g','r']
    mkr = ['s','^','o','*']
    ####### Plot error bar RMSE ######
    axr = plt.subplot()
    factor = 0.2

    #ldf = linkmstd.set_index(['link','dow']).round(2) ## changed on 2023
    ldf = linkmstd.set_index(['link','year']).round(2).groupby(['link','year']).mean()
    
    ls = ldf.sort_values(by='meanMAPE',ascending=True)
    ldfs = ldf.reset_index().set_index('link')
    L = linkmstd.groupby('link').mean().sort_values(by='meanMAPE',ascending=True).index.tolist() ####

    xxr = np.arange(0,len(L))
    for idx,item in enumerate(L):
        try:
            tdf = ldfs.loc[item].sort_values(by='meanMAPE',ascending=True)#.sort_index()
            up =  -len(tdf)/10
            low =  len(tdf)/10
            xvalue = idx + np.arange(up,low,factor)
            for i in range(len(tdf)):
                
                axr.errorbar(xvalue[i],tdf.meanMAPE.values[i],tdf.stdMAPE.values[i],capsize=2,linestyle='None',lw=2,
                             c=clr[yearIndex.index(tdf.year.values[i])],marker=mkr[yearIndex.index(tdf.year.values[i])])
        except:
            tdf = ldfs.loc[item]
            #up =  -len(tdf)/20
            #low =  len(tdf)/20
            axr.errorbar(idx,tdf.meanMAPE,tdf.stdMAPE,capsize=2,linestyle='None',lw=2,
                             c=clr[yearIndex.index(int(tdf.year))],marker=mkr[yearIndex.index(int(tdf.year))])

            
    plt.ylabel("Mean and standard deviation of MAPE")
    #plt.legend(loc='upper left')
    plt.xticks(xxr,L,rotation=90)
    plt.xlabel('Link id') 
    plt.ylim(0,100)
    plt.grid(True)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='k', lw=2,marker='s'),
                    Line2D([0], [0], color='b', lw=2,marker='^'),
                    Line2D([0], [0], color='g', lw=2,marker='o'),
                    Line2D([0], [0], color='r', lw=2,marker='*')]
    axr.legend(custom_lines, ['2016', '2017', '2018','2019'])
    plt.show()


    ## Data extraction for BRP model

if True:
    brp_16m = pd.read_csv('real_pred_df16_brp.csv',index_col=0)
    brp_17m = pd.read_csv('real_pred_df17_brp.csv',index_col=0)
    brp_18m = pd.read_csv('real_pred_df18_brp.csv',index_col=0)
    brp_19m = pd.read_csv('real_pred_df19_brp.csv',index_col=0)

    brp_16m['hod'] = brp_16m.tslice.apply(lambda x:int(x[0:2]))
    brp_17m['hod'] = brp_17m.tslice.apply(lambda x:int(x[0:2]))
    brp_18m['hod'] = brp_18m.tslice.apply(lambda x:int(x[0:2]))
    brp_19m['hod'] = brp_19m.tslice.apply(lambda x:int(x[0:2]))

if True:
    b = brp_16m.groupby(['Link','dow','tslice']).mean()
    c = b.droplevel('tslice').copy()
    d = [j.reset_index().groupby(['Link','dow','hod']).sum() for i,j in c.groupby('hod')]
    geh_df16m = pd.concat(d)
    #geh_df16m['GEH'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
    #p = geh_df16m.groupby(level=[0,2]).mean()
    geh_df16m_avg = geh_df16m.groupby(level=[0, 1]).mean()
    geh_df16m_avg['GEH'] = GEH(geh_df16m_avg.RealX,geh_df16m_avg.PredX)

    brp16 = geh_df16m_avg.groupby(level=[0]).mean()
    brp16['GEH16'] = GEH(brp16.RealX,brp16.PredX)

#if True:
    b = brp_17m.groupby(['Link','dow','tslice']).mean()
    c = b.droplevel('tslice').copy()
    d = [j.reset_index().groupby(['Link','dow','hod']).sum() for i,j in c.groupby('hod')]
    geh_df17m = pd.concat(d)
    #geh_df16m['GEH'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
    #p = geh_df16m.groupby(level=[0,2]).mean()
    geh_df17m_avg = geh_df17m.groupby(level=[0, 1]).mean()
    geh_df17m_avg['GEH'] = GEH(geh_df17m_avg.RealX,geh_df17m_avg.PredX)

    brp17 = geh_df17m_avg.groupby(level=[0]).mean()
    brp17['GEH17'] = GEH(brp17.RealX,brp17.PredX)

#if True:
    b = brp_18m.groupby(['Link','dow','tslice']).mean()
    c = b.droplevel('tslice').copy()
    d = [j.reset_index().groupby(['Link','dow','hod']).sum() for i,j in c.groupby('hod')]
    geh_df18m = pd.concat(d)
    #geh_df16m['GEH'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
    #p = geh_df16m.groupby(level=[0,2]).mean()
    geh_df18m_avg = geh_df18m.groupby(level=[0, 1]).mean()
    geh_df18m_avg['GEH'] = GEH(geh_df18m_avg.RealX,geh_df18m_avg.PredX)

    brp18 = geh_df18m_avg.groupby(level=[0]).mean()
    brp18['GEH18'] = GEH(brp18.RealX,brp18.PredX)

#if True:
    b = brp_19m.groupby(['Link','dow','tslice']).mean()
    c = b.droplevel('tslice').copy()
    d = [j.reset_index().groupby(['Link','dow','hod']).sum() for i,j in c.groupby('hod')]
    geh_df19m = pd.concat(d)
    #geh_df16m['GEH'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
    #p = geh_df16m.groupby(level=[0,2]).mean()
    geh_df19m_avg = geh_df19m.groupby(level=[0, 1]).mean()
    geh_df19m_avg['GEH'] = GEH(geh_df19m_avg.RealX,geh_df19m_avg.PredX)

    brp19 = geh_df19m_avg.groupby(level=[0]).mean()
    brp19['GEH19'] = GEH(brp19.RealX,brp19.PredX)
    
    geh_brp_all = pd.concat([brp16['GEH16'],brp17['GEH17'],brp18['GEH18'],brp19['GEH19']],axis='columns')

print(geh_brp_all.round(2).fillna('-').to_latex())

### plot GEH stats 2016 heatmap
if True:
    _com = geh_brp_all[geh_brp_all.index.isin(linkmstd['link'].unique())]
    b_geh16 = geh_df16m.copy()
    b_geh16['GEH16'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
    b_geh16 = b_geh16.reset_index().drop(['RealX', 'PredX', 'CorrCoeff' ],axis=1)
    b_geh16c = b_geh16[b_geh16.Link.isin(linkmstd['link'].unique())].copy()
    b_geh16_pivot = b_geh16c.pivot(index=['Link','dow'], columns='hod', values='GEH16').dropna()
    bgeh16 = b_geh16_pivot.reset_index()
    


Geh_Heatmap(bgeh16)
mapePlot_MRP(bmpe16)
mapePlot_MRP(brmse16)
### 4 plots * 2 MAPE RMSE 4 years heatmap
##if True:
##    for itemdf in bmpeL+brmseL:
##        itemdf.rename(columns={'Unnamed: 0':'link'},inplace=True)
##        mapePlot_MRP(itemdf)

if True:
    
    
    temp_b16 = pd.DataFrame()
    temp_b16['meanGEH'] = bgeh16.groupby(['Link']).mean().drop('dow',axis=1).mean(axis=1)
    temp_b16['stdGEH'] = bgeh16.groupby(['Link']).mean().drop('dow',axis=1).std(axis=1)
    temp_b16['year'] = [2016] * len(temp_b16)
    temp_b16.reset_index(inplace=True)
    
   
    b_geh17 = geh_df17m.copy()
    b_geh17['GEH17'] = GEH(geh_df17m.RealX,geh_df17m.PredX)
    b_geh17 = b_geh17.reset_index().drop(['RealX', 'PredX', 'CorrCoeff' ],axis=1)
    b_geh17c = b_geh17[b_geh17.Link.isin(linkmstd['link'].unique())].copy()
    b_geh17_pivot = b_geh17c.pivot(index=['Link','dow'], columns='hod', values='GEH17').dropna()
    bgeh17 = b_geh17_pivot.reset_index()

    temp_b17 = pd.DataFrame()
    temp_b17['meanGEH'] = bgeh17.groupby(['Link']).mean().drop('dow',axis=1).mean(axis=1)
    temp_b17['stdGEH'] = bgeh17.groupby(['Link']).mean().drop('dow',axis=1).std(axis=1)
    temp_b17['year'] = [2017] * len(temp_b17)
    temp_b17.reset_index(inplace=True)

   
    b_geh18 = geh_df18m.copy()
    b_geh18['GEH18'] = GEH(geh_df18m.RealX,geh_df18m.PredX)
    b_geh18 = b_geh18.reset_index().drop(['RealX', 'PredX', 'CorrCoeff' ],axis=1)
    b_geh18c = b_geh18[b_geh18.Link.isin(linkmstd['link'].unique())].copy()
    b_geh18_pivot = b_geh18c.pivot(index=['Link','dow'], columns='hod', values='GEH18').dropna()
    bgeh18 = b_geh18_pivot.reset_index()

    temp_b18 = pd.DataFrame()
    temp_b18['meanGEH'] = bgeh18.groupby(['Link']).mean().drop('dow',axis=1).mean(axis=1)
    temp_b18['stdGEH'] = bgeh18.groupby(['Link']).mean().drop('dow',axis=1).std(axis=1)
    temp_b18['year'] = [2018] * len(temp_b18)
    temp_b18.reset_index(inplace=True)
    
   
    b_geh19 = geh_df19m.copy()
    b_geh19['GEH19'] = GEH(geh_df19m.RealX,geh_df19m.PredX)
    b_geh19 = b_geh19.reset_index().drop(['RealX', 'PredX', 'CorrCoeff' ],axis=1)
    b_geh19c = b_geh19[b_geh19.Link.isin(linkmstd['link'].unique())].copy()
    b_geh19_pivot = b_geh19c.pivot(index=['Link','dow'], columns='hod', values='GEH19').dropna()
    bgeh19 = b_geh19_pivot.reset_index()

    temp_b19 = pd.DataFrame()
    temp_b19['meanGEH'] = bgeh19.groupby(['Link']).mean().drop('dow',axis=1).mean(axis=1)
    temp_b19['stdGEH'] = bgeh19.groupby(['Link']).mean().drop('dow',axis=1).std(axis=1)
    temp_b19['year'] = [2019] * len(temp_b19)
    temp_b19.reset_index(inplace=True)

if True:
    b_ly_mstd_geh = pd.concat([temp_b16,temp_b17,temp_b18,temp_b19])
    plt.rcParams.update({'font.size': 22})
    clr = ['k','b','g','r']
    mkr = ['s','^','o','*']
    ####### Plot error bar RMSE ######
    axr = plt.subplot()
    factor = 0.2

    #ldf = b_ly_mstd_geh.set_index(['Link','dow']).round(2) ## changed on 2023
    ldf = b_ly_mstd_geh.set_index(['Link','year']).round(2).groupby(['Link','year']).mean()
    
    ls = ldf.sort_values(by='meanGEH',ascending=True)
    ldfs = ldf.reset_index().set_index('Link')
    L = b_ly_mstd_geh.groupby('Link').mean().sort_values(by='meanGEH',ascending=True).index.tolist() ####

    xxr = np.arange(0,len(L))
    for idx,item in enumerate(L):
        try:
            tdf = ldfs.loc[item].sort_values(by='meanGEH',ascending=True)#.sort_index()
            up =  -len(tdf)/10
            low =  len(tdf)/10
            xvalue = idx + np.arange(up,low,factor)
            for i in range(len(tdf)):
                
                axr.errorbar(xvalue[i],tdf.meanGEH.values[i],tdf.stdGEH.values[i],capsize=2,linestyle='None',lw=2,
                             c=clr[yearIndex.index(tdf.year.values[i])],marker=mkr[yearIndex.index(tdf.year.values[i])])
        except:
            tdf = ldfs.loc[item]
            #up =  -len(tdf)/20
            #low =  len(tdf)/20
            axr.errorbar(idx,tdf.meanGEH,tdf.stdGEH,capsize=2,linestyle='None',lw=2,
                             c=clr[yearIndex.index(int(tdf.year))],marker=mkr[yearIndex.index(int(tdf.year))])

            
    plt.ylabel("Mean and standard deviation of GEH")
    #plt.legend(loc='upper left')
    plt.xticks(xxr,L,rotation=90)
    plt.xlabel('Link id') 
    plt.ylim(0,15)
    plt.grid(True)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='k', lw=2,marker='s'),
                    Line2D([0], [0], color='b', lw=2,marker='^'),
                    Line2D([0], [0], color='g', lw=2,marker='o'),
                    Line2D([0], [0], color='r', lw=2,marker='*')]
    axr.legend(custom_lines, ['2016', '2017', '2018','2019'])
    plt.show()

   

### read raw data
if 1 == 1:
    plt.rcParams.update({'font.size': 22})
    r16 = pd.read_csv('rawdata16y.csv')
    r17 = pd.read_csv('rawdata17y.csv')
    r18 = pd.read_csv('rawdata18y.csv')
    r19 = pd.read_csv('rawdata19y.csv')
    raw = pd.concat([r16,r17,r18,r19],ignore_index=True)
    rdf= raw.drop('Unnamed: 0',axis=1).sort_values(['Pmean'])#.drop(['link','dow'],axis=1)
    axr = plt.subplot()
    xxr =  np.arange(0,len(rdf))
    rdf.drop_duplicates(inplace=True)
    
    axr.errorbar(xxr,rdf.Rmean.values,rdf.Rstd.values,capsize=0,linestyle='None',c='b',marker=' ',label='Observed sample mean and standard deviation')
    axr.errorbar(xxr,rdf.Pmean.values,rdf.Pstd.values,capsize=0,linestyle='None',c='r',marker='o',alpha=0.3,label='Estimated sample mean and standard deviation')
    #axr.plot(xxr,rdf.Pmean.values,linestyle='None',c='r',marker='o',alpha=0.4,label='Expection of the estimated distribution')
    
        #axr.errorbar(xxr + factor,erdf18.mean(axis=1)[ixr],erdf18.std(axis=1)[ixr],capsize=5,linestyle='None',c='y',marker='^',label='MAPE for 2018')

    #plt.title("Mean and standard deviation of MAPE error for each link for three different calender year")
    plt.ylabel("Link counts")
    plt.legend(loc='upper left')
    #plt.xticks(xxr,erdf16.index.values[ixr],rotation=90)
    plt.xlabel('Random variables') 
    plt.ylim(0,500)
    #plt.grid(True)
    plt.show()


if True:
    cr16 = pd.read_csv('crdf16y.csv')
    cr17 = pd.read_csv('crdf17y.csv')
    cr18 = pd.read_csv('crdf18y.csv')
    cr19 = pd.read_csv('crdf19y.csv')

    dci16 = pd.read_csv('noData16y.csv')
    dci17 = pd.read_csv('noData17y.csv')
    dci18 = pd.read_csv('noData18y.csv')
    dci19 = pd.read_csv('noData19y.csv')
    
    crL = [cr19, cr18,cr17,cr16 ]
    cr = cr16.set_index(['link','dow'])
    d = dci16.set_index(['link','dow'])#.rename(columns={'link':'Link'})
    b = bmpe16.set_index(['link','dow'])
    d16 = d[(d != 0).all(1)]
    c16 = cr[(cr != 0).all(1)]
    plt.scatter(d16.values.flatten(),b.values.flatten())
    plt.show()
if True:
    # 4581-4580 2922-2923
    #good_links GEH16 < 5
    #bad_links GEH16 > 5
    a16 = brp16.reset_index()
    a16 = a16[a16.Link.isin(L)]
    
    gL =  a16[a16.GEH16 < 5].Link.to_list()
    #gL = ['4581-4580', '2922-2923','2924-2925','2919-2920','2903-2902','2919-2911']
    bL = a16[a16.GEH16 > 5].Link.to_list()

    
    dl16 = d16.reset_index(level=0)
    gl_16 = dl16[dl16.link.isin(gL)]
    
    bl_16 = dl16[dl16.link.isin(bL)]

if True:
    plt.rcParams.update({'font.size': 25})
    
    gl_bl_df = pd.DataFrame(bl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1).mean(axis=0),columns=['Type2_links_BRP'])
    gl_bl_df['Type1_links_BRP'] = gl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1).mean(axis=0)
    #gl_bl_df
    gl_bl_df1 = pd.read_csv('gl_bl_df_MRP.csv',index_col=0)
    
    _df = pd.concat([gl_bl_df,gl_bl_df1],axis=1)
    _df.rename(columns = {'Type2_links_BRP': 'Type2 links of BRP model', 'Type1_links_BRP': 'Type1 links of BRP model',
                          'Type2_links_MRP': 'Type2 links of MRP model', 'Type1_links_MRP': 'Type1 links of MRP model',}, inplace = True)
    ax = _df.plot(ylabel='DCI',style=['r-.o','g-.o','r--s','g--s'])#,xticks=gl_bl_df.index) # xlabel='Time horizon (HH:MM)',
    
    ax.set_xticks(np.arange(_df.shape[0]))
    al_ticks = gl_bl_df.index.to_list()
    #ax.set_xticklabels(gl_bl_df.index.to_list())
    #plt.xticks(np.arange(gl_bl_df.shape[0]),gl_bl_df.index.to_list(),rotation=90)
    #ax.set_xticks(gl_bl_df.index)
    ax.set_xticks(list(np.arange(_df.shape[0]))[0::2])
    ax.set_xticklabels(al_ticks[0::2],rotation=90)
    ax.set_xlabel('Hours of a day (HH:MM)')
    plt.grid(True)
    
    #plt.plot(bl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1).mean(axis=0),c='r')
    #plt.plot(gl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1).mean(axis=0),c='g')
    plt.show()
    
    gl_16.set_index('link').T.plot()
    plt.show()
    bl_16.set_index('link').T.plot()
    plt.show()
    # 2904-2905 4405-4404
    #
if True:
    bldf = bl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1)
    gldf = gl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1)

if True:
    g16_brp = brp_16m.groupby(['Link','dow','tslice'])
    lol_brp = [[t[0][0],t[0][1],str(t[0][2]),error(t[1].RealX,t[1].PredX)[0],error(t[1].RealX,t[1].PredX)[1]] for t in g16_brp ]
    g16df_brp = pd.DataFrame(lol_brp,columns=['Link','dow','tslice','rmse','mape'])
    #g16df_brp_ = g16df_brp[(g16df_brp.rmse < 100) & (g16df_brp.mape < 100)].copy()
    g16df_brp_ = g16df_brp.copy()#g16df_brp[(g16df_brp.mape < 100)].copy()
    g16df_brp_mape = g16df_brp_.pivot(index=['Link','dow'], columns='tslice', values='mape').dropna()
    g16df_brp_mape_ = g16df_brp_mape.reset_index().rename(columns={'Link':'link'})

##    g16_mrp = rp_16m.groupby(['Link','dow','tslice'])
##    lol_mrp = [[t[0][0],t[0][1],str(t[0][2]),error(t[1].RealX,t[1].PredX)[0],error(t[1].RealX,t[1].PredX)[1]] for t in g16_mrp ]
##    g16df_mrp = pd.DataFrame(lol_mrp,columns=['Link','dow','tslice','rmse','mape'])

def RPlot(df):
    plt.rcParams.update({'font.size': 24})
    #wkday = np.asarray(df.index.dayofweek < 5) 
    wkday = np.asarray(df.dow == 0) 
    #index_holidays = df.loc[df.index.isin(AU_holidays)]
    #holi = np.asarray(index_holidays)
    #print(wkday)
    #print(holi)
    #wkendSat = np.asarray(df.index.dayofweek >= 5)
    wkendSat = np.asarray(df.dow == 1)
    #wkendSun = np.asarray(df.index.dayofweek == 6)
    #print(wkday)
    inx = wkday.astype(int)
    #print(inx)
    #index = np.where(np.array(timeDiffInt) == 0, 0, 1)
    X = df.RealX.values#X
    Y = df.PredX.values
    fig, ax = plt.subplots()

    ax.scatter(X[wkday], Y[wkday], label='Weekdays', c='b',marker='o')
    # scatter warning points in red (c='r')
    ax.scatter(X[wkendSat], Y[wkendSat], label='Weekends', c='r',marker='s')

    Title = "Estimation in the year 2016 for link id " + str(df.Link.values[0]) + " at " + str(df.tslice.values[0])#str(f[14:16]) + ':' + str(f[16:18])
    
    plt.title(Title)
    plt.legend(loc='upper left')
    plt.xlabel('Observed link count') # (Link Id: 2920-2919)')
    mpe = round(np.mean(np.abs((X- Y) /X)) * 100,2)
    #yl = 'Estimated link count'
    yl ='Estimated link count (MAPE = ' + str(mpe) + ')' 
    plt.ylabel(yl)

    res = sm.OLS(Y,sm.add_constant(X)).fit()
    X_plot = np.linspace(0,np.amax(X) + 10,10)
    #plt.plot(X_plot, X_plot*res.params[1] + res.params[0])
    plt.plot(X_plot, X_plot)
    plt.grid(True)
    plt.show()
if True:
    dfXW = brp_16m.copy()
    linkO_L = ['2920-2919','2919-2911','2924-2925','2922-2923']#'2924-2925'#'4580-4581'#'2919-2911'#'2924-2925' #'2920-2919'#
    tslice_list = ['07:15','13:30','16:45','20:15']
    for linkO in linkO_L:
        for ts in tslice_list:
            dfX = dfXW[(dfXW.Link == linkO) & (dfXW.tslice == ts)]
            RPlot(dfX)


def ScatterPlot(X,Y,C):
    HCor = np.where(C >= 0.8)
    MCor = np.where((C >= .5) & (C < .8))
    #ACor = np.where((C >= .5) & (C < .7))
    PCor = np.where(C < .5)
    x = X #* 100
    y = Y
    fig, ax = plt.subplots()
    #ax.scatter(x,y,c='b')
    
    
    #ax.scatter(x[ACor], y[ACor], label='0.7 > Correlation >= 0.5 ', c='y')
    ax.scatter(x[PCor], y[PCor], label='Correlation<0.5',c='r',marker='s')
    ax.scatter(x[MCor], y[MCor], label='0.8>Correlation>=0.5', c='g',marker='*')
    ax.scatter(x[HCor], y[HCor], label='Correlation>=0.8', c='b',marker='.',alpha=0.6)
    
    Title = "Data completeness index vs MAPE for all time slices over three years in each outgoing link"
    #plt.title(Title)
    plt.legend(loc='upper left')
    plt.xlabel("Data completeness index") # (Link Id: 2920-2919)')
    #mpe = round(mpe,2)
    plt.ylabel('Mean absolute percentage error')
    #plt.title()
    #res = sm.OLS(Y,sm.add_constant(X)).fit()
    #X_plot = np.linspace(0,np.amax(X) + 10,10)
    #plt.plot(X_plot, X_plot*res.params[1] + res.params[0])
    #plt.grid(True)
    plt.ylim(0,100)
    plt.xlim(0,1)
    plt.show()
