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
def error(yS,xR):
    rms = sqrt(mean_squared_error(yS, xR))
    mape = np.mean(np.abs((xR - yS) /xR)) * 100
    return rms,mape

def mapePlot_MRP(df):
#if 1 == 1:
    df['dow'] = df.dow.apply(lambda x: 0 if x <5 else 1)
    plt.rcParams.update({'font.size': 19})
    name =[x for x in globals() if globals()[x] is df][0]
    yr = str(20) + name[-2] + name[-1]
    if name[0] == 'm':
        errtype = "MAPE"
    else:
        errtype = "RMSE"
    edata = df.drop(['link','dow'],axis=1)
    te = edata.astype(float)
    #ie = te.mean(axis=1).values.argsort()
    #te = te.iloc[ie]

    grid_kws = {"height_ratios": (.9, .01), "hspace": .4}

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

    y2_pos = 0.5  
    y2_val = []
    y2_pos_list = []
    ax2.set_ylabel('Weekend indicator', color='blue')
    ax2.set_yticks(numy2)
    ax2.set_yticklabels(df.dow.to_list())
##    for k,j in itertools.groupby(df.dow.to_list()):
##        y2_val.append(k)
##        y2_pos = y2_pos + len(list(j))
##        y2_pos_list.append(y2_pos)
##    ax2.set_yticks(y2_pos_list)
##    ax2.set_yticklabels(y2_val)
        
    for t in ax2.get_yticklabels():
        txt = t.get_text()
        #print(t.get_text())
        if int(txt) == 0:
            t.set_color('black')
        else:
            t.set_color('blue')
            
    list_ylab = [i if i ==1 else ' ' for i in df.dow.to_list()]
    ax2.set_yticklabels(list_ylab)  
    #ax2.set_yticklabels([])#
    plt.setp(ax2.get_yticklabels(), rotation=90, fontsize='small')
    plt.tight_layout()
    
    ##ax2.set_ylabel('Day of a week', color='blue')
    ##ax2.set_yticks(numy2)
    ##ax2.set_yticklabels(df.dow.to_list())
    ##plt.setp(ax2.get_yticklabels(), rotation=90, fontsize='small')
    plt.show()
def Geh_Heatmap(df):
#if 1 == 1:
    hours_str = ['00:00','01:00', '02:00', '03:00', '04:00', '05:00','06:00', '07:00', '08:00', '09:00','10:00', '11:00','12:00', '13:00', '14:00', '15:00','16:00',
            '17:00', '18:00', '19:00','20:00',  '21:00', '22:00', '23:00', '24:00']
    plt.rcParams.update({'font.size': 18})
    name =[x for x in globals() if globals()[x] is df][0]
    yr = str(20) + name[-2] + name[-1]
    #if name[0] == 'm':
    #    errtype = "MAPE"
    #else:
    errtype = "GEH"
    plt.rcParams.update({'font.size': 19})
    edata = df.drop(['Link','dow'],axis=1)
    te = edata.astype(float)
    #ie = te.mean(axis=1).values.argsort()
    #te = te.iloc[ie]

    grid_kws = {"height_ratios": (.9, .01), "hspace": .04}

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
    y2_pos = 0.5  
    y2_val = []
    y2_pos_list = []
    ax2.set_ylabel('Weekend indicator', color='blue')
    ax2.set_yticks(numy2)
    ax2.set_yticklabels(df.dow.to_list())
##    for k,j in itertools.groupby(df16.dow.to_list()):
##        y2_val.append(k)
##        y2_pos = y2_pos + len(list(j))
##        y2_pos_list.append(y2_pos)
##    ax2.set_yticks(y2_pos_list)
##    ax2.set_yticklabels(y2_val)
        
    for t in ax2.get_yticklabels():
        txt = t.get_text()
        #print(t.get_text())
        if int(txt) == 0:
            t.set_color('black')
        else:
            t.set_color('blue')
    #ax2.set_yticklabels([])#
    list_ylab = [i if i ==1 else ' ' for i in df.dow.to_list()]
    ax2.set_yticklabels(list_ylab)
    plt.setp(ax2.get_yticklabels(), rotation=90, fontsize='small')
    plt.tight_layout()
    plt.show()   


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



AU_holidays = holidays.Australia()
plt.rcParams.update({'font.size': 16})
def GEH(M,C):
    return np.sqrt(2 * (M-C)**2/(M+C))

##def Export_table(df):
##    
##    b = df.groupby(['Link','dow','tslice']).mean()
##    c = b.droplevel('tslice').copy()
##    
##    d = [j.reset_index().groupby(['Link','dow','hod']).sum() for i,j in c.groupby('hod')]
##    geh_df16m = pd.concat(d)
##    #geh_df16m['GEH'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
##    
##    #p = geh_df16m.groupby(level=[0,2]).mean()
##    
##    geh_df16m_avg = geh_df16m.groupby(level=[0, 1]).mean()
##    geh_df16m_avg['GEH'] = GEH(geh_df16m_avg.RealX,geh_df16m_avg.PredX)
##
##    p = geh_df16m_avg.groupby(level=[0]).mean()
##    p['GEH'] = GEH(p.RealX,p.PredX)
    
    



    
#create mrp plots
if True: 
    plt.rcParams.update({'font.size': 20})

    mpe19 = pd.read_csv('erdfBi19d.csv')
    mpe18 = pd.read_csv('erdfBi18d.csv')
    mpe17 = pd.read_csv('erdfBi17d.csv')
    mpe16 = pd.read_csv('erdfBi16d.csv')

    rmse19 = pd.read_csv('rmseBi19d.csv')
    rmse18 = pd.read_csv('rmseBi18d.csv')
    rmse17 = pd.read_csv('rmseBi17d.csv')
    rmse16 = pd.read_csv('rmseBi16d.csv')

    #cr16 = pd.read_csv('crdf16d.csv')
    #cr17 = pd.read_csv('crdf17d.csv')
    #cr18 = pd.read_csv('crdf18d.csv')
    #cr19 = pd.read_csv('crdf19d.csv')

    #crL = [cr19, cr18,cr17,cr16 ]


    mpeL = [mpe19, mpe18, mpe17, mpe16]
    rmseL = [rmse19, rmse18, rmse17, rmse16]
    #uLinks = list(mpe19['link'].unique()) + list(mpe18['link'].unique()) + list(mpe17['link'].unique()) + list(mpe16['link'].unique())
    #uLinks = list(np.unique(uLinks))

    


if True:
    Links = mpe19[['link','dow']].values.tolist() + mpe18[['link','dow']].values.tolist() + mpe17[['link','dow']].values.tolist() + mpe16[['link','dow']].values.tolist() 
    uLinks = [list(x) for x in set(tuple(x) for x in Links)] #list(k for k,_ in itertools.groupby(Links))
    linkmstd = pd.DataFrame()
    posxticks =  np.arange(0,len(uLinks))
    xticks = []

    #axm = plt.subplot()
    factor = 0.2

    yearIndex = [2016,2017,2018,2019]
    for xxr,link in enumerate(uLinks):
        xticks.append(str(link[0]) + ' ' + str(link[1]))
        for i,tempdf in enumerate(mpeL):
            year = 2019 - i
            tempdfM = mpeL[i]
            tempdfR = rmseL[i]
            tempM = tempdfM[(tempdfM.link == link[0]) & (tempdfM.dow == link[1])].drop(['link','dow'],axis=1)
            tempR = tempdfR[(tempdfR.link == link[0]) & (tempdfR.dow == link[1])].drop(['link','dow'],axis=1)
            if len(tempM) > 0 and len(tempR) > 0:
                linkmstd = linkmstd.append({'link':link[0],'dow':int(link[1]),'year':int(year),'meanMAPE':tempM.mean(axis=1).values[0],'stdMAPE':tempM.std(axis=1).values[0],
                             'meanRMSE':tempR.mean(axis=1).values[0],'stdRMSE':tempR.std(axis=1).values[0]},ignore_index=True)
                lab = 'MAPE for ' + str(year)#2016'

                         
    linkmstd['dow'] =linkmstd.dow.astype(int)
    linkmstd['year'] =linkmstd.year.astype(int)
    linkmstdf = linkmstd.set_index(['link','dow','year']).round(2)
    linkmstdf.to_csv('linkmstdf_MRP.csv')
    #erdfBi = erdfBi16.groupby(['link']).agg(lambda x: x.unique().mean()) ### Very important   
    #rmsedf = rmsedf16.groupby(['link']).agg(lambda x: x.unique().mean()) ### Very important




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

### GEH data extraction MRP model

if True:
    rp_16m = pd.read_csv('real_pred_df16_mrp.csv',index_col=0)
    rp_17m = pd.read_csv('real_pred_df17_mrp.csv',index_col=0)
    rp_18m = pd.read_csv('real_pred_df18_mrp.csv',index_col=0)
    rp_19m = pd.read_csv('real_pred_df19_mrp.csv',index_col=0)

    rp_16m['hod'] = rp_16m.tslice.apply(lambda x:int(x[0:2]))
    rp_17m['hod'] = rp_17m.tslice.apply(lambda x:int(x[0:2]))
    rp_18m['hod'] = rp_18m.tslice.apply(lambda x:int(x[0:2]))
    rp_19m['hod'] = rp_19m.tslice.apply(lambda x:int(x[0:2]))

if True:
    b = rp_16m.groupby(['Link','dow','tslice']).mean()
    c = b.droplevel('tslice').copy()
    d = [j.reset_index().groupby(['Link','dow','hod']).sum() for i,j in c.groupby('hod')]
    geh_df16m = pd.concat(d)
    #geh_df16m['GEH'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
    #p = geh_df16m.groupby(level=[0,2]).mean()
    geh_df16m_avg = geh_df16m.groupby(level=[0, 1]).mean()
    geh_df16m_avg['GEH'] = GEH(geh_df16m_avg.RealX,geh_df16m_avg.PredX)

    mrp16 = geh_df16m_avg.groupby(level=[0]).mean()
    mrp16['GEH16'] = GEH(mrp16.RealX,mrp16.PredX)

#if True:
    b = rp_17m.groupby(['Link','dow','tslice']).mean()
    c = b.droplevel('tslice').copy()
    d = [j.reset_index().groupby(['Link','dow','hod']).sum() for i,j in c.groupby('hod')]
    geh_df17m = pd.concat(d)
    #geh_df16m['GEH'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
    #p = geh_df16m.groupby(level=[0,2]).mean()
    geh_df17m_avg = geh_df17m.groupby(level=[0, 1]).mean()
    geh_df17m_avg['GEH'] = GEH(geh_df17m_avg.RealX,geh_df17m_avg.PredX)

    mrp17 = geh_df17m_avg.groupby(level=[0]).mean()
    mrp17['GEH17'] = GEH(mrp17.RealX,mrp17.PredX)

#if True:
    b = rp_18m.groupby(['Link','dow','tslice']).mean()
    c = b.droplevel('tslice').copy()
    d = [j.reset_index().groupby(['Link','dow','hod']).sum() for i,j in c.groupby('hod')]
    geh_df18m = pd.concat(d)
    #geh_df16m['GEH'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
    #p = geh_df16m.groupby(level=[0,2]).mean()
    geh_df18m_avg = geh_df18m.groupby(level=[0, 1]).mean()
    geh_df18m_avg['GEH'] = GEH(geh_df18m_avg.RealX,geh_df18m_avg.PredX)

    mrp18 = geh_df18m_avg.groupby(level=[0]).mean()
    mrp18['GEH18'] = GEH(mrp18.RealX,mrp18.PredX)

#if True:
    b = rp_19m.groupby(['Link','dow','tslice']).mean()
    c = b.droplevel('tslice').copy()
    d = [j.reset_index().groupby(['Link','dow','hod']).sum() for i,j in c.groupby('hod')]
    geh_df19m = pd.concat(d)
    #geh_df16m['GEH'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
    #p = geh_df16m.groupby(level=[0,2]).mean()
    geh_df19m_avg = geh_df19m.groupby(level=[0, 1]).mean()
    geh_df19m_avg['GEH'] = GEH(geh_df19m_avg.RealX,geh_df19m_avg.PredX)

    mrp19 = geh_df19m_avg.groupby(level=[0]).mean()
    mrp19['GEH19'] = GEH(mrp19.RealX,mrp19.PredX)
    
    geh_mrp_all = pd.concat([mrp16['GEH16'],mrp17['GEH17'],mrp18['GEH18'],mrp19['GEH19']],axis='columns')

print(geh_mrp_all.round(2).fillna('-').to_latex())
## Data extraction for BRP model

### plot GEH stats 2016 heatmap
if True:
    _com = geh_mrp_all[geh_mrp_all.index.isin(linkmstd['link'].unique())]
    m_geh16 = geh_df16m.copy()
    m_geh16['GEH16'] = GEH(geh_df16m.RealX,geh_df16m.PredX)
    m_geh16 = m_geh16.reset_index().drop(['RealX', 'PredX', 'CorrCoeff' ],axis=1)
    m_geh16c = m_geh16[m_geh16.Link.isin(linkmstd['link'].unique())].copy()
    m_geh16_pivot = m_geh16c.pivot(index=['Link','dow'], columns='hod', values='GEH16').dropna()
    mgeh16 = m_geh16_pivot.reset_index()
    mgeh16['dow'] = mgeh16.dow.apply(lambda x: 0 if x <5 else 1)
    
Geh_Heatmap(mgeh16)

### 4 plots * 2 MAPE RMSE 4 years heatmap
mapePlot_MRP(mpe16)
mapePlot_MRP(rmse16)



if True:
    
    
    temp_m16 = pd.DataFrame()
    temp_m16['meanGEH'] = mgeh16.groupby(['Link']).mean().drop('dow',axis=1).mean(axis=1)
    temp_m16['stdGEH'] = mgeh16.groupby(['Link']).mean().drop('dow',axis=1).std(axis=1)
    temp_m16['year'] = [2016] * len(temp_m16)
    temp_m16.reset_index(inplace=True)
    
   
    m_geh17 = geh_df17m.copy()
    m_geh17['GEH17'] = GEH(geh_df17m.RealX,geh_df17m.PredX)
    m_geh17 = m_geh17.reset_index().drop(['RealX', 'PredX', 'CorrCoeff' ],axis=1)
    m_geh17c = m_geh17[m_geh17.Link.isin(linkmstd['link'].unique())].copy()
    m_geh17_pivot = m_geh17c.pivot(index=['Link','dow'], columns='hod', values='GEH17').dropna()
    mgeh17 = m_geh17_pivot.reset_index()

    temp_m17 = pd.DataFrame()
    temp_m17['meanGEH'] = mgeh17.groupby(['Link']).mean().drop('dow',axis=1).mean(axis=1)
    temp_m17['stdGEH'] = mgeh17.groupby(['Link']).mean().drop('dow',axis=1).std(axis=1)
    temp_m17['year'] = [2017] * len(temp_m17)
    temp_m17.reset_index(inplace=True)

   
    m_geh18 = geh_df18m.copy()
    m_geh18['GEH18'] = GEH(geh_df18m.RealX,geh_df18m.PredX)
    m_geh18 = m_geh18.reset_index().drop(['RealX', 'PredX', 'CorrCoeff' ],axis=1)
    m_geh18c = m_geh18[m_geh18.Link.isin(linkmstd['link'].unique())].copy()
    m_geh18_pivot = m_geh18c.pivot(index=['Link','dow'], columns='hod', values='GEH18').dropna()
    mgeh18 = m_geh18_pivot.reset_index()

    temp_m18 = pd.DataFrame()
    temp_m18['meanGEH'] = mgeh18.groupby(['Link']).mean().drop('dow',axis=1).mean(axis=1)
    temp_m18['stdGEH'] = mgeh18.groupby(['Link']).mean().drop('dow',axis=1).std(axis=1)
    temp_m18['year'] = [2018] * len(temp_m18)
    temp_m18.reset_index(inplace=True)
    
   
    m_geh19 = geh_df19m.copy()
    m_geh19['GEH19'] = GEH(geh_df19m.RealX,geh_df19m.PredX)
    m_geh19 = m_geh19.reset_index().drop(['RealX', 'PredX', 'CorrCoeff' ],axis=1)
    m_geh19c = m_geh19[m_geh19.Link.isin(linkmstd['link'].unique())].copy()
    m_geh19_pivot = m_geh19c.pivot(index=['Link','dow'], columns='hod', values='GEH19').dropna()
    mgeh19 = m_geh19_pivot.reset_index()

    temp_m19 = pd.DataFrame()
    temp_m19['meanGEH'] = mgeh19.groupby(['Link']).mean().drop('dow',axis=1).mean(axis=1)
    temp_m19['stdGEH'] = mgeh19.groupby(['Link']).mean().drop('dow',axis=1).std(axis=1)
    temp_m19['year'] = [2019] * len(temp_m19)
    temp_m19.reset_index(inplace=True)

if True:
    m_ly_mstd_geh = pd.concat([temp_m16,temp_m17,temp_m18,temp_m19])
    plt.rcParams.update({'font.size': 22})
    clr = ['k','b','g','r']
    mkr = ['s','^','o','*']
    ####### Plot error bar RMSE ######
    axr = plt.subplot()
    factor = 0.2

    #ldf = m_ly_mstd_geh.set_index(['Link','dow']).round(2) ## changed on 2023
    ldf = m_ly_mstd_geh.set_index(['Link','year']).round(2).groupby(['Link','year']).mean()
    
    ls = ldf.sort_values(by='meanGEH',ascending=True)
    ldfs = ldf.reset_index().set_index('Link')
    L = m_ly_mstd_geh.groupby('Link').mean().sort_values(by='meanGEH',ascending=True).index.tolist() ####

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
##    for itemdf in mpeL+rmseL:
##        mapePlot_MRP(itemdf)

        
    r16 = pd.read_csv('rawdata16d.csv')
    r17 = pd.read_csv('rawdata17d.csv')
    r18 = pd.read_csv('rawdata18d.csv')
    r19 = pd.read_csv('rawdata19d.csv')
    raw = pd.concat([r16,r17,r18,r19],ignore_index=True)
    rdf= raw.drop('Unnamed: 0',axis=1).sort_values(['Pmean'])#.drop(['link','dow'],axis=1)
    axr = plt.subplot()
    xxr =  np.arange(0,len(rdf))
    rdf.drop_duplicates(inplace=True)
    
    axr.errorbar(xxr,rdf.Rmean.values,rdf.Rstd.values,capsize=0,linestyle='None',c='b',marker=' ',label='Observed sample mean and standard deviation')
    axr.errorbar(xxr,rdf.Pmean.values,rdf.Pstd.values,capsize=0,linestyle='None',c='r',marker='o',alpha=0.3,label='Estimated sample mean and standard deviation')
    
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
##    cr16 = pd.read_csv('crdf16d.csv')
##    cr17 = pd.read_csv('crdf17d.csv')
##    cr18 = pd.read_csv('crdf18d.csv')
##    cr19 = pd.read_csv('crdf19d.csv')
##
    dci16 = pd.read_csv('noData16d.csv')
    dci17 = pd.read_csv('noData17d.csv')
    dci18 = pd.read_csv('noData18d.csv')
    dci19 = pd.read_csv('noData19d.csv')
    
#    crL = [cr19, cr18,cr17,cr16 ]
#    cr = cr16.set_index(['link','dow'])
    d = dci16.set_index(['link','dow'])#.rename(columns={'link':'Link'})
    b = mpe16.set_index(['link','dow'])
    d16 = d[(d != 0).all(1)]
#    c16 = cr[(cr != 0).all(1)]
#    plt.scatter(d16.values.flatten(),b.values.flatten())
#    plt.show()
if True:
    # 4581-4580 2922-2923
    #good_links GEH16 < 5
    #bad_links GEH16 > 5
    a16 = mrp16.reset_index()
    a16 = a16[a16.Link.isin(L)]
    
    gL =  a16[a16.GEH16 < 5].Link.to_list()
    #gL = ['4581-4580', '2922-2923','2924-2925','2919-2920','2903-2902','2919-2911']
    bL = a16[a16.GEH16 > 5].Link.to_list()
    dl16 = d16.reset_index(level=0)
    gl_16 = dl16[dl16.link.isin(gL)]
    gl_16.set_index('link').T.plot()
    plt.show()
    bl_16 = dl16[dl16.link.isin(bL)]
    bl_16.set_index('link').T.plot()
    plt.show()
    # 2904-2905 4405-4404
if True:
    plt.plot(bl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1).mean(axis=0),c='r')
    plt.plot(gl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1).mean(axis=0),c='g')
    plt.show()
if True:
    bldf = bl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1)
    gldf = gl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1)    #
    gl_bl_df = pd.DataFrame(bl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1).mean(axis=0),columns=['Type2_links_MRP'])
    gl_bl_df['Type1_links_MRP'] = gl_16.reset_index().groupby(['link']).mean().drop('dow',axis=1).mean(axis=0)
    gl_bl_df.to_csv('gl_bl_df_MRP.csv')
    
if True:
    g16_brp = brp_16m.groupby(['Link','dow','tslice'])
    lol_brp = [[t[0][0],t[0][1],str(t[0][2]),error(t[1].RealX,t[1].PredX)[0],error(t[1].RealX,t[1].PredX)[1]] for t in g16_brp ]
    g16df_brp = pd.DataFrame(lol_brp,columns=['Link','dow','tslice','rmse','mape'])
    #g16df_brp_ = g16df_brp[(g16df_brp.rmse < 100) & (g16df_brp.mape < 100)].copy()
    g16df_brp_ = g16df_brp.copy()#g16df_brp[(g16df_brp.mape < 100)].copy()
    g16df_brp_mape = g16df_brp_.pivot(index=['Link','dow'], columns='tslice', values='mape').dropna()
    g16df_brp_mape_ = g16df_brp_mape.reset_index().rename(columns={'Link':'link'})

    g16_mrp = rp_16m.groupby(['Link','dow','tslice'])
    lol_mrp = [[t[0][0],t[0][1],str(t[0][2]),error(t[1].RealX,t[1].PredX)[0],error(t[1].RealX,t[1].PredX)[1]] for t in g16_mrp ]
    g16df_mrp = pd.DataFrame(lol_mrp,columns=['Link','dow','tslice','rmse','mape'])
