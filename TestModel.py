#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:21:58 2018

@author: Zhehao Li
"""
import sys, time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import datetime
from sklearn import linear_model
from pylab import mpl
import scipy.optimize as opt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

import BackTracking
from ConvertBond import CBond


##################################################################################
'''
1. 原始数据准备
'''
##################################################################################
cbond_parameter = pd.read_csv('parameters.csv',encoding='GBK')
cbond_price = pd.read_csv('cbond_price.csv',encoding='GBK')
stock_price = pd.read_csv('stock_price.csv',encoding='GBK')
strike_price = pd.read_csv('strike_price.csv',encoding='GBK')
nationbond = pd.read_csv('nationbond.csv')


strike_price = strike_price.set_index('Unnamed: 0')
strike_price.columns = pd.DatetimeIndex(strike_price.columns)
strike_price = strike_price.T

stock_price = stock_price.set_index('Unnamed: 0')
stock_price = stock_price.drop('Stock_Code',axis=1)
stock_price.columns = pd.DatetimeIndex(stock_price.columns)
stock_price = stock_price.T

nationbond = nationbond.drop(0)
nationbond = nationbond.set_index('Unnamed: 0')
nationbond.columns = ['EDBCLOSE']
nationbond.index = pd.DatetimeIndex(nationbond.index)
nationbond = nationbond.T
nationbond = nationbond.apply(lambda x:np.float(x))


cbond_price = cbond_price.set_index('Unnamed: 0')
cbond_price.columns = pd.DatetimeIndex(cbond_price.columns)
cbond_price = cbond_price.T

cbond_parameter = cbond_parameter.set_index('Code')
cbond_parameter = cbond_parameter.T
#index 大小写转换
new_index=[]
for ind in cbond_parameter.index:
    new_index.append(ind.upper())
    
cbond_parameter.index = new_index

##################################################################################
'''
2. 运行程序
'''
##################################################################################

start = datetime.datetime.now()
rr = BackTracking.Main(cbond_parameter,stock_price,cbond_price,strike_price,nationbond,lsm=False)
end = datetime.datetime.now()
print(end-start)
rr.to_csv('return__3.csv',encoding='UTF-8')

##################################################################################
'''
3. 提取收益数据
'''
##################################################################################

rr = pd.read_csv('return__3.csv')
rr = rr.set_index('Unnamed: 0')
rr.index = pd.DatetimeIndex(rr.index)

##################################################################################
'''
4. 函数区
'''
##################################################################################

def getMaxDownList(datax):
    maxdownlist=[]
    for i in range(0, len(datax)):
        temp = (max(datax[:i+1]) - datax[i])/max(datax[:i+1])
        maxdownlist.append(temp)
    return max(maxdownlist)


def Performance(return_data):
    return_avg = return_data.mean(axis=1)
    return_avg = return_avg.dropna()
    
    return_std = return_avg.std()*np.sqrt(250)
    
    temp_return = return_avg+1
    return_cum = temp_return.cumprod()
    return_cum = return_cum/return_cum[0]
    
    periods = np.float(str(return_cum.index[-1] - return_cum.index[0])[:-14])
    earning = (return_cum[-1]-1)*365/periods
    sharpe_ratio = (earning - 0.03)/return_std
    maxdown = getMaxDownList(return_cum.values)
    
    performance = {'年化收益':earning,'年波动率':return_std,'夏普比率':sharpe_ratio,'最大回撤':maxdown}
    data = {'日平均收益':return_avg,'累计收益':return_cum}
    return performance,data


def PerformancePlot(return_avg,return_cum,index_1,index_2,label_name,file_name):
    b = return_cum[0]
    c = index_1[0]
    d = index_2[0]
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.grid(False)
    ax1.bar(return_avg.index, return_avg, width=2,linewidth=2,color='yellowgreen',label='日均收益',zorder=1)
    ax1.set_ylabel('日均收益')
    ax1.set_ylim(-0.2,0.2)
    ax1.legend(loc='upper right')
    
    ax2 = ax1.twinx()
    ax2.grid(True)
    ax2.plot(index_1.index, index_1, linewidth=1,label='上证综指',zorder=5)
    ax2.plot(index_2.index, index_2*c/d, linewidth=1,label='深证成指',zorder=6)
    ax2.plot(return_cum.index,return_cum*c/b,color='purple',linewidth=1.5,label=label_name,zorder=7)
    ax2.set_ylabel('指数')
    #ax2.set_ylim(0,10000)
    ax2.legend(loc='upper left')
    ax2.set_xlabel('时间')
    plt.savefig(file_name+'.jpg',dpi=1000)


##################################################################################
'''
5. 绘图区
'''
##################################################################################

index_SH = pd.read_csv('index_SH.csv')
index_SZ = pd.read_csv('index_SZ.csv')

index_SH.columns = ['TRADE_DT','CLOSE']
index_SH = index_SH.set_index('TRADE_DT')
index_SH.index = pd.DatetimeIndex(index_SH.index)
index_SH = index_SH.dropna()

index_SZ.columns = ['TRADE_DT','CLOSE']
index_SZ = index_SZ.set_index('TRADE_DT')
index_SZ.index = pd.DatetimeIndex(index_SZ.index)
index_SZ = index_SZ.dropna()



#2007-10-17 6124点(最高点)
#2008-02-04 4672点 -24%
#2008-11-7 1748点(最低点)
#2009-04-08 2347点 +34%


rr_1 = rr.iloc[265:520,:]
perf_1,dat_1 = Performance(rr_1)

PerformancePlot(dat_1['日平均收益'],dat_1['累计收益'],index_SH.iloc[373:680,0],index_SZ.iloc[373:680,0],label_name='日净值曲线',file_name='R20080204_20090224')


# 2010-05-17 - 2014-10-28
rr_2 = rr.iloc[820:1900,:]
perf_2,dat_2 = Performance(rr_2)

PerformancePlot(dat_2['日平均收益'],dat_2['累计收益'],index_SH.iloc[928:2200,0],index_SZ.iloc[928:2200,0],label_name='日净值曲线',file_name='R20100517_20141028')


#2015-06-12 5718.19 (最高点)
#2015-07-01 4214.15 -22%
#至今

rr_3 = rr.iloc[2063:,:]
perf_3,dat_3 = Performance(rr_3)

PerformancePlot(dat_3['日平均收益'],dat_3['累计收益'],index_SH.iloc[2171:,0],index_SZ.iloc[2171:,0],label_name='日净值曲线',file_name='R20150701_')



#2017-08-11 3208.54
#2018-08-17 2669

rr_4 = rr.iloc[-250:,:]
perf_4,dat_4 = Performance(rr_4)

PerformancePlot(dat_4['日平均收益'],dat_4['累计收益'],index_SH.iloc[-251:,0],index_SZ.iloc[-251:,0],label_name='日净值曲线',file_name='R20170811_')

#2018-02-05

rr_5 = rr.iloc[-130:,:]
perf_5,dat_5 = Performance(rr_5)

PerformancePlot(dat_5['日平均收益'],dat_5['累计收益'],index_SH.iloc[-131:,0],index_SZ.iloc[-131:,0],label_name='日净值曲线',file_name='R20180205_')




