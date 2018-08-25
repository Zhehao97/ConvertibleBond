#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 13:23:10 2018

@author: ZhehaoLi
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
#1. 提取收益数据
'''
##################################################################################

portfolio_r = pd.read_csv('portfolio_return_ss300.csv') #投资组合收益
stock_r = pd.read_csv('stock_return_ss300.csv') #股票空头日收益
cbond_r = pd.read_csv('cbond_return_ss300.csv') #转债多头日收益

portfolio_r = portfolio_r.set_index('Unnamed: 0')
stock_r = stock_r.set_index('Unnamed: 0')
cbond_r = cbond_r.set_index('Unnamed: 0')

portfolio_r.index = pd.DatetimeIndex(portfolio_r.index)
stock_r.index = pd.DatetimeIndex(stock_r.index)
cbond_r.index = pd.DatetimeIndex(cbond_r.index)

# 提取指数行情
index_SH = pd.read_csv('index_SH.csv') #上证综指
index_SZ = pd.read_csv('index_SZ.csv') #深证成指

index_SH.columns = ['TRADE_DT','CLOSE']
index_SH = index_SH.set_index('TRADE_DT')
index_SH.index = pd.DatetimeIndex(index_SH.index)
index_SH = index_SH.dropna()

index_SZ.columns = ['TRADE_DT','CLOSE']
index_SZ = index_SZ.set_index('TRADE_DT')
index_SZ.index = pd.DatetimeIndex(index_SZ.index)
index_SZ = index_SZ.dropna()

portfolio_r.mean(axis=1)

##################################################################################
'''
#2. 绘图区
'''
##################################################################################

'''转债多头和股票空头拆分收益'''

perf_s,dat_s = BackTracking.Performance(stock_r)
perf_c,dat_c = BackTracking.Performance(cbond_r)

#股票多头收益
BackTracking.PerformancePlot(dat_s['日平均收益'],dat_s['累计收益'],index_SH,index_SZ,label_name='股票空头日净值曲线',file_name='沪深300近11年股票空头收益')
#转债多头收益
BackTracking.PerformancePlot(dat_c['日平均收益'],dat_c['累计收益'],index_SH,index_SZ,label_name='转债多头日净值曲线',file_name='沪深300近11年转债多头收益')



##################################################################################
##################################################################################

'''投资组合合并收益'''

perf,dat = BackTracking.Performance(portfolio_r)

BackTracking.PerformancePlot(dat['日平均收益'],dat['累计收益'],index_SH,index_SZ,label_name='日净值曲线',file_name='ss300近11年组合收益')


'''以下是分区间回测'''

#2007-10-17 6124点(最高点)
#2008-02-04 4672点 -24%
#2008-11-7 1748点(最低点)
#2009-04-08 2347点 +34%

rr_1 = portfolio_r.iloc[265:520,:]
perf_1,dat_1 = BackTracking.Performance(rr_1)

BackTracking.PerformancePlot(dat_1['日平均收益'],dat_1['累计收益'],index_SH.iloc[373:680,0],index_SZ.iloc[373:680,0],label_name='日净值曲线',file_name='R20080204_20090224')


# 2010-05-17 - 2014-10-28
rr_2 = portfolio_r.iloc[820:1900,:]
perf_2,dat_2 = BackTracking.Performance(rr_2)

BackTracking.PerformancePlot(dat_2['日平均收益'],dat_2['累计收益'],index_SH.iloc[928:2200,0],index_SZ.iloc[928:2200,0],label_name='日净值曲线',file_name='R20100517_20141028')


#2015-06-12 5718.19 (最高点)
#2015-07-01 4214.15 -22%
#至今

rr_3 = portfolio_r.iloc[2063:,:]
perf_3,dat_3 = BackTracking.Performance(rr_3)

BackTracking.PerformancePlot(dat_3['日平均收益'],dat_3['累计收益'],index_SH.iloc[2171:,0],index_SZ.iloc[2171:,0],label_name='日净值曲线',file_name='R20150701_')



#2017-08-11 3208.54
#2018-08-17 2669

rr_4 = portfolio_r.iloc[-250:,:]
perf_4,dat_4 = BackTracking.Performance(rr_4)

BackTracking.PerformancePlot(dat_4['日平均收益'],dat_4['累计收益'],index_SH.iloc[-251:,0],index_SZ.iloc[-251:,0],label_name='日净值曲线',file_name='R20170811_')

#2018-02-05

rr_5 = portfolio_r.iloc[-130:,:]
perf_5,dat_5 = BackTracking.Performance(rr_5)

BackTracking.PerformancePlot(dat_5['日平均收益'],dat_5['累计收益'],index_SH.iloc[-131:,0],index_SZ.iloc[-131:,0],label_name='日净值曲线',file_name='R20180205_')

