#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 08:54:18 2018

@author: Zhehao Li
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import time
from sklearn import linear_model
from pylab import mpl
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


#大族转债


Time = {}
Price = {}
Number = {}

Time['start'] = pd.to_datetime('2018-02-06')
Time['end'] = pd.to_datetime('2024-02-06')

Price['strike'] = 52.50
Price['facevalue'] = 100
Price['resale_trigger'] = 0.7 * Price['strike']
Price['redeem_trigger'] = 1.3 * Price['strike']

Price['resale'] = 103
Price['redeem'] = 104
Price['riskfree'] = 0.015

cp = np.array([0.002,0.004,0.006,0.008,0.016,0.02])
Bond_Coupon = pd.Series(cp,index=pd.date_range(Time['start'],Time['end'],freq='365D',closed='right'))



stock_price = pd.read_csv('/Users/trevor/Downloads/大族激光.csv')
stock_price = stock_price.set_index('S_INFO_WINDCODE')

stock_change = stock_price.iloc[:,1:]/stock_price.shift(periods=1,axis=1).iloc[:,1:] - 1

stock_change.columns[-1]

Datetime = pd.date_range(start=Time['start'],end=stock_change.columns[-1],freq='D')

std = []
for day in Datetime:
    period = 
    std(stock_change.values)



std = np.array(std)
#两年历史数据年化波动率
Price['volatility'] = std*np.sqrt(252)



Time['now'] = pd.to_datetime(time.strftime("%Y-%m-%d"))
Price['now'] = 41.15
Price['volatility'] = 0.384204#计算波动率


