#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:17:16 2018

@author: trevor
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

import backtracking


cbond_price = pd.read_csv('cbond_price.csv',encoding='GBK')
stock_price = pd.read_csv('stock_price.csv',encoding='GBK')

cbond_price = cbond_price.drop([0,1],axis=0)
stock_price = stock_price.drop([0,1],axis=0)

cbond_price = cbond_price.set_index('Unnamed: 0')
stock_price = stock_price.set_index('Unnamed: 0')

cbond_parameter = pd.read_csv('cbond_parameter.csv',encoding='GBK')
cbond_parameter = cbond_parameter.drop([0,4],axis=0)
cbond_parameter = cbond_parameter.set_index('Unnamed: 0')


FVK = cbond_parameter['LATESTPAR']/cbond_parameter['CLAUSE_CONVERSION2_SWAPSHAREPRICE']




portfolio_value = pd.DataFrame(index=cbond_price.index,columns=cbond_price.columns)
postion = np.zeros(cbond_price.shape[0])
trading_cost = np.zeros(cbond_price.shape[0])



for day in cbond_price.columns:
    for c in range(cbond_price.shape[0]):
        if cbond_price[day].isna()[c]:
            continue
        else:
            excess = stock_price[day][c]*FVK[c] - cbond_price[day][c]
            print(excess)
            if excess > 0 :
                if postion[c] == 0 and cbond_price[day][c]<130: 
                    postion[c] = 1
                    trading_cost[c] = cbond_price[day][c]*5/100000
                    portfolio_value[day][c] = postion[c]*(cbond_price[day][c] - trading_cost[c])
                elif postion[c] == 1:
                    portfolio_value[day][c] = postion[c]*(cbond_price[day][c] - trading_cost[c])
            elif excess < 0:
                if postion[c] == 0:
                    continue
                elif postion[c] == 1:
                    postion[c] = 0
                    portfolio_value[day][c] = cbond_price[day][c] - trading_cost[c]


Return = portfolio_value/portfolio_value.shift(periods=1,axis=1)-1

backtracking.getFeature(Return.iloc[3,:])


