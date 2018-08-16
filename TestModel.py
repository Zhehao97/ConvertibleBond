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




cbond_parameter = pd.read_csv('cbond_parameter.csv',encoding='GBK')
cbond_price = pd.read_csv('cbond_price.csv',encoding='GBK')
stock_price = pd.read_csv('stock_price.csv',encoding='GBK')
nationbond = pd.read_csv('nationbond.csv')


stock_price = stock_price.set_index('Unnamed: 0')
stock_price.columns = pd.DatetimeIndex(stock_price.columns)
stock_price = stock_price.drop('A16256.SZ',axis=0)
stock_price = stock_price.drop(stock_price.columns[-1],axis=1)

nationbond = nationbond.set_index('Unnamed: 0')
nationbond.columns = pd.DatetimeIndex(nationbond.columns)

cbond_price = cbond_price.set_index('Unnamed: 0')
cbond_price.columns = pd.DatetimeIndex(cbond_price.columns)
cbond_price = cbond_price.drop('127005.SZ',axis=0)
cbond_price = cbond_price.drop(cbond_price.columns[-1],axis=1)


cbond_parameter = cbond_parameter.set_index('Unnamed: 0')
cbond_parameter['CLAUSE_PUTOPTION_RESELLINGPRICE']=103
cbond_parameter = cbond_parameter.drop('127005.SZ',axis=0)


start = datetime.datetime.now()
vv = BackTracking.ExecuteModel(cbond_parameter,stock_price,cbond_price,nationbond,s=3)
end = datetime.datetime.now()
print(end-start)

vv.to_csv('value3.csv',encoding='UTF-8')



'''
value1 = pd.read_csv('value1.csv')
value1 = value1.set_index('Unnamed: 0')
value1.columns = pd.DatetimeIndex(value1.columns)

value1 = value1.dropna(axis=0,how='all')


Return_rate = value1/value1.shift(axis=1,periods=1) - 1


BackTracking.getFeature(Return_rate.iloc[0,:])
'''

