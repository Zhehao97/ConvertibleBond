#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:21:58 2018

@author: Zhehao Li
"""

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
vv,rr = BackTracking.ExecuteModel(cbond_parameter,stock_price,cbond_price,nationbond,s=1)
end = datetime.datetime.now()
print(end-start)


vv.to_csv('value1.csv',encoding='UTF-8')
rr.to_csv('return_rate1.csv',encoding='UTF-8')



'''
rr2 = pd.read_csv('return_rate2.csv')
rr2 = rr2.set_index('Unnamed: 0')
rr2 = rr2.dropna(axis=0,how='all')

rr3 = pd.read_csv('return_rate3.csv')
rr3 = rr3.set_index('Unnamed: 0')
rr3 = rr3.dropna(axis=0,how='all')

rr4 = pd.read_csv('return_rate4.csv')
rr4 = rr4.set_index('Unnamed: 0')
rr4 = rr4.dropna(axis=0,how='all')

vv2 = pd.read_csv('2.csv')
rr2 = rr2.set_index('Unnamed: 0')
rr2 = rr2.dropna(axis=0,how='all')

rr3 = pd.read_csv('return_rate3.csv')
rr3 = rr3.set_index('Unnamed: 0')
rr3 = rr3.dropna(axis=0,how='all')

rr4 = pd.read_csv('return_rate4.csv')
rr4 = rr4.set_index('Unnamed: 0')
rr4 = rr4.dropna(axis=0,how='all')


rr3+rr2

frame = [rr2,rr3,rr4]
return_rate = pd.concat(frame)
return_rate.columns = pd.DatetimeIndex(return_rate.columns)
'''


