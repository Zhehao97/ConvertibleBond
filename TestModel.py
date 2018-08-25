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



#在沪深300成分中的转债
cbond_index = pd.read_excel('可转债_指数标的物.xlsx')
cbond_index = cbond_index.reset_index()

c1 = cbond_index.iloc[:,:2]
c2 = cbond_index.iloc[:,2:]
c2.columns = ['Stock_Code','Stock_Name']

count_stock = pd.merge(c2,c1,on='Stock_Code',how='left')
count_stock = count_stock.dropna(axis=0)

count_stock['index'].count()

stock_price['125002.SZ']


cbond_price_SS300 = pd.DataFrame(index=cbond_price.index)

for c_code in count_stock['index'].values:
    temp_columns = cbond_price.columns[cbond_price.columns == c_code]
    cbond_price_SS300[temp_columns] = cbond_price[temp_columns]

##################################################################################
'''
2. 运行程序
'''
##################################################################################

start = datetime.datetime.now()
crr,srr = BackTracking.Main(cbond_parameter,stock_price,cbond_price_SS300,strike_price,nationbond,lsm=False)
end = datetime.datetime.now()
print(end-start)

portfolio_return = crr+srr #投资组合收益等于转债多头收益+股票空头收益

crr.to_csv('cbond_return_ss300.csv',encoding='UTF-8') 
srr.to_csv('stock_return_ss300.csv',encoding='UTF-8')
portfolio_return.to_csv('portfolio_return_ss300.csv',encoding='UTF-8')


##################################################################################