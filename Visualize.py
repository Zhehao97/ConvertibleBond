#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:30:19 2018

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
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 



everyday_data = pd.read_csv('everyday_data.csv')
pricing_data1 = pd.read_csv('data1.csv') 
pricing_data2 = pd.read_csv('data2.csv') 


everyday_data = everyday_data.set_index('TRADE_DT')
pricing_data1 = pricing_data1.set_index('TRADE_DT')
pricing_data2 = pricing_data2.set_index('TRADE_DT')


frame = [pricing_data1,pricing_data2]
pricing_data = pd.concat(frame,axis=0,join='outer')


pricing_data.index = pd.DatetimeIndex(pricing_data.index)
everyday_data.index = pd.DatetimeIndex(everyday_data.index)

pricing_data.to_csv('data.csv')


sigma_difference = (everyday_data['VOLATILITY'] - pricing_data['SIGMA'])



sigma_difference[sigma_difference>0.2].count()

sigma_difference[sigma_difference<-0].count()



###########################################################################################
'''
验证delta值可近似用BSM定价模型解析式求导得到
'''
temp_stock_price = stock_price.iloc[:,-121:]

temp_stock_price = temp_stock_price.sort_values(by=['002008.SZ'],axis=1)

temp_LSM = pricing_data['LSM_PRICE'].sort_values()

temp_BSM = pricing_data['BSM_PRICE'].sort_values()

temp_cbond = cbond_price.sort_values(by=['128035.SZ'],axis=1)

#画图

fig = plt.figure(figsize=(20,5))
ax2 = fig.add_subplot(111)
ax2.grid(True)
ax2.plot(temp_stock_price.values.reshape(-1),temp_LSM.values.reshape(-1), linewidth=1.2,label='LSM定价',zorder=11)
ax2.plot(temp_stock_price.values.reshape(-1),temp_BSM.values.reshape(-1), linewidth=0.8,label='BSM定价',zorder=12)
ax2.plot(temp_stock_price.values.reshape(-1),temp_cbond.values.reshape(-1),color='blue',linewidth=1.5,label='可转债价格',zorder=22)
ax2.set_ylabel('转债价格与正股价格关系')
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('转债价格与正股价格.jpg',dpi=1000)


###########################################################################################



lsm_difference_percent = (pricing_data['LSM_PRICE'] - cbond_price)/cbond_price
bsm_difference_percent = (pricing_data['BSM_PRICE'] - cbond_price)/cbond_price
bond_difference_percent = (pricing_data['BOND_PRICE'] - cbond_price)/cbond_price

lsm_bsm_difference = pricing_data['BSM_PRICE']/pricing_data['LSM_PRICE']

lsm_bsm_difference

np.mean(lsm_bsm_difference.values)

np.var(lsm_bsm_difference.values)

np.std(lsm_bsm_difference.values)


#画图
c = stock_price.iloc[0,-1]
d = cbond_price.iloc[0,-1]

fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.bar(lsm_difference_percent.columns, lsm_difference_percent.iloc[0,:],color='yellowgreen', width=0.8,label='LSM定价偏差',zorder=1)
ax1.set_ylim(0,0.5)
ax1.set_ylabel('定价偏差')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
ax2.plot(pricing_data.index, pricing_data['LSM_PRICE'], linewidth=1.2,label='LSM定价',zorder=11)
ax2.plot(pricing_data.index, pricing_data['BSM_PRICE'], linewidth=0.8,label='BSM定价',zorder=12)
ax2.plot(pricing_data.index, pricing_data['BOND_PRICE'], linewidth=0.8,label='债券定价',zorder=13)
ax2.plot(stock_price.columns[500:],stock_price.iloc[0,500:]*d/c,color='red',linewidth=1.5,label='正股价格',zorder=21)
ax2.plot(cbond_price.columns,cbond_price.iloc[0,:],color='blue',linewidth=1.5,label='可转债价格',zorder=22)
ax2.set_ylabel('标的物价格&定价结果')
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('可转债定价测试.jpg',dpi=1000)

