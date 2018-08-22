#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:33:25 2018

@author: Zhehao Li
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
from sklearn import linear_model
from pandas.tseries.offsets import Day, MonthEnd
from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 



###############################################################################
'''
1. 情绪面指标
'''
###############################################################################

stock = pd.read_excel('stock_price.xlsx')
stock_close = pd.pivot_table(stock_price,values='S_DQ_PRECLOSE',index='S_INFO_WINDCODE',columns='TRADE_DT')

index = pd.read_excel('index_price.xlsx')

#1-1. 半年线以上股票数量占比
###############################################################################












###############################################################################
'''
2. 资金面指标
'''
###############################################################################











###############################################################################
'''
3. 衍生品指标
'''
###############################################################################












###############################################################################
'''
4. 利率
'''
###############################################################################