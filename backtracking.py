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


#格力转债


Time = {}
Price = {}

Time['now'] = pd.to_datetime(time.strftime("%Y-%m-%d"))
Time['start'] = pd.to_datetime('2018-02-06')
Time['end'] = pd.to_datetime('2024-02-06')

Price['resale'] = 103
Price['redeem'] = 104
Price['now'] = 41.15
Price['riskfree'] = 0.015
Price['volatility'] = 0.384204

Price['resale_trigger'] = 36.75
Price['redeem_trigger'] = 68.25
Price['strike'] = 52.50
Price['facevalue'] = 100

cp = np.array([0.002,0.004,0.006,0.008,0.016,0.02])
Bond_Coupon = pd.Series(cp,index=pd.date_range(Time['start'],Time['end'],freq='365D',closed='right'))

