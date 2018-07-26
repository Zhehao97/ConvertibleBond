# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:09:25 2018

@author: Zhehao Li
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn import linear_model
from pylab import mpl
from time import time

mpl.rcParams['font.sans-serif'] = ['FangSong'] #
mpl.rcParams['axes.unicode_minus'] = False 

Time = {}
Stock_Price = {}
Bond_Price = {}

# Bond name : 大族转债

Time['now'] = pd.to_datetime('2018-07-26')
Time['start'] = pd.to_datetime('2017-03-17')
Time['end'] = pd.to_datetime('2023-03-17')

Stock_Price['resale'] = 103
Stock_Price['redeem'] = 104
Stock_Price['now'] = 49.98
Stock_Price['riskfree'] = 0.015
Stock_Price['volatility'] = 0.3

Bond_Price['resale'] = 36.75
Bond_Price['redeem'] = 68.25
Bond_Price['strike'] = 52.50
Bond_Price['facevalue'] = 100

cp = np.array([0.002,0.004,0.008,0.012,0.016,0.02])
Bond_Coupon = pd.Series(cp,index=pd.date_range(Time['start'],Time['end'],freq='365D',closed='right'))

def RemainTime(Time,Bond_Coupon):
    now = Time['now']
    expired = Bond_Coupon.index[-1]
    return str(expired-now)[:-14]


def BondValue(Time,Stock_Price,Bond_Price,Bond_Coupon):
    now = Time['now']
    FV = Bond_Price['facevalue']
    R = Stock_Price['riskfree']
    period = np.float(RemainTime(Time,Bond_Coupon))/365
    coupon = Bond_Coupon[now<Bond_Coupon.index]
    bondvalue = FV/(1+R)**period
    for i in range(len(coupon)):
        p = np.float(RemainTime(Time,Bond_Coupon))/365
        bondvalue = bondvalue + FV * coupon[i]/(1+R)**p
    return bondvalue


def BSM(Time,Stock_Price,Bond_Price):
    FV = Bond_Price['facevalue']
    K = Bond_Price['strike']
    S = Stock_Price['now']
    R = Stock_Price['riskfree']
    sigma = Stock_Price['volatility']
    period = np.float(RemainTime(Time,Bond_Coupon))/365
    d1 = (np.log(S/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))
    d2 = d1 - sigma * np.sqrt(period)
    Call = ((S * st.norm.cdf(d1) - K * np.exp(-R*period)) * st.norm.cdf(d2))*FV/K
    return Call
    
def BSM_Model(Time,Stock_Price,Bond_Price,Bond_Coupon):
    return BSM(Time,Stock_Price,Bond_Price) + BondValue(Time,Stock_Price,Bond_Price,Bond_Coupon)


def MonteCarlo(Time,Stock_Price,Bond_Coupon,paths=5000):
    R = Stock_Price['riskfree']
    sigma = Stock_Price['volatility']
    period = np.int(RemainTime(Time,Bond_Coupon))
    Price_paths = np.zeros[paths,period+1]
    Price_paths[:,0]= Stock_Price['now']
    dt = 1/365
    np.random.seed(1111)
    for t in range(1, period+1):
        z = np.random.standard_normal(paths)
        Price_paths[:,t] = Price_paths[:,t-1] * np.exp((R-0.5*sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return Price_paths



    
    
    