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
from scipy.optimize import fsolve

from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

Time = {}
Price = {}
# Bond name : 大族转债

Time['now'] = pd.to_datetime('2018-08-02')
Time['start'] = pd.to_datetime('2018-02-06')
Time['end'] = pd.to_datetime('2024-02-06')

Price['resale'] = 103
Price['redeem'] = 104
Price['now'] = 48.19
Price['riskfree'] = 0.015
Price['volatility'] = 0.43671

Price['resale_trigger'] = 36.75
Price['redeem_trigger'] = 68.25
Price['strike'] = 52.50
Price['facevalue'] = 100

cp = np.array([0.002,0.004,0.006,0.008,0.016,0.02])
Bond_Coupon = pd.Series(cp,index=pd.date_range(Time['start'],Time['end'],freq='365D',closed='right'))

def RemainTime(T0,T,datatype):
    if datatype == 'int':
        return np.int(str(T-T0)[:-14])
    elif datatype == 'float':
        return np.float(str(T-T0)[:-14])/365

def BondValue(Time,Price,Bond_Coupon):
    T0 = Time['now']
    T = Bond_Coupon.index[-1]
    FV = Price['facevalue']
    R = Price['riskfree']
    period = RemainTime(T0,T,'float')
    coupon = Bond_Coupon[T0<Bond_Coupon.index]
    bondvalue = FV*np.exp(-R*period)
    for i in range(len(coupon)):
        t = coupon.index[i]
        p = RemainTime(T0,t,'float')
        bondvalue = bondvalue + FV*coupon[i]*np.exp(-R*p)
    return bondvalue


def BSM(Time,Price):
    T0 = Time['now']
    T = Bond_Coupon.index[-1]
    FV = Price['facevalue']
    K = Price['strike']
    S = Price['now']
    R = Price['riskfree']
    sigma = Price['volatility']
    period = RemainTime(T0,T,'float')
    d1 = (np.log(S/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))
    d2 = d1 - sigma * np.sqrt(period)
    Call = (S * st.norm.cdf(d1) - K * np.exp(-R*period) * st.norm.cdf(d2))*FV/K
    return Call

    
def BSM_Model(Time,Price,Bond_Coupon):
    return BSM(Time,Price) + BondValue(Time,Price,Bond_Coupon)



def MonteCarlo(Time,Price,Bond_Coupon,paths=5000):
    R = Price['riskfree']
    sigma = Price['volatility']
    T0 = Time['now']
    T = Bond_Coupon.index[-1]
    period = RemainTime(T0,T,'int')
    Price_paths = np.zeros((paths,period+1))
    Price_paths[:,0]= Price['now']
    dt = 1/365
    np.random.seed(1111)
    for t in range(1, period+1):
        z = np.random.standard_normal(paths)
        Price_paths[:,t] = Price_paths[:,t-1] * np.exp((R-0.5*sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return Price_paths

'''
path = MonteCarlo(Time,Price,Bond_Coupon,paths=5000)
plt.figure(figsize=(10,7))
plt.grid(True)
plt.xlabel('Time step')
plt.ylabel('index level')
for i in range(path.shape[1]):
    plt.plot(path[i])
plt.show()
'''

def Resale(Time,Price,Bond_Coupon,S0):
    period = np.float(RemainTime(Time,Bond_Coupon))/365
    FV = Price['facevalue']
    P_resale = Price['resale']
    sigma = Price['volatility']  
    R = Price['riskfree']
    BV = BondValue(Time,Price,Bond_Coupon)
    def okfine(x):
        return ((S0 * st.norm.cdf((np.log(S0/x) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))) - x * np.exp(-R*period)) * st.norm.cdf((np.log(S0/x) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))- sigma * np.sqrt(period)))*FV/x+BV-P_resale
    sol = fsolve(okfine,30)
    return sol
  

def CouponValue(Bond_Coupon,Price,T0,T):
    r = Price['riskfree']
    FV = Price['facevalue']
    temp_Coupon = Bond_Coupon[Bond_Coupon.index <= T]
    temp_Coupon = temp_Coupon[temp_Coupon.index >= T0]
    discounted_value = 0
    for day in temp_Coupon.index:
        period = RemainTime(T0,day)
        discounted_value = discounted_value + FV*temp_Coupon[day]*np.exp(-r*period)
    return discounted_value 

#尚未考虑转股冷却时间 半年
def LSM_Model(Time,Price,Bond_Coupon,paths=5000):
    r = Price['riskfree']
    now = Time['now']
    FV = Price['facevalue']
    coupon_end = Bond_Coupon[-1]
    trig_resale = Price['resale_trigger']
    trig_redeem = Price['redeem_trigger']
    
    DateIndex = pd.date_range(start=now,end=Bond_Coupon.index[-1],freq='D')
    Price_Path = MonteCarlo(Time,Price,Bond_Coupon,paths)
    Price_Path = pd.DataFrame(data=Price_Path,columns=DateIndex)
    
    K = pd.Series(Price['strike'],index=Price_Path.index) #store the strike price for each path
    Path_Value = pd.Series(0,index=Price_Path.index) #store the value of the convert bond along each path
    Expired_Value = pd.Serie(0,index=Price_Path.index)
    
    for path in range(Price_Path.shape[0]):
        for step in range(Price_Path.shape[1]): #Loop in steps
            S = Price_Path.iloc[path,step]
            if S <= trig_resale:
                K[path] = Resale(Time,Price,Bond_Coupon,S)
                continue
            elif S >= trig_redeem:
                day = Price_Path.columns[step]
                period = RemainTime(now,day)
                strike_value = S * Price['facevalue']/K[path] * np.exp(-r*period) #discounted value
                coupon_value = CouponValue(Bond_Coupon,Price,now,day) #Return discounted value
                Path_Value[path] = strike_value+coupon_value
                break
        if Path_Value[path] == 0:
            stock_value = Price_Path.iloc[path,-1]*100/K[path]
            bond_value = FV*(1+coupon_end)
            Expired_Value[path] = max(stock_value,bond_value)
    Path_Value = Path_Value[Path_Value>0]
    Expired_Value = Expired_Value[Expired_Value>0]
    
    # LSM price the option
    
            
            
                                    
                
        
    


