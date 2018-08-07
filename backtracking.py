# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:36:43 2018

@author: Zhehao Li
"""
import re
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

##################################################################################

from WindPy import *
#导入数据
w.start()

cbond_par = w.wss("128035.SZ", "latestpar,carrydate,maturitydate,coupontxt,clause_conversion_code,clause_conversion2_swapshareprice,clause_calloption_redeemspan,clause_calloption_triggerproportion,clause_putoption_putbacktriggerspan,clause_putoption_redeem_triggerproportion,clause_putoption_resellingprice","tradeDate=20180806")
stock_price = w.wsd("002008.SZ", "close", "2016-01-01", "2018-08-07", "PriceAdj=F")

w.close()

##################################################################################

#处理导入的数据
cbond = pd.DataFrame(data=cbond_par.Data, index=cbond_par.Fields, columns=cbond_par.Codes)
cbond = cbond.T
cbond = cbond.iloc[0,:] # 转换为Series
stock = pd.DataFrame(data=stock_price.Data, index=stock_price.Codes, columns=stock_price.Times)

##################################################################################

def ExtractCoupon(cbond_data): #cbond_data 是 Seires类型
    txt = cbond_data['COUPONTXT']
    pattern = re.compile(r'\d*\.*\d*\%') #匹配百分数
    temp = re.findall(pattern, txt)
    coupon = [float(a.strip('%'))/100 for a in temp] #转换为浮点小数
    coupon = np.array(coupon)
    start_date = pd.to_datetime(cbond_data['CARRYDATE'])
    end_date = pd.to_datetime(cbond_data['MATURITYDATE'])
    bond_coupon = pd.Series(coupon, index=pd.date_range(start_date,end_date,freq='365D',closed='right'))
    return bond_coupon


def BasicParameters(cbond):
    bond_coupon = ExtractCoupon(cbond)    
    time = {}
    time['start'] = pd.to_datetime(cbond['CARRYDATE']) #起息日期
    time['end'] = pd.to_datetime(cbond['MATURITYDATE']) #摘牌日期
    price = {}
    price['facevalue'] = cbond['LATESTPAR'] #面值
    price['strike'] = cbond['CLAUSE_CONVERSION2_SWAPSHAREPRICE'] #转股价格
    price['resale_trig'] = cbond['CLAUSE_PUTOPTION_REDEEM_TRIGGERPROPORTION'] * price['strike']/100 #回售触发价格
    price['redeem_trig'] = cbond['CLAUSE_CALLOPTION_TRIGGERPROPORTION'] * price['strike']/100 #赎回触发价格
    price['resale'] = cbond['CLAUSE_PUTOPTION_RESELLINGPRICE'] #回售价格
    number = {}
    number['resale'] = cbond['CLAUSE_PUTOPTION_PUTBACKTRIGGERSPAN'] #回售触发天数限制
    number['redeem'] = cbond['CLAUSE_CALLOPTION_REDEEMSPAN'] #赎回触发天数限制
    return bond_coupon,time,price,number
    


def MarkToMarket(stock,Time):    
    #股票日涨跌幅
    stock_change = stock.iloc[:,1:]/stock.shift(periods=1,axis=1).iloc[:,1:] - 1
    #转债起息日起的时间序列
    Datetime = pd.date_range(start=Time['start'],end=stock_change.columns[-1],freq='D')
    #计算TTM年化波动率
    volatility = []
    for i in range(1,Datetime.shape[0]+1):
        temp_change = np.array(stock_change.iloc[:,-(252+i):-i])
        temp_std = np.std(temp_change)
        volatility.append(temp_std*np.sqrt(250))
    volatility = np.array(volatility)    
    #将波动率制成表格
    volatility_data = pd.DataFrame(data=volatility,index=Datetime)
    volatility_data = volatility_data.reset_index()
    volatility_data.columns = ['TRADE_DT','SIGMA']
    #处理正股价格数据
    stock_data = stock.T
    stock_data = stock_data.reset_index()
    stock_data.columns = ['TRADE_DT','PRICE']
    stock_data['TRADE_DT'] = pd.DatetimeIndex(stock_data['TRADE_DT'])
    #波动率数据和正股价格数据合并
    update_data = pd.merge(volatility_data, stock_data, how='left', on='TRADE_DT')
    update_data = update_data.dropna()
    return update_data


#全局静态变量
Bond_Coupon,Time,Price,Number = BasicParameters(cbond)

#全局动态变量
everyday_data = MarkToMarket(stock,Time)



'''
Time['now'] = 
Price['now'] = 
Price['volatility'] = 
'''




