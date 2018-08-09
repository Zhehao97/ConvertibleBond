#!/usr/bin/env python3
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

from Convert_Bond import CBond


##################################################################################
'''
from WindPy import *
#导入数据
w.start()

cbond_par = w.wss("128035.SZ", "latestpar,carrydate,maturitydate,coupontxt,clause_conversion_code,clause_conversion2_swapshareprice,clause_calloption_redeemspan,clause_calloption_triggerproportion,clause_putoption_putbacktriggerspan,clause_putoption_redeem_triggerproportion,clause_putoption_resellingprice","tradeDate=20180806")
stock_price = w.wsd("002008.SZ", "close", "2016-01-01", "2018-08-07", "PriceAdj=F")
sixyear_bond = w.edb("M0057947", "2017-12-01", "2018-08-08","Fill=Previous")
cbond_price = w.wsd("128035.SZ", "close", "2018-02-06", "2018-08-07", "")

w.close()

#处理导入的数据
cbond = pd.DataFrame(data=cbond_par.Data, index=cbond_par.Fields, columns=cbond_par.Codes)
cbond = cbond.T
cbond = cbond.iloc[0,:] # 转换为Series
stock = pd.DataFrame(data=stock_price.Data, index=stock_price.Codes, columns=stock_price.Times)
nationbond = pd.DataFrame(data=sixyear_bond.Data, index=sixyear_bond.Codes, columns=sixyear_bond.Times)
cbond_price = pd.DataFrame(data=cbond_price.Data, index=cbond_price.Codes, columns=cbond_price.Times)
'''
##################################################################################


cbond_parameter = pd.read_pickle('cbond_parameter.pkl')
cbond_price = pd.read_pickle('cbond_price.pkl')
stock_price = pd.read_pickle('stock_price.pkl')
nationbond = pd.read_pickle('nationbond.pkl')


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


def BasicParameters(cbond_data):
    bond_coupon = ExtractCoupon(cbond_data)    
    Time = {}
    Time['start'] = pd.to_datetime(cbond_data['CARRYDATE']) #起息日期
    Time['end'] = pd.to_datetime(cbond_data['MATURITYDATE']) #摘牌日期
    price = {}
    price['facevalue'] = cbond_data['LATESTPAR'] #面值
    price['strike'] = cbond_data['CLAUSE_CONVERSION2_SWAPSHAREPRICE'] #转股价格
    price['resale_trigger'] = cbond_data['CLAUSE_PUTOPTION_REDEEM_TRIGGERPROPORTION'] * price['strike']/100 #回售触发价格
    price['redeem_trigger'] = cbond_data['CLAUSE_CALLOPTION_TRIGGERPROPORTION'] * price['strike']/100 #赎回触发价格
    price['resale'] = cbond_data['CLAUSE_PUTOPTION_RESELLINGPRICE'] #回售价格
    price['riskfree'] = 0.015
    number = {}
    number['resale'] = cbond_data['CLAUSE_PUTOPTION_PUTBACKTRIGGERSPAN'] #回售触发天数限制
    number['redeem'] = cbond_data['CLAUSE_CALLOPTION_REDEEMSPAN'] #赎回触发天数限制
    return bond_coupon,Time,price,number
    

def StockVolatility(stock_data, start_day, period=252): #计算每天正股波动率
    #股票日涨跌幅, stock_data 是dataframe格式
    stock_data.columns = pd.DatetimeIndex(stock_data.columns)
    stock_change = stock_data.iloc[:,1:]/stock_data.shift(periods=1,axis=1).iloc[:,1:] - 1
    #转债起息日起的时间序列
    Datetime = pd.date_range(start=start_day, end=stock_change.columns[-1],freq='D')
    #计算指数移动加权平均值
    EWMA_avg = []
    for i in range(Datetime.shape[0]):
        temp_change = stock_change.T[Datetime[i]>stock_change.columns] #series
        temp_change = temp_change.values #numpy_array
        temp_std = np.std(temp_change)
        volatility.append(temp_std*np.sqrt(period))
    volatility = np.array(volatility)  
    volatility_data = pd.DataFrame(data=volatility,index=Datetime)
    volatility_data = volatility_data.reset_index()
    volatility_data.columns = ['TRADE_DT','SIGMA']
    return volatility_data



def MarkToMarket(nationbond,stock_price,Time):
    #处理正股价格数据
    stock_data = stock_price.T
    stock_data = stock_data.reset_index()
    stock_data.columns = ['TRADE_DT','PRICE']
    stock_data['TRADE_DT'] = pd.DatetimeIndex(stock_data['TRADE_DT'])
    #计算TTM年化波动率
    volatility_data = StockVolatility(stock_price, Time['start'], period=252)
    #处理六年期国债数据
    nationbond_data = nationbond.T
    nationbond_data = nationbond_data.reset_index()
    nationbond_data.columns = ['TRADE_DT','INTEREST']
    nationbond_data['TRADE_DT'] = pd.DatetimeIndex(nationbond_data['TRADE_DT'])
    #波动率数据和正股价格数据合并
    update_data = pd.merge(volatility_data, stock_data, how='left', on='TRADE_DT')
    update_data = pd.merge(update_data, nationbond_data, how='left',on='TRADE_DT')
    update_data = update_data.dropna()
    return update_data   



###############################################################################

# 计算每日delta 

def ComputeDelta(TEST_price,BSM_price,Time,Price):
    T0 = Time['now']
    T = Time['end']
    period = np.float(str(T-T0)[:-14])/365
    R = Price['riskfree']
    S0 = Price['now']
    K = Price['strike']
    sigma = Price['volatility']
    FV = Price['facevalue']
    coef_list = np.array(BSM_price)/np.array(TEST_price)
    coef = np.mean(coef_list)
    d1 = (np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))
    delta = st.norm.cdf(d1) * coef
    return delta


##################################################################################



def ExecuteModel(Time,Price,Bond_Coupon,Number,everyday_data):
    
    LSM_Price = []
    BSM_Price = []
    BOND_Price = []
    Delta = []
    for day in everyday_data.columns[-2:]:
        Time['now'] = day
        Price['volatility'] = everyday_data[day][0]
        Price['now'] = everyday_data[day][1]
        Price['riskfree'] = everyday_data[day][2]/100
        #调用CBond类计算模型定价
        obj = CBond(Time,Price,Bond_Coupon,Number,5000)
        value = obj.Summary()
        LSM_Price.append(value['LSM定价:'])
        BSM_Price.append(value['BSM定价:'])
        BOND_Price.append(value['债券价值:'])
        #更新每日delta值
        delta = ComputeDelta(LSM_Price,BSM_Price,Time,Price)
        Delta.append(delta)
        
    
    LSM_Price = np.array(LSM_Price)
    BSM_Price = np.array(BSM_Price)
    BOND_Price = np.array(BOND_Price)
    Delta = np.array(Delta)
    
    MODEL = pd.DataFrame(data=LSM_Price,index=everyday_data.columns[-2:])
    MODEL.columns = ['LSM_PRICE']
    MODEL['BSM_PRICE'] = BSM_Price
    MODEL['BOND_PRICE'] = BOND_Price
    MODEL['DELTA'] = Delta
    
    return MODEL

##################################################################################






##################################################################################

#全局静态变量(因变量)
Bond_Coupon,Time,Price,Number = BasicParameters(cbond_parameter)

#全局动态变量(因变量)
everyday_data = MarkToMarket(nationbond,stock_price,Time)#转债上市以来的正股价格和波动率
everyday_data = everyday_data.set_index('TRADE_DT')
everyday_data = everyday_data.T


model = ExecuteModel(Time,Price,Bond_Coupon,Number,everyday_data)



