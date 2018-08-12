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
import scipy.optimize as opt 
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


'''
##################################################################################


cbond_parameter = pd.read_csv('cbond_parameter.csv',encoding='GBK')
cbond_price = pd.read_csv('cbond_price.csv')
stock_price = pd.read_csv('stock_price.csv')
nationbond = pd.read_csv('nationbond.csv')


stock_price = stock_price.set_index('Unnamed: 0')
nationbond = nationbond.set_index('Unnamed: 0')
cbond_price = cbond_price.set_index('Unnamed: 0')
cbond_parameter = cbond_parameter.set_index('Unnamed: 0')
cbond_parameter['CLAUSE_PUTOPTION_RESELLINGPRICE']=104




##################################################################################

def ExtractCoupon(cbond_data): #cbond_data 是 Seires类型
    txt = cbond_data['COUPONTXT'][0]
    pattern = re.compile(r'\d*\.*\d*\%') #匹配百分数
    temp = re.findall(pattern, txt)
    coupon = [float(a.strip('%'))/100 for a in temp] #转换为浮点小数
    coupon = np.array(coupon)
    start_date = pd.to_datetime(cbond_data['CARRYDATE'][0])
    end_date = pd.to_datetime(cbond_data['MATURITYDATE'][0])
    bond_coupon = pd.Series(coupon, index=pd.date_range(start_date,end_date,freq='365D',closed='right'))
    return bond_coupon


def BasicParameters(cbond_data):
    bond_coupon = ExtractCoupon(cbond_data)    
    Time = {}
    Time['start'] = pd.to_datetime(cbond_data['CARRYDATE'][0]) #起息日期
    Time['end'] = pd.to_datetime(cbond_data['MATURITYDATE'][0]) #摘牌日期
    price = {}
    price['facevalue'] = cbond_data['LATESTPAR'][0] #面值
    price['strike'] = cbond_data['CLAUSE_CONVERSION2_SWAPSHAREPRICE'][0]#转股价格
    price['resale_trigger'] = cbond_data['CLAUSE_PUTOPTION_REDEEM_TRIGGERPROPORTION'][0] * price['strike']/100 #回售触发价格
    price['redeem_trigger'] = cbond_data['CLAUSE_CALLOPTION_TRIGGERPROPORTION'][0] * price['strike']/100 #赎回触发价格
    price['resale'] = cbond_data['CLAUSE_PUTOPTION_RESELLINGPRICE'][0] #回售价格
    number = {}
    number['resale'] = cbond_data['CLAUSE_PUTOPTION_PUTBACKTRIGGERSPAN'][0] #回售触发天数限制
    number['redeem'] = cbond_data['CLAUSE_CALLOPTION_REDEEMSPAN'][0] #赎回触发天数限制
    return bond_coupon,Time,price,number
    

###############################################################################


def StockVolatility(stock_data,period=20): #计算每天正股波动率
    #股票日涨跌幅, stock_data 是dataframe格式
    stock_return = stock_data.iloc[:,1:]/stock_data.shift(periods=1,axis=1).iloc[:,1:] - 1
    stock_return = stock_return.T
    #EWMA模型计算指数移动加权平均值
    Avg = stock_return.ewm(span = period).mean()
    Std = stock_return.ewm(span = period).std()
    Avg = Avg*np.sqrt(250)
    Std = Std*np.sqrt(250)
    EWMA = {'AVG':Avg,'STD':Std}
    return EWMA



def MarkToMarket(nationbond,stock_price, cbond_price,Time): #跟踪转债上市后ß每日数据
    #处理正股价格数据
    stock_data = stock_price.T # stock_price （历史两年数据）
    stock_data = stock_data.reset_index()
    stock_data.columns = ['TRADE_DT','PRICE']
    stock_data['TRADE_DT'] = pd.DatetimeIndex(stock_data['TRADE_DT'])
    #计算TTM年化收益率期望和波动率
    EWMA = StockVolatility(stock_price)
    #计算收益率期望
    expectation_data = EWMA['AVG']
    expectation_data = expectation_data.reset_index()
    expectation_data.columns = ['TRADE_DT','EXPECTATION']
    expectation_data['TRADE_DT'] = pd.DatetimeIndex(expectation_data['TRADE_DT'])
    #计算收益率波动率
    volatility_data = EWMA['STD']
    volatility_data = volatility_data.reset_index()
    volatility_data.columns = ['TRADE_DT','VOLATILITY']
    volatility_data['TRADE_DT'] = pd.DatetimeIndex(volatility_data['TRADE_DT'])
    #处理六年期国债数据
    nationbond_data = nationbond.T
    nationbond_data = nationbond_data.reset_index()
    nationbond_data.columns = ['TRADE_DT','INTEREST']
    nationbond_data['TRADE_DT'] = pd.DatetimeIndex(nationbond_data['TRADE_DT'])
    #处理债券价格是数据
    cbond_data = cbond_price.T
    cbond_data = cbond_data.reset_index()
    cbond_data.columns = ['TRADE_DT','CBOND_PRICE']
    cbond_data['TRADE_DT'] = pd.DatetimeIndex(cbond_data['TRADE_DT'])
    #波动率数据,正股价格数据,转债价格,国债利率/无风险利率数据合并合并
    update_data = pd.merge(expectation_data,volatility_data,how='left',on='TRADE_DT')
    update_data = pd.merge(update_data, stock_data, how='left', on='TRADE_DT')
    update_data = pd.merge(update_data, nationbond_data, how='left', on='TRADE_DT')
    update_data = pd.merge(cbond_data, update_data, how='left', on='TRADE_DT')
    return update_data   




###############################################################################

# 计算每日delta 

def Coefficient(LSM,BSM):
    coef_list = np.array(LSM)/np.array(BSM)
    coef = np.mean(coef_list)
    return coef



def ComputeDelta(LSM_price,BSM_price,Time,Price):
    T0 = Time['now']
    T = Time['end']
    period = np.float(str(T-T0)[:-14])/365
    R = Price['riskfree']
    S0 = Price['now']
    K = Price['strike']
    sigma = Price['volatility']
    #确定LSM模型和BSM模型之间的比例系数
    coef = Coefficient(LSM_price,BSM_price)
    d1 = (np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))
    delta = st.norm.cdf(d1) * coef
    return delta


def ComputeSigma(LSM_price,BSM_price,Price,Time,CBond_Price,Bond_Value,pre_sigma):

    T0 = Time['now']
    T = Bond_Coupon.index[-1]
    S0 = Price['now']
    FV = Price['facevalue']
    K = Price['strike']
    R = Price['riskfree']
    BV = Bond_Value
    CBP = CBond_Price 
    period = np.float(str(T-T0)[:-14])/365
    #确定LSM模型和BSM模型之间的比例系数
    coef = Coefficient(LSM_price,BSM_price)
    print(coef)
    def okfine(sigma):
        return (S0 * st.norm.cdf((np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))) - K * np.exp(-R*period) * st.norm.cdf((np.log(S0/K) + (R - 0.5*sigma**2) * period)/(sigma * np.sqrt(period))))* FV/K * coef + BV - CBP
    np.random.seed(np.int(pre_sigma))
    sol = opt.root(okfine,np.random.rand(1)/2)
    return sol.x



##################################################################################



def ExecuteModel(Time,Price,Bond_Coupon,Number,everyday_data):
    
    LSM_Price = []
    BSM_Price = []
    BOND_Price = []
    Delta = []
    Sigma = [0.079] #随机初始化隐含波动率
    for d in range(308,616):
        Time['now'] = everyday_data.index[d]
        Price['volatility'] = everyday_data['VOLATILITY'][d]
        Price['now'] = everyday_data['PRICE'][d]
        Price['riskfree'] = everyday_data['INTEREST'][d]/100
        Price['mean'] = everyday_data['EXPECTATION'][d]
        
        #调用CBond类计算模型定价
        obj = CBond(Time,Price,Bond_Coupon,Number,5000)
        value = obj.Summary()
        LSM_Price.append(value['LSM定价:'])
        BSM_Price.append(value['BSM定价:'])
        BOND_Price.append(value['债券价值:'])
        
        #更新每日delta值
        delta = ComputeDelta(LSM_Price,BSM_Price,Time,Price)
        Delta.append(delta)
        
        #更新每日隐含波动率sigma
        CBond_Price = everyday_data['CBOND_PRICE'][d]
        sigma = ComputeSigma(LSM_Price,BSM_Price,Price,Time,CBond_Price,value['债券价值:'],Sigma[-1])
        Sigma.append(sigma)
        
        
    LSM_Price = np.array(LSM_Price)
    BSM_Price = np.array(BSM_Price)
    BOND_Price = np.array(BOND_Price)
    Delta = np.array(Delta)
    Sigma = np.array(Sigma[1:]) #舍弃第一个初始化波动率
    
    MODEL = pd.DataFrame(data=LSM_Price,index=everyday_data.index[308:616])
    MODEL.columns = ['LSM_PRICE']
    MODEL['BSM_PRICE'] = BSM_Price
    MODEL['BOND_PRICE'] = BOND_Price
    MODEL['DELTA'] = Delta
    MODEL['SIGMA']  = Sigma
    
    return MODEL



##################################################################################




#全局静态变量(因变量)
Bond_Coupon,Time,Price,Number = BasicParameters(cbond_parameter)


#全局动态变量(因变量)
everyday_data = MarkToMarket(nationbond,stock_price,cbond_price,Time)#转债上市以来的正股价格和波动率
everyday_data = everyday_data.set_index('TRADE_DT')
everyday_data = everyday_data.dropna()

everyday_data


start = datetime.datetime.now()
model = ExecuteModel(Time,Price,Bond_Coupon,Number,everyday_data)
end = datetime.datetime.now()
print(end-start)


model.to_csv('data4.csv')


##################################################################################

