# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:36:43 2018

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

from ShowProcess import Process
from ConvertBond import CBond


##################################################################################


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
    price['strike'] = cbond_data['CLAUSE_CONVERSION2_SWAPSHAREPRICE']#转股价格
    price['resale_trigger'] = cbond_data['CLAUSE_PUTOPTION_REDEEM_TRIGGERPROPORTION'] * price['strike']/100 #回售触发价格
    price['redeem_trigger'] = cbond_data['CLAUSE_CALLOPTION_TRIGGERPROPORTION'] * price['strike']/100 #赎回触发价格
    price['resale'] = cbond_data['CLAUSE_PUTOPTION_RESELLINGPRICE'] #回售价格
    number = {}
    number['resale'] = cbond_data['CLAUSE_PUTOPTION_PUTBACKTRIGGERSPAN'] #回售触发天数限制
    number['redeem'] = cbond_data['CLAUSE_CALLOPTION_REDEEMSPAN'] #赎回触发天数限制
    return bond_coupon,Time,price,number
    


###############################################################################


def StockVolatility(stock_data,period=20): #计算每天正股波动率
    #股票日涨跌幅, stock_data 是dataframe格式
    stock_return = stock_data/stock_data.shift(periods=1) - 1
    #EWMA模型计算指数移动加权平均值
    Std = stock_return.ewm(span = period).std()
    std = Std[-1]*np.sqrt(250)
    return std


###############################################################################

 



def ComputeDelta(lsm_price,bsm_price,Time,Price):
    T0 = Time['now']
    T = Time['end']
    period = np.float(str(T-T0)[:-14])/365
    R = Price['riskfree']
    S0 = Price['now']
    K = Price['strike']
    sigma = Price['volatility']
    #确定LSM模型和BSM模型之间的比例系数
    coef = lsm_price/bsm_price
    d1 = (np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))
    delta = st.norm.cdf(d1) * coef
    return delta


def ComputeSigma(lsm_price,bsm_price,bond_price,cb_price,Price,Time):

    T0 = Time['now']
    T = Time['end']
    S0 = Price['now']
    FV = Price['facevalue']
    K = Price['strike']
    R = Price['riskfree']
    BV = bond_price
    CBP = cb_price 
    period = np.float(str(T-T0)[:-14])/365
    #确定LSM模型和BSM模型之间的比例系数
    coef = lsm_price/bsm_price
    def okfine(sigma):
        return (S0 * st.norm.cdf((np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))) - K * np.exp(-R*period) * st.norm.cdf((np.log(S0/K) + (R - 0.5*sigma**2) * period)/(sigma * np.sqrt(period))))* FV/K * coef + BV - CBP
    np.random.seed(2222)
    sol = opt.root(okfine,np.random.rand(1))
    return np.float(sol.x)



##################################################################################

def RunModel(cbond_par,stock_data,nbond_data,now,c_price):
    #导入基本参数
    Bond_Coupon,Time,Price,Number = BasicParameters(cbond_par)
    #更新每日参数
    Time['now'] = now
    Price['volatility'] = StockVolatility(stock_data,period=20) #回滚以前的数据
    Price['now'] = stock_data[now]
    Price['riskfree'] = nbond_data[now][0]/100
    #调用CBond类计算模型定价
    obj = CBond(Time,Price,Bond_Coupon,Number,1000)
    #######################
    value = obj.Summary()
    lsm_price = value['LSM定价:']
    bsm_price = value['BSM定价:']
    bond_price = value['债券价值:']    
    #更新每日delta值
    delta = ComputeDelta(lsm_price,bsm_price,Time,Price)        
    #更新每日隐含波动率sigma
    sigma = ComputeSigma(lsm_price,bsm_price,bond_price,c_price,Price,Time)
    Par = {'delta':delta,'sigma':sigma,'volatility':Price['volatility']}
    return Par,Price
 
##################################################################################



##################################################################################

def ExecuteModel(cbond_parameter,stock_price,cbond_price,nbond_data,s):
    max_steps = cbond_price.shape[1]
    process_bar = Process(max_steps)
    #数据表的时间差
    diff = stock_price.shape[1]-cbond_price.shape[1]
    #建仓/平仓信号
    Signal = np.zeros(cbond_price.shape[0])
    #可转债和正股持仓仓位
    c_postion = np.zeros(cbond_price.shape[0])
    s_postion = np.zeros(cbond_price.shape[0])
    #总资本&现金
    Capital = np.ones(cbond_price.shape[0])*100000 #10万本金
    Cash = np.zeros(cbond_price.shape[0])
    #资产净值
    Value = pd.DataFrame(index=cbond_price.index, columns=cbond_price.columns)
    in_date = {}
     
    #for s in range(cbond_price.shape[0]):
    for d in range(cbond_price.shape[1]):
        process_bar.show_process()
        time.sleep(0.01)
        #初始化参数
        now = cbond_price.columns[d]
        c_price = cbond_price.iloc[s,d] #转债当日价格
        
        if pd.isna(c_price):
            continue
        else:
            s_price = stock_price[cbond_price.columns[d]][s] #正股当日价格
            name = cbond_price.index[s] #转债名称
            cbond_par = cbond_parameter.iloc[s,:] #转债条款
            stock_data = stock_price.iloc[s,:d+diff+1] #正股历史数据
            
            #运行定价模型计算delta和sigma
            Par,Price = RunModel(cbond_par,stock_data,nbond_data,now,c_price)
            
            #跟踪信号        
            delta = Par['delta']
            sigma = Par['sigma']
            volatility = Par['volatility']
            sigma_diff = volatility-sigma
            FV = Price['facevalue']
            K = Price['strike']
            
            
            if Signal[s] == 0:
                if sigma_diff >= 0.2: #建仓
                    in_date[name] = now
                    Signal[s]=1 #将信号调整为持仓
                    #设置仓位
                    c_postion[s] = 200 
                    s_postion[s] = -200*delta*FV/K
                    #计算现金
                    Cash[s] = Capital[s] - c_price * c_postion[s] - s_price * s_postion[s]
                    #计算资本净值
                    Value.iloc[s,d] = c_price*c_postion[s] + s_price*s_postion[s] + Cash[s]
                    continue
                else:
                    continue #保留昨日收益
            
            elif Signal[s] == 1:
                if sigma_diff <= 0.1: #平仓
                    period = np.float(str(now-in_date[name])[:-14]) #建仓时长
                    s_coupon = s_postion[s]*(0.08/365)*period  #融券利息
                    #平仓日资本净值
                    Value.iloc[s,d] = c_price*c_postion[s] + s_price*s_postion[s] + Cash[s] + s_coupon
                    Capital[s] = Value.iloc[s,d] #将总资本更新为平仓后资本
                    Signal[s] = 0 #将信号设置为空仓
                    #将仓位归零
                    c_postion[s] = 0 
                    s_postion[s] = 0
                    continue
                else: 
                    period = np.float(str(now-in_date[name])[:-14]) #建仓时长
                    s_coupon = s_postion[s]*(0.08/365)*period  #融券利息
                    #计算资本净值
                    Value.iloc[s,d] = c_price*c_postion[s] + s_price*s_postion[s] + Cash[s] + s_coupon

    return Value



##################################################################################

#计算最大回撤
def getMaxDownList(datax):
    data = datax.copy()
    data = list(data)
    maxdownlist = []
    for i in range(0, len(data)):
        temp = (max(data[0:(i+1)]) - data[i])/max(data[0:(i+1)])
        maxdownlist.append(temp)
    return(max(maxdownlist))


#计算策略指标
def getFeature(return_data,risk_free=0.03):
    Return_Rate = return_data.fillna(0)
    #计算年化波动率
    std = Return_Rate.std()*np.sqrt(250)
    #计算累积收益率
    Return = Return_Rate + 1
    Return = Return.cumprod()
    #计算最大回撤
    maxdown = getMaxDownList(Return)
    #计算年化收益率
    span = np.float(str(Return_Rate.index[-1] - Return_Rate.index[0])[:-14])/365
    rate = (Return[-1] - 1)/span
    #计算夏普比率
    sharpe_ratio = (rate - risk_free)/std #0.03六年期国债
    Feature = {'年化收益率:':rate, '夏普比率:':sharpe_ratio, '最大回撤:':maxdown}
    return Feature

##################################################################################


###########################################################################################




