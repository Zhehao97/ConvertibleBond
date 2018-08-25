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
    pattern = re.compile(r'\d+\.*\d+\%') #匹配百分数
    temp = re.findall(pattern, txt)
    new_temp = list(set(temp)) #去除重复元素
    new_temp.sort(key=temp.index) #保持原有排序
    coupon = [float(a.strip('%'))/100 for a in new_temp] #转换为浮点小数
    coupon = np.array(coupon)
    start_date = pd.to_datetime(cbond_data['CARRYDATE'])
    end_date = pd.to_datetime(cbond_data['MATURITYDATE'])
    date_index = pd.date_range(start_date,end_date,freq='365D',closed='right')
    if len(date_index) > len(coupon) > 0:
        bond_coupon = pd.Series(coupon[-1], index=date_index)
    elif len(date_index) == len(coupon):
        bond_coupon = pd.Series(coupon, index=date_index)
    else:
        bond_coupon = pd.Series(0.008, index=date_index)
    return bond_coupon


def BasicParameters(cbond_data,strike):
    bond_coupon = ExtractCoupon(cbond_data)    
    Time = {}
    Time['start'] = pd.to_datetime(cbond_data['CARRYDATE']) #起息日期
    Time['end'] = pd.to_datetime(cbond_data['MATURITYDATE']) #摘牌日期
    price = {}
    price['strike'] = strike 
    price['facevalue'] = cbond_data['LATESTPAR'] #面值
    price['resale_trigger'] = cbond_data['CLAUSE_PUTOPTION_REDEEM_TRIGGERPROPORTION'] * price['strike']/100 #回售触发价格
    price['redeem_trigger'] = cbond_data['CLAUSE_CALLOPTION_TRIGGERPROPORTION'] * price['strike']/100 #赎回触发价格
    price['resale'] = cbond_data['CLAUSE_PUTOPTION_RESELLINGPRICE'] #回售价格
    #转股价格
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


def ComputeDelta(lsm_price,bsm_price,Time,Price,lsm=True):
    T0 = Time['now']
    T = Time['end']
    period = np.float(str(T-T0)[:-14])/365
    R = Price['riskfree']
    S0 = Price['now']
    K = Price['strike']
    sigma = Price['volatility']
    d1 = (np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))
    if lsm == True:
        coef = lsm_price/bsm_price
    elif lsm == False:
        coef = 1        
    delta = st.norm.cdf(d1)*coef
    return delta


def ComputeSigma(lsm_price,bsm_price,bond_price,cb_price,Price,Time,lsm=True):
    T0 = Time['now']
    T = Time['end']
    S0 = Price['now']
    FV = Price['facevalue']
    K = Price['strike']
    R = Price['riskfree']
    BV = bond_price
    CBP = cb_price 
    period = np.float(str(T-T0)[:-14])/365
    if lsm == True:
        coef = lsm_price/bsm_price
    elif lsm == False:
        coef = 1  
    def okfine(sigma):
        return (S0 * st.norm.cdf((np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))) - K * np.exp(-R*period) * st.norm.cdf((np.log(S0/K) + (R - 0.5*sigma**2) * period)/(sigma * np.sqrt(period))))* FV/K * coef + BV - CBP
    np.random.seed(2222)
    sol = opt.root(okfine,np.random.rand(1))
    return np.float(sol.x)

##################################################################################

def RunModel(cbond_par,stock_data,nbond_data,now,c_price,strike,lsm=True):
    #导入基本参数
    Bond_Coupon,Time,Price,Number = BasicParameters(cbond_par,strike)
    #更新每日参数
    Time['now'] = now
    Price['volatility'] = StockVolatility(stock_data,period=20) #回滚以前的数据
    Price['now'] = stock_data[now]
    Price['riskfree'] = nbond_data[now]/100
    #调用CBond类计算模型定价
    obj = CBond(Time,Price,Bond_Coupon,Number,1000)
    #######################
    value = obj.Summary(lsm)
    bsm_price = value['BSM定价:']
    bond_price = value['债券价值:']    
    if lsm == True:
        lsm_price = value['LSM定价:']
    else:
        lsm_price = 0
    #更新每日delta值
    delta = ComputeDelta(lsm_price,bsm_price,Time,Price,lsm)        
    #更新每日隐含波动率sigma
    sigma = ComputeSigma(lsm_price,bsm_price,bond_price,c_price,Price,Time,lsm)
    Par = {'delta':delta,'sigma':sigma,'volatility':Price['volatility']}
    return Par,Price
 
##################################################################################

def ComputePosition(capital,delta,c_price,s_price,coef):
    x = capital/(c_price - s_price*delta*coef)
    c_position = x 
    s_position = -x*delta*coef
    return [c_position, s_position]


##################################################################################


def Main(cbond_parameter,stock_data,cbond_data,strike_data,nbond_data,lsm):
    max_steps = cbond_data.shape[1] #时间
    process_bar = Process(max_steps)
    
    diff = stock_data.shape[0]-cbond_data.shape[0]    #数据表的时间差
    Signal = np.zeros(cbond_parameter.shape[1]) #建仓/平仓信号
    Count = np.zeros(cbond_parameter.shape[1])
    #可转债和正股持仓仓位
    #c_position = np.zeros(cbond_data.shape[0])
    #s_position = np.zeros(cbond_data.shape[0])
    #s_principle = np.zeros(cbond_data.shape[0])
    c_proportion = np.zeros(cbond_parameter.shape[1])
    s_proportion = np.zeros(cbond_parameter.shape[1])
    #总资本
    #Capital = np.ones(cbond_data.shape[0])*1000 #10万本金
    #Cash = np.zeros(cbond_data.shape[0])
    #资产净值
    #Value = pd.DataFrame(index=cbond_data.index, columns=cbond_data.columns)
    c_Return = pd.DataFrame(index=cbond_data.index, columns=cbond_data.columns)
    s_Return = pd.DataFrame(index=cbond_data.index, columns=cbond_data.columns)
    in_date = {}
     
    for s in range(cbond_data.shape[1]): #列是转债代码
        process_bar.show_process()
        time.sleep(0.01)
        for d in range(1,cbond_data.shape[0]-1): #行是日期

            #初始化参数
            name = cbond_data.columns[s] #转债名称
            now = cbond_data.index[d] #日期
            c_price = cbond_data[name][d] #转债当日价格cc
            
            if (c_price == 100 or pd.isna(c_price) or pd.isna(cbond_data[name][d-1]) or cbond_data[name][d-1]==0):
                continue
            else:
                #s_price = stock_data[cbond_data.columns[d]][s] #正股当日价格
                strike = strike_data[name][d]
                if strike == 0:
                    continue
                else:
                    cbond_par = cbond_parameter[name] #转债条款
                    stock_history = stock_data[name][:d+diff+1] #正股历史数据
                    MA120 = stock_history.ewm(span=120).mean()
                    MA60 = stock_history.ewm(span=60).mean()
                    
                    #运行定价模型计算delta和sigma
                    Par,Price = RunModel(cbond_par,stock_history,nbond_data,now,c_price,strike,lsm)
                    
                    c_return = cbond_data[name][d]/cbond_data[name][d-1]-1
                    s_return = stock_data[name][d]/stock_data[name][d-1]-1
                    #跟踪信号        
                    delta = Par['delta']
                    sigma_diff = Par['volatility']-Par['sigma']
                    FVK = Price['facevalue']/Price['strike']
                    
                    if Signal[s] == 0:
                        if (sigma_diff >= 0.2 and MA60.values[-1]<MA120.values[-1]): #建仓
                            in_date[name] = now
                            Signal[s] = 1 #将信号调整为持仓
                            Count[s] = 1 #计数器
                            #设置仓位
                            c_proportion[s] = 1
                            s_proportion[s] = c_proportion[s] * FVK * delta
                            #c_position[s],s_position[s] = ComputePosition(Capital[s],delta,c_price,s_price,FVK)
                            #计算资本净值
                            #Value.iloc[s,d] = c_price*c_position[s] + s_price*s_position[s]
                            #计算融券本金
                            #s_principle[s] = s_price*s_position[s]
                            continue
                        else:
                            continue #保留昨日收益
                    
                    elif Signal[s] == 1:
                        if sigma_diff <= 0.1: #平仓
                            c_Return[name][d] = c_return * c_proportion[s] / (c_proportion[s] + s_proportion[s])
                            s_Return[name][d] = - s_return * s_proportion[s] / (c_proportion[s] + s_proportion[s])
                            #period = np.float(str(now-in_date[name])[:-14]) #建仓时长
                            #s_coupon = s_principle[s]*(0.08/365)*period  #融券利息
                            #平仓日资本净值
                            #Value.iloc[s,d] = c_price*c_position[s] + s_price*s_position[s] + s_coupon
                            #Capital[s] = Value.iloc[s,d] #将总资本更新为平仓后资本
                            Signal[s] = 0 #将信号设置为空仓
                            Count[s] = 0
                            #c_position[s],s_position[s] = 0
                            c_proportion[s] = 0
                            s_proportion[s] = 0 #将仓位归零
                            continue
                        else:
                            c_Return[name][d] = c_return * c_proportion[s] / (c_proportion[s] + s_proportion[s])
                            s_Return[name][d] = - s_return * s_proportion[s] / (c_proportion[s] + s_proportion[s])
                            #period = np.float(str(now-in_date[name])[:-14]) #建仓时长
                            #s_coupon = s_principle[s]*(0.08/365)*period  #融券利息
                            #计算资本净值
                            #Value.iloc[s,d] = c_price*c_position[s] + s_price*s_position[s] + s_coupon
                            Count[s] += 1 #记录持仓天数
                            if Count[s] == 20: #根据最新delta值调整持仓比例
                                #c_position[s],s_position[s] = ComputePosition(Capital[s],delta,c_price,s_price,FVK)
                                c_proportion[s] = 1
                                s_proportion[s] = c_proportion[s] * FVK * delta
    return c_Return,s_Return #Value

##################################################################################


def getMaxDownList(datax):
    maxdownlist=[]
    for i in range(0, len(datax)):
        temp = (max(datax[:i+1]) - datax[i])/max(datax[:i+1])
        maxdownlist.append(temp)
    return max(maxdownlist)


def Performance(return_data):
    return_avg = return_data.mean(axis=1)
    return_avg = return_avg.fillna(0)
    return_avg = return_avg[return_avg<1]

    return_std = return_avg.std()*np.sqrt(250)
    
    temp_return = return_avg+1
    return_cum = temp_return.cumprod()
    return_cum = return_cum/return_cum[0]
    
    periods = np.float(str(return_cum.index[-1] - return_cum.index[0])[:-14])
    earning = (return_cum[-1]-1)*365/periods
    sharpe_ratio = (earning - 0.03)/return_std
    maxdown = getMaxDownList(return_cum.values)
    
    performance = {'年化收益':earning,'年波动率':return_std,'夏普比率':sharpe_ratio,'最大回撤':maxdown}
    data = {'日平均收益':return_avg,'累计收益':return_cum}
    return performance,data


def PerformancePlot(return_avg,return_cum,index_1,index_2,label_name,file_name):
    b = return_cum[0]
    c = index_1.values[0]
    d = index_2.values[0]
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.grid(False)
    ax1.bar(return_avg.index, return_avg, width=2,linewidth=2,color='yellowgreen',label='日均收益',zorder=1)
    ax1.set_ylabel('日均收益')
    ax1.set_ylim(-0.2,0.2)
    ax1.legend(loc='upper right')
    
    ax2 = ax1.twinx()
    ax2.grid(True)
    ax2.plot(index_1.index, index_1, linewidth=1,label='上证综指',zorder=5)
    ax2.plot(index_2.index, index_2*c/d, linewidth=1,label='深证成指',zorder=6)
    ax2.plot(return_cum.index,return_cum*c/b,color='purple',linewidth=1.5,label=label_name,zorder=7)
    ax2.set_ylabel('指数')
    #ax2.set_ylim(0,10000)
    ax2.legend(loc='upper left')
    ax2.set_xlabel('时间')
    plt.savefig(file_name+'.jpg',dpi=1000)

##################################################################################


