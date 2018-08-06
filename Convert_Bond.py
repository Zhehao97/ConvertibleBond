# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:09:25 2018
@author: Zhehao Li
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import time
from sklearn import linear_model
from pylab import mpl
from time import time
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

 
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
Price['now'] = 43.91
Price['riskfree'] = 0.015
Price['volatility'] = 0.384204

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
<<<<<<< HEAD


=======


>>>>>>> 5a3a108599c1a4a0b8ab04bc8fa92be086c29a1e
def BondValue(T0,Price,Bond_Coupon):
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


def BSM(T0,S0,Price):
    T = Bond_Coupon.index[-1]
    FV = Price['facevalue']
    K = Price['strike']
    R = Price['riskfree']
    sigma = Price['volatility']
    period = RemainTime(T0,T,'float')
    d1 = (np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))
    d2 = (np.log(S0/K) + (R - 0.5*sigma**2) * period)/(sigma * np.sqrt(period))
    Call = (S0 * st.norm.cdf(d1) - K * np.exp(-R*period) * st.norm.cdf(d2))*FV/K
    return Call

    
def BSM_Model(T0,S0,Price,Bond_Coupon):
    return BSM(T0,S0,Price) + BondValue(T0,Price,Bond_Coupon)


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


def Resale(Price,Bond_Coupon,T0,S0):
    T = Bond_Coupon.index[-1]
    period = RemainTime(T0,T,'float')
    FV = Price['facevalue']
    P_resale = Price['resale']
    sigma = Price['volatility']  
    R = Price['riskfree']
    BV = BondValue(T0,Price,Bond_Coupon)
    def okfine(K):
        return (S0 * st.norm.cdf((np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))) - K * np.exp(-R*period) * st.norm.cdf((np.log(S0/K) + (R - 0.5*sigma**2) * period)/(sigma * np.sqrt(period))))*FV/K+(BV-P_resale)
    sol = fsolve(okfine,1)
    return sol
 

def CouponValue(Bond_Coupon,Price,T0,T):
    r = Price['riskfree']
    FV = Price['facevalue']
    temp_Coupon = Bond_Coupon[Bond_Coupon.index <= T]
    temp_Coupon = temp_Coupon[temp_Coupon.index >= T0]
    discounted_value = 0
    for day in temp_Coupon.index:
        period = RemainTime(T0,day,'float')
        discounted_value = discounted_value + FV*temp_Coupon[day]*np.exp(-r*period)
    return discounted_value 


#多元非线性回归
def PolyRegression(X,Y): #X,Y numpy array type
    quadratic = PolynomialFeatures(degree=2)
    X_train = quadratic.fit_transform(X.reshape(-1,1))
    X_test = X_train
    Y_train = Y.reshape(-1,1)
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    Y_test = regressor.predict(X_test)
    return Y_test
<<<<<<< HEAD
=======



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
    Redeem_Value = pd.Series(0,index=Price_Path.index) #store the value of the convert bond along each path
    Expired_Value = pd.DataFrame(0,index=Price_Path.index,columns=Price_Path.columns)
    
    for path in range(Price_Path.shape[0]):
        for step in range(Price_Path.shape[1]-1): #遍历到倒数第二天
            S = Price_Path.iloc[path,step]
            if S <= trig_resale:
                #K[path] = Resale(Price=Price,Bond_Coupon=Bond_Coupon,T0=step,S0=S)
                continue
            elif S >= trig_redeem:
                day = Price_Path.columns[step]
                period = RemainTime(now,day,'float')
                strike_value = S * Price['facevalue']/K[path] * np.exp(-r*period) #discounted value
                coupon_value = CouponValue(Bond_Coupon,Price,now,day) #Return discounted value
                Redeem_Value[path] = strike_value+coupon_value
                break
        if Redeem_Value[path] == 0:
            stock_value = Price_Path.iloc[path,-1]*FV/K[path]
            bond_value = FV*(1+coupon_end)
            Expired_Value.iloc[path,-1] = max(stock_value,bond_value)
    Redeem_Value = Redeem_Value[Redeem_Value>0]
    Expired_K = K[Expired_Value.iloc[:,-1]>0]
    Expired_Price = Price_Path[Expired_Value.iloc[:,-1]>0]
    Expired_Value = Expired_Value[Expired_Value.iloc[:,-1]>0]
    
    temp_K = Expired_K.values.reshape(-1,1)
    # 反向传播算法
    for j in range(1,Price_Path.shape[1]):
        temp_y = Expired_Value.iloc[:,-j].values * np.exp(-r/365) #向前一天贴现
        temp_x = Expired_Price.iloc[:,-(j+1)].values
        temp_y = temp_y.reshape(-1,1)
        temp_x = temp_x.reshape(-1,1)
        predict_y = PolyRegression(temp_x,temp_y) #非线性回归后得到的预测持有价值
        
        assert temp_x.shape == s temp_K.shape
        temp_convert = temp_x*FV/temp_K
        # 逐个元素比较持有期权的预测现价 与 转股价值的大小
        predict_y = predict_y.reshape(-1,1)
        temp_convert = temp_convert.reshape(-1,1)
        
        temp = np.zeros(predict_y.shape)
        temp[predict_y>temp_convert] = predict_y[predict_y>temp_convert]
        temp[predict_y<temp_convert] = temp_convert[predict_y<temp_convert] 
        
        Expired_Value.iloc[:,-(j+1)] = temp
    #计算平均价格
    mean_value = (Redeem_Value.sum() + Expired_Value.iloc[:,0].sum())/Price_Path.shape[0]
    return mean_value


                     
ff = LSM_Model(Time,Price,Bond_Coupon,paths=5000)

print(ff)
    
>>>>>>> 5a3a108599c1a4a0b8ab04bc8fa92be086c29a1e




def LSM_Model(Time,Price,Bond_Coupon,paths=500):
    
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
    Redeem_Value = pd.Series(0,index=Price_Path.index) #store the value of the convert bond along each path
    Expired_Value = pd.DataFrame(0,index=Price_Path.index,columns=Price_Path.columns)
    
    for path in range(Price_Path.shape[0]):
        for step in range(180,Price_Path.shape[1]): #遍历到倒数第二天,冷却半年
            S = Price_Path.iloc[path,step]
            if S <= trig_resale:
                #K[path] = Resale(Price=Price,Bond_Coupon=Bond_Coupon,T0=step,S0=S)
                continue
            elif S >= trig_redeem:
                day = Price_Path.columns[step]
                period = RemainTime(now,day,'float')
                strike_value = S * Price['facevalue']/K[path] * np.exp(-r*period) #discounted value
                coupon_value = CouponValue(Bond_Coupon,Price,now,day) #Return discounted value
                Redeem_Value[path] = strike_value+coupon_value
                break
        if Redeem_Value[path] == 0:
            stock_value = Price_Path.iloc[path,-1]*FV/K[path]
            bond_value = FV*(1+coupon_end)
            Expired_Value.iloc[path,-1] = max(stock_value,bond_value)
    Redeem_Value = Redeem_Value[Redeem_Value>0]
    Expired_K = K[Expired_Value.iloc[:,-1]>0]
    Expired_Price = Price_Path[Expired_Value.iloc[:,-1]>0]
    Expired_Value = Expired_Value[Expired_Value.iloc[:,-1]>0]
    
    temp_K = Expired_K.values.reshape(-1,1)
    # 反向传播算法
    for j in range(1,Price_Path.shape[1]):
        temp_y = Expired_Value.iloc[:,-j].values * np.exp(-r/365) #向前一天贴现
        temp_x = Expired_Price.iloc[:,-(j+1)].values
        temp_y = temp_y.reshape(-1,1)
        temp_x = temp_x.reshape(-1,1)
        predict_y = PolyRegression(temp_x,temp_y) #非线性回归后得到的预测持有价值
        
        temp_convert = temp_x*FV/temp_K
        # 逐个元素比较持有期权的预测现价 与 转股价值的大小
        predict_y = predict_y.reshape(-1,1)
        temp_convert = temp_convert.reshape(-1,1)
        
        temp = np.zeros(predict_y.shape)
        temp[predict_y>temp_convert] = predict_y[predict_y>temp_convert]
        temp[predict_y<temp_convert] = temp_convert[predict_y<temp_convert] 
        
        Expired_Value.iloc[:,-(j+1)] = temp
    #计算平均价格
    mean_value = (Redeem_Value.sum() + Expired_Value.iloc[:,0].sum())/Price_Path.shape[0]
    return mean_value


                     
ff = LSM_Model(Time,Price,Bond_Coupon,paths=5000)

print(ff)
    
