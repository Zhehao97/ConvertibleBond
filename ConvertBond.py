#!/usr/bin/env python3
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
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


class CBond:
    def __init__(self, Time, Price, Bond_Coupon, Number, path):
        self.Time = Time
        self.Price = Price
        self.Bond_Coupon = Bond_Coupon
        self.path = path
        self.Number = Number
    
    def RemainTime(self,T0,T,datatype):
        if datatype == 'int':
            return np.int(str(T-T0)[:-14])
        elif datatype == 'float':
            return np.float(str(T-T0)[:-14])/365
    
    
    def BondValue(self):
        T0 = self.Time['now']
        T = self.Bond_Coupon.index[-1]
        FV = self.Price['facevalue']
        R = self.Price['riskfree']
        period = self.RemainTime(T0,T,'float')
        coupon = self.Bond_Coupon[T0<self.Bond_Coupon.index]
        bondvalue = FV*np.exp(-R*period)
        for i in range(len(coupon)):
            t = coupon.index[i]
            p = self.RemainTime(T0,t,'float')
            bondvalue = bondvalue + FV*coupon[i]*np.exp(-R*p)
        return bondvalue
    
    
    def BSM(self):
        T0 = self.Time['now']
        T = self.Bond_Coupon.index[-1]
        S0 = self.Price['now']
        FV = self.Price['facevalue']
        K = self.Price['strike']
        R = self.Price['riskfree']
        sigma = self.Price['volatility']
        period = self.RemainTime(T0,T,'float')
        d1 = (np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))
        d2 = (np.log(S0/K) + (R - 0.5*sigma**2) * period)/(sigma * np.sqrt(period))
        Call = (S0 * st.norm.cdf(d1) - K * np.exp(-R*period) * st.norm.cdf(d2))*FV/K
        return Call
    
        
    def BSM_Model(self):
        return self.BSM() + self.BondValue()
    
    
    def MonteCarlo(self):
        paths = self.path
        sigma = self.Price['volatility']
        R = self.Price['riskfree']
        T0 = self.Time['now']
        T = self.Bond_Coupon.index[-1]
        period = self.RemainTime(T0,T,'int')
        Price_paths = np.zeros((paths,period+1))
        Price_paths[:,0]= self.Price['now']
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
    
    
    def Resale(self,T0,S0):
        T = self.Bond_Coupon.index[-1]
        period = self.RemainTime(T0,T,'float')
        FV = self.Price['facevalue']
        P_resale = self.Price['resale']
        sigma = self.Price['volatility']  
        R = self.Price['riskfree']
        BV = self.BondValue(T0)
        def okfine(K):
            return (S0 * st.norm.cdf((np.log(S0/K) + (R + 0.5*sigma**2) * period)/(sigma * np.sqrt(period))) - K * np.exp(-R*period) * st.norm.cdf((np.log(S0/K) + (R - 0.5*sigma**2) * period)/(sigma * np.sqrt(period))))*FV/K+(BV-P_resale)
        sol = fsolve(okfine,1)
        return sol
     
    
    def CouponValue(self,T0,T):
        r = self.Price['riskfree']
        FV = self.Price['facevalue']
        if T>=self.Bond_Coupon.index[0]:
            temp_Coupon = self.Bond_Coupon[self.Bond_Coupon.index <= T]
            Coupon = temp_Coupon.values[-1]
        else:
            Coupon = 0
        period = self.RemainTime(T0,T,'float')
        #按照债券面值加当期应计利息的价格赎回
        discounted_value = discounted_value =  FV * (1+Coupon) * np.exp(-r*period)
        return discounted_value 
    
    
    #多元非线性回归
    def PolyRegression(self,X,Y): #X,Y numpy array type
        quadratic = PolynomialFeatures(degree=2)
        X_train = quadratic.fit_transform(X.reshape(-1,1))
        X_test = X_train
        Y_train = Y.reshape(-1,1)
        regressor = LinearRegression()
        regressor.fit(X_train,Y_train)
        Y_test = regressor.predict(X_test)
        return Y_test
    
    

    def LSM_Model(self):
        R = self.Price['riskfree'] # risk free rate
        now = self.Time['now']
        FV = self.Price['facevalue']
        coupon_end = self.Bond_Coupon[-1]
        
        trig_resale = self.Price['resale_trigger'] # resale price trigger 
        trig_redeem = self.Price['redeem_trigger'] # redeem price trigger
        
        if pd.isna(trig_resale): # check if the resale price is negative
            trig_resale = -1000000
            #trig_resale_num = 1000000
            if pd.isna(trig_redeem):
                trig_redeem = 1000000
                trig_redeem_num = 1000000
            else:
                trig_redeem_num = self.Number['redeem']
        #MonteCarlo Simulation
        Price_Path = self.MonteCarlo()
        DateIndex = pd.date_range(start=now,end=self.Bond_Coupon.index[-1],freq='D')
        Price_Path = pd.DataFrame(data=Price_Path,columns=DateIndex)
        
        K = pd.Series(self.Price['strike'],index=Price_Path.index) #store the strike price for each path
        Redeem_Value = pd.Series(0,index=Price_Path.index) #store the value of the convert bond along each path
        Expired_Value = pd.DataFrame(0,index=Price_Path.index,columns=Price_Path.columns)
        
        for path in range(Price_Path.shape[0]):
            resale_count = 0 # count the number of resale triggers
            redeem_count = 0 # count the number of resale triggers
            for step in range(180,Price_Path.shape[1]-1):
                S = Price_Path.iloc[path,step]
                if S <= trig_resale:
                    resale_count = resale_count + 1
                    '''
                    if resale_count >= trig_resale_num:
                        K[path] = Resale(Price=Price,Bond_Coupon=Bond_Coupon,T0=step,S0=S)
                        continue
                    '''
                elif S >= trig_redeem:
                    redeem_count = redeem_count + 1
                    if redeem_count >= trig_redeem_num:
                        T = Price_Path.columns[step]
                        period = self.RemainTime(now,T,'float')
                        strike_value = S * self.Price['facevalue']/K[path] * np.exp(-R*period) #discounted value
                        coupon_value = self.CouponValue(now,T) #Return discounted value
                        Redeem_Value[path] = max(strike_value,coupon_value)
                        break
            if Redeem_Value[path] == 0:
                stock_value = Price_Path.iloc[path,-1]*FV/K[path]
                bond_value = FV*(1+coupon_end)
                Expired_Value.iloc[path,-1] = max(stock_value,bond_value)
                
        Redeem_Value = Redeem_Value[Redeem_Value>0]
        Expired_K = K[Expired_Value.iloc[:,-1]>0]
        Expired_Price = Price_Path[Expired_Value.iloc[:,-1]>0] # stock price path
        Expired_Value = Expired_Value[Expired_Value.iloc[:,-1]>0] # convertible bond price path
        
        temp_K = Expired_K.values.reshape(-1,1)
        
        # backward proporagtion 
        if Expired_Value.iloc[:,-1].count() != 0:
            for j in range(1,Price_Path.shape[1]):
                temp_y = Expired_Value.iloc[:,-j].values * np.exp(-R/365) # discounting
                temp_x = Expired_Price.iloc[:,-(j+1)].values
                temp_y = temp_y.reshape(-1,1)
                temp_x = temp_x.reshape(-1,1)
                predict_y = self.PolyRegression(temp_x,temp_y) # non-linear regression to predict the price
                
                temp_convert = temp_x*FV/temp_K
                predict_y = predict_y.reshape(-1,1)
                temp_convert = temp_convert.reshape(-1,1)
                
                temp = np.zeros(predict_y.shape)
                temp[predict_y>temp_convert] = predict_y[predict_y>temp_convert]
                temp[predict_y<temp_convert] = temp_convert[predict_y<temp_convert] 
                
                Expired_Value.iloc[:,-(j+1)] = temp
            # redeem_percent = Redeem_Value.count()/Price_Path.shape[0]
            # calclulate mean price
            mean_value = (Redeem_Value.sum() + Expired_Value.iloc[:,0].sum())/Price_Path.shape[0]
        else:
            mean_value = Redeem_Value.sum()/Redeem_Value.count()
            
        #cbond = {'赎回概率':redeem_percent,'定价':mean_value}
        return mean_value
    
    def Summary(self,lsm=True):
        BSM_price = self.BSM_Model()
        bond_value = self.BondValue()
        if lsm == True:
            LSM_price = self.LSM_Model()
            paramerter = {'BSM定价:':BSM_price,'债券价值:':bond_value,'LSM定价:':LSM_price}
        elif lsm == False:
            paramerter = {'BSM定价:':BSM_price,'债券价值:':bond_value}
        return paramerter



