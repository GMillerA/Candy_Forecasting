# -*- coding: utf-8 -*-
"""
Time Series Analysis of Candy Production Kaggle Dataset
Created on Sun Feb 11 11:45:40 2018
Some code sections applied from:
    https://www.kaggle.com/stevebroll/visualization-of-seasonality/notebook
    https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    https://www.kaggle.com/muonneutrino/wikipedia-traffic-data-exploration/notebook
@author: Galen Miller
"""
##Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##Import Data
data = pd.read_csv('candy_production.csv')
print(data.head())
print('\n Data Types:')
print(data.dtypes)

##Create month and year columns by converting timestamps to datetime objects
data['Date'] = pd.to_datetime(data['observation_date'])
data['Month'] = ''
data['Year'] = ''
for i in range(0,len(data)):
    data.loc[i,('Month')] = data['Date'][i].month
    data.loc[i,('Year')] = data['Date'][i].year
data.head()
df = data.set_index('Date')
df = df.drop('observation_date', axis=1)
##Plot initial data
plt.plot(df)
plt.show()

sns.tsplot(data=data, time = 'Year',condition="Month", unit = 'Month', value="IPG3113N")   
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,title = 'Month')
plt.show()

##Eliminating trends
ts_log = np.log(df)
plt.plot(ts_log)
plt.show()

##Checking for stationarity 
##Compare Mean and Variance

series = pd.Series.from_csv('candy_production.csv', header=0)
X = series.values
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
#Variance 1 is much greater than variance 2
#indicates non-stionarity

##Check histograms and data plots
X = series.values
X = np.log(X)
plt.hist(X)
plt.show()
plt.plot(X)
plt.show()
##graphs show a distinct trend
##Check mean and variance in log transformed data
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
#log transform has corrected for most of the trend

##Augmented Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
##Cannot reject null hypothesis
    
##Calculate Rolling Averages
moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()

##Eliminate rolling average from data set 
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
ts_log_moving_avg_diff.head(12)

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        
X = ts_log_moving_avg_diff.iloc[:,0]
test_stationarity(ts_log_moving_avg_diff)

##Eliminating seasonality
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.show()
#Lots of variance but differencing eliminates the trend

ts_log_diff.dropna(inplace=True)
X = ts_log_diff.iloc[:,0]
test_stationarity(X)

##Forecasting Data using ARIMA
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#Need to find p and q parameters for ARIMA forecasting
#p is the x-value when the ACF line crosses the confidence interval first
#q is the x-value when the PACF line crosses the confidence interval first
#Both appear to be 3

from statsmodels.tsa.arima_model import ARIMA
#ARIMA(order=(p,d,q))
model = ARIMA(ts_log, order=(3, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
plt.show()

model = ARIMA(ts_log, order=(0, 1, 3))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

model = ARIMA(ts_log, order=(3, 1, 3))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(ts_log.iloc[:,0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(df)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-df)**2)/len(df)))

