# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:25:18 2019

@author: Shyam Parmar
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import  seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

dataset = pd.read_csv("AirPassengers.csv")
#Parse strings to datetime type
dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format = True)
indexedDataset = dataset.set_index(['Month'])

plt.xlabel("Date")
plt.ylabel("Number of Air passengers")
plt.plot(indexedDataset)

## Check Stationarity
def test_stationarity(timeseries):
    
    #Determining Rolling Statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    movingAverage = timeseries.rolling(window = 12).mean()
    movingSTD = timeseries.rolling(window = 12).std()
    
    #Plot Rolling Statistics
    orig = plt.plot(indexedDataset, color = 'blue', label = 'Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title("Rolling Mean and Standard deviation")
    plt.show(block = False)
    
    # Perform Dickey Fuller test
    print("Results of Dickey Fuller Test")
    dftest = adfuller(indexedDataset['#Passengers'], autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Static', 'p-value', '#Lags Used', 'Number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)
    
test_stationarity(indexedDataset)

# Since the data is non-stationary, we'll perform different transformations to make it stationary

#Estimating trend
indexedDataset_logscale = np.log(indexedDataset)  #log transformation is one of the step for making data stationary
plt.plot(indexedDataset_logscale)

movingAverage = indexedDataset_logscale.rolling(window = 12).mean()
movingSTD = indexedDataset_logscale.rolling(window = 12).std()                   
plt.plot(indexedDataset_logscale)
plt.plot(movingAverage, color = 'red')

datasetLogscaleMinusMovingAverage = indexedDataset_logscale - movingAverage

#Reamove Nan-Values   
datasetLogscaleMinusMovingAverage.dropna(inplace = True)
    
test_stationarity(datasetLogscaleMinusMovingAverage)

#Weighted average
#We are doing this because we need t see the trend present in the time series and that's why we'vw calculated
#weighted average of time series

exponentialDecayWeightedAverage = indexedDataset_logscale.ewm(halflife=12, min_periods = 0, adjust = True).mean()
plt.plot(indexedDataset_logscale)
plt.plot(exponentialDecayWeightedAverage, color = 'red')

datasetLogscaleMinusMovingExponentialDecayAverage = indexedDataset_logscale - exponentialDecayWeightedAverage
test_stationarity(datasetLogscaleMinusMovingExponentialDecayAverage) #Checking again and again to see if we've removed non-stationarity

#Now the data is stationary so we'll shift the data into time series so that we can use it for forecasting
datasetLogDiffShifting = indexedDataset_logscale - indexedDataset_logscale.shift()
plt.plot(datasetLogDiffShifting)

datasetLogDiffShifting.dropna(inplace = True)
test_stationarity(datasetLogDiffShifting)

decomposition = seasonal_decompose(indexedDataset_logscale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logscale, label = 'Original')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend, label = 'Trend')
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(seasonal, label = 'Seasonality')
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(residual, label = 'Residuals')
plt.legend(loc = 'best')
plt.tight_layout()

decomposedLogData = residual
decomposedLogData.dropna(inplace = True)
test_stationarity(decomposedLogData)

# ACF and PACF graphs

##In order to calculate the value of p we need to calculate PACF graph and to calculate value of q we need to
##calculate ACF graph
lag_acf = acf(datasetLogDiffShifting, nlags = 20)
lag_pacf = pacf(datasetLogDiffShifting, nlags = 20, method = 'ols') #ols = ordinary least square method

#Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y = 0, linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle = '--', color = 'gray')
plt.title('Autocorrelation function')

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y = 0, linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle = '--', color = 'gray')
plt.title('Partial Autocorrelation function')
plt.tight_layout()

#AR (Auto Regressive) model
model = ARIMA(indexedDataset_logscale, order = (2,1,0)) #p=1, d=2, q=2. Play around these values to decrease RSS score
results_AR = model.fit(disp = -1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color = 'red')
plt.title('RSS : %.4f'% sum((results_AR.fittedvalues - datasetLogDiffShifting["#Passengers"])**2)) #RSS = Residual Sum of Squares. Greater the RSS the bad it is.
print('Plotting AR model')

#MA (Moving Average) model
model = ARIMA(indexedDataset_logscale, order = (0,1,2)) #p=0, d=2, q=2
results_MA = model.fit(disp = -1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color = 'red')
plt.title('RSS : %.4f'% sum((results_MA.fittedvalues - datasetLogDiffShifting["#Passengers"])**2))
print('Plotting MA model')         

model = ARIMA(indexedDataset_logscale, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
                                                                              
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())

#Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(indexedDataset_logscale['#Passengers'].ix[0], index=indexedDataset_logscale['#Passengers'].index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-indexedDataset["#Passengers"])**2)/len(indexedDataset["#Passengers"])))