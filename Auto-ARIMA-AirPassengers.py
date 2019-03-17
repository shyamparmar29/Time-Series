# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:34:32 2019

@author: Shyam Parmar
"""
import pandas as pd
import matplotlib.pylab as plt
from math import sqrt
from pyramid.arima import auto_arima
from sklearn.metrics import mean_squared_error


#load the data
data = pd.read_csv('AirPassengers.csv')

#divide into train and validation set
train = data[:int(0.7*(len(data)))]
valid = data[int(0.7*(len(data))):]

#preprocessing (since arima takes univariate series as input)
train.drop('Month',axis=1,inplace=True)
valid.drop('Month',axis=1,inplace=True)

#plotting the data
train['#Passengers'].plot()
valid['#Passengers'].plot()

#building the model
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#calculate rmse
rms = sqrt(mean_squared_error(valid,forecast))
print(rms)

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()

