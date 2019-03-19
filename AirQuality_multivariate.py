# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:09:10 2019

@author: Shyam Parmar
"""

import pandas as pd
from math import sqrt
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.metrics import mean_squared_error

#read the data
df = pd.read_csv("AirQualityUCI.csv", parse_dates=[['Date', 'Time']])

#check the dtypes
df.dtypes

#The data type of the Date_Time column is object and we need to change it to datetime. 
#Also, for preparing the data, we need the index to have datetime.
df['Date_Time'] = pd.to_datetime(df.Date_Time , format = '%d/%m/%Y %H.%M.%S')
data = df.drop(['Date_Time'], axis=1)
data.index = df.Date_Time

#The next step is to deal with the missing values. Since the missing values in the data are replaced with  
#a value -200, we will have to impute the missing value with a better number. Consider this – if the present 
#dew point value is missing, we can safely assume that it will be close to the value of the previous hour. 
#Makes sense, right? Here, I will impute -200 with the previous value.

#missing value treatment
cols = data.columns
for j in cols:
    for i in range(0,len(data)):
       if data[j][i] == -200:
           data[j][i] = data[j][i-1]

#checking stationarity

#since the test works for only 12 variables, I have randomly dropped
#in the next iteration, I would drop another and check the eigenvalues
johan_test_temp = data.drop([ 'CO(GT)'], axis=1)
coint_johansen(johan_test_temp,-1,1).eig

#creating the train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model
model = VAR(endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))

#The predictions are in the form of an array, where each list represents the predictions of the row.
# We will transform this into a more presentable format.

#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,13):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#check rmse
for i in cols:
    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid[i])))
    
#After the testing on validation set, lets fit the model on the complete dataset
    #make final predictions
model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)