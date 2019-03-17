# Time-Series

# ARIMA
ARIMA is a very popular statistical method for time series forecasting. ARIMA stands for Auto-Regressive Integrated Moving Averages. ARIMA models work on the following assumptions –

1. The data series is stationary, which means that the mean and variance should not vary with time. A series can be made stationary by using log transformation or differencing the series.
2. The data provided as input must be a univariate series, since arima uses the past values to predict the future values.

The general steps to implement an ARIMA model are –

1. Load the data: The first step for model building is of course to load the dataset
2. Preprocessing: Depending on the dataset, the steps of preprocessing will be defined. This will include creating timestamps, converting the dtype of date/time column, making the series univariate, etc.
3. Make series stationary: In order to satisfy the assumption, it is necessary to make the series stationary. This would include checking the stationarity of the series and performing required transformations
4. Determine d value: For making the series stationary, the number of times the difference operation was performed will be taken as the d value
5. Create ACF and PACF plots: This is the most important step in ARIMA implementation. ACF PACF plots are used to determine the input parameters for our ARIMA model
6. Determine the p and q values: Read the values of p and q from the plots in the previous step
7. Fit ARIMA model: Using the processed data and parameter values we calculated from the previous steps, fit the ARIMA model
8. Predict values on validation set: Predict the future values
9. Calculate RMSE: To check the performance of the model, check the RMSE value using the predictions and actual values on the validation set




# Auto ARIMA
Although ARIMA is a very powerful model for forecasting time series data, the data preparation and parameter tuning processes end up being really time consuming. Before implementing ARIMA, you need to make the series stationary, and determine the values of p and q using the plots we discussed above. Auto ARIMA makes this task really simple for us as it eliminates steps 3 to 6 we saw in the previous section. Below are the steps you should follow for implementing auto ARIMA:

1. Load the data: This step will be the same. Load the data into your notebook
2. Preprocessing data: The input should be univariate, hence drop the other columns
3. Fit Auto ARIMA: Fit the model on the univariate series
4. Predict values on validation set: Make predictions on the validation set
5. Calculate RMSE: Check the performance of the model using the predicted values against the actual values
We completely bypassed the selection of p and q feature as you can see.
