import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv("Google_Stock_Price_Train.csv")
# print(data_train.head())

training_set = data_train.iloc[:,1:2].values
# print(training_set)
# print(training_set.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)
# print(scaled_training_set)

x_train = []
y_train = []
for i in range(40,198):
    x_train.append(scaled_training_set[i-40:i,0])
    y_train.append(scaled_training_set[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)
# print(x_train.shape)
# print(y_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
# print(x_train.shape)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences= True, input_shape= (x_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(x_train, y_train, batch_size=32)

data_test = pd.read_csv("Google_Stock_Price_Test.csv")
actual_stock_price = data_test.iloc[:,1:2].values
data_total = pd.concat((data_train['Open'], data_test['Open']), axis = 0)
inputs = data_total[len(data_total)-len(data_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(40,60):
    X_test.append(inputs[i-40:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predict_price = regressor.predict(X_test)
predict_price = scaler.inverse_transform(predict_price)

plt.plot(actual_stock_price, color='red', label='Actual Google Stock Price')
plt.plot(predict_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
