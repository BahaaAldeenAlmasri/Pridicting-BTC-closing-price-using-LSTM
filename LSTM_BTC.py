# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:46:40 2024

@author: BAHAA ALDEEN
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime

def str_to_datetime(s):
  split = s.split('/')
  year, month, day = int(split[2]), int(split[0]), int(split[1])
  return datetime.datetime(year=year, month=month, day=day)


# Load your Bitcoin price data into a pandas dataframe 'df'
df = pd.read_csv('C:/Users/BAHAA ALDEEN/Downloads/Binance_BTCUSDT_d.csv')

# Select relevant features (e.g., closing price)
df = df[['Date', 'Close']]
df['Date'] = df['Date'].apply(str_to_datetime)
plt.plot(df['Date'] ,df['Close'])
plt.show()

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))


# Define the number of past closing prices to consider (lookback window)
look_back = 4 # Experiment with different values

# Create sequences
sequences = []
for i in range(look_back, len(scaled_data)):
  sequence = scaled_data[i-look_back:i, :]
  sequences.append(sequence)

# Convert sequences to numpy arrays
sequences = np.array(sequences)
sequences =  sequences.reshape(sequences.shape[0], look_back)
sequences = pd.DataFrame(sequences)

X_train, X_test, y_train, y_test = train_test_split(sequences.iloc[:, :-1].values, 
                                                    sequences.iloc[:, -1].values,
                                                    test_size=0.2, shuffle=False)

#df['Date'].iloc[-488:]
plt.plot(y_test)

# Modeling
model = keras.Sequential()
model.add(LSTM(units=32, return_sequences=True, input_shape=(look_back-1 , 1)))
model.add(LSTM(units=16))
model.add(Dense(units=1))  # Output layer with 1 unit for predicted price
model.compile(loss='mse', optimizer='adam')  # Mean squared error loss and Adam optimizer
model.fit(X_train, y_train, epochs=100, batch_size=32)  # Train for 100 epochs with a batch size of 32

loss = model.evaluate(X_test, y_test)
print(f"LOSS: {loss}")

# Make predictions on test data
predicted_prices = model.predict(X_test)

# Invert scaling for predicted prices
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# Calculate Mean Squared Error (MSE)
mse = round(mean_squared_error(actual_prices, predicted_prices),2)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Root Mean Squared Error (RMSE)
rmse = round(np.sqrt(mse),2)
print(f"Root Mean Squared Error (RMSE): {rmse}")



# Invert scaling for y_test (actual prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual prices vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
