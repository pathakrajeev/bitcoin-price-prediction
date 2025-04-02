import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import streamlit as st

# Define the path to the default file (hardcoded)
default_file_path = 'bitcoin_price_Training - Training.csv'

# Load dataset from the hardcoded file
train_data = pd.read_csv(default_file_path)
test_data = pd.read_csv('bitcoin_price_1week_Test - Test.csv')

# Convert the 'Date' column to datetime format
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data.set_index('Date', inplace=True)

# Feature engineering
train_data['Price Change'] = train_data['Close'].pct_change()
train_data['SMA_10'] = train_data['Close'].rolling(window=10).mean()
train_data['SMA_30'] = train_data['Close'].rolling(window=30).mean()
train_data['Volatility'] = train_data['Close'].rolling(window=10).std()

# Relative Strength Index (RSI)
delta = train_data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
train_data['RSI'] = 100 - (100 / (1 + rs))

# Bollinger Bands
train_data['Middle Band'] = train_data['Close'].rolling(window=20).mean()
train_data['Upper Band'] = train_data['Middle Band'] + 2 * train_data['Close'].rolling(window=20).std()
train_data['Lower Band'] = train_data['Middle Band'] - 2 * train_data['Close'].rolling(window=20).std()

# Moving Average Convergence Divergence (MACD)
train_data['EMA_12'] = train_data['Close'].ewm(span=12, adjust=False).mean()
train_data['EMA_26'] = train_data['Close'].ewm(span=26, adjust=False).mean()
train_data['MACD'] = train_data['EMA_12'] - train_data['EMA_26']
train_data['Signal Line'] = train_data['MACD'].ewm(span=9, adjust=False).mean()

# Drop NaN values
train_data.dropna(inplace=True)

# Prepare data for ML models
features = ['Price Change', 'SMA_10', 'SMA_30', 'Volatility', 'RSI', 'Upper Band', 'Lower Band', 'MACD', 'Signal Line']
X = train_data[features].values
y = train_data['Close'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest with Hyperparameter Tuning
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=3, scoring='neg_mean_squared_error')
rf_grid_search.fit(X_train_scaled, y_train)
rf_best_model = rf_grid_search.best_estimator_
rf_pred = rf_best_model.predict(X_test_scaled)

# Train Gradient Boosting with Hyperparameter Tuning
gb_param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
gb_grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_param_grid, cv=3, scoring='neg_mean_squared_error')
gb_grid_search.fit(X_train_scaled, y_train)
gb_best_model = gb_grid_search.best_estimator_
gb_pred = gb_best_model.predict(X_test_scaled)

# Calculate errors
rf_mse = mean_squared_error(y_test, rf_pred)
gb_mse = mean_squared_error(y_test, gb_pred)
st.write(f'Random Forest Best Parameters: {rf_grid_search.best_params_}')
st.write(f'Random Forest Mean Squared Error: {rf_mse}')
st.write(f'Gradient Boosting Best Parameters: {gb_grid_search.best_params_}')
st.write(f'Gradient Boosting Mean Squared Error: {gb_mse}')

# LSTM Model Preparation
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_lstm, y_train, epochs=20, batch_size=32)

y_pred_lstm = model.predict(X_test_lstm)
st.write("LSTM Model MAE:", mean_absolute_error(y_test, y_pred_lstm))

# Visualizations with Streamlit
st.subheader("Random Forest Predictions")
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(rf_pred, label='Random Forest Predicted Prices', color='green', linestyle='--')
plt.title('Random Forest Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot()

st.subheader("Gradient Boosting Predictions")
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(gb_pred, label='Gradient Boosting Predicted Prices', color='orange', linestyle='-.')
plt.title('Gradient Boosting Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot()

st.subheader("LSTM Predictions")
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred_lstm, label='LSTM Predicted Prices', color='red', linestyle=':')
plt.title('LSTM Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot()

# Visualize all predictions in a single plot
st.subheader("Model Comparison: All Predictions")
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(rf_pred, label='Random Forest Predicted Prices', color='green')
plt.plot(gb_pred, label='Gradient Boosting Predicted Prices', color='orange')
plt.plot(y_pred_lstm, label='LSTM Predicted Prices', color='red')
plt.title('Bitcoin Price Prediction with Hyperparameter Tuning and LSTM')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot()
