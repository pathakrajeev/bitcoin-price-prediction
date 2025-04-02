import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
from textblob import TextBlob  # For sentiment analysis

# Define the default file path
default_file_path = 'bitcoin_price_Training - Training.csv'

# Streamlit layout
st.title('Bitcoin Price Prediction')

# Display the welcome message
st.markdown("### Welcome to the Bitcoin Prediction page! Please select the training data. If no selection is made, the default training dataset will be used.")

# Option for the user to upload a file or use the default one
uploaded_file = st.file_uploader("Choose a file to upload", type=["csv"])

# If no file is uploaded, use the default file
if uploaded_file is None:
    st.warning("No file uploaded, using default dataset.")
    train_data = pd.read_csv(default_file_path)
else:
    train_data = pd.read_csv(uploaded_file)

# Load test data (you can also upload it the same way as the training file if needed)
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

# Display top 10 rows of training data
st.subheader("Top 10 Rows of the Training Data")
st.write(train_data.head(10))

# Define and train models only after Submit button is clicked
if st.button("Submit"):

    # Create progress bar
    progress_bar = st.progress(0)

    # Train Random Forest with Hyperparameter Tuning
    st.write("Training Random Forest Model...")
    rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=3, scoring='neg_mean_squared_error')
    rf_grid_search.fit(X_train_scaled, y_train)
    rf_best_model = rf_grid_search.best_estimator_
    rf_pred = rf_best_model.predict(X_test_scaled)

    # Update progress bar
    progress_bar.progress(33)

    # Train Gradient Boosting with Hyperparameter Tuning
    st.write("Training Gradient Boosting Model...")
    gb_param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    gb_grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_param_grid, cv=3, scoring='neg_mean_squared_error')
    gb_grid_search.fit(X_train_scaled, y_train)
    gb_best_model = gb_grid_search.best_estimator_
    gb_pred = gb_best_model.predict(X_test_scaled)

    # Update progress bar
    progress_bar.progress(66)

    # LSTM Model Preparation
    st.write("Training LSTM Model...")
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

    # Update progress bar
    progress_bar.progress(100)

    # Visualize Random Forest Predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(rf_pred, label='Random Forest Predicted Prices', color='green', linestyle='--', linewidth=2)
    ax.set_title('Random Forest Bitcoin Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # Visualize Gradient Boosting Predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(gb_pred, label='Gradient Boosting Predicted Prices', color='orange', linestyle='-.', linewidth=2)
    ax.set_title('Gradient Boosting Bitcoin Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # Visualize LSTM Predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(y_pred_lstm, label='LSTM Predicted Prices', color='red', linestyle=':', linewidth=2)
    ax.set_title('LSTM Bitcoin Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # Visualizing all model predictions together
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(rf_pred, label='Random Forest Predicted Prices', color='green', linestyle='--', linewidth=2)
    ax.plot(gb_pred, label='Gradient Boosting Predicted Prices', color='orange', linestyle='-.', linewidth=2)
    ax.plot(y_pred_lstm, label='LSTM Predicted Prices', color='red', linestyle=':', linewidth=2)
    ax.set_title('Bitcoin Price Prediction with Random Forest, Gradient Boosting, and LSTM')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # High/Low Price Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_data['High'], label='High Prices', color='green', linestyle='-', linewidth=2)
    ax.plot(train_data['Low'], label='Low Prices', color='red', linestyle='-', linewidth=2)
    ax.set_title('Bitcoin High and Low Prices')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # Sentiment Analysis: Assuming we have a list of Bitcoin-related news headlines
    # For example, these could be fetched from an API or manually provided
    example_news = [
        "Bitcoin hits new all-time high, investors are excited",
        "Bitcoin prices crash, market sentiment turns negative",
        "Bitcoin adoption increasing worldwide, bullish outlook"
    ]
    
    # Calculate sentiment polarity for each news item
    sentiments = []
    for news in example_news:
        sentiment = TextBlob(news).sentiment.polarity
        sentiments.append(sentiment)

    # Display sentiment analysis results
    sentiment_df = pd.DataFrame({'News': example_news, 'Sentiment Polarity': sentiments})
    st.subheader("Sentiment Analysis on Bitcoin News Headlines")
    st.write(sentiment_df)

    # Display Prediction
    st.subheader("Bitcoin Price Prediction for the Next Week")
    future_prediction_rf = rf_best_model.predict(scaler.transform(train_data[features].tail(1)))
    future_prediction_gb = gb_best_model.predict(scaler.transform(train_data[features].tail(1)))
    future_prediction_lstm = model.predict(np.reshape(scaler.transform(train_data[features].tail(1)), (1, train_data[features].shape[1], 1)))

    st.write(f"Random Forest Prediction: {future_prediction_rf[0]:.2f} USD")
    st.write(f"Gradient Boosting Prediction: {future_prediction_gb[0]:.2f} USD")
    st.write(f"LSTM Prediction: {future_prediction_lstm[0][0]:.2f} USD")
