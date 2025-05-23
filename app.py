import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
from textblob import TextBlob  # For sentiment analysis

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500&display=swap');

        .custom-header {
            background-color: #1c1c1c;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-family: 'Montserrat', sans-serif;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        }
        .custom-header img {
            height: 50px;
            margin-right: 10px;
            vertical-align: middle;
        }
    </style>
    <div class="custom-header">
        <img src="https://img.icons8.com/color/48/bitcoin.png" alt="Bitcoin Logo">
        <span style="font-size: 24px;">Bitcoin Price Prediction Dashboard</span><br>
        <span style="font-size: 14px; color: #ccc;">Rajeev Ranjan & Samunder Singh</span>
    </div>
    <br>
""", unsafe_allow_html=True)

# Define the default file path for training data
default_file_path = 'bitcoin_price_Training - Training.csv'

# Streamlit layout
st.title('Bitcoin Price Prediction')
st.markdown("### Welcome to the Bitcoin Prediction page! The models will predict the Bitcoin price for that date, and you can also view the models’ predictions over time.")

# Load training data from default file
train_data = pd.read_csv(default_file_path)

# Convert the 'Date' column to datetime format and set it as the index
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data.set_index('Date', inplace=True)

# Feature engineering
train_data['Price Change'] = train_data['Close'].pct_change()
train_data['SMA_10'] = train_data['Close'].rolling(window=10).mean()
train_data['SMA_30'] = train_data['Close'].rolling(window=30).mean()
train_data['Volatility'] = train_data['Close'].rolling(window=10).std()

# Relative Strength Index (RSI)
delta = train_data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
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

# Drop rows with NaN values created by feature engineering
train_data.dropna(inplace=True)

# Prepare features and target variable
features = ['Price Change', 'SMA_10', 'SMA_30', 'Volatility', 'RSI', 'Upper Band', 'Lower Band', 'MACD', 'Signal Line']
X = train_data[features].values
y = train_data['Close'].values

# Split the data without shuffling to keep the time order intact and store the date indexes for test set
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, train_data.index, test_size=0.2, random_state=42, shuffle=False)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display top 10 rows of the training data
st.subheader("Top 10 Rows of the Training Data")
st.write(train_data.head(10))

# ----------------------- Plot Historical Bitcoin Price Over Time -----------------------
st.subheader("Historical Bitcoin Price Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_data.index, train_data['Close'], label='Close Price', color='blue', linewidth=2)
ax.set_title('Bitcoin Price Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
st.pyplot(fig)

# Let user select a date for prediction based on the available date range
min_date = train_data.index.min().date()
max_date = train_data.index.max().date()
selected_date = st.date_input("Select a date for prediction", value=max_date, min_value=min_date, max_value=max_date)
st.write(f"Selected Date: {selected_date}")

if st.button("Submit"):
    # Create a progress bar
    progress_bar = st.progress(0)

    # ----------------------- Random Forest Model -----------------------
    st.write("Training Random Forest Model...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                                  rf_param_grid, cv=3, scoring='neg_mean_squared_error')
    rf_grid_search.fit(X_train_scaled, y_train)
    rf_best_model = rf_grid_search.best_estimator_
    rf_pred = rf_best_model.predict(X_test_scaled)
    progress_bar.progress(33)

    # ----------------------- Gradient Boosting Model -----------------------
    st.write("Training Gradient Boosting Model...")
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb_grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42),
                                  gb_param_grid, cv=3, scoring='neg_mean_squared_error')
    gb_grid_search.fit(X_train_scaled, y_train)
    gb_best_model = gb_grid_search.best_estimator_
    gb_pred = gb_best_model.predict(X_test_scaled)
    progress_bar.progress(66)

    # ----------------------- LSTM Model -----------------------
    st.write("Training LSTM Model...")
    X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    lstm_model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
    y_pred_lstm = lstm_model.predict(X_test_lstm)
    st.write("LSTM Model MAE:", mean_absolute_error(y_test, y_pred_lstm))
    progress_bar.progress(100)

    # ----------------------- Plot Predictions Over Time -----------------------
    st.subheader("Predicted Bitcoin Prices Over Time by Model")

    # Plot for Random Forest
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates_test, y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(dates_test, rf_pred, label='Random Forest Predictions', color='green', linestyle='--', linewidth=2)
    ax.set_title('Random Forest: Bitcoin Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # Plot for Gradient Boosting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates_test, y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(dates_test, gb_pred, label='Gradient Boosting Predictions', color='orange', linestyle='-.', linewidth=2)
    ax.set_title('Gradient Boosting: Bitcoin Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # Plot for LSTM
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates_test, y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(dates_test, y_pred_lstm, label='LSTM Predictions', color='red', linestyle=':', linewidth=2)
    ax.set_title('LSTM: Bitcoin Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # ----------------------- Sentiment Analysis -----------------------
    st.subheader("Sentiment Analysis on Bitcoin News Headlines")
    example_news = [
        "Bitcoin hits new all-time high, investors are excited",
        "Bitcoin prices crash, market sentiment turns negative",
        "Bitcoin adoption increasing worldwide, bullish outlook"
    ]
    sentiments = [TextBlob(news).sentiment.polarity for news in example_news]
    sentiment_df = pd.DataFrame({'News': example_news, 'Sentiment Polarity': sentiments})
    st.write(sentiment_df)

    # ----------------------- Price Prediction for the Selected Date -----------------------
    st.subheader("Bitcoin Price Prediction for the Selected Date")
    selected_date = pd.to_datetime(selected_date)
    if selected_date in train_data.index:
        features_for_date = train_data.loc[selected_date, features].values.reshape(1, -1)
    else:
        st.error("Selected date is not available in the dataset. Please select a valid date.")
        st.stop()

    features_scaled = scaler.transform(features_for_date)
    prediction_rf = rf_best_model.predict(features_scaled)
    prediction_gb = gb_best_model.predict(features_scaled)
    features_lstm = np.reshape(features_scaled, (1, features_scaled.shape[1], 1))
    prediction_lstm = lstm_model.predict(features_lstm)

    st.write(f"Random Forest Prediction: {prediction_rf[0]:.2f} USD")
    st.write(f"Gradient Boosting Prediction: {prediction_gb[0]:.2f} USD")
    st.write(f"LSTM Prediction: {prediction_lstm[0][0]:.2f} USD")
    
    # ----------------------- Confidence Intervals for Models -----------------------
    st.subheader("Confidence Intervals for Predictions")
    
    def bootstrap_confidence_interval(model, X, num_samples=100, alpha=0.05):
        predictions = []
        for _ in range(num_samples):
            sample_indices = np.random.choice(len(X), len(X), replace=True)
            sample_X = X[sample_indices]
            predictions.append(model.predict(sample_X))
        predictions = np.array(predictions)
        lower_bound = np.percentile(predictions, (alpha/2) * 100, axis=0)
        upper_bound = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
        return lower_bound, upper_bound

    rf_lower, rf_upper = bootstrap_confidence_interval(rf_best_model, X_test_scaled)
    gb_lower, gb_upper = bootstrap_confidence_interval(gb_best_model, X_test_scaled)

    # Displaying Random Forest Confidence Interval
    st.write("Random Forest Confidence Interval")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates_test, y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(dates_test, rf_pred, label='Predicted Prices (RF)', color='green', linestyle='--', linewidth=2)
    ax.fill_between(dates_test, rf_lower, rf_upper, color='green', alpha=0.2, label='Confidence Interval')
    ax.set_title('Random Forest Confidence Interval')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # Displaying Gradient Boosting Confidence Interval
    st.write("Gradient Boosting Confidence Interval")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates_test, y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(dates_test, gb_pred, label='Predicted Prices (GB)', color='orange', linestyle='-.', linewidth=2)
    ax.fill_between(dates_test, gb_lower, gb_upper, color='orange', alpha=0.2, label='Confidence Interval')
    ax.set_title('Gradient Boosting Confidence Interval')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # ----------------------- Interactive Feature Adjustment -----------------------
    st.subheader("Interactive Feature Adjustment for Scenario Simulation")
    rsi_slider = st.slider("RSI", min_value=0, max_value=100, value=50)
    macd_slider = st.slider("MACD", min_value=-5.0, max_value=5.0, value=0.0)
    sentiment_slider = st.slider("Sentiment Polarity", min_value=-1.0, max_value=1.0, value=0.0)

    scenario_features = scaler.transform([[0, 0, 0, 0, rsi_slider, 0, 0, macd_slider, 0]])
    rf_scenario_pred = rf_best_model.predict(scenario_features)
    gb_scenario_pred = gb_best_model.predict(scenario_features)

    st.write(f"Random Forest Prediction for Adjusted Scenario: {rf_scenario_pred[0]:.2f} USD")
    st.write(f"Gradient Boosting Prediction for Adjusted Scenario: {gb_scenario_pred[0]:.2f} USD")

    # ----------------------- Sensitivity Analysis for Sentiment -----------------------
    st.subheader("Sensitivity Analysis: Forecasting Based on Sentiment Trends")
    sentiment_range = np.linspace(-1.0, 1.0, 20)
    sensitivity_analysis = []

    for sentiment in sentiment_range:
        adjusted_features = scaler.transform([[0, 0, 0, 0, 50, 0, 0, 0, sentiment]])
        rf_sent_pred = rf_best_model.predict(adjusted_features)
        gb_sent_pred = gb_best_model.predict(adjusted_features)
        sensitivity_analysis.append({'Sentiment': sentiment, 
                                     'Random Forest Prediction': rf_sent_pred[0], 
                                     'Gradient Boosting Prediction': gb_sent_pred[0]})

    sensitivity_df = pd.DataFrame(sensitivity_analysis)
    st.write("Sensitivity Analysis Data")
    st.write(sensitivity_df)

    # Plotting Sensitivity to Sentiment Trends
    st.write("Sensitivity Analysis Visualization")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sensitivity_df['Sentiment'], sensitivity_df['Random Forest Prediction'], label='Random Forest', color='green')
    ax.plot(sensitivity_df['Sentiment'], sensitivity_df['Gradient Boosting Prediction'], label='Gradient Boosting', color='orange')
    ax.set_title("Model Predictions vs. Sentiment Trends")
    ax.set_xlabel("Sentiment Polarity")
    ax.set_ylabel("Bitcoin Price Prediction (USD)")
    ax.legend()
    st.pyplot(fig)
    st.markdown("""
    <br><br>
    <hr>
    <center>
        <small>Created with ❤️ using Streamlit | © 2025 Rajeev and Samunder</small>
    </center>
""", unsafe_allow_html=True)

