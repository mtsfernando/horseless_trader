import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from streamlit_lottie import st_lottie
import json
import os

model_cache = {}

def load_lottie_file(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

def fetch_data(symbol):
    price_st = st.empty()
    loading_st = st.empty()
    loading_st.write(f'Fetching data for {selected_stock}...')
    stock = yf.Ticker(symbol)
    data = stock.history(period="5y")
    loading_st.empty()
    price_st.write(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
    return data[['Close']]

def preprocess_data(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss

    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['Close'].rolling(window=20).std() * 2)
    df['Momemtum'] = df['Close'] - df['Close'].shift(4)
    df['Volatility'] = df['Close'].rolling(window=21).std()

    df.dropna(inplace=True)
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    return scaled_data, scaler

def prepare_data(scaled_data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps: i])
        y.append(scaled_data[i, 0])

    print(f"X Array Shape: {np.array(X).shape}, Y Array Shape: {np.array(y).shape}")
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_predictions_df(predictions):
    days = pd.date_range(start=pd.Timestamp.now() + pd.DateOffset(1), periods=10).strftime('%Y-%m-%d').tolist()
    prediction_df = pd.DataFrame({
        'Date':days,
        'Predicted Price': predictions
    })
    return prediction_df

def plot_predictions(prediction_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Predicted Price'],
        mode='lines+markers',
        name='Predicted Price'
    ))

    fig.update_layout(
        title=f'{selected_stock} Predicted Price for Next 10 Days',
        xaxis_title='Date',
        yaxis_title='Price ($USD)',
        template='plotly_dark'
    )

    st.plotly_chart(fig)

def run_predictions(data):
    data = preprocess_data(data)
    scaled_data, scaler = normalize_data(data)

    X, y = prepare_data(scaled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, batch_size=32, epochs=5)

    loss = model.evaluate(X_test, y_test, verbose=0)
    st.write(f'Model Evaluation - MSE: {loss:.4f}')

    model_cache[selected_stock] = (model, scaler)
    st.write('Predicting for 10 days...')

    predictions = []
    input_sequence = scaled_data[-60:]

    for day in range(10):
        input_sequence = input_sequence.reshape(1, -1, 1)
        predicted_price = model.predict(input_sequence)[0][0]
        predictions.append(predicted_price)
        input_sequence = np.append(input_sequence[0][1:], [[predicted_price]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))[:,0]
    return predictions

def update_selection():
    st.session_state.predict_running = False

def footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #e6e6fa;
            color: #4f4f4f;
            text-align: center;
            padding: 10px;
            font-size: 20px;
            border-top: 1px solid #dcdcdc;
        }
        </style>
        <div class="footer">
            <p>
                Created by MTS | <a href="https://github.com/mtsfernando" target="_blank">GitHub</a> | Built with the Lankan Spirit
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

#Init predict state
if 'predict_running' not in st.session_state:
    st.session_state.predict_running = False

horse_lt_file_path = os.path.join("assets", "horse-icon.json")
horse_lt_json = load_lottie_file(horse_lt_file_path)

loader_lt_file_path = os.path.join("assets", "among-us.json")
loader_lt_json = load_lottie_file(loader_lt_file_path)

col1, col2 = st.columns([1, 4])
with col1:
    st_lottie(horse_lt_json, speed=1)

with col2:
    st.title('Horseless Trader')

stock_list = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
selected_stock = st.selectbox('Select a stock', stock_list, key='stock_select', on_change=update_selection)

data = fetch_data(selected_stock)

if st.button("Predict", disabled=st.session_state.predict_running):
    st.session_state.predict_running = True

if st.session_state.predict_running:
    pg_bar = st.progress(10, text='Predicting...')
    predictions = run_predictions(data)
    pg_bar.progress(60, text='Building DataFrame...')
    prediction_df = build_predictions_df(predictions)
    pg_bar.progress(80, text='Building View...')
    st.write('Predicted Price')
    st.table(prediction_df)
    pg_bar.progress(90, text='Plotting Graph...')
    plot_predictions(prediction_df)
    pg_bar.progress(100, text='Done')

footer()
