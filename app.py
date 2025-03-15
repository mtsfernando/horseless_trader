import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from streamlit_lottie import st_lottie
import os

from data_fetcher import fetch_data
from data_processor import preprocess_data, normalize_data, prepare_data
from model_builder import build_model
from visualization import build_predictions_df, plot_predictions
from utils import load_lottie_file, footer

model_cache = {}

def run_predictions(data):
    data = preprocess_data(data)
    scaled_data, scaler = normalize_data(data)

    X, y = prepare_data(scaled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, batch_size=32, epochs=5)

    loss = model.evaluate(X_test, y_test, verbose=0)
    st.metric(label='MSE', value=f'{loss:.4f}')

    model_cache[selected_stock] = (model, scaler)

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

footer()

#Init predict state
if 'predict_running' not in st.session_state:
    st.session_state.predict_running = False

horse_lt_file_path = os.path.join("assets", "horse-icon.json")
horse_lt_json = load_lottie_file(horse_lt_file_path)

loader_lt_file_path = os.path.join("assets", "among-us.json")
loader_lt_json = load_lottie_file(loader_lt_file_path)

gradient_lt_file_path = os.path.join("assets", "gradient-loader.json")
gradient_lt_json = load_lottie_file(gradient_lt_file_path)

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
    pg_bar = st.progress(10, text='Doing some magic...')
    predictions = run_predictions(data)
    pg_bar.progress(60, text='Building DataFrame...')
    prediction_df = build_predictions_df(predictions)
    pg_bar.progress(80, text='Building View...')
    st.divider()
    col1, col2 = st.columns([1, 7])
    with col1:
        st_lottie(gradient_lt_json, speed=1)
    with col2:
        st.header('Predicted Price')
    st.table(prediction_df)
    pg_bar.progress(90, text='Plotting Graph...')
    st.divider()
    col1, col2 = st.columns([1, 7])
    with col1:
        st_lottie(gradient_lt_json, speed=1)
    with col2:
        st.header('10-Day Prediction')
    plot_predictions(prediction_df)
    pg_bar.progress(100, text='Done')


