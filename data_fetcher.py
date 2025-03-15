import streamlit as st
import yfinance as yf

def fetch_data(symbol):
    loading_st = st.empty()
    loading_st.write(f'Fetching data for {symbol}...')
    stock = yf.Ticker(symbol)
    data = stock.history(period="5y")
    loading_st.empty()
    st.metric(label="Latest price", value=f"${data['Close'].iloc[-1]:.2f}")
    return data[['Close']]
