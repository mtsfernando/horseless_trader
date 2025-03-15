import streamlit as st
import plotly.graph_objects as go
import pandas as pd

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
        xaxis_title='Date',
        yaxis_title='Price ($USD)',
        template='plotly_dark'
    )

    st.plotly_chart(fig)