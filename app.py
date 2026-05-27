import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

st.set_page_config(page_title="Stock Price Forecaster", layout="wide")
st.title("Stock Price Forecaster")
st.caption("Enter any publicly traded ticker to forecast price trends using Prophet time series modeling.")

# --- USER INPUT ---
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Stock Ticker", value="MSFT").upper()
with col2:
    forecast_days = st.slider("Forecast Horizon (days)", min_value=30, max_value=365, value=180, step=30)

if st.button("Run Forecast"):

    with st.spinner(f"Downloading 5 years of {ticker} price data..."):
        raw_data = yf.download(ticker, start="2021-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))

        if raw_data.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the symbol and try again.")
            st.stop()

        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)

        df_daily = raw_data.reset_index()
        df_daily['Smoothed_Close'] = df_daily['Close'].rolling(window=20, min_periods=1).mean()

        df_model = pd.DataFrame()
        df_model['ds'] = pd.to_datetime(df_daily['Date'])
        df_model['y'] = df_daily['Smoothed_Close'].astype(float)
        df_model = df_model.dropna().reset_index(drop=True)

    with st.spinner("Training Prophet forecasting model..."):
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            interval_width=0.80
        )
        model.fit(df_model)

        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        forecast = model.predict(future)

    # --- BACKTEST METRICS ---
    historical_preds = forecast[forecast['ds'].isin(df_model['ds'])]['yhat'].values
    actual_values = df_model['y'].values
    mae = mean_absolute_error(actual_values, historical_preds)
    mape = mean_absolute_percentage_error(actual_values, historical_preds) * 100

    st.subheader("Model Accuracy")
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
    m2.metric("Mean Absolute % Error (MAPE)", f"{mape:.2f}%")
    m3.metric("Training Data Points", f"{len(df_model):,} days")

    # --- SCENARIO MODELING ---
    last_date = df_model['ds'].max()
    last_val = float(df_model[df_model['ds'] == last_date]['y'].values[0])
    future_forecast = forecast[forecast['ds'] > last_date]
    future_dates = future_forecast['ds'].reset_index(drop=True)

    rates = {
        'Base Case (+8% Annual)': (0.08, '#ff7f0e'),
        'Upside Case (+15% Annual)': (0.15, '#2ca02c'),
        'Downside Case (-5% Annual)': (-0.05, '#d62728')
    }

    # --- CHART ---
    st.subheader(f"{ticker} Price Forecast")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_daily['Date'], y=df_daily['Close'],
        name="Raw Close Price", mode='lines',
        line=dict(color='#e0e0e0', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df_model['ds'], y=df_model['y'],
        name="20-Day Moving Average", mode='lines',
        line=dict(color='#111111', width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        name="Prophet Forecast", mode='lines',
        line=dict(color='#0078d4', dash='dash', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([future_forecast['ds'], future_forecast['ds'].iloc[::-1]]),
        y=pd.concat([future_forecast['yhat_upper'], future_forecast['yhat_lower'].iloc[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 120, 212, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name="80% Confidence Interval"
    ))

    for scenario_name, (annual_rate, color) in rates.items():
        daily_rate = (1 + annual_rate) ** (1/365) - 1
        values = [last_val * ((1 + daily_rate) ** d) for d in range(1, len(future_dates) + 1)]
        plot_dates = pd.concat([pd.Series([last_date]), future_dates])
        plot_values = [last_val] + values

        fig.add_trace(go.Scatter(
            x=plot_dates, y=plot_values,
            name=scenario_name, mode='lines',
            line=dict(color=color, width=2)
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        hovermode="x unified",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- FORECAST TABLE ---
    st.subheader("Projected Price Targets")
    final = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6).copy()
    final.columns = ['Date', 'Forecast', 'Lower Bound (80%)', 'Upper Bound (80%)']
    final['Date'] = final['Date'].dt.strftime('%b %Y')
    final[['Forecast', 'Lower Bound (80%)', 'Upper Bound (80%)']] = final[['Forecast', 'Lower Bound (80%)', 'Upper Bound (80%)']].applymap(lambda x: f"${x:.2f}")
    st.dataframe(final, use_container_width=True, hide_index=True)

    st.caption("Disclaimer: This tool is for educational purposes only and does not constitute financial advice.")
