import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

st.set_page_config(page_title="Stock Price Forecaster", layout="wide")

st.title("📈 Stock Price Forecaster")
st.caption("Enter any publicly traded stock ticker to forecast future price trends using Prophet time series modeling.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Forecast Settings")
    ticker = st.text_input(
        "Stock Ticker Symbol",
        value="MSFT",
        max_chars=10,
        help="Enter any valid stock ticker, e.g. AAPL, NVDA, TSLA, AMZN"
    ).upper().strip()

    forecast_days = st.slider(
        "Forecast Horizon (days)",
        min_value=30,
        max_value=365,
        value=180,
        step=30,
        help="How many days into the future to forecast"
    )

    history_years = st.selectbox(
        "Historical Data Window",
        options=[2, 3, 5],
        index=1,
        help="How many years of historical data to train on"
    )

    smoothing_window = st.slider(
        "Moving Average Window (days)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Smoothing window applied to close price before modeling"
    )

    base_growth = st.number_input("Base Case Annual Growth (%)", value=8.0, step=1.0)
    upside_growth = st.number_input("Upside Case Annual Growth (%)", value=15.0, step=1.0)
    downside_growth = st.number_input("Downside Case Annual Growth (%)", value=-5.0, step=1.0)

    run = st.button("Run Forecast", use_container_width=True)

if not run:
    st.info("Configure your settings in the sidebar and click **Run Forecast** to begin.")
    st.stop()

# --- VALIDATE TICKER ---
if not ticker:
    st.error("Please enter a valid stock ticker symbol.")
    st.stop()

# --- DATA DOWNLOAD ---
start_date = (pd.Timestamp.today() - pd.DateOffset(years=history_years)).strftime("%Y-%m-%d")
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

with st.spinner(f"Downloading {history_years} years of {ticker} price data..."):
    raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if raw_data.empty:
        st.error(f"No data found for **'{ticker}'**. Please check the ticker symbol and try again.")
        st.stop()

    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    df_daily = raw_data.reset_index()

    if 'Close' not in df_daily.columns:
        st.error("Could not retrieve closing price data for this ticker.")
        st.stop()

    df_daily['Smoothed_Close'] = df_daily['Close'].rolling(
        window=smoothing_window, min_periods=1
    ).mean()

    df_model = pd.DataFrame({
        'ds': pd.to_datetime(df_daily['Date']),
        'y': df_daily['Smoothed_Close'].astype(float)
    }).dropna().reset_index(drop=True)

st.success(f"Loaded {len(df_model):,} trading days of {ticker} data.")

# --- MODEL TRAINING ---
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

st.subheader("Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ticker", ticker)
m2.metric("Training Days", f"{len(df_model):,}")
m3.metric("MAE", f"${mae:.2f}")
m4.metric("MAPE", f"{mape:.2f}%")

# --- SCENARIO SETUP ---
last_date = df_model['ds'].max()
last_val = float(df_model[df_model['ds'] == last_date]['y'].values[0])
future_forecast = forecast[forecast['ds'] > last_date]
future_dates = future_forecast['ds'].reset_index(drop=True)

rates = {
    f'Base Case ({base_growth:+.0f}% Annual)': (base_growth / 100, '#ff7f0e'),
    f'Upside Case ({upside_growth:+.0f}% Annual)': (upside_growth / 100, '#2ca02c'),
    f'Downside Case ({downside_growth:+.0f}% Annual)': (downside_growth / 100, '#d62728')
}

# --- MAIN CHART ---
st.subheader(f"{ticker} Price Forecast — Next {forecast_days} Days")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_daily['Date'], y=df_daily['Close'],
    name="Raw Close Price", mode='lines',
    line=dict(color='#e0e0e0', width=1)
))

fig.add_trace(go.Scatter(
    x=df_model['ds'], y=df_model['y'],
    name=f"{smoothing_window}-Day Moving Average", mode='lines',
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
    daily_rate = (1 + annual_rate) ** (1 / 365) - 1
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
    height=560,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- FORECAST TABLE ---
st.subheader("Projected Price Targets")
final = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6).copy()
final.columns = ['Date', 'Forecast', 'Lower Bound (80%)', 'Upper Bound (80%)']
final['Date'] = final['Date'].dt.strftime('%b %Y')
for col in ['Forecast', 'Lower Bound (80%)', 'Upper Bound (80%)']:
    final[col] = final[col].apply(lambda x: f"${x:.2f}")
st.dataframe(final, use_container_width=True, hide_index=True)

st.caption("⚠️ Disclaimer: This tool is for educational purposes only and does not constitute financial advice.")
