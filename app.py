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
        value="AAPL",
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
        index=2,
        help="How many years of historical data to train on"
    )

    test_months = st.slider(
        "Holdout Test Period (months)",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
        help="Most recent months withheld from training to evaluate true forecast accuracy"
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

if not ticker:
    st.error("Please enter a valid stock ticker symbol.")
    st.stop()

# --- DATA DOWNLOAD ---
start_date = (pd.Timestamp.today() - pd.DateOffset(years=history_years)).strftime("%Y-%m-%d")
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

with st.spinner(f"Downloading {history_years} years of {ticker} price data..."):
    raw_data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )

    if raw_data.empty:
        st.error(f"No data found for **'{ticker}'**. Please check the ticker symbol and try again.")
        st.stop()

    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    df_daily = raw_data.reset_index()

    if 'Date' not in df_daily.columns and 'Datetime' not in df_daily.columns:
        df_daily = raw_data.copy()
        df_daily.index.name = 'Date'
        df_daily = df_daily.reset_index()

    date_col = 'Date' if 'Date' in df_daily.columns else 'Datetime'

    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)

    for col in df_daily.columns:
        if df_daily[col].dtype == object:
            try:
                df_daily[col] = df_daily[col].apply(
                    lambda x: x[0] if hasattr(x, '__len__') and not isinstance(x, str) else x
                )
            except:
                pass

    if 'Close' not in df_daily.columns:
        st.error("Could not retrieve closing price data for this ticker.")
        st.stop()

    df_daily['Close'] = pd.to_numeric(df_daily['Close'], errors='coerce')
    df_daily['Smoothed_Close'] = df_daily['Close'].rolling(
        window=smoothing_window, min_periods=1
    ).mean()

    df_full = pd.DataFrame({
        'ds': pd.to_datetime(df_daily[date_col]),
        'y': df_daily['Smoothed_Close'].astype(float)
    }).dropna().reset_index(drop=True)

# --- TRAIN / TEST SPLIT ---
cutoff_date = pd.Timestamp.today() - pd.DateOffset(months=test_months)
df_train = df_full[df_full['ds'] < cutoff_date].reset_index(drop=True)
df_test  = df_full[df_full['ds'] >= cutoff_date].reset_index(drop=True)

if len(df_train) < 60:
    st.error("Not enough training data. Try increasing the historical window or reducing the holdout period.")
    st.stop()

st.success(
    f"Loaded {len(df_full):,} total trading days — "
    f"**{len(df_train):,} training** / **{len(df_test):,} holdout test**"
)

# --- MODEL TRAINING ---
with st.spinner("Training Prophet model on historical data only..."):
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=True,
        interval_width=0.80
    )
    model.fit(df_train)

    total_periods = len(df_test) + forecast_days
    future = model.make_future_dataframe(periods=total_periods, freq='D')
    forecast = model.predict(future)

# --- HOLDOUT METRICS ---
forecast_indexed = forecast.set_index('ds')
test_preds = forecast_indexed.loc[
    forecast_indexed.index.isin(df_test['ds']), 'yhat'
].values
test_actuals = df_test['y'].values[:len(test_preds)]

if len(test_preds) > 0:
    mae  = mean_absolute_error(test_actuals, test_preds)
    mape = mean_absolute_percentage_error(test_actuals, test_preds) * 100
else:
    mae, mape = 0, 0

# --- METRICS ---
st.subheader("Model Performance — Evaluated on Unseen Holdout Data")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ticker", ticker)
m2.metric("Training Days", f"{len(df_train):,}")
m3.metric("Holdout Days", f"{len(df_test):,}")
m4.metric("MAE", f"${mae:.2f}")
m5.metric("MAPE", f"{mape:.2f}%")

st.caption(
    f"Model trained on data before **{cutoff_date.strftime('%b %d, %Y')}** only. "
    f"MAE and MAPE measured against **{len(df_test):,} days** the model never saw during training."
)

# --- SCENARIO SETUP ---
last_full_date = df_full['ds'].max()
last_full_val  = float(df_full[df_full['ds'] == last_full_date]['y'].values[0])
future_only    = forecast[forecast['ds'] > last_full_date]
future_dates   = future_only['ds'].reset_index(drop=True)

rates = {
    f'Base Case ({base_growth:+.0f}% Annual)'        : (base_growth / 100,    '#ff7f0e'),
    f'Upside Case ({upside_growth:+.0f}% Annual)'    : (upside_growth / 100,  '#2ca02c'),
    f'Downside Case ({downside_growth:+.0f}% Annual)': (downside_growth / 100,'#d62728')
}

# --- MAIN CHART ---
st.subheader(f"{ticker} Forecast — Train/Test Split + {forecast_days}-Day Future Outlook")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_daily[date_col], y=df_daily['Close'],
    name="Raw Close Price", mode='lines',
    line=dict(color='#e0e0e0', width=1)
))

fig.add_trace(go.Scatter(
    x=df_train['ds'], y=df_train['y'],
    name="Training Data (Smoothed)", mode='lines',
    line=dict(color='#111111', width=2)
))

fig.add_trace(go.Scatter(
    x=df_test['ds'], y=df_test['y'],
    name="Holdout Actuals (Unseen)", mode='lines',
    line=dict(color='#9467bd', width=2.5)
))

fig.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat'],
    name="Prophet Forecast", mode='lines',
    line=dict(color='#0078d4', dash='dash', width=2)
))

if len(future_only) > 0:
    fig.add_trace(go.Scatter(
        x=pd.concat([future_only['ds'], future_only['ds'].iloc[::-1]]),
        y=pd.concat([future_only['yhat_upper'], future_only['yhat_lower'].iloc[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 120, 212, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name="80% Confidence Interval"
    ))

fig.add_trace(go.Scatter(
    x=[str(cutoff_date.date()), str(cutoff_date.date())],
    y=[df_full['y'].min(), df_full['y'].max()],
    mode='lines',
    name="Train / Test Cutoff",
    line=dict(color='gray', width=1.5, dash='dot'),
    showlegend=True
))

for scenario_name, (annual_rate, color) in rates.items():
    if len(future_dates) > 0:
        daily_rate = (1 + annual_rate) ** (1/365) - 1
        values = [last_full_val * ((1 + daily_rate) ** d) for d in range(1, len(future_dates) + 1)]
        plot_dates = pd.concat([pd.Series([last_full_date]), future_dates])
        plot_values = [last_full_val] + values
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
    height=580,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- FORECAST TABLE ---
if len(future_only) > 0:
    st.subheader("Monthly Price Targets")
    future_only_copy = future_only.copy()
    future_only_copy['Month'] = future_only_copy['ds'].dt.to_period('M')
    final = future_only_copy.groupby('Month').last().reset_index()
    final = final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6).copy()
    final.columns = ['Date', 'Forecast', 'Lower Bound (80%)', 'Upper Bound (80%)']
    final['Date'] = final['Date'].dt.strftime('%b %Y')
    for col in ['Forecast', 'Lower Bound (80%)', 'Upper Bound (80%)']:
        final[col] = final[col].apply(lambda x: f"${x:.2f}")
    st.dataframe(final, use_container_width=True, hide_index=True)

st.caption("⚠️ Disclaimer: This tool is for educational purposes only and does not constitute financial advice.")
