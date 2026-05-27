# 📈 Stock Price Forecaster

An interactive web application that forecasts stock prices using Facebook's Prophet 
time series model, with a proper train/test split to evaluate real out-of-sample 
accuracy. Built with Python and deployed on Streamlit.

🔗 **Live App**: [your-app-url.streamlit.app]([https://your-app-url.streamlit.app](https://aoy3jftblzvvfeijjywwsu.streamlit.app))

---

## What It Does

Most forecasting projects evaluate their model on the same data it trained on, 
which produces artificially low error rates. This app withholds the most recent 
months of price data completely from training and uses them only to measure how 
accurately the model predicted prices it had never seen — the same standard used 
in professional financial modeling.

### Key Features

- **True holdout evaluation** — model trains on historical data only, then is 
  tested against a user-defined period of unseen prices
- **Dynamic ticker input** — forecast any publicly traded stock on demand
- **Multi-scenario modeling** — base, upside, and downside growth projections 
  with user-defined annual growth rate assumptions
- **Confidence intervals** — Prophet's 80% prediction bands visualized across 
  the forecast horizon
- **Model confidence label** — automatic classification of forecast strength 
  (Strong / Moderate / Weak) based on MAPE thresholds
- **Normalized MAE** — error expressed as a percentage of average holdout price 
  so results are comparable across different-priced stocks
- **Monthly price targets table** — aggregated end-of-month forecast values with 
  upper and lower bounds
- **Fully adjustable parameters** — historical window, holdout period, smoothing 
  window, forecast horizon, and growth assumptions all controlled via sidebar

---

## How the Model Works

1. **Data ingestion** — pulls historical daily price data via the `yfinance` API
2. **Smoothing** — applies a rolling moving average (default 20-day) to reduce 
   short-term noise before modeling
3. **Train/test split** — withholds the most recent N months as a holdout set 
   the model never sees during training
4. **Prophet training** — fits a decomposable time series model with yearly 
   seasonality and multiplicative growth on the training set only
5. **Forecast generation** — projects prices through the holdout period and 
   beyond into a user-defined future horizon
6. **Holdout evaluation** — computes MAE and MAPE by comparing Prophet's 
   predictions against actual holdout prices
7. **Scenario overlay** — applies user-defined compounding growth rates to 
   generate base, upside, and downside projections from the last known price

---

## Model Performance (Example — AAPL, 6-Month Holdout)

| Metric | Value |
|--------|-------|
| Training Days | 1,132 |
| Holdout Days | 122 |
| MAE | $10.66 |
| MAPE | 3.92% |
| Normalized MAE | ~5.1% of avg price |

> Performance varies significantly by ticker. Stocks that experienced trend 
> reversals or high volatility during the holdout period will show higher error 
> rates — the app flags these automatically with a warning label.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Ingestion | yfinance, pandas |
| Modeling | Prophet (Facebook/Meta) |
| Evaluation | scikit-learn (MAE, MAPE) |
| Visualization | Plotly |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |
| Language | Python 3.12 |

---


