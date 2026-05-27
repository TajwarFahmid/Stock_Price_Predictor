---

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Ticker | AAPL | Any valid symbol | Stock to forecast |
| Forecast Horizon | 180 days | 30–365 | Days to project forward |
| Historical Window | 5 years | 2–5 | Training data range |
| Holdout Period | 6 months | 1–12 | Months withheld for evaluation |
| Moving Average | 20 days | 5–50 | Smoothing window |
| Base Growth | 8% | Any | Annual base case assumption |
| Upside Growth | 15% | Any | Annual upside case assumption |
| Downside Growth | -5% | Any | Annual downside case assumption |

---

## Limitations

- Prophet is a trend and seasonality model — it does not incorporate 
  fundamental data, earnings, macroeconomic indicators, or market sentiment
- Wide confidence intervals on longer horizons reflect genuine uncertainty, 
  not model weakness
- High MAPE on certain tickers indicates a trend reversal during the holdout 
  period that the model could not anticipate — this is expected behavior, 
  not a bug
- Past price trends do not guarantee future performance

---

## Disclaimer

This tool is for educational and portfolio purposes only. It does not constitute 
financial advice. Do not make investment decisions based on this model's output.

---

## Author

**Tajwar Fahmid**  
B.S. Data Science — University of Texas at Arlington, May 2026  
[linkedin.com/in/tajwar-fahmid](https://www.linkedin.com/in/tajwar-fahmid-0a1b7720b/) · 
[github.com/TajwarFahmid](https://github.com/TajwarFahmid) · 
[Tableau Portfolio](https://public.tableau.com/app/profile/tajwar.fahmid8295/vizzes)
