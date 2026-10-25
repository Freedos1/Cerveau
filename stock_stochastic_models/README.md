# Stochastic Processes Problem Set – Modelling Stock Price Movements

Monte Carlo comparison of three stochastic models for daily stock prices of
**Apple (AAPL), JPMorgan Chase (JPM) and Coca-Cola (KO)**:

| Model | SDE | Why it is included |
|---|---|---|
| Geometric Brownian Motion (GBM) | dS = μS dt + σS dW | Benchmark random walk; ADF tests cannot reject a unit root |
| Ornstein–Uhlenbeck on log price (OU) | dX = κ(θ − X) dt + σ dW, X = ln S | Tests the mean-reversion hypothesis (most plausible for KO) |
| Merton jump-diffusion | dS/S = (μ − λk) dt + σ dW + (J − 1) dN | Returns are fat-tailed (excess kurtosis 5–7) |

## Deliverables

| File | What it is |
|---|---|
| `Stock_Price_Stochastic_Models.ipynb` | **File 1** – executed notebook with README cells, step-by-step walkthrough, tables and figures |
| `stochastic_models.py` | Library: data cleaning, diagnostics, the three models (`fit` / `simulate`), risk metrics, backtest, plots |
| `run_analysis.py` | End-to-end pipeline that writes `figures/` and `results/` |
| `Stochastic_Stock_Price_Modeling_Report.docx` / `.pdf` | **File 2** – final report in Word and PDF (identical content: figures with captions, APA references) |
| `build_report.py`, `build_report_docx.js`, `build_notebook.py` | Regenerate the PDF + Word report and the notebook |
| `data/kaggle_sp500_AAPL_JPM_KO.csv` | Daily prices, 2013-02-08 – 2018-02-07, from the Kaggle *S&P 500 stock data* dataset (Nugent, 2018) |

## How to run

```bash
pip install -r requirements.txt
python run_analysis.py              # ~1–2 min: simulations, risk metrics, backtest -> figures/, results/
python build_report.py              # rebuild the PDF and Word report (Word needs Node.js + `npm install docx`)
jupyter notebook Stock_Price_Stochastic_Models.ipynb   # or VS Code; on Google Colab just upload the .ipynb and Run All
```

Options: `python run_analysis.py --source yfinance --start 2023-01-01` downloads current prices
from Yahoo Finance instead of using the bundled Kaggle file; `--n-paths 5000` increases the
number of Monte Carlo runs (default 1,000); `--skip-backtest` skips the slow part.
All randomness is seeded, so results are reproducible.

## What is computed

* **Preprocessing** – de-duplication, non-positive prices → missing, forward-fill of ≤3-day gaps, robust-z anomaly flagging (flagged moves are real news and are kept).
* **Diagnostics** – Jarque–Bera, ADF, AR(1) half-life, Ljung–Box on squared returns.
* **Calibration** – MLE (GBM, Merton) and exact AR(1) regression (OU) on the last two years.
* **Simulation** – exact transition laws, 1,000 paths × 180 trading days per model and stock; statistics at 30/60/180 days: median, 90% interval, P(loss), P(drop > 10%), VaR95, ES95.
* **Drift sensitivity** – GBM with historical vs 8% vs 0% drift.
* **Rolling backtest** – 70 forecast origins per stock; 90%-interval coverage, CRPS, Brier score.

## Headline results

* GBM and Merton 30/60-day intervals are well calibrated for AAPL and JPM (coverage 86–90% for a nominal 90%); too wide for KO.
* OU is overconfident out of sample (coverage 69–93%) – mean reversion in price levels is not supported.
* Merton fits daily returns much better (LR 85–121) but forecasts like GBM beyond a month.
* Long-horizon risk is driven by the drift assumption: AAPL's 180-day P(loss) is 13% with its historical drift vs 40% with an 8% drift.
