# Stochastic Processes Problem Set: Modelling Stock Price Movements

A quantitative study of **Apple Inc. (AAPL)** daily prices (13 Jul 2021 – 13 Jul 2026) with three stochastic
models: Geometric Brownian Motion, a trend-stationary Ornstein–Uhlenbeck process, and Merton jump-diffusion.
Each model is calibrated on the latest two years, simulated with 10,000-path Monte Carlo over 30, 60 and 180
trading days, and evaluated out of sample.

## Deliverables
| File | Description |
|---|---|
| `Stochastic_Stock_Models.ipynb` | **File 1**: executed notebook with README, code walk-through, tables and figures |
| `stock_models.py`, `plots.py` | Modular source code that the notebook imports (also runnable as a script) |
| `Stochastic_Stock_Models_Report.docx` / `.pdf` | **File 2**: final report with captioned figures and APA references |
| `figures/` | All 8 figures (PNG, 200 dpi) |
| `data/AAPL_raw.csv`, `data/AAPL_clean.csv` | Raw yfinance export and cleaned close series |
| `results_AAPL.json` | Every estimated parameter and risk metric, machine-readable |
| `build_report.js` | Generates the .docx report from `results_AAPL.json` and `figures/` |

## How to run
```bash
pip install -r requirements.txt
python stock_models.py                       # offline: uses data/AAPL_raw.csv, writes figures/ and results_AAPL.json
jupyter notebook Stochastic_Stock_Models.ipynb   # or: Kernel → Restart & Run All
```
To refresh the data (internet required):
```bash
python stock_models.py --source yfinance --tickers AAPL MSFT KO --start 2021-07-13
```
To rebuild the report: `node build_report.js` (needs the `docx` npm package), then convert to PDF with
`soffice --headless --convert-to pdf Stochastic_Stock_Models_Report.docx`.

## Data source
AAPL adjusted OHLCV downloaded from Yahoo Finance with `yfinance` and published as a CSV in the public repository
[Jonasmiles/AAPL-STOCK-ANALYSIS-2021-2026](https://github.com/Jonasmiles/AAPL-STOCK-ANALYSIS-2021-2026).
The analysis environment could not reach Yahoo Finance directly, so this export (last date 13 Jul 2026) was used.
`stock_models.download_yfinance()` reproduces the download wherever Yahoo Finance is reachable.
