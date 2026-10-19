"""Build Stock_Price_Stochastic_Models.ipynb from the cells below.

Run `python build_notebook.py`, then execute it with
`jupyter nbconvert --to notebook --execute --inplace Stock_Price_Stochastic_Models.ipynb`.
"""

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Modelling Stock Price Movements with Stochastic Processes

**Stocks:** Apple (AAPL), JPMorgan Chase (JPM), Coca-Cola (KO)
**Models:** Geometric Brownian Motion (GBM), Ornstein-Uhlenbeck on log price (OU), Merton jump-diffusion

## README - how to run

| Step | Command |
|---|---|
| Install dependencies | `pip install -r requirements.txt` |
| Run this notebook | open it in Jupyter / VS Code / Colab and *Run All* |
| Or run the pipeline as a script | `python run_analysis.py` (add `--source yfinance` for fresh Yahoo Finance data, `--n-paths 5000` for more simulations) |
| Rebuild the PDF report | `python build_report.py` |

**Files**

* `stochastic_models.py` - data cleaning, diagnostics, the three models (calibration + simulation), risk metrics, backtest and plotting functions.
* `run_analysis.py` - the end-to-end pipeline; writes `figures/` and `results/`.
* `data/kaggle_sp500_AAPL_JPM_KO.csv` - daily prices for the three stocks, extracted from the Kaggle *S&P 500 stock data* dataset (Nugent, 2018), 8 Feb 2013 - 7 Feb 2018. It is bundled so the analysis runs offline and is exactly reproducible.

All randomness is seeded (`SEED = 42`), so re-running gives identical numbers."""),

    code("""import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

import stochastic_models as sm
import run_analysis as cfg

sm.set_plot_style()
pd.set_option("display.precision", 4)
pd.set_option("display.width", 140)"""),

    md("""## 1. Data collection and preprocessing

`load_prices_kaggle` reads the long-format Kaggle file and pivots it to one close-price column per ticker.
`clean_prices` then sorts by date, removes duplicate dates, treats non-positive prices as missing,
forward-fills gaps of up to three days, drops incomplete rows, and **flags** (without deleting)
returns whose robust z-score exceeds 6. Large equity moves are usually real news, and they are
exactly what the jump model has to learn from."""),

    code("""raw = sm.load_prices_kaggle(cfg.KAGGLE_CSV, cfg.TICKERS)
prices, cleaning_report = sm.clean_prices(raw)
print({k: v for k, v in cleaning_report.items() if k != "flagged_returns"})
print("Flagged large moves:", cleaning_report["flagged_returns"])
prices.tail()"""),

    code("""calib = prices.iloc[-(cfg.CALIBRATION_DAYS + 1):]   # last two years used for calibration
returns = sm.log_returns(prices)
ax = sm.plot_price_history(prices, calib.index[0])
ax.set_title("Daily closing prices, rebased to 100");"""),

    md("""## 2. Diagnostics that motivate the model choice

* **Jarque-Bera** p-values near zero and large excess kurtosis -> returns are fat-tailed, which a pure GBM cannot produce; this motivates the **jump-diffusion**.
* **ADF** test on log price: a low p-value would support **mean reversion (OU)**; a high p-value means we cannot reject a random walk (**GBM**).
* **Ljung-Box on squared returns**: a low p-value signals volatility clustering, a feature none of the three models has (a limitation)."""),

    code("""diag = pd.DataFrame({t: sm.diagnostics(calib[t]) for t in cfg.TICKERS}).T
diag[["ann_drift_log", "ann_vol", "skew", "excess_kurtosis", "jb_pvalue",
      "adf_pvalue", "half_life_days", "ljungbox_sq_ret_p", "max_drawdown"]]"""),

    code("""sm.plot_return_diagnostics(returns.loc[calib.index[1]:]);"""),

    md("""## 3. Models

| Model | SDE | Calibration |
|---|---|---|
| GBM | $dS_t = \\mu S_t\\,dt + \\sigma S_t\\,dW_t$ | MLE from log returns |
| OU (log price) | $dX_t = \\kappa(\\theta - X_t)\\,dt + \\sigma\\,dW_t,\\; X_t = \\ln S_t$ | OLS on the exact AR(1) discretisation |
| Merton | $dS_t/S_{t^-} = (\\mu - \\lambda k)\\,dt + \\sigma\\,dW_t + (J-1)\\,dN_t,\\; \\ln J \\sim N(\\mu_J, \\sigma_J^2)$ | MLE of the Poisson mixture of normals |

All three are simulated with their **exact** transition laws, so there is no time-discretisation error.
Below, each model is calibrated on the last two years of AAPL and simulated once as a demonstration."""),

    code("""rng = np.random.default_rng(cfg.SEED)
s0 = float(prices["AAPL"].iloc[-1])
demo = {}
for name, model in sm.make_models().items():
    model.fit(calib["AAPL"])
    print(f"{name:7s}", {k: round(v, 4) for k, v in model.params().items()})
    demo[name] = model.simulate(s0, 180, cfg.N_PATHS, rng)
sm.plot_fan_charts(demo, "AAPL (demo, x-axis in trading days)");"""),

    md("""## 4. Full Monte Carlo study, risk metrics and backtest

`run_analysis.main()` repeats the steps above for every stock and model with 1,000 paths each over
30, 60 and 180 trading days, computes the risk metrics, runs a GBM drift-sensitivity check and the
rolling-origin backtest (70 forecast origins per stock), and saves every figure and table.
It takes about 1-2 minutes."""),

    code("""cfg.main([])"""),

    code("""params = pd.read_csv("results/model_parameters.csv")
params"""),

    code("""metrics = pd.read_csv("results/forecast_risk_metrics.csv")
metrics.pivot_table(index=["ticker", "horizon_days"], columns="model",
                    values=["median", "prob_loss", "var_95", "es_95"]).round(3)"""),

    code("""for f in ["fig3_fan_AAPL", "fig3_fan_JPM", "fig3_fan_KO",
          "fig4_terminal_distributions", "fig5_probability_of_loss"]:
    display(Image(f"figures/{f}.png", width=820))"""),

    md("""### Drift sensitivity

The historical drift is the least reliable input: a two-year sample cannot pin down an expected return.
GBM is re-simulated with the same volatility and three drifts."""),

    code("""pd.read_csv("results/gbm_drift_sensitivity.csv")[
    ["ticker", "drift", "mu", "median", "prob_loss", "prob_drop_10", "var_95", "es_95"]].round(3)"""),

    md("""### Out-of-sample backtest

At each origin, every model is calibrated only on the preceding two years and its forecast is scored
against the realised price. A well-calibrated model's 90% interval should contain the realised price
about 90% of the time; CRPS (lower is better) scores the whole forecast distribution; the Brier score
scores the predicted probability of a loss."""),

    code("""display(Image("figures/fig6_backtest_example.png", width=820))
pd.read_csv("results/backtest_summary.csv").round(3)"""),

    code("""display(Image("figures/fig7_backtest_summary.png", width=820))"""),

    md("""## 5. Key takeaways

* **GBM and Merton** produce well-calibrated 30/60-day intervals for AAPL and JPM (coverage 86-90% for a nominal 90%); for KO they are too wide (99-100%) because KO was calmer after calibration than during it.
* **OU** intervals are too narrow (coverage down to 69% for JPM): the ADF test cannot reject a unit root, so the fitted mean reversion is mostly an artefact of the sample and the model is overconfident.
* **Merton** adds a large likelihood improvement and fatter tails at short horizons, but its forecast scores are almost identical to GBM's: over 30+ days the jumps average out.
* **Drift dominates long-horizon risk**: AAPL's 180-day probability of loss moves from about 13% (historical drift) to about 40% (8% drift). Short-horizon risk (VaR, ES) depends mostly on volatility and is far more robust.

See `Stochastic_Stock_Price_Modeling_Report.pdf` for the full discussion and recommendations."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "Stock_Price_Stochastic_Models.ipynb")
print("Notebook written.")
