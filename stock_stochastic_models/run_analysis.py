"""
run_analysis.py
===============

End-to-end pipeline: data -> diagnostics -> calibration -> Monte Carlo
simulation -> risk metrics -> backtest -> figures and tables.

Usage
-----
    python run_analysis.py                    # bundled Kaggle data (offline)
    python run_analysis.py --source yfinance  # fresh data from Yahoo Finance
    python run_analysis.py --n-paths 5000     # more Monte Carlo runs

Outputs are written to ./figures (PNG) and ./results (CSV / JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stochastic_models as sm

HERE = Path(__file__).resolve().parent

# ----------------------------- configuration -------------------------------
TICKERS = ["AAPL", "JPM", "KO"]          # growth tech, large bank, defensive staple
KAGGLE_CSV = HERE / "data" / "kaggle_sp500_AAPL_JPM_KO.csv"
CALIBRATION_DAYS = 504                   # two years of trading days
HORIZONS = [30, 60, 180]                 # forecast horizons in trading days
N_PATHS = 1000                           # Monte Carlo runs per model and ticker
BACKTEST_HORIZONS = [30, 60]
BACKTEST_STEP = 10                       # trading days between forecast origins
SEED = 42                                # reproducibility


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["kaggle", "yfinance"], default="kaggle")
    p.add_argument("--start", default="2019-01-01", help="yfinance start date")
    p.add_argument("--end", default=None, help="yfinance end date (default: today)")
    p.add_argument("--n-paths", type=int, default=N_PATHS)
    p.add_argument("--skip-backtest", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    """Run the whole pipeline; `argv` lets a notebook pass CLI options."""
    args = parse_args(argv)
    fig_dir, res_dir = HERE / "figures", HERE / "results"
    fig_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)
    sm.set_plot_style()
    rng = np.random.default_rng(SEED)

    # ---- 1. Data collection and cleaning -----------------------------------
    if args.source == "kaggle":
        raw = sm.load_prices_kaggle(KAGGLE_CSV, TICKERS)
    else:
        end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
        raw = sm.load_prices_yfinance(TICKERS, args.start, end)
    prices, cleaning = sm.clean_prices(raw)
    print(f"Data: {cleaning['start']} to {cleaning['end']}, {cleaning['rows_out']} trading days")

    calib = prices.iloc[-(CALIBRATION_DAYS + 1):]          # most recent two years
    returns = sm.log_returns(prices)

    # ---- 2. Diagnostics (full sample and calibration window) ---------------
    diag = pd.DataFrame({t: sm.diagnostics(prices[t]) for t in TICKERS}).T
    diag_calib = pd.DataFrame({t: sm.diagnostics(calib[t]) for t in TICKERS}).T
    diag.to_csv(res_dir / "diagnostics_full_sample.csv")
    diag_calib.to_csv(res_dir / "diagnostics_calibration_window.csv")

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    sm.plot_price_history(prices, calib.index[0], ax=ax)
    fig.savefig(fig_dir / "fig1_price_history.png"); plt.close(fig)
    fig = sm.plot_return_diagnostics(returns)
    fig.savefig(fig_dir / "fig2_return_diagnostics.png"); plt.close(fig)

    # ---- 3. Calibration and forward Monte Carlo simulation -----------------
    max_h = max(HORIZONS)
    future_dates = pd.bdate_range(prices.index[-1], periods=max_h + 1)
    params, metrics, sims_by_ticker = {}, [], {}
    for t in TICKERS:
        s0 = float(prices[t].iloc[-1])
        sims_by_ticker[t] = {}
        params[t] = {}
        for name, model in sm.make_models().items():
            model.fit(calib[t])
            params[t][name] = model.params()
            paths = model.simulate(s0, max_h, args.n_paths, rng)
            sims_by_ticker[t][name] = paths
            for h in HORIZONS:
                metrics.append({"ticker": t, "model": name, "s0": s0,
                                **sm.terminal_risk_metrics(paths, h)})
        fig = sm.plot_fan_charts(sims_by_ticker[t], t, dates=future_dates,
                                 history=prices[t].iloc[-120:])
        fig.savefig(fig_dir / f"fig3_fan_{t}.png"); plt.close(fig)

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(res_dir / "forecast_risk_metrics.csv", index=False)
    param_rows = [{"ticker": t, "model": m, **p} for t in params for m, p in params[t].items()]
    pd.DataFrame(param_rows).to_csv(res_dir / "model_parameters.csv", index=False)

    # Drift sensitivity: the historical drift is the least reliable input, so
    # re-run GBM with the same volatility but a long-run equity drift (8%) and
    # with zero drift, and compare the 180-day risk numbers.
    sens = []
    for t in TICKERS:
        hist = sm.GBM().fit(calib[t])
        for label, mu in (("historical", hist.mu), ("8% long-run", 0.08), ("zero", 0.0)):
            paths = sm.GBM(mu=mu, sigma=hist.sigma).simulate(
                float(prices[t].iloc[-1]), max_h, args.n_paths, rng)
            sens.append({"ticker": t, "drift": label, "mu": mu,
                         **sm.terminal_risk_metrics(paths, max_h)})
    pd.DataFrame(sens).to_csv(res_dir / "gbm_drift_sensitivity.csv", index=False)

    fig = sm.plot_terminal_histograms(sims_by_ticker, max_h)
    fig.savefig(fig_dir / "fig4_terminal_distributions.png"); plt.close(fig)
    fig = sm.plot_prob_loss(sims_by_ticker, max_h)
    fig.savefig(fig_dir / "fig5_probability_of_loss.png"); plt.close(fig)

    # ---- 4. Out-of-sample check: single-origin illustration ----------------
    origin = len(prices) - max_h - 1
    showcase = {}
    for t in TICKERS:
        window = prices[t].iloc[origin - CALIBRATION_DAYS:origin + 1]
        s0 = float(prices[t].iloc[origin])
        sims = {name: model.fit(window).simulate(s0, max_h, args.n_paths, rng)
                for name, model in sm.make_models().items()}
        realized = prices[t].iloc[origin:origin + max_h + 1]
        showcase[t] = {name: float((p[:, -1] <= realized.iloc[-1]).mean())
                       for name, p in sims.items()}
        if t == TICKERS[0]:
            fig = sm.plot_fan_charts(sims, t, dates=realized.index,
                                     history=prices[t].iloc[origin - 120:origin + 1],
                                     realized=realized)
            fig.savefig(fig_dir / "fig6_backtest_example.png"); plt.close(fig)

    # ---- 5. Rolling-origin backtest ----------------------------------------
    summary = None
    if not args.skip_backtest:
        frames = []
        for t in TICKERS:
            bt = sm.rolling_backtest(prices[t], CALIBRATION_DAYS, BACKTEST_HORIZONS,
                                     BACKTEST_STEP, args.n_paths, SEED)
            bt.insert(0, "ticker", t)
            frames.append(bt)
            print(f"Backtest {t}: {bt['origin'].nunique()} origins")
        bt_all = pd.concat(frames, ignore_index=True)
        bt_all.to_csv(res_dir / "backtest_detail.csv", index=False)
        summary = sm.summarize_backtest(bt_all)
        summary.to_csv(res_dir / "backtest_summary.csv", index=False)
        fig = sm.plot_backtest_summary(summary)
        fig.savefig(fig_dir / "fig7_backtest_summary.png"); plt.close(fig)

    # ---- 6. Machine-readable summary for the report ------------------------
    out = {
        "source": args.source, "n_paths": args.n_paths, "seed": SEED,
        "calibration_start": calib.index[0], "calibration_end": calib.index[-1],
        "cleaning": cleaning, "parameters": params,
        "showcase_origin": prices.index[origin],
        "showcase_realized_percentile": showcase,
    }
    with open(res_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2, default=sm.to_serializable)

    pd.set_option("display.width", 160)
    print(diag_calib.round(4))
    print(pd.DataFrame(param_rows).round(4))
    print(metrics.round(4))
    if summary is not None:
        print(summary.round(4))


if __name__ == "__main__":
    # Render to files only. Set here, not at import, so that importing this
    # module from a notebook keeps inline plots working.
    matplotlib.use("Agg")
    main()
