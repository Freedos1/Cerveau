"""
plots.py
========
Figures for the stochastic stock-price study. ``make_all`` writes every
figure used in the report to ``figures/`` as 200-dpi PNG files.

Colour roles: each model keeps one fixed hue in every figure (GBM blue,
OU orange, Merton aqua); observed prices are drawn in neutral ink.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import stock_models as sm

MODEL_COLORS = {"GBM": "#2a78d6", "OU (trend)": "#eb6834", "Merton JD": "#1baf7a"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0,
    "font.family": "DejaVu Sans",
})


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _future_dates(last: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last, periods=n + 1)


def fig_price_history(prices: pd.Series, res: dict, out: Path) -> None:
    r = sm.log_returns(prices)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(prices.index, prices.values, color=INK, lw=1.4)
    for a in (a1, a2):
        a.axvspan(res["calib_start"], res["calib_end"], color=MODEL_COLORS["GBM"], alpha=0.07, lw=0)
    a1.text(res["calib_start"], prices.max() * 0.98, "  calibration window (504 days)", color=INK_2, fontsize=8.5, va="top")
    a1.set_ylabel("Adjusted close (USD)")
    a1.set_title(f"{prices.name} daily adjusted close and log returns")
    a2.plot(r.index, r.values, color=INK_2, lw=0.6)
    big = r[np.abs(r) > 0.07]
    a2.scatter(big.index, big.values, s=18, color=MODEL_COLORS["OU (trend)"], zorder=3,
               edgecolor="white", linewidth=1, label="|return| > 7%")
    a2.set_ylabel("Daily log return")
    a2.legend(loc="upper left")
    a2.xaxis.set_major_locator(mdates.YearLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.align_ylabels()
    _save(fig, out, "fig1_price_history.png")


def fig_return_distribution(res: dict, out: Path) -> None:
    r = sm.log_returns(res["calib"]).values
    gbm, mer = res["params"]["GBM"], res["params"]["Merton JD"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.hist(r, bins=70, density=True, color=GRID, edgecolor="white", linewidth=0.5, label="Observed")
    x = np.linspace(r.min() * 1.1, r.max() * 1.1, 600)
    a1.plot(x, stats.norm.pdf(x, (gbm.mu - gbm.sigma**2 / 2) * sm.DT, gbm.sigma * np.sqrt(sm.DT)),
            color=MODEL_COLORS["GBM"], label="GBM (normal)")
    m = mer.mu - 0.5 * mer.sigma**2 - mer.lam * mer.k
    a1.plot(x, np.exp(sm._merton_logpdf(x, m, mer.sigma, mer.lam, mer.mu_j, mer.sigma_j)),
            color=MODEL_COLORS["Merton JD"], label="Merton jump-diffusion")
    a1.set_title("Daily log returns vs fitted densities")
    a1.set_xlabel("Daily log return")
    a1.set_ylabel("Density")
    a1.legend()

    (osm, osr), (slope, icpt, _) = stats.probplot(r, dist="norm")
    a2.scatter(osm, osr, s=10, color=INK_2, alpha=0.7, linewidth=0)
    a2.plot(osm, slope * osm + icpt, color=MODEL_COLORS["GBM"], lw=1.5, label="Normal reference")
    a2.set_title("Normal Q-Q plot: fat tails at both ends")
    a2.set_xlabel("Theoretical normal quantile")
    a2.set_ylabel("Observed quantile")
    a2.legend(loc="upper left")
    _save(fig, out, "fig2_return_distribution.png")


def fig_sample_paths(res: dict, out: Path, n_show: int = 100, history_days: int = 126) -> None:
    calib, sims = res["calib"], res["sims"]
    hist = calib.iloc[-history_days:]
    n_days = next(iter(sims.values())).shape[1] - 1
    fut = _future_dates(calib.index[-1], n_days)
    ymax = max(np.quantile(p[:, -1], 0.99) for p in sims.values())
    ymin = min(np.quantile(p.min(axis=1), 0.01) for p in sims.values())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (name, paths) in zip(axes, sims.items()):
        c = MODEL_COLORS[name]
        ax.plot(fut, paths[:n_show].T, color=c, alpha=0.12, lw=0.6)
        lo, q25, q75, hi = np.quantile(paths, [0.05, 0.25, 0.75, 0.95], axis=0)
        ax.fill_between(fut, lo, hi, color=c, alpha=0.12, lw=0, label="90% interval")
        ax.fill_between(fut, q25, q75, color=c, alpha=0.22, lw=0, label="50% interval")
        ax.plot(fut, paths.mean(axis=0), color=INK, lw=2, label="Mean")
        ax.plot(fut, np.median(paths, axis=0), color=INK, lw=1.5, ls="--", label="Median")
        ax.plot(hist.index, hist.values, color=INK_2, lw=1.3, label="Observed")
        ax.set_title(name)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Simulated {n_days}-day price paths ({n_show} of {paths.shape[0]:,} shown) with mean, median and intervals",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    _save(fig, out, "fig3_sample_paths.png")


def fig_fan_comparison(res: dict, out: Path) -> None:
    calib, sims = res["calib"], res["sims"]
    n_days = next(iter(sims.values())).shape[1] - 1
    fut = _future_dates(calib.index[-1], n_days)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    hist = calib.iloc[-126:]
    ax.plot(hist.index, hist.values, color=INK, lw=1.4, label="Observed")
    for name, paths in sims.items():
        c = MODEL_COLORS[name]
        lo, med, hi = np.quantile(paths, [0.05, 0.5, 0.95], axis=0)
        ax.plot(fut, lo, color=c, lw=1.2, ls=":")
        ax.plot(fut, hi, color=c, lw=1.2, ls=":")
        ax.plot(fut, med, color=c, lw=2,
                label=f"{name}: median ${med[-1]:.0f}, 90% [{lo[-1]:.0f}, {hi[-1]:.0f}] at day {n_days}")
    ax.set_title("Model comparison: median forecast (solid) and 90% interval (dotted bounds)")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _save(fig, out, "fig4_model_comparison_fan.png")


def fig_terminal_hist(res: dict, out: Path, horizons=(30, 60, 180)) -> None:
    sims, s0 = res["sims"], res["s0"]
    fig, axes = plt.subplots(1, len(horizons), figsize=(13, 3.9))
    for ax, h in zip(axes, horizons):
        allp = np.concatenate([p[:, h] for p in sims.values()])
        bins = np.linspace(np.quantile(allp, 0.002), np.quantile(allp, 0.998), 70)
        for name, paths in sims.items():
            ax.hist(paths[:, h], bins=bins, density=True, histtype="step", lw=1.8,
                    color=MODEL_COLORS[name], label=name)
        ax.axvline(s0, color=INK, lw=1.2, ls="--")
        ax.text(s0, ax.get_ylim()[1] * 0.97, f" today ${s0:.0f}", fontsize=8, color=INK_2, va="top")
        ax.set_title(f"Price after {h} trading days")
        ax.set_xlabel("Price (USD)")
        ax.set_yticks([])
    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    _save(fig, out, "fig5_terminal_histograms.png")


def fig_risk_vs_horizon(res: dict, out: Path) -> None:
    sims, s0 = res["sims"], res["s0"]
    n_days = next(iter(sims.values())).shape[1] - 1
    hs = np.arange(5, n_days + 1, 5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for name, paths in sims.items():
        ret = paths[:, hs] / s0 - 1
        c = MODEL_COLORS[name]
        a1.plot(hs, (ret < 0).mean(axis=0) * 100, color=c, label=name)
        a2.plot(hs, -np.quantile(ret, 0.05, axis=0) * 100, color=c, label=name)
    a1.set_title("Probability of a loss vs horizon")
    a1.set_ylabel("P(price below today) (%)")
    a2.set_title("95% Value-at-Risk vs horizon")
    a2.set_ylabel("Loss not exceeded with 95% confidence (%)")
    for a in (a1, a2):
        a.set_xlabel("Horizon (trading days)")
        a.legend()
    _save(fig, out, "fig6_risk_vs_horizon.png")


def fig_holdout(prices: pd.Series, res: dict, out: Path) -> None:
    ho = res["holdout"]
    train, test, sims = ho["train"], ho["test"], ho["sims"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    hist = train.iloc[-126:]
    ax.plot(hist.index, hist.values, color=INK_2, lw=1.3, label="Calibration data")
    for name, paths in sims.items():
        c = MODEL_COLORS[name]
        lo, med, hi = np.quantile(paths, [0.05, 0.5, 0.95], axis=0)
        ax.plot(test.index, lo, color=c, lw=1.2, ls=":")
        ax.plot(test.index, hi, color=c, lw=1.2, ls=":")
        ax.plot(test.index, med, color=c, lw=1.8, label=f"{name} median (dotted: 90% band)")
    ax.plot(test.index, test.values, color=INK, lw=2, label="Realised price (out of sample)")
    ax.set_title(f"Hold-out test: models fitted to data up to {train.index[-1]:%d %b %Y}, "
                 f"then simulated {len(test) - 1} days")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _save(fig, out, "fig7_holdout_test.png")


def fig_backtest(res: dict, out: Path) -> None:
    bt = res["backtest"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, (name, g) in zip(axes, bt.groupby("model", sort=False)):
        ax.hist(g["pit"], bins=np.linspace(0, 1, 11), color=MODEL_COLORS[name], edgecolor="white", linewidth=2)
        ax.axhline(len(g) / 10, color=INK, lw=1, ls="--", label="Uniform (ideal)")
        cov = g["inside90"].mean() * 100
        ax.set_title(f"{name}: 90% coverage {cov:.0f}%")
        ax.set_xlabel("PIT of realised 30-day price")
    axes[0].set_ylabel("Number of forecasts")
    axes[0].legend(loc="upper right", fontsize=8)
    _save(fig, out, "fig8_backtest_pit.png")


def make_all(prices: pd.Series, res: dict, out: Path) -> None:
    out = Path(out)
    out.mkdir(exist_ok=True)
    fig_price_history(prices, res, out)
    fig_return_distribution(res, out)
    fig_sample_paths(res, out)
    fig_fan_comparison(res, out)
    fig_terminal_hist(res, out)
    fig_risk_vs_horizon(res, out)
    fig_holdout(prices, res, out)
    fig_backtest(res, out)
