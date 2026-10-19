"""
stochastic_models.py
====================

Stochastic-process models for simulating stock price paths.

This module holds everything the analysis needs, grouped as:

1. Data handling     - loading (Kaggle CSV or yfinance), cleaning, log returns.
2. Diagnostics       - statistical tests that motivate the choice of models.
3. Models            - Geometric Brownian Motion (GBM), Ornstein-Uhlenbeck on
                       log price (OU), and Merton jump-diffusion (MJD). Each
                       model has `fit(prices)` (calibration from history) and
                       `simulate(s0, n_steps, n_paths, rng)` (Monte Carlo).
4. Monte Carlo       - risk metrics of simulated terminal prices.
5. Backtesting       - rolling-origin out-of-sample evaluation.
6. Plotting          - the figures used in the report.

Run `python run_analysis.py` to reproduce every figure and table.

Conventions
-----------
* Time is measured in years; one trading day is dt = 1/252.
* Prices are daily closing prices. All model parameters are annualised.
* Simulated paths are arrays of shape (n_paths, n_steps + 1); column 0 is S0.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

TRADING_DAYS_PER_YEAR = 252
DT = 1.0 / TRADING_DAYS_PER_YEAR

# Colour per model, fixed so a model has the same colour in every figure.
MODEL_COLORS = {"GBM": "#2a78d6", "OU": "#eb6834", "Merton": "#1baf7a"}
TICKER_COLORS = {"AAPL": "#2a78d6", "JPM": "#eb6834", "KO": "#1baf7a"}
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
REALIZED_COLOR = "#e34948"


# ---------------------------------------------------------------------------
# 1. Data handling
# ---------------------------------------------------------------------------

def load_prices_kaggle(csv_path: str | Path, tickers: list[str]) -> pd.DataFrame:
    """Load closing prices from the Kaggle "S&P 500 stock data" CSV layout.

    The file is in long format (date, open, high, low, close, volume, Name).
    Returns a wide DataFrame indexed by date with one close-price column per ticker.
    """
    raw = pd.read_csv(csv_path, parse_dates=["date"])
    raw = raw[raw["Name"].isin(tickers)]
    # Duplicate (date, ticker) rows would break the pivot; keep the last one.
    raw = raw.drop_duplicates(subset=["date", "Name"], keep="last")
    wide = raw.pivot(index="date", columns="Name", values="close")
    return wide[tickers].sort_index()


def load_prices_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily closes from Yahoo Finance (needs network access)."""
    import yfinance as yf  # imported lazily: optional dependency

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    closes = data["Close"]
    if isinstance(closes, pd.Series):  # a single ticker comes back as a Series
        closes = closes.to_frame(tickers[0])
    return closes[tickers].sort_index()


def clean_prices(prices: pd.DataFrame, max_ffill_days: int = 3,
                 outlier_z: float = 6.0) -> tuple[pd.DataFrame, dict]:
    """Clean a wide price table and report what was done.

    Steps
    -----
    1. Sort by date and drop duplicate dates.
    2. Treat non-positive prices as missing (log prices need S > 0).
    3. Forward-fill gaps of at most `max_ffill_days` (e.g. a missing quote);
       rows still incomplete afterwards are dropped.
    4. Flag (but keep) daily log returns whose robust z-score exceeds
       `outlier_z`. Large moves in equities are usually genuine (earnings,
       macro news) and are exactly what the jump model should see, so they
       are reported for manual review rather than deleted.
    """
    report: dict = {"rows_in": int(len(prices))}
    df = prices.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    report["duplicate_dates_removed"] = report["rows_in"] - int(len(df))

    report["non_positive_prices"] = int((df <= 0).sum().sum())
    df = df.where(df > 0)

    report["missing_before_fill"] = int(df.isna().sum().sum())
    df = df.ffill(limit=max_ffill_days)
    rows_before_drop = len(df)
    df = df.dropna()
    report["rows_dropped_incomplete"] = int(rows_before_drop - len(df))

    log_ret = np.log(df).diff().dropna()
    median = log_ret.median()
    mad = (log_ret - median).abs().median() * 1.4826  # MAD -> sigma for a normal
    robust_z = (log_ret - median) / mad
    flagged = robust_z.abs() > outlier_z
    report["flagged_returns"] = {
        col: [(d.strftime("%Y-%m-%d"), round(float(log_ret.loc[d, col]), 4))
              for d in log_ret.index[flagged[col]]]
        for col in df.columns
    }
    report["rows_out"] = int(len(df))
    report["start"] = df.index[0].strftime("%Y-%m-%d")
    report["end"] = df.index[-1].strftime("%Y-%m-%d")
    return df, report


def log_returns(prices: pd.Series | pd.DataFrame):
    """Daily log returns r_t = ln(S_t / S_{t-1})."""
    return np.log(prices).diff().dropna()


# ---------------------------------------------------------------------------
# 2. Diagnostics used to justify the models
# ---------------------------------------------------------------------------

def diagnostics(price: pd.Series) -> dict:
    """Summary statistics and tests for one price series.

    * Annualised drift and volatility of log returns (GBM inputs).
    * Skewness / excess kurtosis and the Jarque-Bera normality test: fat tails
      argue for a jump component.
    * Augmented Dickey-Fuller test on log price: rejecting a unit root would
      support mean reversion (OU); failing to reject supports a random walk.
    * AR(1) mean-reversion half-life of log price.
    * Ljung-Box on squared returns: volatility clustering, which none of the
      three constant-volatility models captures (a stated limitation).
    """
    r = log_returns(price)
    log_p = np.log(price)
    jb_stat, jb_p = stats.jarque_bera(r)
    with warnings.catch_warnings():  # statsmodels return-type FutureWarning
        warnings.simplefilter("ignore", FutureWarning)
        adf_stat, adf_p, *_ = adfuller(log_p, regression="c", autolag="AIC")
    b = np.polyfit(log_p.values[:-1], log_p.values[1:], 1)[0]
    half_life_days = np.log(2) / -np.log(b) if 0 < b < 1 else np.inf
    lb_sq = acorr_ljungbox(r ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    lb_r = acorr_ljungbox(r, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    return {
        "n_obs": int(len(r)),
        "ann_drift_log": float(r.mean() * TRADING_DAYS_PER_YEAR),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "jb_pvalue": float(jb_p),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "ar1_coef": float(b),
        "half_life_days": float(half_life_days),
        "ljungbox_ret_p": float(lb_r),
        "ljungbox_sq_ret_p": float(lb_sq),
        "max_drawdown": float((price / price.cummax() - 1).min()),
    }


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------

@dataclass
class GBM:
    """Geometric Brownian Motion:  dS = mu * S dt + sigma * S dW.

    Exact solution used for simulation (no discretisation error):
        S_{t+dt} = S_t * exp((mu - sigma^2 / 2) dt + sigma * sqrt(dt) * Z).
    Calibration is the maximum-likelihood estimate from i.i.d. normal log returns.
    """
    mu: float = 0.0
    sigma: float = 0.2
    name: str = field(default="GBM", repr=False)

    def fit(self, prices: pd.Series) -> "GBM":
        r = log_returns(prices).values
        self.sigma = float(r.std(ddof=1) / np.sqrt(DT))
        # E[log return] = (mu - sigma^2/2) dt  ->  mu = mean/dt + sigma^2/2
        self.mu = float(r.mean() / DT + 0.5 * self.sigma ** 2)
        return self

    def simulate(self, s0: float, n_steps: int, n_paths: int,
                 rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((n_paths, n_steps))
        increments = (self.mu - 0.5 * self.sigma ** 2) * DT + self.sigma * np.sqrt(DT) * z
        return _paths_from_log_increments(s0, increments)

    def params(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma}


@dataclass
class OrnsteinUhlenbeck:
    """Mean-reverting Ornstein-Uhlenbeck process on the log price X = ln S:

        dX = kappa * (theta - X) dt + sigma dW.

    This is the one-factor model of Schwartz (1997). The price reverts towards
    exp(theta) with speed kappa (half-life ln2 / kappa years).

    Calibration: the exact discretisation is an AR(1),
        X_{t+1} = a + b X_t + e,   b = exp(-kappa dt),  a = theta (1 - b),
        Var(e) = sigma^2 (1 - b^2) / (2 kappa),
    so an OLS regression of X_{t+1} on X_t gives (kappa, theta, sigma).
    If the fitted b >= 1 there is no mean reversion in the sample; kappa is then
    floored at `min_kappa` so the model degenerates gracefully towards a
    random walk instead of failing.
    """
    kappa: float = 1.0
    theta: float = 0.0
    sigma: float = 0.2
    min_kappa: float = 1e-3
    name: str = field(default="OU", repr=False)

    def fit(self, prices: pd.Series) -> "OrnsteinUhlenbeck":
        x = np.log(prices.values)
        b, a = np.polyfit(x[:-1], x[1:], 1)
        resid = x[1:] - (a + b * x[:-1])
        resid_var = resid.var(ddof=2)
        if 0 < b < 1:
            self.kappa = float(-np.log(b) / DT)
            self.theta = float(a / (1 - b))
        else:
            self.kappa = self.min_kappa
            self.theta = float(x.mean())
        b_used = np.exp(-self.kappa * DT)
        self.sigma = float(np.sqrt(resid_var * 2 * self.kappa / (1 - b_used ** 2)))
        return self

    def simulate(self, s0: float, n_steps: int, n_paths: int,
                 rng: np.random.Generator) -> np.ndarray:
        b = np.exp(-self.kappa * DT)
        step_sd = self.sigma * np.sqrt((1 - b ** 2) / (2 * self.kappa))
        x = np.empty((n_paths, n_steps + 1))
        x[:, 0] = np.log(s0)
        z = rng.standard_normal((n_paths, n_steps))
        for t in range(n_steps):  # exact transition, step by step
            x[:, t + 1] = self.theta + (x[:, t] - self.theta) * b + step_sd * z[:, t]
        return np.exp(x)

    def half_life_days(self) -> float:
        return float(np.log(2) / self.kappa * TRADING_DAYS_PER_YEAR)

    def params(self) -> dict:
        return {"kappa": self.kappa, "theta_price": float(np.exp(self.theta)),
                "sigma": self.sigma, "half_life_days": self.half_life_days()}


@dataclass
class MertonJumpDiffusion:
    """Merton (1976) jump-diffusion:

        dS / S- = (mu - lam * k) dt + sigma dW + (J - 1) dN,

    where N is a Poisson process with intensity lam (jumps per year), the log
    jump size ln J ~ Normal(mu_j, sigma_j^2) and k = E[J - 1] = exp(mu_j + sigma_j^2/2) - 1.
    The compensator lam * k keeps mu as the total expected return, which makes
    the drift directly comparable to GBM.

    Calibration is maximum likelihood. A daily log return is a Poisson mixture
    of normals:
        f(r) = sum_n P(N = n) * Normal(r; alpha*dt + n*mu_j, sigma^2*dt + n*sigma_j^2),
    with alpha = mu - sigma^2/2 - lam*k, truncated at `max_jumps` per day.

    Identification: on two years of daily data the unconstrained likelihood
    prefers very frequent, tiny jumps (lam at any upper bound), i.e. it uses
    the jump term as a generic fat-tail mixture. Capping lam at `max_lam`
    (default 12 per year, about one per month: earnings releases plus a few
    macro shocks) keeps jumps economically meaningful "rare, large" events.
    """
    mu: float = 0.0
    sigma: float = 0.2
    lam: float = 5.0
    mu_j: float = 0.0
    sigma_j: float = 0.03
    max_jumps: int = 10
    max_lam: float = 12.0
    min_sigma_j: float = 0.005
    name: str = field(default="Merton", repr=False)

    def _neg_log_likelihood(self, theta: np.ndarray, r: np.ndarray) -> float:
        alpha, sigma, lam, mu_j, sigma_j = theta
        n = np.arange(self.max_jumps + 1)
        pois = stats.poisson.pmf(n, lam * DT)                  # (K,)
        mean = alpha * DT + n * mu_j                           # (K,)
        sd = np.sqrt(sigma ** 2 * DT + n * sigma_j ** 2)       # (K,)
        dens = stats.norm.pdf(r[:, None], mean[None, :], sd[None, :]) @ pois
        return -np.sum(np.log(dens + 1e-300))

    def fit(self, prices: pd.Series) -> "MertonJumpDiffusion":
        r = log_returns(prices).values
        sample_vol = r.std(ddof=1) / np.sqrt(DT)
        bounds = [(-3.0, 3.0),                 # alpha (annual drift of the diffusion part)
                  (0.02, 2.0),                 # sigma (diffusion volatility)
                  (0.1, self.max_lam),         # lam   (jumps per year)
                  (-0.2, 0.2),                 # mu_j  (mean log jump)
                  (self.min_sigma_j, 0.3)]     # sigma_j (log jump volatility)
        best = None
        # Several starting points: the mixture likelihood is multi-modal.
        for lam0 in (2.0, 8.0, 25.0):
            for sj0 in (0.02, 0.05):
                lam0 = min(lam0, self.max_lam)
                sj0 = max(sj0, self.min_sigma_j)
                x0 = [r.mean() / DT, 0.8 * sample_vol, lam0, 0.0, sj0]
                res = optimize.minimize(self._neg_log_likelihood, x0, args=(r,),
                                        method="L-BFGS-B", bounds=bounds)
                if best is None or res.fun < best.fun:
                    best = res
        alpha, self.sigma, self.lam, self.mu_j, self.sigma_j = map(float, best.x)
        self.mu = alpha + 0.5 * self.sigma ** 2 + self.lam * self.jump_mean()
        self.log_likelihood = float(-best.fun)
        return self

    def jump_mean(self) -> float:
        """k = E[J - 1], the average relative jump size."""
        return float(np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1)

    def simulate(self, s0: float, n_steps: int, n_paths: int,
                 rng: np.random.Generator) -> np.ndarray:
        drift = (self.mu - 0.5 * self.sigma ** 2 - self.lam * self.jump_mean()) * DT
        diffusion = self.sigma * np.sqrt(DT) * rng.standard_normal((n_paths, n_steps))
        n_jumps = rng.poisson(self.lam * DT, size=(n_paths, n_steps))
        # The sum of n i.i.d. Normal(mu_j, sigma_j^2) log jumps is Normal(n mu_j, n sigma_j^2).
        jumps = n_jumps * self.mu_j + np.sqrt(n_jumps) * self.sigma_j * \
            rng.standard_normal((n_paths, n_steps))
        return _paths_from_log_increments(s0, drift + diffusion + jumps)

    def params(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma, "lam": self.lam,
                "mu_j": self.mu_j, "sigma_j": self.sigma_j,
                "total_vol": float(np.sqrt(self.sigma ** 2 +
                                           self.lam * (self.mu_j ** 2 + self.sigma_j ** 2)))}


def _paths_from_log_increments(s0: float, increments: np.ndarray) -> np.ndarray:
    """Turn (n_paths, n_steps) log increments into price paths starting at s0."""
    log_paths = np.concatenate([np.zeros((increments.shape[0], 1)),
                                np.cumsum(increments, axis=1)], axis=1)
    return s0 * np.exp(log_paths)


def make_models() -> dict:
    """Fresh, unfitted instances of the three models, keyed by display name."""
    return {"GBM": GBM(), "OU": OrnsteinUhlenbeck(), "Merton": MertonJumpDiffusion()}


# ---------------------------------------------------------------------------
# 4. Monte Carlo summaries and risk metrics
# ---------------------------------------------------------------------------

def terminal_risk_metrics(paths: np.ndarray, horizon: int) -> dict:
    """Risk statistics of the simulated price after `horizon` trading days.

    * prob_loss      : P(S_T < S_0)
    * prob_drop_10   : P(S_T < 0.9 S_0)
    * var_95 / es_95 : 95% Value-at-Risk and Expected Shortfall of the simple
                       return, reported as positive loss fractions.
    * mean_se        : Monte Carlo standard error of the mean price.
    """
    s0 = paths[0, 0]
    s_t = paths[:, horizon]
    ret = s_t / s0 - 1
    q05 = np.quantile(ret, 0.05)
    return {
        "horizon_days": horizon,
        "mean": float(s_t.mean()),
        "mean_se": float(s_t.std(ddof=1) / np.sqrt(len(s_t))),
        "median": float(np.median(s_t)),
        "p05": float(np.quantile(s_t, 0.05)),
        "p95": float(np.quantile(s_t, 0.95)),
        "expected_return": float(ret.mean()),
        "prob_loss": float((s_t < s0).mean()),
        "prob_drop_10": float((ret < -0.10).mean()),
        "var_95": float(-q05),
        "es_95": float(-ret[ret <= q05].mean()),
    }


def crps_sample(samples: np.ndarray, observed: float) -> float:
    """Continuous Ranked Probability Score of an ensemble (lower is better).

    CRPS = E|X - y| - 0.5 E|X - X'|, computed in O(n log n) from sorted samples.
    """
    x = np.sort(samples)
    n = len(x)
    term1 = np.abs(x - observed).mean()
    # E|X - X'| = 2 / n^2 * sum_i (2i - n - 1) x_(i)
    term2 = 2.0 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2
    return float(term1 - 0.5 * term2)


# ---------------------------------------------------------------------------
# 5. Rolling-origin backtest
# ---------------------------------------------------------------------------

def rolling_backtest(price: pd.Series, calib_days: int, horizons: list[int],
                     step: int, n_paths: int, seed: int) -> pd.DataFrame:
    """Out-of-sample evaluation of all models over many forecast origins.

    At each origin t (every `step` days), each model is calibrated on the
    previous `calib_days` prices only, simulated forward, and scored against
    the realised price h days later. Returns one row per (origin, model, horizon).
    """
    rng = np.random.default_rng(seed)
    max_h = max(horizons)
    rows = []
    for t in range(calib_days, len(price) - max_h, step):
        window = price.iloc[t - calib_days:t + 1]
        s0 = float(price.iloc[t])
        for name, model in make_models().items():
            model.fit(window)
            paths = model.simulate(s0, max_h, n_paths, rng)
            for h in horizons:
                sim = paths[:, h]
                realized = float(price.iloc[t + h])
                lo, hi = np.quantile(sim, [0.05, 0.95])
                p_loss = float((sim < s0).mean())
                rows.append({
                    "origin": price.index[t], "model": name, "horizon": h,
                    "s0": s0, "realized": realized,
                    "in_90_interval": bool(lo <= realized <= hi),
                    "pit": float((sim <= realized).mean()),  # probability integral transform
                    "crps_pct": crps_sample(sim / s0, realized / s0) * 100,
                    "brier_loss": (p_loss - float(realized < s0)) ** 2,
                    "median_abs_err_pct": abs(np.median(sim) / realized - 1) * 100,
                })
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    """Average backtest scores per ticker, model and horizon."""
    return (bt.groupby(["ticker", "model", "horizon"])
              .agg(origins=("realized", "size"),
                   coverage_90=("in_90_interval", "mean"),
                   crps_pct=("crps_pct", "mean"),
                   brier=("brier_loss", "mean"),
                   median_abs_err_pct=("median_abs_err_pct", "mean"))
              .reset_index())


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def set_plot_style() -> None:
    """Quiet, report-friendly matplotlib defaults."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def plot_price_history(prices: pd.DataFrame, calib_start, ax=None):
    """Prices rebased to 100, with the calibration window shaded."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.2))
    rebased = prices / prices.iloc[0] * 100
    for col in rebased.columns:
        ax.plot(rebased.index, rebased[col], color=TICKER_COLORS.get(col), lw=1.3, label=col)
    ax.axvspan(calib_start, prices.index[-1], color=GRID, alpha=0.6, lw=0)
    ax.text(calib_start, ax.get_ylim()[1], " calibration window", va="top",
            fontsize=8, color=INK_MUTED)
    ax.set_ylabel("Price, rebased (start = 100)")
    ax.legend(loc="upper left", ncol=3)
    return ax


def plot_return_diagnostics(returns: pd.DataFrame):
    """Histogram of daily log returns vs fitted normal, plus a normal Q-Q plot."""
    import matplotlib.pyplot as plt
    tickers = list(returns.columns)
    fig, axes = plt.subplots(2, len(tickers), figsize=(7.5, 4.6))
    for j, t in enumerate(tickers):
        r = returns[t].values * 100
        ax = axes[0, j]
        ax.hist(r, bins=60, density=True, color=TICKER_COLORS.get(t), alpha=0.75,
                edgecolor="white", linewidth=0.3)
        grid = np.linspace(r.min(), r.max(), 300)
        ax.plot(grid, stats.norm.pdf(grid, r.mean(), r.std()), color=INK, lw=1.2,
                label="Normal fit")
        ax.set_title(f"{t}: daily log returns")
        ax.set_xlabel("Return (%)")
        if j == 0:
            ax.set_ylabel("Density")
            ax.legend(loc="upper left", fontsize=7)
        ax = axes[1, j]
        (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
        ax.scatter(osm, osr, s=6, color=TICKER_COLORS.get(t), alpha=0.7, lw=0)
        ax.plot(osm, slope * osm + intercept, color=INK, lw=1)
        ax.set_xlabel("Normal quantiles")
        if j == 0:
            ax.set_ylabel("Sample quantiles (%)")
    fig.tight_layout()
    return fig


def plot_fan_charts(sims: dict, ticker: str, dates: pd.DatetimeIndex | None = None,
                    history: pd.Series | None = None, realized: pd.Series | None = None,
                    n_sample_paths: int = 40):
    """One panel per model: sample paths, 90% / 50% bands, mean and median."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(sims), figsize=(7.5, 2.9), sharey=True)
    for ax, (name, paths) in zip(axes, sims.items()):
        color = MODEL_COLORS[name]
        x = np.arange(paths.shape[1]) if dates is None else dates
        for p in paths[:n_sample_paths]:
            ax.plot(x, p, color=color, alpha=0.18, lw=0.5)
        q05, q25, q50, q75, q95 = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        ax.fill_between(x, q05, q95, color=color, alpha=0.15, lw=0, label="90% interval")
        ax.fill_between(x, q25, q75, color=color, alpha=0.25, lw=0, label="50% interval")
        ax.plot(x, paths.mean(axis=0), color=INK, lw=1.4, label="Mean")
        ax.plot(x, q50, color=INK, lw=1.2, ls="--", label="Median")
        if history is not None:
            ax.plot(history.index, history.values, color=INK_MUTED, lw=1.0, label="History")
        if realized is not None:
            ax.plot(realized.index, realized.values, color=REALIZED_COLOR, lw=1.8,
                    label="Realised")
        ax.set_title(f"{ticker} - {name}")
        if dates is None:
            ax.set_xlabel("Trading days ahead")
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axes[0].set_ylabel("Price (USD)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return fig


def plot_terminal_histograms(sims_by_ticker: dict, horizon: int):
    """Overlaid terminal-price densities (one panel per ticker) at `horizon` days."""
    import matplotlib.pyplot as plt
    tickers = list(sims_by_ticker)
    fig, axes = plt.subplots(1, len(tickers), figsize=(7.5, 2.8))
    for ax, t in zip(axes, tickers):
        sims = sims_by_ticker[t]
        s0 = next(iter(sims.values()))[0, 0]
        all_t = np.concatenate([p[:, horizon] for p in sims.values()])
        bins = np.linspace(*np.quantile(all_t, [0.002, 0.998]), 36)
        for name, paths in sims.items():
            ax.hist(paths[:, horizon], bins=bins, density=True, histtype="step",
                    color=MODEL_COLORS[name], lw=1.6, label=name)
        ax.axvline(s0, color=INK_MUTED, lw=1, ls=":", label="Current price")
        ax.set_title(f"{t}: price after {horizon} days")
        ax.set_xlabel("Price (USD)")
        ax.set_yticks([])
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def plot_prob_loss(sims_by_ticker: dict, max_h: int):
    """Probability that the price is below today's price, by horizon and model."""
    import matplotlib.pyplot as plt
    tickers = list(sims_by_ticker)
    fig, axes = plt.subplots(1, len(tickers), figsize=(7.5, 2.6), sharey=True)
    h = np.arange(1, max_h + 1)
    for ax, t in zip(axes, tickers):
        for name, paths in sims_by_ticker[t].items():
            p = (paths[:, 1:max_h + 1] < paths[:, [0]]).mean(axis=0)
            ax.plot(h, p * 100, color=MODEL_COLORS[name], lw=1.8, label=name)
        ax.axhline(50, color=INK_MUTED, lw=0.8, ls=":")
        ax.set_title(t)
        ax.set_xlabel("Horizon (trading days)")
    axes[0].set_ylabel("P(loss) (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_backtest_summary(summary: pd.DataFrame):
    """90%-interval coverage and CRPS by ticker and model (rolling backtest)."""
    import matplotlib.pyplot as plt
    horizons = sorted(summary["horizon"].unique())
    tickers = list(dict.fromkeys(summary["ticker"]))
    models = list(MODEL_COLORS)
    fig, axes = plt.subplots(2, len(horizons), figsize=(7.5, 4.4))
    width = 0.26
    xs = np.arange(len(tickers))
    for j, h in enumerate(horizons):
        sub = summary[summary["horizon"] == h]
        for metric, row, ylabel in (("coverage_90", 0, "90% interval coverage (%)"),
                                    ("crps_pct", 1, "Mean CRPS (% of S0)")):
            ax = axes[row, j]
            for k, m in enumerate(models):
                vals = [sub[(sub.ticker == t) & (sub.model == m)][metric].iloc[0] for t in tickers]
                vals = np.array(vals) * (100 if metric == "coverage_90" else 1)
                ax.bar(xs + (k - 1) * width, vals, width=width * 0.92,
                       color=MODEL_COLORS[m], label=m)
            if metric == "coverage_90":
                ax.axhline(90, color=INK, lw=1, ls="--")
                ax.set_ylim(0, 105)
                ax.set_title(f"{h}-day horizon")
            ax.set_xticks(xs, tickers)
            if j == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.05))
    return fig


def to_serializable(obj):
    """Helper for json.dump of numpy / pandas values."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(type(obj))
