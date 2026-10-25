"""
stock_models.py
===============
Stochastic models of stock prices: data handling, calibration, Monte Carlo
simulation, risk metrics and backtesting.

The module implements three continuous-time models:

1. Geometric Brownian Motion (GBM)
       dS_t = mu * S_t dt + sigma * S_t dW_t

2. Trend-stationary Ornstein-Uhlenbeck process on the log price
       ln S_t = a + g*t + Y_t,     dY_t = -theta * Y_t dt + sigma * dW_t
   The log price reverts toward an exponential trend line rather than a
   flat level. Setting g = 0 gives the classic OU (Vasicek) model, which
   reverts to a constant level.

3. Merton (1976) jump-diffusion
       dS_t / S_{t-} = (mu - lambda*k) dt + sigma dW_t + (J - 1) dN_t
   N_t is a Poisson process with intensity lambda, ln J ~ N(mu_J, sigma_J^2)
   and k = E[J - 1] = exp(mu_J + sigma_J^2 / 2) - 1.

All simulations use the exact transition laws (no Euler discretisation
error), so the time step only controls how finely each path is sampled.

Usage
-----
    python stock_models.py                 # full pipeline on data/AAPL_raw.csv
    python stock_models.py --source yfinance --tickers AAPL MSFT KO

Time is measured in years and one trading day is dt = 1/252.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"


# ---------------------------------------------------------------------------
# 1. Data collection and preprocessing
# ---------------------------------------------------------------------------

def load_yfinance_csv(path: str | Path, ticker: str = "AAPL") -> pd.DataFrame:
    """Read a CSV exported with ``yfinance.download(...).to_csv()``.

    yfinance (>= 0.2.5) writes a three-row header (Price / Ticker / Date).
    The function also accepts an ordinary one-row header (Kaggle style).
    """
    path = Path(path)
    head = pd.read_csv(path, nrows=3, header=None)
    if str(head.iloc[1, 0]).strip().lower() == "ticker":
        df = pd.read_csv(path, header=[0, 1], index_col=0, skiprows=[2])
        df.columns = [c[0] for c in df.columns]          # keep the field name
    else:
        df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce")
    df.attrs["ticker"] = ticker
    return df


def download_yfinance(tickers: list[str], start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Download daily, split- and dividend-adjusted prices from Yahoo Finance."""
    import yfinance as yf  # imported lazily so the offline pipeline needs no network

    out = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.attrs["ticker"] = t
        out[t] = df
    return out


def clean_prices(df: pd.DataFrame, price_col: str = "Close", z_flag: float = 6.0) -> tuple[pd.Series, dict]:
    """Clean a daily price table and return (close series, cleaning report).

    Steps
    -----
    * drop rows with an unparseable date and sort chronologically;
    * remove duplicated dates (keep the last record);
    * treat non-positive prices as missing;
    * fill isolated gaps (at most 2 consecutive days) by time interpolation
      and drop anything longer, since inventing a long gap would bias the
      volatility estimate;
    * flag, but keep, daily log returns more than ``z_flag`` robust standard
      deviations from the median. Large moves in a listed stock are usually
      real (earnings, macro shocks), and deleting them would remove exactly
      the tail risk the jump model is meant to capture.
    """
    report: dict = {"rows_raw": int(len(df))}
    df = df[~df.index.isna()].sort_index()
    report["bad_dates"] = report["rows_raw"] - int(len(df))

    dup = df.index.duplicated(keep="last")
    report["duplicate_dates"] = int(dup.sum())
    df = df[~dup]

    s = df[price_col].astype(float)
    s = s.where(s > 0)
    report["missing_or_nonpositive"] = int(s.isna().sum())
    s = s.interpolate(method="time", limit=2, limit_area="inside").dropna()

    r = np.log(s).diff().dropna()
    mad = stats.median_abs_deviation(r, scale="normal")
    z = (r - r.median()) / mad
    flagged = z[np.abs(z) > z_flag]
    report["flagged_outliers"] = {d.strftime("%Y-%m-%d"): round(float(v), 4) for d, v in r[flagged.index].items()}
    report["rows_clean"] = int(len(s))
    report["start"] = s.index[0].strftime("%Y-%m-%d")
    report["end"] = s.index[-1].strftime("%Y-%m-%d")
    s.name = df.attrs.get("ticker", price_col)
    return s, report


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()


def describe_returns(r: pd.Series) -> dict:
    """Descriptive statistics and stylised-fact tests for daily log returns."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    jb = stats.jarque_bera(r)
    lb_r = acorr_ljungbox(r, lags=[10], return_df=True)
    lb_r2 = acorr_ljungbox(r**2, lags=[10], return_df=True)
    return {
        "n_obs": int(len(r)),
        "mean_daily": float(r.mean()),
        "std_daily": float(r.std(ddof=1)),
        "ann_mean_log_return": float(r.mean() * TRADING_DAYS),
        "ann_volatility": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "min": float(r.min()),
        "max": float(r.max()),
        "jarque_bera_stat": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
        "ljungbox_r_p": float(lb_r["lb_pvalue"].iloc[0]),
        "ljungbox_r2_p": float(lb_r2["lb_pvalue"].iloc[0]),
    }


def stationarity_tests(prices: pd.Series) -> dict:
    """ADF tests on the log price: constant only, and constant + trend.

    Failing to reject a unit root supports a random-walk (GBM) view. Rejecting
    it in the trend specification supports mean reversion around a trend.
    """
    from statsmodels.tsa.stattools import adfuller

    x = np.log(prices.values)
    out = {}
    for reg in ("c", "ct"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            stat, p, *_ = adfuller(x, regression=reg, autolag="AIC")
        out[f"adf_{reg}_stat"] = float(stat)
        out[f"adf_{reg}_p"] = float(p)
    return out


# ---------------------------------------------------------------------------
# 2. Model parameters and calibration
# ---------------------------------------------------------------------------

@dataclass
class GBMParams:
    mu: float       # annual drift of the price (arithmetic)
    sigma: float    # annual volatility


@dataclass
class OUParams:
    theta: float    # speed of mean reversion (per year)
    sigma: float    # annual volatility of the log-price deviation
    a: float        # intercept of the log-price trend
    g: float        # trend slope (per year); 0 means classic OU
    t0: float       # time (years) of the last observation, relative to the start of the sample
    y0: float       # deviation from trend at the last observation

    @property
    def half_life_days(self) -> float:
        return float(np.log(2) / self.theta * TRADING_DAYS)


@dataclass
class MertonParams:
    mu: float       # total annual drift of the price (includes jump compensation)
    sigma: float    # annual diffusion volatility
    lam: float      # jumps per year
    mu_j: float     # mean log jump size
    sigma_j: float  # std of log jump size

    @property
    def k(self) -> float:
        return float(np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0)

    @property
    def total_vol(self) -> float:
        return float(np.sqrt(self.sigma**2 + self.lam * (self.mu_j**2 + self.sigma_j**2)))


def fit_gbm(r: pd.Series) -> GBMParams:
    """Maximum likelihood for GBM: log returns are i.i.d. N((mu - sigma^2/2)dt, sigma^2 dt)."""
    sigma = r.std(ddof=1) / np.sqrt(DT)
    mu = r.mean() / DT + 0.5 * sigma**2
    return GBMParams(float(mu), float(sigma))


def fit_ou(prices: pd.Series, trend: bool = True) -> OUParams:
    """Fit the (trend-stationary) OU model through its exact AR(1) form.

    With x_t = ln S_t, first remove a linear time trend by OLS (or only the
    mean when ``trend=False``). The deviation y_t then follows
        y_{t+1} = b * y_t + e_t,   b = exp(-theta*dt),
        Var(e) = sigma^2 (1 - b^2) / (2 theta),
    so theta and sigma follow directly from the AR(1) regression.
    """
    x = np.log(prices.values)
    t = np.arange(len(x)) * DT
    if trend:
        g, a = np.polyfit(t, x, 1)
    else:
        g, a = 0.0, float(x.mean())
    y = x - (a + g * t)
    b = float(np.dot(y[:-1], y[1:]) / np.dot(y[:-1], y[:-1]))   # zero-mean AR(1)
    b = min(max(b, 1e-6), 1 - 1e-6)                             # keep theta > 0
    resid = y[1:] - b * y[:-1]
    theta = -np.log(b) / DT
    sigma = resid.std(ddof=1) * np.sqrt(2 * theta / (1 - b**2))
    return OUParams(float(theta), float(sigma), float(a), float(g), float(t[-1]), float(y[-1]))


def _merton_logpdf(r: np.ndarray, m: float, s: float, lam: float, mu_j: float, s_j: float, n_max: int = 8) -> np.ndarray:
    """Log density of one Merton log return over dt (Poisson mixture of normals)."""
    n = np.arange(n_max + 1)[:, None]
    w = stats.poisson.pmf(n, lam * DT)
    mean = m * DT + n * mu_j
    sd = np.sqrt(s**2 * DT + n * s_j**2)
    return np.log((w * stats.norm.pdf(r[None, :], mean, sd)).sum(axis=0) + 1e-300)


def fit_merton(r: pd.Series, k_thresh: float = 3.0, lam_max: float = 12.0) -> MertonParams:
    """Calibrate Merton jump-diffusion by maximum likelihood.

    Starting values come from a threshold rule: returns more than
    ``k_thresh`` robust standard deviations from the median count as jumps.
    The MLE then refines all five parameters jointly (L-BFGS-B with bounds,
    several starting points).

    Identification: left unconstrained, the likelihood keeps rising as lambda
    grows, because many tiny "jumps" mimic volatility clustering rather than
    rare discontinuities. The jump intensity is therefore capped at
    ``lam_max`` (default 12 per year, about one large move a month) so the
    jump component keeps its intended meaning.
    """
    x = r.values
    med = np.median(x)
    rob = stats.median_abs_deviation(x, scale="normal")
    is_jump = np.abs(x - med) > k_thresh * rob
    n_j = max(int(is_jump.sum()), 1)
    s0 = x[~is_jump].std(ddof=1) / np.sqrt(DT)
    lam0 = n_j / (len(x) * DT)
    mu_j0 = float(x[is_jump].mean()) if is_jump.any() else 0.0
    s_j0 = float(x[is_jump].std(ddof=1)) if is_jump.sum() > 1 else 2 * rob
    m0 = x[~is_jump].mean() / DT

    def nll(p):
        m, s, lam, mj, sj = p
        return -_merton_logpdf(x, m, s, lam, mj, sj).sum()

    bounds = [(-5, 5), (1e-3, 2.0), (1e-2, lam_max), (-0.3, 0.3), (1e-3, 0.3)]
    starts = [[m0, s0, min(lam0, lam_max), mu_j0, max(s_j0, 1e-3)],
              [m0, 0.8 * s0, 0.5 * lam_max, 0.0, 3 * rob],
              [m0, 0.6 * s0, lam_max, -rob, 2 * rob]]
    res = min((optimize.minimize(nll, p0, method="L-BFGS-B", bounds=bounds) for p0 in starts),
              key=lambda o: o.fun)
    m, s, lam, mj, sj = res.x
    # m is the drift of the continuous part of ln S. Convert it to the total
    # arithmetic price drift: mu = m + sigma^2/2 + lambda*k.
    k = np.exp(mj + 0.5 * sj**2) - 1
    mu = m + 0.5 * s**2 + lam * k
    return MertonParams(float(mu), float(s), float(lam), float(mj), float(sj))


def model_loglik(r: pd.Series, prices: pd.Series, gbm: GBMParams, ou: OUParams, mer: MertonParams) -> pd.DataFrame:
    """Log-likelihood, AIC and BIC of each model on the calibration sample."""
    x = r.values
    n = len(x)
    ll_gbm = stats.norm.logpdf(x, (gbm.mu - 0.5 * gbm.sigma**2) * DT, gbm.sigma * np.sqrt(DT)).sum()

    lx = np.log(prices.values)
    t = np.arange(len(lx)) * DT
    y = lx - (ou.a + ou.g * t)
    b = np.exp(-ou.theta * DT)
    sd = ou.sigma * np.sqrt((1 - b**2) / (2 * ou.theta))
    ll_ou = stats.norm.logpdf(y[1:], b * y[:-1], sd).sum()

    m = mer.mu - 0.5 * mer.sigma**2 - mer.lam * mer.k
    ll_mer = _merton_logpdf(x, m, mer.sigma, mer.lam, mer.mu_j, mer.sigma_j).sum()

    rows = []
    for name, ll, k in [("GBM", ll_gbm, 2), ("OU (trend)", ll_ou, 4), ("Merton JD", ll_mer, 5)]:
        rows.append({"model": name, "loglik": ll, "params": k,
                     "AIC": 2 * k - 2 * ll, "BIC": k * np.log(n) - 2 * ll})
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# 3. Monte Carlo simulation (exact discretisations)
# ---------------------------------------------------------------------------

def simulate_gbm(s0: float, p: GBMParams, n_days: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Return an array (n_paths, n_days + 1) of GBM price paths."""
    z = rng.standard_normal((n_paths, n_days))
    incr = (p.mu - 0.5 * p.sigma**2) * DT + p.sigma * np.sqrt(DT) * z
    return s0 * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)]))


def simulate_ou(s0: float, p: OUParams, n_days: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate the trend-stationary OU model, starting exactly at today's price s0."""
    b = np.exp(-p.theta * DT)
    sd = p.sigma * np.sqrt((1 - b**2) / (2 * p.theta))
    # Anchor the deviation to the observed price so every path starts at s0.
    y = np.empty((n_paths, n_days + 1))
    y[:, 0] = np.log(s0) - (p.a + p.g * p.t0)
    z = rng.standard_normal((n_paths, n_days))
    for i in range(n_days):
        y[:, i + 1] = b * y[:, i] + sd * z[:, i]
    t = p.t0 + np.arange(n_days + 1) * DT
    return np.exp(p.a + p.g * t + y)


def simulate_merton(s0: float, p: MertonParams, n_days: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate Merton jump-diffusion paths.

    The number of jumps per day is Poisson(lambda*dt); given n jumps, their total
    log size is N(n*mu_J, n*sigma_J^2), so jumps are sampled exactly.
    """
    m = p.mu - 0.5 * p.sigma**2 - p.lam * p.k
    z = rng.standard_normal((n_paths, n_days))
    n_j = rng.poisson(p.lam * DT, size=(n_paths, n_days))
    jump = n_j * p.mu_j + np.sqrt(n_j) * p.sigma_j * rng.standard_normal((n_paths, n_days))
    incr = m * DT + p.sigma * np.sqrt(DT) * z + jump
    return s0 * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)]))


def simulate_all(s0: float, params: dict, n_days: int, n_paths: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Run every model with its own, reproducible random stream."""
    sims = {}
    funcs = {"GBM": simulate_gbm, "OU (trend)": simulate_ou, "Merton JD": simulate_merton}
    for i, (name, f) in enumerate(funcs.items()):
        rng = np.random.default_rng(seed + i)
        sims[name] = f(s0, params[name], n_days, n_paths, rng)
    return sims


# ---------------------------------------------------------------------------
# 4. Risk metrics
# ---------------------------------------------------------------------------

def terminal_stats(paths: np.ndarray, s0: float) -> dict:
    """Distribution summary and risk measures of the simulated horizon return."""
    st = paths[:, -1]
    ret = st / s0 - 1
    q05 = np.quantile(ret, 0.05)
    run_max = np.maximum.accumulate(paths, axis=1)
    max_dd = ((run_max - paths) / run_max).max(axis=1)
    return {
        "mean_price": float(st.mean()),
        "median_price": float(np.median(st)),
        "p05_price": float(np.quantile(st, 0.05)),
        "p95_price": float(np.quantile(st, 0.95)),
        "expected_return": float(ret.mean()),
        "prob_loss": float((ret < 0).mean()),
        "prob_loss_gt_10": float((ret < -0.10).mean()),
        "prob_gain_gt_10": float((ret > 0.10).mean()),
        "VaR95": float(-q05),                        # loss not exceeded with 95% confidence
        "CVaR95": float(-ret[ret <= q05].mean()),    # mean loss in the worst 5%
        "median_max_drawdown": float(np.median(max_dd)),
    }


def summary_table(sims: dict[str, np.ndarray], s0: float, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for name, paths in sims.items():
        for h in horizons:
            d = terminal_stats(paths[:, : h + 1], s0)
            rows.append({"model": name, "horizon": h, **d})
    return pd.DataFrame(rows).set_index(["model", "horizon"])


# ---------------------------------------------------------------------------
# 5. Out-of-sample evaluation
# ---------------------------------------------------------------------------

def calibrate_all(prices: pd.Series) -> dict:
    r = log_returns(prices)
    return {"GBM": fit_gbm(r), "OU (trend)": fit_ou(prices, trend=True), "Merton JD": fit_merton(r)}


def kupiec_pof(n: int, x: int, p: float) -> float:
    """Kupiec (1995) proportion-of-failures test; returns the p-value.

    n trials, x observed exceedances, p expected exceedance rate.
    """
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        phat = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    return float(1 - stats.chi2.cdf(lr, 1))


def rolling_backtest(prices: pd.Series, window: int = 504, horizon: int = 30, step: int = 10,
                     n_paths: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Rolling-origin backtest of the forecast distributions.

    At each origin the models are recalibrated on the previous ``window`` days
    and simulated ``horizon`` days ahead. For every forecast we record the
    probability integral transform (PIT): the share of simulated prices below
    the realised price. A well-calibrated model gives uniformly distributed
    PIT values, and its 90% interval should contain about 90% of outcomes.
    Origins are spaced ``step`` days apart, so neighbouring 30-day windows
    overlap and the observations are not independent; the coverage tests are
    indicative rather than exact.
    """
    rows = []
    origins = range(window, len(prices) - horizon, step)
    for k, i in enumerate(origins):
        hist = prices.iloc[i - window: i + 1]
        params = calibrate_all(hist)
        s0 = float(hist.iloc[-1])
        realised = float(prices.iloc[i + horizon])
        sims = simulate_all(s0, params, horizon, n_paths, seed=seed + 10 * k)
        for name, paths in sims.items():
            st = paths[:, -1]
            lo, hi = np.quantile(st, [0.05, 0.95])
            rows.append({"origin": prices.index[i], "model": name, "s0": s0, "realised": realised,
                         "pit": float((st < realised).mean()), "lo90": lo, "hi90": hi,
                         "inside90": bool(lo <= realised <= hi),
                         "log_score": float(np.log(stats.gaussian_kde(np.log(st))(np.log(realised))[0] + 1e-12))})
    return pd.DataFrame(rows)


def backtest_summary(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in bt.groupby("model", sort=False):
        n, x = len(g), int((~g["inside90"]).sum())
        rows.append({"model": name, "forecasts": n, "coverage90": g["inside90"].mean(),
                     "kupiec_p": kupiec_pof(n, x, 0.10),
                     "mean_log_score": g["log_score"].mean(),
                     "pit_ks_p": float(stats.kstest(g["pit"], "uniform").pvalue)})
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# 6. Command-line pipeline
# ---------------------------------------------------------------------------

def run_pipeline(prices: pd.Series, calib_days: int = 504, horizons=(30, 60, 180),
                 n_paths: int = 10_000, seed: int = 42, make_plots: bool = True) -> dict:
    """End-to-end analysis for one stock. Returns every table as a DataFrame or dict."""
    ticker = prices.name
    calib = prices.iloc[-(calib_days + 1):]
    r = log_returns(calib)
    s0 = float(calib.iloc[-1])

    params = calibrate_all(calib)
    params_plain_ou = fit_ou(calib, trend=False)

    sims = simulate_all(s0, params, max(horizons), n_paths, seed)
    summ = summary_table(sims, s0, list(horizons))

    # Hold-out check: calibrate without the last 180 days and compare the
    # forecast with the realised path.
    h_out = max(horizons)
    train = prices.iloc[-(calib_days + h_out + 1): -h_out]
    test = prices.iloc[-(h_out + 1):]
    holdout_sims = simulate_all(float(train.iloc[-1]), calibrate_all(train), h_out, n_paths, seed + 100)

    bt = rolling_backtest(prices, window=calib_days, horizon=30, step=10)

    result = {
        "ticker": ticker, "s0": s0, "calib_start": calib.index[0], "calib_end": calib.index[-1],
        "returns_stats": describe_returns(r), "full_returns_stats": describe_returns(log_returns(prices)),
        "adf": stationarity_tests(calib), "params": params, "plain_ou": params_plain_ou,
        "fit": model_loglik(r, calib, *params.values()), "summary": summ, "sims": sims,
        "holdout": {"train": train, "test": test, "sims": holdout_sims},
        "backtest": bt, "backtest_summary": backtest_summary(bt), "calib": calib,
    }
    if make_plots:
        import plots
        plots.make_all(prices, result, FIG_DIR)
    return result


def _to_jsonable(res: dict) -> dict:
    out = {k: res[k] for k in ("ticker", "s0", "returns_stats", "full_returns_stats", "adf")}
    out["calib_window"] = [str(res["calib_start"].date()), str(res["calib_end"].date())]
    out["params"] = {k: {**asdict(v), **({"half_life_days": v.half_life_days} if isinstance(v, OUParams) else {}),
                         **({"k": v.k, "total_vol": v.total_vol} if isinstance(v, MertonParams) else {})}
                     for k, v in res["params"].items()}
    out["plain_ou"] = {**asdict(res["plain_ou"]), "half_life_days": res["plain_ou"].half_life_days}
    out["fit"] = res["fit"].reset_index().to_dict(orient="records")
    out["summary"] = res["summary"].reset_index().to_dict(orient="records")
    out["backtest_summary"] = res["backtest_summary"].reset_index().to_dict(orient="records")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["csv", "yfinance"], default="csv")
    ap.add_argument("--csv", default=str(DATA_DIR / "AAPL_raw.csv"))
    ap.add_argument("--tickers", nargs="+", default=["AAPL"])
    ap.add_argument("--start", default="2021-07-13")
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.source == "csv":
        raw = {args.tickers[0]: load_yfinance_csv(args.csv, args.tickers[0])}
    else:
        raw = download_yfinance(args.tickers, args.start)

    FIG_DIR.mkdir(exist_ok=True)
    for t, df in raw.items():
        prices, report = clean_prices(df)
        prices.to_csv(DATA_DIR / f"{t}_clean.csv", header=["Close"])
        print(f"\n=== {t} ===\nCleaning report: {json.dumps(report, indent=2)}")
        res = run_pipeline(prices, n_paths=args.paths, make_plots=not args.no_plots)
        res_json = _to_jsonable(res)
        res_json["cleaning"] = report
        (HERE / f"results_{t}.json").write_text(json.dumps(res_json, indent=2, default=str))
        pd.set_option("display.width", 200)
        print(res["fit"].round(2))
        print(res["summary"][["mean_price", "median_price", "p05_price", "p95_price",
                              "prob_loss", "VaR95", "CVaR95"]].round(3))
        print(res["backtest_summary"].round(3))


if __name__ == "__main__":
    main()
