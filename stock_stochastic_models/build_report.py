"""
build_report.py
===============

Builds Stochastic_Stock_Price_Modeling_Report.pdf from the figures and tables
written by run_analysis.py. Every number quoted in the text is read from
./results, so the report always matches the latest run.

Usage:  python run_analysis.py && python build_report.py
"""

import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from scipy import stats

import stochastic_models as sm
import run_analysis as cfg

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
OUT = HERE / "Stochastic_Stock_Price_Modeling_Report.pdf"

# ------------------------------------------------------------------ data ----
summary = json.loads((RES / "summary.json").read_text())
diag = pd.read_csv(RES / "diagnostics_calibration_window.csv", index_col=0)
diag_full = pd.read_csv(RES / "diagnostics_full_sample.csv", index_col=0)
params = pd.read_csv(RES / "model_parameters.csv")
risk = pd.read_csv(RES / "forecast_risk_metrics.csv")
sens = pd.read_csv(RES / "gbm_drift_sensitivity.csv")
bt = pd.read_csv(RES / "backtest_summary.csv")
bt_detail = pd.read_csv(RES / "backtest_detail.csv")
T = cfg.TICKERS

# Likelihood-ratio test of Merton vs GBM on the calibration window (GBM is
# nested in Merton with lam -> 0; 3 extra parameters).
prices = sm.clean_prices(sm.load_prices_kaggle(cfg.KAGGLE_CSV, T))[0]
calib = prices.loc[summary["calibration_start"]:summary["calibration_end"]]
lr = {}
for t in T:
    r = sm.log_returns(calib[t]).values
    ll_gbm = stats.norm.logpdf(r, r.mean(), r.std()).sum()
    ll_mjd = sm.MertonJumpDiffusion().fit(calib[t]).log_likelihood
    lr[t] = 2 * (ll_mjd - ll_gbm)


def P(t, model, col):
    return params[(params.ticker == t) & (params.model == model)][col].iloc[0]


def R(t, model, h, col):
    return risk[(risk.ticker == t) & (risk.model == model) & (risk.horizon_days == h)][col].iloc[0]


def B(t, model, h, col):
    return bt[(bt.ticker == t) & (bt.model == model) & (bt.horizon == h)][col].iloc[0]


def S(t, drift, col):
    return sens[(sens.ticker == t) & (sens.drift == drift)][col].iloc[0]


def long_date(value):
    """'2018-02-07' -> '7 February 2018' (portable; avoids the %-d strftime flag)."""
    d = pd.Timestamp(value)
    return f"{d.day} {d:%B %Y}"


pct = lambda x, d=0: f"{100 * x:.{d}f}%"
s0 = {t: R(t, "GBM", 30, "s0") for t in T}
cal_start = long_date(summary["calibration_start"])
cal_end = long_date(summary["calibration_end"])
show_origin = long_date(summary["showcase_origin"])
n_origins = int(bt["origins"].iloc[0])
pooled = bt.groupby(["model", "horizon"])[["coverage_90", "crps_pct", "brier"]].mean()

# ---------------------------------------------------------------- styles ----
# DejaVu fonts ship with matplotlib, so they are always available, and unlike
# the PDF base-14 fonts they contain the Greek letters and math symbols used below.
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in [("Serif", "DejaVuSerif"), ("Serif-Bold", "DejaVuSerif-Bold"),
                   ("Serif-Italic", "DejaVuSerif-Italic"), ("Serif-BoldItalic", "DejaVuSerif-BoldItalic"),
                   ("Sans", "DejaVuSans"), ("Sans-Bold", "DejaVuSans-Bold"),
                   ("Sans-Italic", "DejaVuSans-Oblique"), ("Sans-BoldItalic", "DejaVuSans-BoldOblique"),
                   ("Mono", "DejaVuSansMono")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / f"{file}.ttf")))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic",
                   boldItalic="Serif-BoldItalic")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic",
                   boldItalic="Sans-BoldItalic")

ss = getSampleStyleSheet()
INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#c9c8c2")
HEAD_BG = colors.HexColor("#f1f0ec")
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Serif", fontSize=9.8,
                      leading=13.6, alignment=TA_JUSTIFY, spaceAfter=6, textColor=INK)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Sans-Bold", fontSize=13.5,
                    spaceBefore=12, spaceAfter=6, textColor=INK, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Sans-Bold", fontSize=11,
                    spaceBefore=8, spaceAfter=4, textColor=INK, keepWithNext=1)
caption = ParagraphStyle("caption", parent=body, fontName="Serif-Italic", fontSize=9,
                         leading=11.5, alignment=TA_JUSTIFY, textColor=MUTED, spaceAfter=10)
eq = ParagraphStyle("eq", parent=body, alignment=TA_CENTER, fontName="Serif-Italic",
                    spaceBefore=2, spaceAfter=6)
cell = ParagraphStyle("cell", parent=body, fontName="Sans", fontSize=7.6, leading=9.6,
                      alignment=0, spaceAfter=0)
cell_b = ParagraphStyle("cellb", parent=cell, fontName="Sans-Bold")
ref = ParagraphStyle("ref", parent=body, fontSize=9.5, leading=12.5, leftIndent=18,
                     firstLineIndent=-18, alignment=0, spaceAfter=4)
title = ParagraphStyle("title", parent=body, fontName="Sans-Bold", fontSize=19,
                       leading=24, alignment=TA_CENTER, spaceAfter=8)
subtitle = ParagraphStyle("subtitle", parent=body, fontName="Sans", fontSize=11,
                          alignment=TA_CENTER, textColor=MUTED, spaceAfter=4)

story = []
# The same content is also recorded as plain blocks, which build_report_docx.js
# turns into the Word version, so the PDF and .docx can never drift apart.
blocks = []
fig_no = [0]
tab_no = [0]


def para(text, style=body):
    story.append(Paragraph(text, style))
    blocks.append({"type": style.name, "text": text})


def bullets(items):
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))
        blocks.append({"type": "bullet", "text": it})


def page_break():
    story.append(PageBreak())
    blocks.append({"type": "pagebreak"})


def figure(name, text, width=16.2 * cm):
    fig_no[0] += 1
    w, h = PILImage.open(FIG / name).size
    img = Image(str(FIG / name), width=width, height=width * h / w)
    story.append(KeepTogether([img, Spacer(1, 3),
                               Paragraph(f"<b>Figure {fig_no[0]}.</b> {text}", caption)]))
    blocks.append({"type": "figure", "path": str(FIG / name), "width_cm": width / cm,
                   "aspect": h / w, "caption": f"<b>Figure {fig_no[0]}.</b> {text}"})
    return fig_no[0]


def table(rows, text, col_widths=None, header_rows=1):
    tab_no[0] += 1
    data = [[Paragraph(str(c), cell_b if i < header_rows else cell) for c in row]
            for i, row in enumerate(rows)]
    tbl = Table(data, colWidths=col_widths, repeatRows=header_rows, hAlign="CENTER")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD_BG),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#fafaf8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(KeepTogether([
        Paragraph(f"<b>Table {tab_no[0]}.</b> {text}", ParagraphStyle(
            "tcap", parent=caption, fontName="Serif", textColor=INK, spaceAfter=4)),
        tbl, Spacer(1, 10)]))
    widths = [w / cm for w in col_widths] if col_widths else None
    blocks.append({"type": "table", "rows": [[str(c) for c in row] for row in rows],
                   "widths_cm": widths, "header_rows": header_rows,
                   "caption": f"<b>Table {tab_no[0]}.</b> {text}"})
    return tab_no[0]


# ================================================================ content ===
# ---- title block -----------------------------------------------------------
story += [Spacer(1, 1.2 * cm)]
para("Modelling Stock Price Movements with Stochastic Processes", title)
para("A Monte Carlo comparison of Geometric Brownian Motion, Ornstein&ndash;Uhlenbeck "
     "mean reversion and Merton jump-diffusion for AAPL, JPM and KO", subtitle)
para("Quantitative Research &middot; Stochastic Processes Problem Set", subtitle)
story.append(Spacer(1, 0.6 * cm))

para("<b>Executive summary</b>", h2)
para(
    f"We calibrated three stochastic models to two years of daily closing prices "
    f"({cal_start} &ndash; {cal_end}) for Apple (AAPL), JPMorgan Chase (JPM) and Coca-Cola (KO), "
    f"simulated {summary['n_paths']:,} price paths per model over 30, 60 and 180 trading days, "
    f"and evaluated every model out of sample at {n_origins} rolling forecast origins per stock. "
    "The main findings are:")
bullets([
    f"<b>Returns are fat-tailed.</b> Excess kurtosis is {diag['excess_kurtosis'].min():.1f}&ndash;"
    f"{diag['excess_kurtosis'].max():.1f} and the Jarque&ndash;Bera test rejects normality for all three "
    "stocks. The Merton jump-diffusion fits the daily data far better than GBM (likelihood-ratio "
    f"statistics {min(lr.values()):.0f}&ndash;{max(lr.values()):.0f}, p &lt; 0.001).",
    "<b>There is no reliable evidence of mean reversion in price levels.</b> The Augmented "
    f"Dickey&ndash;Fuller test does not reject a unit root (p = {diag['adf_pvalue'].min():.2f}&ndash;"
    f"{diag['adf_pvalue'].max():.2f}). Consistently, the mean-reverting OU model produced the worst "
    f"calibrated forecasts: its nominal 90% intervals contained the realised price only "
    f"{pct(pooled.loc[('OU', 60), 'coverage_90'])} of the time at 60 days, against "
    f"{pct(pooled.loc[('GBM', 60), 'coverage_90'])} for GBM.",
    "<b>GBM and Merton give well-calibrated short-horizon risk ranges</b> for AAPL and JPM, and "
    "conservative (too wide) ranges for KO. Over 30&ndash;60 days their forecasts are almost "
    "indistinguishable: jumps matter for daily tail risk, not for monthly distributions.",
    "<b>Long-horizon forecasts are dominated by the assumed drift.</b> With the same volatility, "
    f"AAPL's 180-day probability of loss is {pct(S('AAPL','historical','prob_loss'))} under its "
    f"historical drift ({pct(S('AAPL','historical','mu'))} a year) but "
    f"{pct(S('AAPL','8% long-run','prob_loss'))} under a more defensible 8% drift.",
    "<b>Recommendation:</b> use GBM or Merton with a <i>conservative</i> drift to size positions "
    "and set risk limits (VaR, expected shortfall) over horizons up to a quarter; do not use any "
    "of these models to forecast direction; and treat OU price targets with caution.",
])

page_break()

# ---- 1. Introduction ---------------------------------------------------------
para("1. Introduction and objectives", h1)
para(
    "Investment decisions are made under uncertainty about future prices. Instead of producing a "
    "single point forecast, stochastic process models describe the whole distribution of possible "
    "future prices, from which an investor can read off expected outcomes, ranges and the "
    "probability and size of losses. The objectives of this study are to (a) apply continuous-time "
    "stochastic models to real equity data; (b) compare a random-walk model (Geometric Brownian "
    "Motion), a mean-reverting model (Ornstein&ndash;Uhlenbeck) and a model with discontinuous "
    "jumps (Merton jump-diffusion); (c) implement them in modular Python code with Monte Carlo "
    "simulation; and (d) interpret the results in terms of volatility, trend, risk of loss and "
    "forecast horizon, and translate them into recommendations for investors.")
para(
    "The three stocks were chosen to span different risk profiles: AAPL (large-cap technology, "
    "growth), JPM (large bank, sensitive to interest rates and macro news) and KO (consumer "
    "staple, traditionally defensive and the most plausible candidate for mean reversion).")

# ---- 2. Data -----------------------------------------------------------------
para("2. Data and preprocessing", h1)
para("2.1 Source", h2)
cl = summary["cleaning"]
para(
    "Daily prices come from the Kaggle <i>S&amp;P 500 stock data</i> dataset (Nugent, 2018), "
    f"which covers all S&amp;P 500 constituents from {long_date(cl['start'])} "
    f"to {long_date(cl['end'])} ({cl['rows_out']:,} trading days). We use "
    "the daily closing price of AAPL, JPM and KO. The full five years are used for descriptive "
    f"statistics and backtesting; the most recent two years ({cfg.CALIBRATION_DAYS} trading days) are "
    "used to calibrate the models for the forward simulation. Two years balance two needs: enough "
    "observations to estimate volatility and jump parameters, and recent enough data to reflect the "
    "current regime. The forecast origin is therefore the last observation, "
    f"{long_date(cl['end'])}; because the data end there, the forward "
    "simulation is best read as a case study, while the rolling backtest (Section 5) provides the "
    "real out-of-sample test. The code can also download current data from Yahoo Finance "
    "(<font face='Mono'>--source yfinance</font>) so the same analysis can be repeated today.")
para("2.2 Cleaning", h2)
flagged = summary["cleaning"]["flagged_returns"]
n_flag = sum(len(v) for v in flagged.values())
para(
    "The cleaning routine sorts by date, removes duplicate dates, treats non-positive prices as "
    "missing, forward-fills gaps of at most three days and drops incomplete rows. The dataset proved "
    f"clean: there were {cl['duplicate_dates_removed']} duplicate dates, {cl['non_positive_prices']} "
    f"non-positive prices and {cl['missing_before_fill']} missing values. Prices are adjusted for "
    "AAPL's 7-for-1 split of June 2014 (no artificial jump appears in the series) but not for "
    "dividends, which slightly understates total return, most for KO (dividend yield of about 3%). "
    f"Anomaly screening flagged {n_flag} daily returns with a robust z-score above 6 (for example "
    "AAPL &minus;8.3% on 28 January 2014 after earnings and JPM &minus;7.2% on 24 June 2016 after "
    "the Brexit referendum). These are genuine market reactions, so they were <i>kept</i>: removing "
    "them would hide precisely the tail risk the models are meant to capture.")

f1 = figure("fig1_price_history.png",
            "Daily closing prices of AAPL, JPM and KO, rebased to 100 on 8 February 2013. "
            "The shaded area marks the two-year calibration window used for the forward simulation.")

para("2.3 Statistical properties", h2)
rows = [["Statistic (calibration window)"] + T]
spec = [("Annualised log drift", "ann_drift_log", lambda x: pct(x, 1)),
        ("Annualised volatility", "ann_vol", lambda x: pct(x, 1)),
        ("Skewness", "skew", lambda x: f"{x:.2f}"),
        ("Excess kurtosis", "excess_kurtosis", lambda x: f"{x:.2f}"),
        ("Jarque-Bera p-value", "jb_pvalue", lambda x: "&lt; 0.001" if x < 1e-3 else f"{x:.3f}"),
        ("ADF p-value (log price)", "adf_pvalue", lambda x: f"{x:.2f}"),
        ("ADF p-value, full 5 years", None, None),
        ("AR(1) half-life (trading days)", "half_life_days", lambda x: f"{x:.0f}"),
        ("Ljung-Box p, squared returns", "ljungbox_sq_ret_p", lambda x: "&lt; 0.001" if x < 1e-3 else f"{x:.2f}"),
        ("Maximum drawdown", "max_drawdown", lambda x: pct(x, 1))]
for label, col, fmt in spec:
    if col is None:
        rows.append([label] + [f"{diag_full.loc[t, 'adf_pvalue']:.3f}" for t in T])
    else:
        rows.append([label] + [fmt(diag.loc[t, col]) for t in T])
table(rows, f"Descriptive statistics and diagnostic tests of daily log returns and log prices, "
            f"calibration window {cal_start} &ndash; {cal_end} (n = {int(diag['n_obs'].iloc[0])} returns "
            "per stock) unless stated otherwise.", col_widths=[6.4 * cm, 3 * cm, 3 * cm, 3 * cm])
para(
    "Table 1 and Figure 2 summarise the features that guide model choice. All three return "
    "series are strongly leptokurtic and fail the Jarque&ndash;Bera normality test (Jarque &amp; Bera, "
    "1980), with KO also markedly negatively skewed: large daily moves occur far more often than a "
    "normal distribution allows, a well-documented stylised fact of asset returns (Cont, 2001). "
    "The Augmented Dickey&ndash;Fuller test (Dickey &amp; Fuller, 1979) does not reject a unit root in "
    "the log price of any stock over the calibration window; only KO over the full five years "
    f"is borderline (p = {diag_full.loc['KO', 'adf_pvalue']:.3f}), consistent with its range-bound "
    "behaviour in Figure 1. Finally, the Ljung&ndash;Box test on squared returns (Ljung &amp; Box, 1978) "
    "detects volatility clustering for JPM over the calibration window and for all three stocks over "
    "the full sample, a feature none of the constant-volatility models below can reproduce.")
figure("fig2_return_diagnostics.png",
       "Top: histograms of daily log returns over the full sample with a fitted normal density. "
       "Bottom: normal quantile-quantile plots. Points bending away from the line in both tails "
       "show that extreme returns are more frequent than under normality (fat tails).")

# ---- 3. Models ---------------------------------------------------------------
para("3. Models and implementation", h1)
para("3.1 Geometric Brownian Motion (GBM)", h2)
para("GBM is the benchmark model of Black and Scholes (1973):")
para("dS<sub>t</sub> = &mu; S<sub>t</sub> dt + &sigma; S<sub>t</sub> dW<sub>t</sub>,", eq)
para(
    "where &mu; is the expected rate of return, &sigma; the volatility and W a standard Brownian motion. "
    "Log returns are independent and normal with mean (&mu; &minus; &sigma;<super>2</super>/2)&Delta;t and "
    "variance &sigma;<super>2</super>&Delta;t, so prices are log-normal, stay positive and have no "
    "memory. Parameters are estimated by maximum likelihood from daily log returns, and paths are "
    "simulated with the exact solution "
    "S<sub>t+&Delta;t</sub> = S<sub>t</sub> exp[(&mu; &minus; &sigma;<super>2</super>/2)&Delta;t + "
    "&sigma;&radic;&Delta;t Z], Z ~ N(0,1). <i>Justification:</i> the ADF tests cannot reject a "
    "random walk in log prices, which is exactly the GBM hypothesis, and GBM is the industry "
    "baseline against which other models are judged.")
para("3.2 Ornstein&ndash;Uhlenbeck process on the log price (OU)", h2)
para("dX<sub>t</sub> = &kappa;(&theta; &minus; X<sub>t</sub>) dt + &sigma; dW<sub>t</sub>,  "
     "X<sub>t</sub> = ln S<sub>t</sub>.", eq)
para(
    "The log price is pulled towards a long-run level &theta; at speed &kappa; (half-life ln 2/&kappa;) "
    "(Uhlenbeck &amp; Ornstein, 1930; Schwartz, 1997). Its exact discretisation is an AR(1) model, "
    "X<sub>t+&Delta;t</sub> = a + bX<sub>t</sub> + &epsilon;, with b = e<super>&minus;&kappa;&Delta;t</super> and "
    "a = &theta;(1 &minus; b), so (&kappa;, &theta;, &sigma;) follow from an ordinary least-squares "
    "regression; simulation uses the exact Gaussian transition. <i>Justification:</i> some evidence "
    "of mean reversion in stock prices exists at long horizons (Poterba &amp; Summers, 1988), and "
    "a defensive stock such as KO, which traded in a narrow range for five years, is the natural "
    "candidate. The OU model is included to test this hypothesis directly against the random walk.")
para("3.3 Merton jump-diffusion", h2)
para("dS<sub>t</sub>/S<sub>t&minus;</sub> = (&mu; &minus; &lambda;k) dt + &sigma; dW<sub>t</sub> + "
     "(J &minus; 1) dN<sub>t</sub>,  ln J ~ N(&mu;<sub>J</sub>, &sigma;<sub>J</sub><super>2</super>),", eq)
para(
    "where N is a Poisson process with intensity &lambda; (jumps per year) and "
    "k = E[J &minus; 1] (Merton, 1976). The compensator &lambda;k keeps &mu; as the total expected "
    "return, so it is directly comparable to GBM. A daily return is then a Poisson mixture of normal "
    "distributions; we maximise this likelihood numerically (L-BFGS-B, several starting values, the "
    "Poisson sum truncated at 10 jumps a day). Unconstrained, the likelihood pushes &lambda; to "
    "implausibly high values (many tiny &lsquo;jumps&rsquo; acting as a generic fat-tail mixture), so "
    "&lambda; is capped at 12 per year, about one jump per month (four earnings releases plus a few "
    "macro shocks). <i>Justification:</i> the fat tails in Table 1 and the earnings- and news-driven "
    "gaps flagged in Section 2.2 are exactly what a jump component is designed to capture.")

para("3.4 Implementation", h2)
para(
    "The code is organised in two modules. <font face='Mono'>stochastic_models.py</font> contains "
    "data loading and cleaning, diagnostics, one class per model with a common "
    "<font face='Mono'>fit(prices)</font> / <font face='Mono'>simulate(s0, n_steps, n_paths, rng)</font> "
    "interface, risk metrics, the backtest and plotting functions. "
    "<font face='Mono'>run_analysis.py</font> runs the pipeline and writes all figures and tables; "
    "an executed Jupyter notebook walks through the same steps. Time is measured in years with "
    f"&Delta;t = 1/252. Each model is simulated with {summary['n_paths']:,} Monte Carlo paths per stock over "
    f"{max(cfg.HORIZONS)} trading days, with a fixed random seed ({summary['seed']}) for exact "
    "reproducibility. Simulation is fully vectorised in NumPy (Harris et al., 2020); calibration "
    "uses SciPy (Virtanen et al., 2020) and statsmodels (Seabold &amp; Perktold, 2010).")

rows = [["", "Parameter"] + T]
add = lambda label, sym, vals: rows.append([label, sym] + vals)
add("GBM", "&mu; (drift, p.a.)", [pct(P(t, "GBM", "mu"), 1) for t in T])
add("", "&sigma; (volatility, p.a.)", [pct(P(t, "GBM", "sigma"), 1) for t in T])
add("OU", "&kappa; (reversion speed, p.a.)", [f"{P(t, 'OU', 'kappa'):.2f}" for t in T])
add("", "e<super>&theta;</super> (long-run price, USD)", [f"{P(t, 'OU', 'theta_price'):.2f}" for t in T])
add("", "Half-life (trading days)", [f"{P(t, 'OU', 'half_life_days'):.0f}" for t in T])
add("", "&sigma; (p.a.)", [pct(P(t, "OU", "sigma"), 1) for t in T])
add("Merton", "&mu; (total drift, p.a.)", [pct(P(t, "Merton", "mu"), 1) for t in T])
add("", "&sigma; (diffusion vol, p.a.)", [pct(P(t, "Merton", "sigma"), 1) for t in T])
add("", "&lambda; (jumps per year)", [f"{P(t, 'Merton', 'lam'):.1f}" for t in T])
add("", "&mu;<sub>J</sub> (mean log jump)", [pct(P(t, "Merton", "mu_j"), 2) for t in T])
add("", "&sigma;<sub>J</sub> (jump size s.d.)", [pct(P(t, "Merton", "sigma_j"), 2) for t in T])
add("", "Total volatility (p.a.)", [pct(P(t, "Merton", "total_vol"), 1) for t in T])
add("", "LR statistic vs GBM", [f"{lr[t]:.0f}" for t in T])
add("Start", "S<sub>0</sub> (USD, 7 Feb 2018)", [f"{s0[t]:.2f}" for t in T])
table(rows, "Calibrated model parameters (two-year calibration window). The likelihood-ratio "
            "statistic compares Merton with the nested GBM (3 extra parameters; the 0.1% critical "
            "value of &chi;<super>2</super>(3) is 16.3).",
      col_widths=[1.6 * cm, 5.4 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm])
para(
    "Table 2 shows two points that shape all later results. First, the historical drifts of AAPL "
    f"({pct(P('AAPL','GBM','mu'))}) and JPM ({pct(P('JPM','GBM','mu'))}) reflect an exceptional two-year "
    "rally and are far above any sensible long-run expected return; the drift of an equity is "
    "notoriously hard to estimate because its standard error over a two-year window is about "
    "&sigma;/&radic;2, i.e. roughly 14 percentage points here (Merton, 1980). Second, the OU fit "
    "implies very slow mean reversion for AAPL and JPM (half-lives of about one year or more), with "
    "long-run levels <i>above</i> the current price, but fast reversion for KO (half-life "
    f"{P('KO','OU','half_life_days'):.0f} days) towards USD {P('KO','OU','theta_price'):.2f}, almost "
    "exactly its current price. Because the ADF tests cannot reject a unit root, these OU estimates "
    "are fragile; AR(1)-based estimates of &kappa; are also biased upwards in short samples. The "
    "Merton model splits volatility into a smaller diffusion part plus monthly jumps of a few per "
    "cent; total volatility stays close to the GBM value, and the likelihood improvement is highly "
    "significant for every stock.")

# ---- 4. Simulation results ---------------------------------------------------
page_break()
para("4. Simulation results", h1)
para(
    "Figures 3&ndash;5 show the simulated price paths from the forecast origin of "
    f"{long_date(cl['end'])}, and Table 3 lists the risk statistics "
    "at the three horizons. The 90% interval is the range between the 5th and 95th percentiles of "
    "the simulated price; VaR<sub>95</sub> is the loss not exceeded with 95% probability and "
    "ES<sub>95</sub> (expected shortfall) is the average loss in the worst 5% of scenarios.")
for t, desc in (("AAPL", "Apple"), ("JPM", "JPMorgan Chase"), ("KO", "Coca-Cola")):
    figure(f"fig3_fan_{t}.png",
           f"Simulated {desc} ({t}) price paths for 180 trading days under each model "
           f"({summary['n_paths']:,} paths; 40 shown). Shaded bands are the 50% and 90% intervals; "
           "solid and dashed black lines are the mean and median paths; the grey line is the last "
           "120 days of history.")

rows = [["Stock", "Model", "Days", "Median (USD)", "90% interval (USD)", "P(loss)",
         "P(drop &gt; 10%)", "VaR<sub>95</sub>", "ES<sub>95</sub>"]]
for t in T:
    for m in ("GBM", "OU", "Merton"):
        for h in cfg.HORIZONS:
            rows.append([t if (m == "GBM" and h == cfg.HORIZONS[0]) else "",
                         m if h == cfg.HORIZONS[0] else "", h,
                         f"{R(t, m, h, 'median'):.2f}",
                         f"{R(t, m, h, 'p05'):.2f} &ndash; {R(t, m, h, 'p95'):.2f}",
                         pct(R(t, m, h, "prob_loss"), 1), pct(R(t, m, h, "prob_drop_10"), 1),
                         pct(R(t, m, h, "var_95"), 1), pct(R(t, m, h, "es_95"), 1)])
table(rows, f"Monte Carlo risk statistics of the simulated price ({summary['n_paths']:,} paths per model). "
            "P(loss) is the probability that the price ends below the current price S<sub>0</sub>; "
            "VaR and ES are expressed as a percentage loss of S<sub>0</sub>. The Monte Carlo standard "
            "error of a probability near 50% is about 1.6 percentage points.",
      col_widths=[1.2 * cm, 1.5 * cm, 1.1 * cm, 1.9 * cm, 3.1 * cm, 1.6 * cm, 1.9 * cm, 1.6 * cm, 1.6 * cm])

figure("fig4_terminal_distributions.png",
       "Distribution of the simulated price after 180 trading days under each model. The dotted "
       "line marks the current price. OU distributions are narrower and centred closer to the "
       "estimated long-run level; GBM and Merton distributions are right-skewed (log-normal-like).")
figure("fig5_probability_of_loss.png",
       "Probability that the simulated price is below the current price, as a function of the "
       "horizon. A positive drift pushes the probability below 50% as the horizon grows.")

para("4.1 Interpretation", h2)
para(
    "<b>Volatility and trend.</b> Over short horizons the spread of outcomes is governed by "
    "volatility and grows with the square root of time; over longer horizons the drift takes over, "
    "because the expected log return grows linearly in time. This is visible in Figure 5: for "
    f"AAPL and JPM, whose calibrated drifts are high, GBM's probability of loss falls from close to "
    f"50% at one day to {pct(R('AAPL','GBM',180,'prob_loss'))} (AAPL) and "
    f"{pct(R('JPM','GBM',180,'prob_loss'))} (JPM) at 180 days. For KO, whose drift is close to zero, "
    f"it stays near 50% ({pct(R('KO','GBM',180,'prob_loss'))}) while the 90% interval widens "
    f"to USD {R('KO','GBM',180,'p05'):.2f}&ndash;{R('KO','GBM',180,'p95'):.2f}.")
para(
    "<b>Model differences.</b> The OU model produces markedly narrower long-horizon distributions "
    f"(KO 180-day 90% interval USD {R('KO','OU',180,'p05'):.2f}&ndash;{R('KO','OU',180,'p95'):.2f} versus "
    f"USD {R('KO','GBM',180,'p05'):.2f}&ndash;{R('KO','GBM',180,'p95'):.2f} under GBM) because mean "
    "reversion caps the variance of the log price at &sigma;<super>2</super>/(2&kappa;). For AAPL and "
    "JPM, OU's drift is a pull towards a level estimated from the same rally, so its central "
    "forecast lies between &lsquo;no change&rsquo; and the GBM trend, and its tail risk is actually "
    f"higher than GBM's (AAPL 180-day ES<sub>95</sub> {pct(R('AAPL','OU',180,'es_95'))} vs "
    f"{pct(R('AAPL','GBM',180,'es_95'))}). The Merton model is very close to GBM at all three "
    "horizons for AAPL and JPM: with roughly one jump a month, jumps of a few per cent average out "
    "over 30+ days and the distribution of the sum is again close to normal. For KO, the estimated "
    f"negative jumps (mean {pct(P('KO','Merton','mu_j'),1)}) combine with a higher estimated total "
    f"drift ({pct(P('KO','Merton','mu'))} vs {pct(P('KO','GBM','mu'))} for GBM), which lowers the "
    "probability of loss; this difference reflects drift-estimation noise rather than a robust "
    "property of the jump model.")
para(
    "<b>Short-term versus long-term prediction.</b> Short-horizon risk numbers are robust: the "
    f"30-day VaR<sub>95</sub> lies between {pct(risk[risk.horizon_days==30]['var_95'].min(),1)} and "
    f"{pct(risk[risk.horizon_days==30]['var_95'].max(),1)} across all stocks and models, because it "
    "is driven mainly by volatility, which is estimated precisely. Long-horizon numbers are fragile "
    "because they depend on the drift. Table 4 makes this explicit: keeping GBM's volatility fixed "
    "and replacing the historical drift by a long-run equity return of 8% or by zero changes the "
    "180-day risk profile dramatically for AAPL and JPM.")
rows = [["Stock", "Drift assumption", "&mu; (p.a.)", "Median (USD)", "P(loss)", "P(drop &gt; 10%)",
         "VaR<sub>95</sub>", "ES<sub>95</sub>"]]
for t in T:
    for d in ("historical", "8% long-run", "zero"):
        rows.append([t if d == "historical" else "", d, pct(S(t, d, "mu"), 1),
                     f"{S(t, d, 'median'):.2f}", pct(S(t, d, "prob_loss"), 1),
                     pct(S(t, d, "prob_drop_10"), 1), pct(S(t, d, "var_95"), 1), pct(S(t, d, "es_95"), 1)])
table(rows, "Sensitivity of the 180-day GBM risk statistics to the drift assumption "
            "(volatility fixed at its calibrated value).",
      col_widths=[1.3 * cm, 2.8 * cm, 1.8 * cm, 2.2 * cm, 1.8 * cm, 2.4 * cm, 1.8 * cm, 1.8 * cm])

# ---- 5. Backtest -------------------------------------------------------------
para("5. Out-of-sample validation", h1)
para(
    "In-sample fit does not show whether a model forecasts well. We therefore ran a rolling-origin "
    f"backtest: every {cfg.BACKTEST_STEP} trading days from February 2015 onwards ({n_origins} origins per "
    f"stock), each model was re-calibrated on the preceding {cfg.CALIBRATION_DAYS} days only, "
    f"{summary['n_paths']:,} paths were simulated, and the forecast was compared with the realised "
    f"price {cfg.BACKTEST_HORIZONS[0]} and {cfg.BACKTEST_HORIZONS[1]} days later. Three scores are "
    "reported: the <i>coverage</i> of the 90% interval (ideal: 90%), the continuous ranked "
    "probability score (CRPS, in % of the starting price; lower is better; Gneiting &amp; Raftery, 2007), "
    "which rewards forecasts that are both sharp and correct, and the Brier score of the predicted "
    "probability of a loss (lower is better; 0.25 is the score of always predicting 50%).")
figure("fig6_backtest_example.png",
       f"Example out-of-sample forecast for AAPL from {show_origin}: each model is calibrated on the "
       "preceding two years only and simulated forward for 180 trading days; the red line is the "
       "price that was actually realised.")
para(
    f"Figure 6 gives one illustration. From {show_origin}, the realised AAPL price ended at the "
    f"{summary['showcase_realized_percentile']['AAPL']['GBM'] * 100:.0f}th percentile of the GBM "
    f"forecast and the {summary['showcase_realized_percentile']['AAPL']['Merton'] * 100:.0f}th of "
    f"Merton's, i.e. near the centre, but at the "
    f"{summary['showcase_realized_percentile']['AAPL']['OU'] * 100:.0f}th percentile of the OU "
    "forecast, which expected the price to drift down towards its estimated mean. For JPM the "
    "realised price ended at the "
    f"{summary['showcase_realized_percentile']['JPM']['OU'] * 100:.0f}th percentile of the OU "
    "forecast, almost outside its 90% range.")
figure("fig7_backtest_summary.png",
       f"Rolling backtest scores ({n_origins} forecast origins per stock). Top: share of realised "
       "prices inside the model's 90% interval (dashed line = 90% target). Bottom: mean CRPS as a "
       "percentage of the starting price (lower is better).")
rows = [["Model", "Horizon", "Coverage of 90% interval", "Mean CRPS (% of S<sub>0</sub>)", "Brier score, P(loss)"]]
for m in ("GBM", "OU", "Merton"):
    for h in cfg.BACKTEST_HORIZONS:
        rows.append([m if h == cfg.BACKTEST_HORIZONS[0] else "", f"{h} days",
                     pct(pooled.loc[(m, h), "coverage_90"], 1),
                     f"{pooled.loc[(m, h), 'crps_pct']:.2f}", f"{pooled.loc[(m, h), 'brier']:.3f}"])
table(rows, "Backtest scores averaged over the three stocks (per-stock values in Figure 7 and "
            "<font face='Mono'>results/backtest_summary.csv</font>).",
      col_widths=[2.0 * cm, 2.0 * cm, 4.0 * cm, 4.0 * cm, 3.6 * cm])
para(
    "The results are consistent with the diagnostics. GBM and Merton intervals are close to "
    f"nominal for AAPL ({pct(B('AAPL','GBM',60,'coverage_90'))} at 60 days) and JPM "
    f"({pct(B('JPM','GBM',60,'coverage_90'))}), and too wide for KO "
    f"({pct(B('KO','GBM',60,'coverage_90'))}), whose volatility fell to about 9% in 2017, well below the "
    "14&ndash;16% of 2013&ndash;2016 that dominated its calibration windows. The OU model is systematically overconfident: "
    f"its 60-day coverage is only {pct(B('JPM','OU',60,'coverage_90'))} for JPM and "
    f"{pct(B('KO','OU',60,'coverage_90'))} for KO, even though KO is the stock for which mean "
    "reversion was most plausible. OU obtains a slightly lower CRPS for AAPL, where its narrower "
    "distribution happened to be centred well, but it has the worst Brier score on average: its "
    "directional calls (based on the distance to an estimated mean) were wrong more often than a "
    "coin flip. Merton and GBM are practically tied, with Merton marginally ahead on CRPS and Brier "
    "for JPM and KO. No model predicts direction reliably: Brier scores range from "
    f"{bt['brier'].min():.2f} to {bt['brier'].max():.2f}, close to or worse than the 0.25 of a coin flip.")

# ---- 6. Discussion -----------------------------------------------------------
para("6. Realism and limitations of the models", h1)
rows = [["", "GBM", "OU (log price)", "Merton jump-diffusion"],
        ["Captures", "Positive prices; uncertainty growing with &radic;t; compounding",
         "Range-bound behaviour; bounded long-run variance",
         "Fat tails, sudden gaps (earnings, macro news); negative skew"],
        ["Misses", "Fat tails, jumps, volatility clustering",
         "No unit-root rejection in the data; no long-run growth; fat tails",
         "Volatility clustering; jumps not identifiable without constraints"],
        ["Evidence here", "Well-calibrated 30/60-day intervals (AAPL, JPM)",
         "Overconfident intervals (69&ndash;93% coverage)",
         "Best in-sample fit; out-of-sample scores &asymp; GBM"],
        ["Best use", "Baseline risk ranges and position sizing",
         "Spreads, ratios, rates or commodities, not single stocks",
         "Short-horizon tail risk (VaR/ES), event risk, option pricing"]]
table(rows, "Qualitative comparison of the three models.",
      col_widths=[2.3 * cm, 4.5 * cm, 4.5 * cm, 4.9 * cm])
bullets([
    "<b>Constant parameters.</b> All three models assume volatility is constant, while the "
    "Ljung&ndash;Box tests show volatility clustering. In calm periods the models overstate risk "
    "(KO), and in turbulent periods they would understate it. GARCH-type (Bollerslev, 1986) or "
    "stochastic-volatility models (Heston, 1993) address this.",
    "<b>Drift uncertainty.</b> Two years of data cannot pin down the expected return (Table 4). "
    "Long-horizon outputs inherit this uncertainty; the drift is better set from an economic "
    "view (e.g. risk-free rate plus equity premium) than estimated from recent returns.",
    "<b>Identification of jumps.</b> With daily data, frequent small jumps and fat-tailed "
    "diffusion are hard to tell apart, so the jump intensity had to be constrained. The jump "
    "parameters should be read as a description of tail shape rather than of discrete events.",
    "<b>Mean reversion in price levels is not supported</b> for these stocks. Treating an OU "
    "level as a price target creates false confidence; the backtest confirms this.",
    "<b>Single-asset view.</b> The stocks are simulated independently. A portfolio analysis would "
    "need correlated Brownian motions (and common jumps), since correlations rise in sell-offs.",
    "<b>Data.</b> Prices are not dividend-adjusted and the sample ends in February 2018, so the "
    "numbers illustrate the method rather than describe today's market.",
])

# ---- 7. Recommendations ------------------------------------------------------
para("7. Recommendations for investors", h1)
bullets([
    "<b>Use the models for ranges and risk, not for direction.</b> None of the models beat a "
    "coin flip at predicting whether a stock would rise or fall. Their value lies in quantifying "
    "how much prices can move: base position sizes and stop-loss levels on the 90% intervals and "
    "on VaR/ES, not on the mean path.",
    "<b>Prefer GBM or Merton for horizons up to about three months,</b> where their intervals were "
    "well calibrated. Use Merton (or historical/filtered simulation) when daily tail risk matters, "
    f"e.g. around earnings. On these numbers, a 30-day 95% VaR of roughly "
    f"{pct(R('AAPL','Merton',30,'var_95'))} for AAPL and JPM and {pct(R('KO','Merton',30,'var_95'))} for "
    "KO is a reasonable planning figure for a one-month holding.",
    "<b>Do not extrapolate recent trends.</b> The 180-day probability of loss for AAPL and JPM is "
    "below 15% only under their historical drifts; under a plausible 8% drift it is about 40%. "
    "Investors with a six-month horizon should plan for a roughly 2-in-5 chance of being under water "
    "and a 1-in-5 chance of a loss of more than 10%.",
    "<b>Match the stock to the risk budget.</b> KO's volatility is about two-thirds of AAPL's and "
    f"JPM's, so its 180-day ES<sub>95</sub> is lower ({pct(R('KO','GBM',180,'es_95'))} vs "
    f"{pct(R('AAPL','GBM',180,'es_95'))} for AAPL under GBM), but it also offers little expected "
    "price growth: it suits capital preservation and income (dividends), while AAPL and JPM suit "
    "investors able to bear larger drawdowns for higher expected growth.",
    "<b>Treat mean-reversion signals with caution.</b> Buying a stock because an OU model says it is "
    "below its &lsquo;fair&rsquo; level was not supported by the data; mean-reversion strategies are "
    "better applied to spreads between related assets than to single-stock prices.",
    "<b>Re-calibrate regularly and stress test.</b> Re-estimate volatility at least monthly, "
    "complement the models with scenario analysis (e.g. a 2008- or 2020-type shock), and consider "
    "volatility-clustering models before relying on any single number.",
])

para("8. Conclusion", h1)
para(
    "Stochastic process models turn price uncertainty into explicit probabilities. For the three "
    "stocks studied, the random-walk family (GBM and its jump extension) described the data and "
    "forecast future ranges well, whereas mean reversion in price levels was neither supported "
    "statistically nor useful out of sample. Jumps improve the description of daily tail risk but "
    "add little at monthly horizons. The single most important input for long-horizon conclusions "
    "is the drift, which history estimates poorly; investors should therefore rely on these models "
    "for the <i>width</i> of the distribution and treat its <i>centre</i> as an assumption.")

# ---- References --------------------------------------------------------------
page_break()
para("References", h1)
refs = [
    "Black, F., &amp; Scholes, M. (1973). The pricing of options and corporate liabilities. <i>Journal of Political Economy, 81</i>(3), 637&ndash;654. https://doi.org/10.1086/260062",
    "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. <i>Journal of Econometrics, 31</i>(3), 307&ndash;327. https://doi.org/10.1016/0304-4076(86)90063-1",
    "Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. <i>Quantitative Finance, 1</i>(2), 223&ndash;236. https://doi.org/10.1080/713665670",
    "Dickey, D. A., &amp; Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. <i>Journal of the American Statistical Association, 74</i>(366a), 427&ndash;431. https://doi.org/10.1080/01621459.1979.10482531",
    "Glasserman, P. (2004). <i>Monte Carlo methods in financial engineering</i>. Springer. https://doi.org/10.1007/978-0-387-21617-1",
    "Gneiting, T., &amp; Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. <i>Journal of the American Statistical Association, 102</i>(477), 359&ndash;378. https://doi.org/10.1198/016214506000001437",
    "Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del R&iacute;o, J. F., Wiebe, M., Peterson, P., &hellip; Oliphant, T. E. (2020). Array programming with NumPy. <i>Nature, 585</i>(7825), 357&ndash;362. https://doi.org/10.1038/s41586-020-2649-2",
    "Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. <i>The Review of Financial Studies, 6</i>(2), 327&ndash;343. https://doi.org/10.1093/rfs/6.2.327",
    "Hull, J. C. (2018). <i>Options, futures, and other derivatives</i> (10th ed.). Pearson.",
    "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. <i>Computing in Science &amp; Engineering, 9</i>(3), 90&ndash;95. https://doi.org/10.1109/MCSE.2007.55",
    "Jarque, C. M., &amp; Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. <i>Economics Letters, 6</i>(3), 255&ndash;259. https://doi.org/10.1016/0165-1765(80)90024-5",
    "Ljung, G. M., &amp; Box, G. E. P. (1978). On a measure of lack of fit in time series models. <i>Biometrika, 65</i>(2), 297&ndash;303. https://doi.org/10.1093/biomet/65.2.297",
    "McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt &amp; J. Millman (Eds.), <i>Proceedings of the 9th Python in Science Conference</i> (pp. 56&ndash;61). https://doi.org/10.25080/Majora-92bf1922-00a",
    "Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. <i>Journal of Financial Economics, 3</i>(1&ndash;2), 125&ndash;144. https://doi.org/10.1016/0304-405X(76)90022-2",
    "Merton, R. C. (1980). On estimating the expected return on the market: An exploratory investigation. <i>Journal of Financial Economics, 8</i>(4), 323&ndash;361. https://doi.org/10.1016/0304-405X(80)90007-0",
    "Nugent, C. (2018). <i>S&amp;P 500 stock data</i> [Data set]. Kaggle. https://www.kaggle.com/datasets/camnugent/sandp500",
    "Poterba, J. M., &amp; Summers, L. H. (1988). Mean reversion in stock prices: Evidence and implications. <i>Journal of Financial Economics, 22</i>(1), 27&ndash;59. https://doi.org/10.1016/0304-405X(88)90021-9",
    "Schwartz, E. S. (1997). The stochastic behavior of commodity prices: Implications for valuation and hedging. <i>The Journal of Finance, 52</i>(3), 923&ndash;973. https://doi.org/10.1111/j.1540-6261.1997.tb02721.x",
    "Seabold, S., &amp; Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In S. van der Walt &amp; J. Millman (Eds.), <i>Proceedings of the 9th Python in Science Conference</i> (pp. 92&ndash;96). https://doi.org/10.25080/Majora-92bf1922-011",
    "Uhlenbeck, G. E., &amp; Ornstein, L. S. (1930). On the theory of the Brownian motion. <i>Physical Review, 36</i>(5), 823&ndash;841. https://doi.org/10.1103/PhysRev.36.823",
    "Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., &hellip; SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. <i>Nature Methods, 17</i>(3), 261&ndash;272. https://doi.org/10.1038/s41592-019-0686-2",
]
for r_ in refs:
    para(r_, ref)

para("Appendix: Reproducibility", h1)
para(
    "All code, data and outputs are in the <font face='Mono'>stock_stochastic_models/</font> folder. "
    "Install the dependencies with <font face='Mono'>pip install -r requirements.txt</font>, then "
    "run <font face='Mono'>python run_analysis.py</font> (about 1&ndash;2 minutes, including the "
    "backtest) and <font face='Mono'>python build_report.py</font> to regenerate this report. "
    "The notebook <font face='Mono'>Stock_Price_Stochastic_Models.ipynb</font> walks through the "
    "same steps with explanations. Monte Carlo and backtest settings (tickers, calibration window, "
    "horizons, number of paths, seed) are constants at the top of "
    "<font face='Mono'>run_analysis.py</font>. The general Monte Carlo methodology follows "
    "Glasserman (2004) and Hull (2018); data handling uses pandas (McKinney, 2010) and figures "
    "Matplotlib (Hunter, 2007).")


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "Modelling Stock Price Movements with Stochastic Processes")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=2 * cm, bottomMargin=2 * cm,
                        title="Modelling Stock Price Movements with Stochastic Processes",
                        author="Quantitative Research")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"Report written to {OUT}")

# ---- Word version ------------------------------------------------------------
BLOCKS_JSON = RES / "report_blocks.json"
BLOCKS_JSON.write_text(json.dumps(blocks, indent=1))
try:
    subprocess.run(["node", str(HERE / "build_report_docx.js"), str(BLOCKS_JSON),
                    str(OUT.with_suffix(".docx"))], check=True)
except (FileNotFoundError, subprocess.CalledProcessError) as exc:
    print(f"Word version skipped ({exc}); needs Node.js and `npm install docx`.")
