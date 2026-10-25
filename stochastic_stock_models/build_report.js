// build_report.js: generates Stochastic_Stock_Models_Report.docx from results_AAPL.json and figures/.
// Usage: node build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  HeadingLevel, WidthType, ShadingType, BorderStyle, LevelFormat, Footer, PageNumber,
  PageBreak,
} = require("docx");

const HERE = __dirname;
const R = JSON.parse(fs.readFileSync(path.join(HERE, "results_AAPL.json"), "utf8"));
const FONT = "Calibri";
const INK = "1F1F1F", ACCENT = "1F4E79", MUTED = "595959";
const PAGE_W = 12240, MARGIN = 1300, CONTENT_W = PAGE_W - 2 * MARGIN;

// ---------- formatting helpers ----------
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const num = (x, d = 2) => Number(x).toFixed(d);
const usd = (x) => "$" + Number(x).toFixed(2);

// Inline markup: **bold**, *italic*
function runs(text, base = {}) {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g).filter(Boolean);
  return parts.map((p) => {
    if (p.startsWith("**")) return new TextRun({ text: p.slice(2, -2), bold: true, ...base });
    if (p.startsWith("*")) return new TextRun({ text: p.slice(1, -1), italics: true, ...base });
    return new TextRun({ text: p, ...base });
  });
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 140, line: 288 },
  alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const Bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(t),
  spacing: { after: 80, line: 276 } });
const Num = (t) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: runs(t),
  spacing: { after: 100, line: 276 } });
const Eq = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
  children: [new TextRun({ text: t, font: "Cambria Math", size: 23 })] });

// APA-style label + title above figures and tables; an optional note below.
function caption(label, title) {
  return [
    new Paragraph({ keepNext: true, spacing: { before: 200, after: 0 }, children: [new TextRun({ text: label, bold: true })] }),
    new Paragraph({ keepNext: true, spacing: { after: 100 }, children: [new TextRun({ text: title, italics: true })] }),
  ];
}
const Note = (t) => new Paragraph({ spacing: { before: 60, after: 220 },
  children: [new TextRun({ text: "Note. ", italics: true, size: 19, color: MUTED }), ...runs(t, { size: 19, color: MUTED })] });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
let figNo = 0;
function Figure(file, title, note, widthIn = 6.4) {
  figNo += 1;
  const img = pngSize(path.join(HERE, "figures", file));
  const wPx = widthIn * 96;
  return [
    ...caption(`Figure ${figNo}`, title),
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [new ImageRun({ type: "png", data: img.data,
      transformation: { width: wPx, height: wPx * img.h / img.w },
      altText: { title: `Figure ${figNo}`, description: title, name: file } })] }),
    Note(note),
  ];
}

let tabNo = 0;
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function DataTable(title, headers, rows, widths, note, leftCols = 1) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, head, shade, last) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    borders: { top: head ? border : noBorder, bottom: border, left: noBorder, right: noBorder },
    shading: head ? { fill: "E8EEF5", type: ShadingType.CLEAR, color: "auto" }
      : shade ? { fill: "F7F7F7", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({ keepNext: !last, alignment: i < leftCols ? AlignmentType.LEFT : AlignmentType.RIGHT,
      children: [new TextRun({ text: String(text), bold: head, size: 19 })] })],
  });
  const table = new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, k) => new TableRow({ cantSplit: true, children: r.map((v, i) => cell(v, i, false, k % 2 === 1, k === rows.length - 1)) }))],
  });
  return [...caption(`Table ${tabNo}`, title), table, note ? Note(note) : new Paragraph({ spacing: { after: 160 }, children: [] })];
}

// ---------- numbers from the analysis ----------
const rs = R.returns_stats, fs5 = R.full_returns_stats, adf = R.adf;
const G = R.params["GBM"], O = R.params["OU (trend)"], M = R.params["Merton JD"], PO = R.plain_ou;
const S0 = R.s0;
const S = {}; R.summary.forEach((r) => { S[`${r.model}|${r.horizon}`] = r; });
const BT = {}; R.backtest_summary.forEach((r) => { BT[r.model] = r; });
const FIT = {}; R.fit.forEach((r) => { FIT[r.model] = r; });
const models = ["GBM", "OU (trend)", "Merton JD"];
const jumpShare = M.lam * (M.mu_j ** 2 + M.sigma_j ** 2) / M.total_vol ** 2;
const outliers = Object.entries(R.cleaning.flagged_outliers);
const s = (m, h) => S[`${m}|${h}`];
const devPct = Math.exp(O.y0) - 1;

// Hold-out figures are computed in the notebook (section 5); they are fixed by the seed.
const HO = { trainEnd: "21 October 2025", s0: 262.03, real: 316.15, realDate: "13 July 2026",
  GBM: { med: 300.32, lo: 205.5, hi: 443.6, pit: 0.58 }, "OU (trend)": { med: 264.09, lo: 228.3, hi: 307.1, pit: 0.98 },
  "Merton JD": { med: 302.98, lo: 208.9, hi: 437.2, pit: 0.58 } };

// ---------- document body ----------
const body = [];

// Title block
body.push(
  new Paragraph({ spacing: { before: 1800, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Modelling Stock Price Movements with Stochastic Processes", bold: true, size: 44, color: ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "Geometric Brownian Motion, Mean-Reverting and Jump-Diffusion Models Applied to Apple Inc. (AAPL)", size: 26, color: MUTED })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Stochastic Processes Problem Set: Final Report", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Quantitative Research Team", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 }, children: [new TextRun({ text: "October 2026", size: 24 })] }),
);
body.push(new Paragraph({ shading: { fill: "F2F5F9", type: ShadingType.CLEAR, color: "auto" }, spacing: { after: 120 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 8 } },
  children: [new TextRun({ text: "Executive summary", bold: true, color: ACCENT })] }));
[
  `We calibrated three stochastic models to two years of AAPL daily prices (8 Jul 2024 – 13 Jul 2026; last close ${usd(S0)}) and simulated 10,000 paths per model over 30, 60 and 180 trading days.`,
  `AAPL returns are clearly non-normal (excess kurtosis ${num(rs.excess_kurtosis, 1)}) and show strong volatility clustering. Jump-diffusion fits the data best by a wide margin (AIC ${num(FIT["GBM"].AIC - FIT["Merton JD"].AIC, 0)} points below GBM).`,
  `GBM and Merton give similar central forecasts (180-day median ${usd(s("GBM", 180).median_price)} and ${usd(s("Merton JD", 180).median_price)}) and probabilities of loss (${pct(s("GBM", 180).prob_loss, 0)} and ${pct(s("Merton JD", 180).prob_loss, 0)}). The mean-reverting model is the outlier: AAPL ended ${pct(devPct, 0)} above its two-year trend, so it predicts a pull-back (180-day median ${usd(s("OU (trend)", 180).median_price)}).`,
  `Out of sample, the GBM and Merton 90% intervals are well calibrated (${pct(BT["GBM"].coverage90, 0)} and ${pct(BT["Merton JD"].coverage90, 0)} coverage). The mean-reverting intervals are too narrow (${pct(BT["OU (trend)"].coverage90, 0)}) and missed the realised price in the 180-day hold-out. No model provides a reliable directional signal.`,
  `Recommendation: use the models to size positions and set risk limits, not to time the market. Budget for a 95% one-month VaR of about ${pct(s("Merton JD", 30).VaR95, 0)}, and hold AAPL as part of a diversified portfolio rather than as a concentrated bet.`,
].forEach((t) => body.push(Bullet(t)));
body.push(new Paragraph({ children: [new PageBreak()] }));


// 1. Introduction
body.push(H1("1. Introduction and Objectives"));
body.push(P("Investors make decisions about prices that are not yet known. Stochastic processes describe the probability distribution of future prices rather than a single forecast, so they make questions such as \"how likely is a loss of more than 10% over the next quarter?\" quantitative. They are the foundation of derivatives pricing (Black & Scholes, 1973; Merton, 1976) and of simulation-based risk management (Glasserman, 2004)."));
body.push(P("This report applies three such models to Apple Inc. (AAPL), a large, liquid technology stock. The objectives are to:"));
[
  "collect and clean at least one year of daily closing prices;",
  "select stochastic models suited to the stock's statistical properties and justify the choice;",
  "implement the stochastic differential equations (SDEs) in modular Python code and simulate future price paths with Monte Carlo over 30, 60 and 180 trading days;",
  "compare the models on volatility, trend, probability of loss and tail risk, and test them out of sample;",
  "translate the findings into practical recommendations for investors.",
].forEach((t) => body.push(Bullet(t)));
body.push(P("All code, data and figures are in the accompanying notebook *Stochastic_Stock_Models.ipynb* and in the modules *stock_models.py* and *plots.py*. Every number in this report is generated by that code with a fixed random seed."));

// 2. Data
body.push(H1("2. Data and Preprocessing"));
body.push(H2("2.1 Source"));
body.push(P(`The dataset contains AAPL daily open, high, low, close and volume from ${R.cleaning.start} to ${R.cleaning.end} (${R.cleaning.rows_raw.toLocaleString()} trading days). The prices were downloaded from Yahoo Finance with the *yfinance* library (Aroussi, 2026) with split and dividend adjustment, and published as a CSV file (Jonasmiles, 2026). The analysis environment could not reach Yahoo Finance directly, so this export was used. It ends on 13 July 2026, and all forecasts start from that date. The function *download_yfinance()* in the code reproduces the download and extends the sample to the present wherever Yahoo Finance is reachable.`));
body.push(P("Models were calibrated on the most recent two years (504 trading days, 8 July 2024 – 13 July 2026). This is long enough for stable volatility estimates and short enough to reflect the current market regime. The earlier three years serve as the history for the rolling out-of-sample backtest (Section 5.4)."));
body.push(H2("2.2 Cleaning"));
body.push(P("The function *clean_prices()* applies the following pipeline:"));
[
  "parse dates, discard unparseable rows and sort chronologically;",
  "remove duplicate dates (keeping the last record) and treat non-positive prices as missing;",
  "interpolate gaps of at most two consecutive days and drop longer gaps, since inventing data over a long gap would understate volatility;",
  "compute daily log returns *r(t) = ln S(t) − ln S(t−1)* and flag returns more than six robust (median absolute deviation) standard deviations from the median.",
].forEach((t) => body.push(Bullet(t)));
body.push(P(`The file had no missing values, duplicates or non-positive prices. Three returns were flagged as anomalies: ${outliers.map(([d, v]) => `${d} (${pct(v, 1)})`).join(", ")}. These match real market events: the November 2022 rally after a soft US inflation report, and the April 2025 tariff announcement followed by the 90-day tariff pause. They were **retained**. Deleting genuine extreme moves would make the data look safer than it is and would remove exactly the information the jump-diffusion model is designed to capture.`));

body.push(...Figure("fig1_price_history.png", "AAPL Adjusted Closing Price and Daily Log Returns, July 2021 – July 2026",
  "The shaded region is the two-year calibration window. Orange markers show daily moves larger than 7%. Volatility comes in bursts (for example April 2025) rather than at a constant rate."));

body.push(H2("2.3 Exploratory statistics"));
body.push(...DataTable("Descriptive Statistics of AAPL Daily Log Returns",
  ["Statistic", "Calibration window (2 yr)", "Full sample (5 yr)"],
  [
    ["Observations", rs.n_obs, fs5.n_obs],
    ["Annualised mean log return", pct(rs.ann_mean_log_return), pct(fs5.ann_mean_log_return)],
    ["Annualised volatility", pct(rs.ann_volatility), pct(fs5.ann_volatility)],
    ["Skewness", num(rs.skewness), num(fs5.skewness)],
    ["Excess kurtosis", num(rs.excess_kurtosis), num(fs5.excess_kurtosis)],
    ["Worst / best day", `${pct(rs.min)} / ${pct(rs.max)}`, `${pct(fs5.min)} / ${pct(fs5.max)}`],
    ["Jarque–Bera p-value (normality)", rs.jarque_bera_p < 1e-4 ? "< 0.0001" : num(rs.jarque_bera_p, 4), fs5.jarque_bera_p < 1e-4 ? "< 0.0001" : num(fs5.jarque_bera_p, 4)],
    ["Ljung–Box p, returns (10 lags)", num(rs.ljungbox_r_p, 3), num(fs5.ljungbox_r_p, 3)],
    ["Ljung–Box p, squared returns (10 lags)", "< 0.0001", "< 0.0001"],
  ], [4040, 2800, 2800],
  "Annualisation uses 252 trading days. Ljung–Box tests the null hypothesis of no autocorrelation."));
body.push(P(`Three stylised facts of equity returns (Cont, 2001) are visible. First, **fat tails**: excess kurtosis is ${num(rs.excess_kurtosis, 1)}, normality is rejected decisively, and the Q–Q plot in Figure 2 bends away from the normal line at both ends. Second, **little linear predictability**: returns show no significant autocorrelation at the 5% level. Third, **volatility clustering**: squared returns are strongly autocorrelated. Augmented Dickey–Fuller tests (Dickey & Fuller, 1979) on the log price do not reject a unit root, with a constant (p = ${num(adf.adf_c_p)}) or with a constant and trend (p = ${num(adf.adf_ct_p)}). This is consistent with a random walk and gives no statistical support for mean reversion.`));
body.push(...Figure("fig2_return_distribution.png", "Distribution of Daily Log Returns Compared With Fitted Model Densities",
  "Left: histogram of the 504 calibration-window returns with the GBM (normal) and Merton jump-diffusion densities. Merton captures the sharper peak and heavier tails. Right: normal Q–Q plot; points beyond the line at both ends indicate fat tails."));

// 3. Models
body.push(H1("3. Model Selection and Implementation"));
body.push(H2("3.1 Choice of models"));
body.push(P("Three models were selected. Each represents a different hypothesis about how the stock moves, and together they span the main modelling families:"));
body.push(Bullet("**Geometric Brownian Motion (GBM)** is the benchmark. It assumes prices follow a random walk in logs, with constant drift and volatility. The unit-root test results and the absence of return autocorrelation support it as a first approximation, and it underlies the Black–Scholes model."));
body.push(Bullet("**Ornstein–Uhlenbeck (OU) mean reversion** tests the opposite hypothesis: that prices are pulled back toward a fundamental level (Uhlenbeck & Ornstein, 1930; Vasicek, 1977). A flat long-run level makes little sense for a growing company, so the log price is modelled as reverting to a linear trend (a trend-stationary OU process). Mean reversion over horizons of months is documented for some stock portfolios (Poterba & Summers, 1988), which makes it a meaningful alternative to test."));
body.push(Bullet("**Merton jump-diffusion** extends GBM with randomly occurring jumps (Merton, 1976). It addresses GBM's most obvious failure on this dataset, the fat tails created by earnings releases and macro shocks such as April 2025."));

body.push(H2("3.2 Model specifications"));
body.push(P("Time *t* is in years and one trading day is Δt = 1/252. *W(t)* is a standard Brownian motion."));
body.push(P("**Geometric Brownian Motion.** The SDE and its exact solution are"));
body.push(Eq("dS(t) = μ S(t) dt + σ S(t) dW(t),      S(t+Δt) = S(t) · exp[(μ − σ²/2)Δt + σ√Δt · Z],  Z ~ N(0,1)"));
body.push(P("Log returns are therefore i.i.d. normal, and the maximum-likelihood estimates are σ̂ = sd(r)/√Δt and μ̂ = mean(r)/Δt + σ̂²/2."));
body.push(P("**Trend-stationary Ornstein–Uhlenbeck.** The log price is the sum of a linear trend and an OU deviation *Y(t)*:"));
body.push(Eq("ln S(t) = a + g·t + Y(t),      dY(t) = −θ Y(t) dt + σ dW(t)"));
body.push(P("θ is the speed of reversion; the half-life of a deviation is ln 2 / θ. The exact discrete form is an AR(1) process, Y(t+Δt) = b·Y(t) + ε with b = e^(−θΔt) and Var(ε) = σ²(1 − b²)/(2θ). Estimation is therefore an OLS trend fit followed by an AR(1) regression on the residuals. Setting g = 0 gives the classic OU process with a constant long-run level."));
body.push(P("**Merton jump-diffusion.** The price follows GBM plus a compound Poisson jump component:"));
body.push(Eq("dS(t)/S(t−) = (μ − λk) dt + σ dW(t) + (J − 1) dN(t),      ln J ~ N(μ_J, σ_J²),  k = e^(μ_J + σ_J²/2) − 1"));
body.push(P("*N(t)* is a Poisson process with λ jumps per year; the λk term keeps the total expected return equal to μ. The daily log-return density is a Poisson-weighted mixture of normals. Its five parameters were estimated by maximum likelihood (L-BFGS-B, several starting points), with starting values from a threshold rule that labels returns beyond three robust standard deviations as jumps (Mancini, 2009)."));
body.push(P(`**Identification issue.** Without a constraint, the Merton likelihood keeps increasing as λ grows (to 100 jumps per year and beyond). The model then uses many small jumps to imitate volatility clustering rather than describing rare discontinuities. The intensity was therefore capped at λ = 12 per year (about one large move per month, in line with quarterly earnings plus macro events). The cap is binding, which signals that a stochastic-volatility model would describe this data better (Section 6).`));

body.push(H2("3.3 Python implementation"));
body.push(P("The code is modular. Each step is a separate, documented function in *stock_models.py*:"));
[
  "**Data**: *load_yfinance_csv()*, *download_yfinance()*, *clean_prices()*, *describe_returns()*, *stationarity_tests()*.",
  "**Calibration**: *fit_gbm()*, *fit_ou()*, *fit_merton()* return typed parameter objects (dataclasses); *model_loglik()* computes log-likelihood, AIC and BIC.",
  "**Simulation**: *simulate_gbm()*, *simulate_ou()* and *simulate_merton()* are vectorised with NumPy and return an array of shape (paths × days). All three use exact transition laws, so there is no Euler discretisation bias. Merton jumps are sampled exactly: the daily jump count is Poisson(λΔt), and the sum of n jumps is N(nμ_J, nσ_J²).",
  "**Risk and evaluation**: *terminal_stats()* (mean, median, quantiles, probability of loss, VaR, CVaR, drawdown), *rolling_backtest()* and *kupiec_pof()*.",
  "**Visualisation**: *plots.py* produces every figure, with one fixed colour per model (GBM blue, OU orange, Merton green).",
].forEach((t) => body.push(Bullet(t)));
body.push(P("Each model was simulated with **10,000 Monte Carlo paths** over 180 trading days; the 30- and 60-day results are read off the same paths. This is ten times the required 1,000 runs and keeps the Monte Carlo error of the 5% quantile below about half a percentage point. Each model uses its own seeded random generator, so results are reproducible and the models do not share random draws."));

body.push(H2("3.4 Calibrated parameters"));
body.push(...DataTable("Calibrated Model Parameters (Calibration Window 8 Jul 2024 – 13 Jul 2026)",
  ["Model", "Parameter", "Estimate", "Interpretation"],
  [
    ["GBM", "μ (drift)", pct(G.mu), "Expected annual price return"],
    ["", "σ (volatility)", pct(G.sigma), "Annualised volatility"],
    ["OU (trend)", "g (trend growth)", pct(O.g), "Annual growth of trend line"],
    ["", "θ (reversion speed)", num(O.theta), `Half-life ${num(O.half_life_days, 1)} trading days`],
    ["", "σ (volatility)", pct(O.sigma), "Volatility of deviations"],
    ["", "Y(0) (current deviation)", `+${num(O.y0, 3)}`, `Price ${pct(devPct)} above trend`],
    ["Plain OU (g = 0)", "θ", num(PO.theta), `Half-life ${num(PO.half_life_days, 0)} trading days`],
    ["Merton JD", "μ (total drift)", pct(M.mu), "Expected annual price return"],
    ["", "σ (diffusion)", pct(M.sigma), "Volatility between jumps"],
    ["", "λ (intensity)", `${num(M.lam, 1)} / yr`, "Cap binding (see 3.2)"],
    ["", "μ_J, σ_J (jump size)", `${pct(M.mu_j, 2)}, ${pct(M.sigma_j, 2)}`, "Mean and spread of log jumps"],
    ["", "Total volatility", pct(M.total_vol), `Jumps: ${pct(jumpShare, 0)} of variance`],
  ], [1900, 2350, 1700, 3690]));
body.push(P(`The drift estimates (about 21–23% per year) reflect AAPL's strong rise over the window, from about $226 to $316. Drift estimates are notoriously imprecise: with σ ≈ 28%, the standard error of a two-year mean return is about 28%/√2 ≈ 20 percentage points, so the true drift could plausibly be anywhere from near zero to above 40%. The volatility estimates agree closely across models at 27–28%. In the Merton fit, jumps average ${pct(M.mu_j, 1)} with a standard deviation of ${pct(M.sigma_j, 1)} and account for about ${pct(jumpShare, 0)} of total return variance.`));
body.push(P(`The trend-OU fit implies a half-life of ${num(O.half_life_days, 0)} trading days. Combined with the current deviation of ${pct(devPct, 0)} above trend, this is what drives the model's bearish forecast. Two caveats apply. First, the ADF test cannot reject a unit root, so the reversion may be a statistical artefact. Second, AR(1) estimates of θ are biased upward in finite samples, which makes mean reversion look stronger than it is. Without a trend, the plain OU half-life is ${num(PO.half_life_days, 0)} days, which confirms that the result depends heavily on the specification.`));
body.push(...DataTable("In-Sample Goodness of Fit",
  ["Model", "Parameters", "Log-likelihood", "AIC", "BIC"],
  models.map((m) => [m, FIT[m].params, num(FIT[m].loglik, 1), num(FIT[m].AIC, 1), num(FIT[m].BIC, 1)]),
  [2840, 1500, 1800, 1750, 1750],
  "Lower AIC/BIC is better. All likelihoods are for the same 504 one-day transitions and are directly comparable."));
body.push(P(`Merton jump-diffusion is clearly the best in-sample fit: its AIC is ${num(FIT["GBM"].AIC - FIT["Merton JD"].AIC, 0)} points lower than GBM's, and a difference above 10 is considered decisive. The trend-OU model barely improves on GBM (ΔAIC = ${num(FIT["GBM"].AIC - FIT["OU (trend)"].AIC, 1)}), and BIC, which penalises extra parameters more heavily, prefers GBM to OU. On daily data, mean reversion explains almost nothing; its effect matters only at longer horizons, where it is hardest to verify.`));

// 4. Simulation results
body.push(H1("4. Simulation Results"));
body.push(P(`All simulations start from the last observed close, S(0) = ${usd(S0)} on 13 July 2026. Figure 3 shows a sample of paths for each model with the mean and median trajectories and the 50% and 90% intervals. Figure 4 overlays the three forecasts, and Figure 5 shows the full distribution of simulated prices at each horizon.`));
body.push(...Figure("fig3_sample_paths.png", "Simulated 180-Day Price Paths by Model",
  "Each panel shows 100 of the 10,000 simulated paths, the mean (solid black) and median (dashed) trajectories, and the 50% and 90% intervals, after the last six months of observed prices. The y-axis is shared across panels."));
body.push(...Figure("fig4_model_comparison_fan.png", "Median Forecast and 90% Interval for the Three Models",
  "Solid lines are medians; dotted lines are the 5th and 95th percentiles. GBM and Merton nearly coincide. The OU forecast is pulled toward the trend line and has a much narrower interval."));
body.push(...Figure("fig5_terminal_histograms.png", "Distribution of Simulated Prices at 30, 60 and 180 Trading Days",
  "The dashed line marks today's price. GBM and Merton distributions are right-skewed (log-normal-like) and widen with the horizon. The OU distribution stops widening once mean reversion dominates."));

const rowsRisk = [];
models.forEach((m) => [30, 60, 180].forEach((h) => {
  const x = s(m, h);
  rowsRisk.push([h === 30 ? m : "", h, usd(x.mean_price), usd(x.median_price), `${num(x.p05_price, 0)} – ${num(x.p95_price, 0)}`,
    pct(x.expected_return), pct(x.prob_loss), pct(x.prob_loss_gt_10), pct(x.VaR95), pct(x.CVaR95)]);
}));
body.push(...DataTable("Simulated Price Distribution and Risk Metrics by Model and Horizon (10,000 Paths)",
  ["Model", "Days", "Mean", "Median", "90% interval", "E[return]", "P(loss)", "P(loss >10%)", "VaR 95%", "CVaR 95%"],
  rowsRisk, [1150, 560, 880, 880, 1150, 900, 850, 1100, 1050, 1120],
  "P(loss) is the probability that the price at the horizon is below today's price. VaR 95% is the loss not exceeded with 95% probability; CVaR 95% (expected shortfall) is the average loss in the worst 5% of outcomes. Both are expressed as a percentage of the initial investment."));
body.push(...Figure("fig6_risk_vs_horizon.png", "Probability of Loss and 95% Value-at-Risk as Functions of the Horizon",
  "For GBM and Merton the probability of loss falls with the horizon because the positive drift accumulates faster than the uncertainty around it, while VaR keeps rising roughly with the square root of time. Under OU both curves bend once the pull toward the trend takes over."));

// 5. Analysis
body.push(H1("5. Analysis and Interpretation"));
body.push(H2("5.1 Volatility and trend behaviour"));
body.push(P(`GBM and Merton have almost the same total volatility (${pct(G.sigma, 0)} and ${pct(M.total_vol, 0)}) and the same drift, so their central forecasts are nearly identical: a 180-day median of ${usd(s("GBM", 180).median_price)} under GBM and ${usd(s("Merton JD", 180).median_price)} under Merton. In both, the mean lies above the median (for example ${usd(s("GBM", 180).mean_price)} vs ${usd(s("GBM", 180).median_price)} for GBM at 180 days). This is the right skew of a log-normal distribution: a few very large gains pull the average up, while the typical outcome is more modest. The two models differ in how risk is distributed. Merton puts more of its variance in rare jumps and less in day-to-day diffusion, so it shows a slightly lower probability of an ordinary loss but a fatter left tail per unit of volatility.`));
body.push(P(`The trend-OU model behaves very differently. Because AAPL ended the window ${pct(devPct, 0)} above its fitted trend, the model expects a reversion: the median falls to about ${usd(s("OU (trend)", 60).median_price)} within 60 days and then rises slowly with the trend, to ${usd(s("OU (trend)", 180).median_price)} at 180 days. Its uncertainty also stops growing: the 90% interval at 180 days (${num(s("OU (trend)", 180).p05_price, 0)}–${num(s("OU (trend)", 180).p95_price, 0)}) is about a third as wide as GBM's (${num(s("GBM", 180).p05_price, 0)}–${num(s("GBM", 180).p95_price, 0)}). This difference is the central practical point: **whether one believes in mean reversion changes the long-horizon risk estimate by a factor of about three.**`));
body.push(H2("5.2 Investment risk"));
body.push(P(`Under GBM and Merton, an investor buying at ${usd(S0)} has roughly a 39–42% chance of being under water after 30 trading days, falling to ${pct(s("Merton JD", 180).prob_loss, 0)}–${pct(s("GBM", 180).prob_loss, 0)} after 180 days. The 95% VaR rises from about 12–13% at one month to 21–23% at nine months, and the average loss in the worst 5% of scenarios (CVaR) reaches ${pct(s("GBM", 180).CVaR95, 0)} under GBM. The typical worst peak-to-trough drawdown along a 180-day path is ${pct(s("GBM", 180).median_max_drawdown, 0)}, so even investors who end up with a gain should expect to sit through a meaningful decline on the way.`));
body.push(P(`Under OU, the probability of loss is much higher (${pct(s("OU (trend)", 30).prob_loss, 0)} at 30 days, ${pct(s("OU (trend)", 60).prob_loss, 0)} at 60 days), but large losses are rarer at long horizons because the trend line supports the price. The models therefore disagree mainly about direction, which is the least predictable component, and agree broadly on the scale of short-term risk: all three put the one-month 95% VaR between ${pct(Math.min(...models.map((m) => s(m, 30).VaR95)), 0)} and ${pct(Math.max(...models.map((m) => s(m, 30).VaR95)), 0)}.`));
body.push(H2("5.3 Short-term vs. long-term prediction"));
body.push(P("Over short horizons the forecast is dominated by volatility: at 30 days the standard deviation of the return (about 10%) is roughly four times the expected return (about 2.5%), so the direction is close to a coin flip. Volatility is estimated precisely, and the models agree, so short-horizon risk statements (VaR, the probability of a 10% drop) are the most trustworthy output. Over long horizons the drift matters more, but it is also the least reliable estimate. A 180-day expected return of +16% under GBM simply extrapolates AAPL's exceptional past two years and should not be read as a forecast. Long-horizon results are therefore best treated as scenario ranges whose width, not centre, carries the information."));

body.push(H2("5.4 Out-of-sample validation"));
body.push(P(`**Hold-out test.** The models were recalibrated on the 504 days ending ${HO.trainEnd} (price ${usd(HO.s0)}) and simulated 180 days forward (Figure 7). The realised price on ${HO.realDate} was ${usd(HO.real)}, a gain of ${pct(HO.real / HO.s0 - 1, 0)}. It fell at the ${num(HO["GBM"].pit * 100, 0)}th percentile of the GBM and Merton forecasts, well inside their 90% intervals ($${num(HO["GBM"].lo, 0)}–${num(HO["GBM"].hi, 0)}). It fell at the ${num(HO["OU (trend)"].pit * 100, 0)}th percentile of the OU forecast, above its upper bound of $${num(HO["OU (trend)"].hi, 0)}. OU correctly anticipated the sideways drift of December 2025 – April 2026 but missed the subsequent breakout.`));
body.push(...Figure("fig7_holdout_test.png", "Hold-Out Test: Forecasts From 21 October 2025 Compared With the Realised Path",
  "Models calibrated only on data up to 21 Oct 2025. The realised path (black) stays within the GBM and Merton 90% bands throughout. It ends above the OU band, which was outside its 90% interval on 11% of the days."));
body.push(P("**Rolling backtest.** One hold-out is a single draw, so the evaluation was repeated at 73 origins spaced 10 trading days apart (July 2023 – May 2026). At each origin, each model was recalibrated on the preceding 504 days and used to forecast 30 days ahead. Two properties were checked: whether the 90% interval contains the outcome about 90% of the time (Kupiec, 1995), and whether the probability integral transform (PIT), the forecast percentile of the realised price, is uniformly distributed (Gneiting et al., 2007)."));
body.push(...DataTable("Rolling 30-Day Backtest of Forecast Distributions (73 Forecasts per Model)",
  ["Model", "90% interval coverage", "Kupiec p-value", "PIT uniformity (KS p)", "Mean log score"],
  models.map((m) => [m, pct(BT[m].coverage90), num(BT[m].kupiec_p, 3), num(BT[m].pit_ks_p, 3), num(BT[m].mean_log_score, 3)]),
  [2140, 1900, 1800, 1950, 1850],
  "Higher log score is better (sharper and well-calibrated). Neighbouring 30-day windows overlap, so the observations are not independent and the p-values are indicative."));
body.push(...Figure("fig8_backtest_pit.png", "Distribution of Probability Integral Transform Values in the Rolling Backtest",
  "A well-calibrated forecaster produces a flat histogram (dashed line). The OU histogram is U-shaped, with too many outcomes in the extreme deciles, which indicates intervals that are too narrow.", 6.2));
body.push(P(`GBM and Merton are well calibrated, with slightly conservative intervals (${pct(BT["GBM"].coverage90, 0)} and ${pct(BT["Merton JD"].coverage90, 0)} coverage for a nominal 90%) and roughly flat PIT histograms. OU covers only ${pct(BT["OU (trend)"].coverage90, 0)} of outcomes, and its U-shaped PIT histogram confirms that it underestimates uncertainty. It nevertheless has the highest average log score, because its sharper forecasts are rewarded whenever the price does revert. With 73 overlapping observations none of these differences is statistically significant (all Kupiec p > 0.15). The honest conclusion is that **no model shows a reliable edge in predicting direction**, while GBM and Merton describe the size of one-month uncertainty well.`));

// 6. Limitations
body.push(H1("6. Realism and Limitations of the Models"));
body.push(...DataTable("Strengths and Weaknesses of Each Model for AAPL",
  ["Model", "Strengths", "Weaknesses"],
  [
    ["GBM", "Simple and analytically tractable; prices stay positive; consistent with the unit-root evidence; well-calibrated 30-day intervals", "Normal returns understate tail risk (kurtosis 10.6 observed); constant volatility ignores clustering; drift estimate very noisy"],
    ["OU (trend)", "Captures pull-backs after overextension; bounded long-horizon uncertainty; sharpest forecasts", "Mean reversion not supported by ADF; θ biased upward; intervals too narrow out of sample; trend line is a modelling choice"],
    ["Merton JD", "Best in-sample fit; models crash and earnings-gap risk explicitly; well-calibrated intervals", "Jump intensity poorly identified (cap binds); jumps independent of each other; still constant diffusion volatility"],
  ], [1500, 4070, 4070], null, 3));
body.push(P("Several limitations are shared by all three models:"));
[
  "**Constant volatility.** The strongest pattern in the data is volatility clustering (Ljung–Box on squared returns, p < 0.0001). After a shock such as April 2025, real risk stays elevated for weeks, while these models assume it returns to average immediately. Stochastic-volatility (Heston) or GARCH-type models address this directly.",
  "**Parameter uncertainty.** Forecasts use point estimates, while the drift in particular is uncertain by about ±20 percentage points per year. Bayesian or bootstrap simulation that also draws the parameters would give wider, more honest long-horizon intervals.",
  "**Stationarity of the regime.** Calibrating on two exceptional years for AAPL builds a strong upward drift into GBM and Merton. A different window would change the central forecasts considerably, while volatility estimates are much more stable.",
  "**Single stock, single data source, and stale endpoint.** The analysis covers one stock and ends on 13 July 2026, because live market data could not be retrieved in the analysis environment. The 30- and 60-day horizons have therefore already elapsed in calendar time; rerunning the pipeline with *--source yfinance* updates every result.",
  "**Physical vs. risk-neutral measure.** All simulations use real-world (historical) drift and are suitable for risk assessment, not for pricing options, which requires replacing μ with the risk-free rate.",
].forEach((t) => body.push(Bullet(t)));

// 7. Recommendations
body.push(H1("7. Recommendations for Investors"));
[
  `**Use the models for risk sizing, not market timing.** The backtest shows the models describe the size of uncertainty well but cannot predict direction. Base position sizes and stop-loss levels on the volatility-driven outputs, for example a one-month 95% VaR of about ${pct(s("Merton JD", 30).VaR95, 0)} and a CVaR of about ${pct(s("Merton JD", 30).CVaR95, 0)}.`,
  `**Plan for tail events.** Jump risk accounts for about ${pct(jumpShare, 0)} of AAPL's return variance, and single-day moves of ±10–14% occurred in the sample. Investors with leverage or short option positions should stress-test against a 15–20% gap move, not just the normal-distribution VaR. Protective puts or collars are worth considering around earnings dates.`,
  "**Match the holding period to the evidence.** Over 6–12 months GBM and Merton give a 70–75% chance of a gain, but that relies on a drift estimate inflated by an exceptional period. Long-term investors should rely on fundamentals for the expected return and use the simulations only for the range of outcomes.",
  `**Treat the mean-reversion signal as a caution, not a sell signal.** AAPL trades ${pct(devPct, 0)} above its two-year trend. This argues against adding aggressively at current levels and for averaging into new positions. It does not justify selling, given the weak statistical support and the missed breakout in the hold-out test.`,
  "**Diversify.** With an annual volatility of about 28% and a typical 180-day drawdown near 17–19%, a concentrated AAPL position carries substantial idiosyncratic risk. Holding it within a diversified portfolio reduces that risk without giving up its expected return.",
  "**Monitor and recalibrate.** Re-estimate the parameters at least monthly, and immediately after volatility shocks. Extending the framework with stochastic volatility would make the risk estimates respond to changing market conditions.",
].forEach((t) => body.push(Num(t)));

// 8. Conclusion
body.push(H1("8. Conclusion"));
body.push(P("Three stochastic models were implemented, calibrated and tested on five years of AAPL daily data. Merton jump-diffusion describes the observed fat tails best, GBM remains a robust and well-calibrated baseline, and the trend-stationary OU model offers a contrasting mean-reversion view that is weakly supported by the data and too confident out of sample. The models agree on the scale of short-term risk and disagree on direction, which is the part of the forecast that is hardest to estimate and that no model predicts reliably. The main value of stochastic modelling for investors is therefore a disciplined, quantitative view of risk: how bad outcomes can get, how likely they are, and how that changes with the investment horizon."));

// References
body.push(H1("References"));
const refs = [
  "Aroussi, R. (2026). *yfinance: Download market data from Yahoo! Finance's API* (Version 1.7.0) [Computer software]. https://github.com/ranaroussi/yfinance",
  "Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy, 81*(3), 637–654. https://doi.org/10.1086/260062",
  "Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance, 1*(2), 223–236. https://doi.org/10.1080/713665670",
  "Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association, 74*(366a), 427–431. https://doi.org/10.1080/01621459.1979.10482531",
  "Glasserman, P. (2004). *Monte Carlo methods in financial engineering*. Springer. https://doi.org/10.1007/978-0-387-21617-1",
  "Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *Journal of the Royal Statistical Society: Series B, 69*(2), 243–268. https://doi.org/10.1111/j.1467-9868.2007.00587.x",
  "Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2",
  "Hull, J. C. (2022). *Options, futures, and other derivatives* (11th ed.). Pearson.",
  "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55",
  "Jonasmiles. (2026). *AAPL-STOCK-ANALYSIS-2021-2026* [Data set]. GitHub. https://github.com/Jonasmiles/AAPL-STOCK-ANALYSIS-2021-2026",
  "Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *The Journal of Derivatives, 3*(2), 73–84. https://doi.org/10.3905/jod.1995.407942",
  "Mancini, C. (2009). Non-parametric threshold estimation for models with stochastic diffusion coefficient and jumps. *Scandinavian Journal of Statistics, 36*(2), 270–296. https://doi.org/10.1111/j.1467-9469.2008.00622.x",
  "McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 56–61). https://doi.org/10.25080/Majora-92bf1922-00a",
  "Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics, 3*(1–2), 125–144. https://doi.org/10.1016/0304-405X(76)90022-2",
  "Poterba, J. M., & Summers, L. H. (1988). Mean reversion in stock prices: Evidence and implications. *Journal of Financial Economics, 22*(1), 27–59. https://doi.org/10.1016/0304-405X(88)90021-9",
  "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 92–96). https://doi.org/10.25080/Majora-92bf1922-011",
  "Uhlenbeck, G. E., & Ornstein, L. S. (1930). On the theory of the Brownian motion. *Physical Review, 36*(5), 823–841. https://doi.org/10.1103/PhysRev.36.823",
  "Vasicek, O. (1977). An equilibrium characterization of the term structure. *Journal of Financial Economics, 5*(2), 177–188. https://doi.org/10.1016/0304-405X(77)90016-2",
  "Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2",
];
refs.forEach((t) => body.push(new Paragraph({ children: runs(t), indent: { left: 720, hanging: 720 }, spacing: { after: 120, line: 276 } })));

// Appendix
body.push(H1("Appendix: Reproducing the Results"));
body.push(P("Install the dependencies with *pip install -r requirements.txt* (NumPy, SciPy, pandas, matplotlib, statsmodels; yfinance is optional). Then either open *Stochastic_Stock_Models.ipynb* and run all cells, or run *python stock_models.py* from a terminal. The pipeline writes the figures to *figures/*, the cleaned data to *data/AAPL_clean.csv*, and every parameter and risk metric to *results_AAPL.json*. The command *python stock_models.py --source yfinance --tickers AAPL MSFT KO* downloads fresh data and repeats the full analysis for several stocks. Random seeds are fixed (seed = 42), so all results are reproducible exactly."));

// ---------- document ----------
const doc = new Document({
  creator: "Quantitative Research Team",
  title: "Modelling Stock Price Movements with Stochastic Processes",
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: ACCENT, font: FONT }, paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: ACCENT, font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 15840 }, margin: { top: 1300, bottom: 1300, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: MUTED })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(HERE, "Stochastic_Stock_Models_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
