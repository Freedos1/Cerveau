"""
bayes_heart.py
--------------
Helper module for the Bayesian heart-disease diagnosis assignment.

Everything that the notebook needs more than once lives here so the notebook
itself can stay focused on the story: load -> clean -> set priors -> fit ->
validate -> refine -> update sequentially.

Sections
    1. Data loading and preprocessing
    2. Prior specifications (weakly informative and literature informed)
    3. Model construction and fitting (PyMC)
    4. Prediction and validation metrics
    5. Prior-data conflict check
    6. Sequential (online) Bayesian updating
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats
from scipy.special import expit, logit
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. Data loading and preprocessing
# ---------------------------------------------------------------------------

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

SITE_FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "Long Beach VA": "processed.va.data",
}

# Continuous features are z-scored, so their coefficients are "per 1 SD".
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Binary indicators (after one-hot encoding). They are centred but NOT scaled,
# so each coefficient stays a log odds ratio for "has it vs does not".
BINARY = [
    "sex", "fbs", "exang", "restecg_abn",
    "cp_atypical", "cp_nonanginal", "cp_asymptomatic",
    "slope_flat", "slope_down",
    "thal_fixed", "thal_reversible",
]

# Number of major vessels coloured by fluoroscopy (0-3), kept as a count.
COUNT = ["ca"]

FEATURES = CONTINUOUS + BINARY + COUNT


def load_site(path: str | Path) -> pd.DataFrame:
    """Read one processed UCI file; '?' marks a missing value."""
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)  # 0 = <50% narrowing, 1-4 = disease
    return df


def load_all_sites(data_dir: str | Path) -> pd.DataFrame:
    """Stack the four collection sites with a 'site' column."""
    frames = []
    for site, fname in SITE_FILES.items():
        d = load_site(Path(data_dir) / fname)
        d["site"] = site
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical codes used in the UCI files.

    Reference levels: typical angina (cp=1), normal resting ECG (restecg=0),
    upsloping ST segment (slope=1) and normal thallium scan (thal=3).
    """
    out = pd.DataFrame(index=df.index)
    for c in ["age", "trestbps", "chol", "thalach", "oldpeak", "sex", "fbs", "exang", "ca"]:
        out[c] = df[c].astype(float)
    # restecg=1 (ST-T abnormality) has only 4 Cleveland cases, so it is pooled
    # with restecg=2 (left ventricular hypertrophy) into one "abnormal" flag.
    out["restecg_abn"] = (df["restecg"] > 0).astype(float)
    out["cp_atypical"] = (df["cp"] == 2).astype(float)
    out["cp_nonanginal"] = (df["cp"] == 3).astype(float)
    out["cp_asymptomatic"] = (df["cp"] == 4).astype(float)
    out["slope_flat"] = (df["slope"] == 2).astype(float)
    out["slope_down"] = (df["slope"] == 3).astype(float)
    out["thal_fixed"] = (df["thal"] == 6).astype(float)
    out["thal_reversible"] = (df["thal"] == 7).astype(float)
    return out[FEATURES]


@dataclass
class Prepared:
    """Container for the train/test design matrices and the fitted transforms."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame
    features: list[str]
    center: pd.Series
    scale: pd.Series
    impute_values: dict = field(default_factory=dict)

    def transform(self, raw: pd.DataFrame) -> np.ndarray:
        """Apply the *training* imputation and scaling to new patients."""
        raw = raw.copy()
        for col, val in self.impute_values.items():
            raw[col] = raw[col].fillna(val)
        enc = encode(raw)
        return ((enc - self.center) / self.scale).to_numpy()


def prepare(raw_train: pd.DataFrame, raw_test: pd.DataFrame) -> Prepared:
    """Impute, encode and scale using statistics from the training rows only.

    Fitting the imputation and scaling on the training rows alone keeps any
    information about the test patients out of the model (no leakage).
    """
    raw_train, raw_test = raw_train.copy(), raw_test.copy()

    # Only 'ca' (4 rows) and 'thal' (2 rows) are missing in Cleveland.
    # Median/mode imputation is reasonable for such a small gap.
    impute_values = {}
    for col in ["trestbps", "chol", "thalach", "oldpeak", "age"]:
        impute_values[col] = raw_train[col].median()
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        impute_values[col] = raw_train[col].mode().iloc[0]
    for col, val in impute_values.items():
        raw_train[col] = raw_train[col].fillna(val)
        raw_test[col] = raw_test[col].fillna(val)

    enc_train = encode(raw_train)
    center = enc_train.mean()
    scale = pd.Series(1.0, index=FEATURES)
    scale[CONTINUOUS] = enc_train[CONTINUOUS].std()

    X_train = ((enc_train - center) / scale).to_numpy()
    X_test = ((encode(raw_test) - center) / scale).to_numpy()
    return Prepared(
        X_train, X_test,
        raw_train["target"].to_numpy(), raw_test["target"].to_numpy(),
        raw_train, raw_test, FEATURES, center, scale, impute_values,
    )


def preprocess(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Prepared:
    """Stratified train/test split followed by `prepare`."""
    raw_train, raw_test = train_test_split(
        df, test_size=test_size, stratify=df["target"], random_state=seed
    )
    return prepare(raw_train, raw_test)


# ---------------------------------------------------------------------------
# 2. Prior specifications
# ---------------------------------------------------------------------------
# Each prior is Normal(mu, sd) on the log-odds scale.
# Continuous features: effect per 1 SD.  Binary: log odds ratio.  ca: per vessel.

@dataclass
class PriorSpec:
    name: str
    intercept: tuple[float, float]           # (mu, sd) for the intercept
    coefs: dict[str, tuple[float, float]]    # feature -> (mu, sd)
    note: str = ""

    def arrays(self, features=FEATURES):
        mu = np.array([self.coefs[f][0] for f in features])
        sd = np.array([self.coefs[f][1] for f in features])
        return mu, sd


# Patel et al. (2010): 37.6% of ~399,000 US patients referred for elective
# angiography had obstructive CAD. Because predictors are centred, the
# intercept is (roughly) the log-odds for an "average" referred patient.
PRIOR_PREVALENCE = 0.376
INTERCEPT_LIT = (float(logit(PRIOR_PREVALENCE)), 0.5)

WEAK = PriorSpec(
    name="M1 weakly informative",
    intercept=(0.0, 2.5),
    coefs={f: (0.0, 2.5) for f in FEATURES},
    note="Normal(0, 2.5) everywhere (Gelman et al., 2008 style default).",
)

LITERATURE = PriorSpec(
    name="M2 literature informed",
    intercept=INTERCEPT_LIT,
    coefs={
        # Demographics: Diamond & Forrester (1979); Genders et al. (2011)
        "age":             (0.40, 0.30),   # ~1.5x odds per SD (~9 years)
        "sex":             (0.92, 0.50),   # men ~2.5x odds
        # Chest pain type vs typical angina: Diamond & Forrester (1979)
        "cp_atypical":     (-1.00, 0.75),
        "cp_nonanginal":   (-1.50, 0.75),
        "cp_asymptomatic": (0.00, 1.50),   # no clean literature analogue
        # Classic risk factors: Wilson et al. (1998), small diagnostic effects
        "trestbps":        (0.20, 0.30),
        "chol":            (0.20, 0.30),
        "fbs":             (0.30, 0.50),
        "restecg_abn":     (0.20, 0.50),
        # Exercise ECG: Gianrossi et al. (1989); Gibbons et al. (2002)
        "thalach":         (-0.30, 0.50),
        "exang":           (0.80, 0.50),
        "oldpeak":         (0.60, 0.40),
        "slope_flat":      (0.50, 0.50),
        "slope_down":      (0.70, 0.60),
        # Imaging: perfusion LR+ ~ 3 (Knuuti et al., 2018); calcification
        "thal_fixed":      (0.70, 0.60),
        "thal_reversible": (1.00, 0.50),
        "ca":              (0.80, 0.40),
    },
    note="Means from published odds/likelihood ratios; SDs widened for transportability.",
)


def refine_spec(base: PriorSpec, conflicts: list[str], widen_to: float = 1.5,
                name: str = "M3 refined") -> PriorSpec:
    """Copy a prior spec, replacing conflicting priors with a centred, wider one."""
    coefs = dict(base.coefs)
    for f in conflicts:
        coefs[f] = (0.0, widen_to)
    return PriorSpec(name=name, intercept=base.intercept, coefs=coefs,
                     note=f"Widened to N(0, {widen_to}): {', '.join(conflicts)}")


# ---------------------------------------------------------------------------
# 3. Model construction and fitting
# ---------------------------------------------------------------------------

def build_model(X, y, spec: PriorSpec | None = None, robust: bool = False,
                eps_prior=(1.0, 19.0), mv_prior=None) -> pm.Model:
    """Bayesian logistic regression.

    Parameters
    ----------
    spec : independent Normal priors (used unless mv_prior is given)
    robust : if True, use a 'mislabel-robust' likelihood
             P(y=1) = eps * 0.5 + (1 - eps) * sigmoid(eta),
             where eps ~ Beta(eps_prior) is the share of essentially random
             labels (data entry errors, borderline angiograms, etc.).
    mv_prior : optional (mean, cov) for [intercept, coefs...]; used for
               sequential updating where yesterday's posterior is today's prior.
    """
    k = X.shape[1]
    coords = {"feature": FEATURES, "param": ["intercept"] + FEATURES}
    with pm.Model(coords=coords) as model:
        Xd = pm.Data("X", X)
        if mv_prior is None:
            mu, sd = spec.arrays()
            alpha = pm.Normal("alpha", mu=spec.intercept[0], sigma=spec.intercept[1])
            beta = pm.Normal("beta", mu=mu, sigma=sd, dims="feature")
        else:
            m, c = mv_prior
            theta = pm.MvNormal("theta", mu=m, cov=c, dims="param")
            alpha = pm.Deterministic("alpha", theta[0])
            beta = pm.Deterministic("beta", theta[1:], dims="feature")
        eta = alpha + pm.math.dot(Xd, beta)
        if robust:
            eps = pm.Beta("eps", *eps_prior)
            p = pm.Deterministic("p", eps * 0.5 + (1 - eps) * pm.math.sigmoid(eta))
        else:
            p = pm.Deterministic("p", pm.math.sigmoid(eta))
        pm.Bernoulli("y_obs", p=p, observed=y, shape=Xd.shape[0])
    return model


def fit(model: pm.Model, draws=2000, tune=1500, chains=4, seed=42,
        target_accept=0.9) -> az.InferenceData:
    """NUTS sampling, keeping the pointwise log-likelihood (needed for LOO)."""
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains,
                          random_seed=seed, target_accept=target_accept,
                          idata_kwargs={"log_likelihood": True}, progressbar=False)
    return idata


# ---------------------------------------------------------------------------
# 4. Prediction and validation metrics
# ---------------------------------------------------------------------------

def posterior_draws(idata, var):
    """Flatten (chain, draw, ...) into (samples, ...)."""
    a = idata.posterior[var].values
    return a.reshape(-1, *a.shape[2:])


def predict_draws(idata, X, robust=False):
    """Posterior probability of disease for each draw and patient: (S, N)."""
    alpha = posterior_draws(idata, "alpha")
    beta = posterior_draws(idata, "beta")
    p = expit(alpha[:, None] + beta @ X.T)
    if robust:
        eps = posterior_draws(idata, "eps")
        p = eps[:, None] * 0.5 + (1 - eps[:, None]) * p
    return p


def evaluate(y, p_draws, threshold=0.5, n_boot=2000, seed=0):
    """Diagnostic metrics from posterior predictive probabilities.

    p_mean is the posterior predictive probability (average over draws).
    lppd = sum_i log( mean_s p(y_i | theta_s) ), the Bayesian test log-likelihood.
    """
    p_draws = np.atleast_2d(p_draws)
    p_mean = p_draws.mean(0)
    yhat = (p_mean >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    lik = np.where(y == 1, p_draws, 1 - p_draws)
    lppd = np.log(lik.mean(0)).sum()

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], p_mean[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "auc": roc_auc_score(y, p_mean),
        "auc_lo": lo, "auc_hi": hi,
        "brier": brier_score_loss(y, p_mean),
        "log_loss": log_loss(y, np.clip(p_mean, 1e-9, 1 - 1e-9)),
        "lppd": lppd,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def beta_binomial_update(a0, b0, successes, n):
    """Conjugate update of a prevalence prior Beta(a0, b0)."""
    return a0 + successes, b0 + n - successes


# ---------------------------------------------------------------------------
# 5. Prior-data conflict check
# ---------------------------------------------------------------------------

def prior_data_conflict(idata_weak, spec: PriorSpec, features=FEATURES):
    """Compare what the data say (posterior under a vague prior) with a prior.

    z = (data_mean - prior_mean) / sqrt(data_sd^2 + prior_sd^2).
    |z| > 2 means the literature prior and the data disagree more than their
    combined uncertainty allows, in the spirit of Evans and Moshonov (2006).
    """
    b = posterior_draws(idata_weak, "beta")
    rows = []
    for j, f in enumerate(features):
        m, s = b[:, j].mean(), b[:, j].std()
        pm_, ps = spec.coefs[f]
        z = (m - pm_) / np.sqrt(s ** 2 + ps ** 2)
        rows.append({"feature": f, "data_mean": m, "data_sd": s,
                     "prior_mean": pm_, "prior_sd": ps, "z": z,
                     "p_two_sided": 2 * stats.norm.sf(abs(z))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Sequential Bayesian updating
# ---------------------------------------------------------------------------

def gaussian_summary(idata):
    """Moment-match the joint posterior of [alpha, beta] to a multivariate Normal."""
    theta = np.column_stack([posterior_draws(idata, "alpha"),
                             posterior_draws(idata, "beta")])
    mean = theta.mean(0)
    cov = np.cov(theta, rowvar=False) + 1e-6 * np.eye(theta.shape[1])
    return mean, cov


def sequential_update(X, y, spec, X_eval, y_eval, n_batches=5, seed=42,
                      draws=1500, tune=1000, chains=4):
    """Feed training patients in batches; each posterior becomes the next prior.

    Returns a list of dicts with the evidence seen so far, the Gaussian
    posterior summary and validation metrics after each update.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    batches = np.array_split(order, n_batches)

    mu0, sd0 = spec.arrays()
    mean = np.r_[spec.intercept[0], mu0]
    cov = np.diag(np.r_[spec.intercept[1], sd0] ** 2)

    # Stage 0: prior only (no patients yet)
    prior_draws = rng.multivariate_normal(mean, cov, size=4000)
    p0 = expit(prior_draws[:, :1] + prior_draws[:, 1:] @ X_eval.T)
    history = [{"n_seen": 0, "mean": mean, "cov": cov,
                "metrics": evaluate(y_eval, p0, n_boot=200), "p_eval": p0}]
    seen = 0
    for i, idx in enumerate(batches, start=1):
        model = build_model(X[idx], y[idx], mv_prior=(mean, cov))
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains,
                              random_seed=seed + i, target_accept=0.9,
                              progressbar=False)
        mean, cov = gaussian_summary(idata)
        seen += len(idx)
        p = predict_draws(idata, X_eval)
        history.append({"n_seen": seen, "mean": mean, "cov": cov,
                        "metrics": evaluate(y_eval, p, n_boot=200), "p_eval": p})
    return history
