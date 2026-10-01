"""Reproducible analysis for the research paper:
Preprocessing strategies and model choice for diabetes prediction (Pima Indians Diabetes data).
Libraries: NumPy, pandas, scikit-learn, XGBoost, Dask, Matplotlib/Seaborn."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import dask
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, brier_score_loss)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)
COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigree", "Age", "Outcome"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = pd.read_csv("pima.csv", header=None, names=COLS)
X, y = df.drop(columns="Outcome"), df["Outcome"]

# ---------- descriptive statistics ----------
desc = df.drop(columns="Outcome").describe().T[["mean", "std", "min", "max"]]
desc["zeros_%"] = (df.drop(columns="Outcome") == 0).mean() * 100
desc.round(2).to_csv("table_descriptives.csv")
print(desc.round(2))
print("prevalence", y.mean().round(4), "n", len(df))

def zeros_to_nan(a):
    a = pd.DataFrame(a, columns=X.columns).copy()
    a[ZERO_AS_MISSING] = a[ZERO_AS_MISSING].replace(0, np.nan)
    return a

def preprocessor(kind):
    to_nan = FunctionTransformer(zeros_to_nan, feature_names_out="one-to-one")
    if kind == "Raw (zeros kept)":
        return [("scale", StandardScaler())]
    if kind == "Median imputation":
        return [("nan", to_nan), ("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if kind == "KNN imputation":
        return [("nan", to_nan), ("scale", StandardScaler()), ("imp", KNNImputer(n_neighbors=5))]
    if kind == "Iterative (MICE-style)":
        return [("nan", to_nan), ("imp", IterativeImputer(max_iter=10, random_state=SEED)), ("scale", StandardScaler())]
    raise ValueError(kind)

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                    class_weight="balanced", random_state=SEED, n_jobs=1),
    "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                                     eval_metric="logloss", random_state=SEED, n_jobs=1),
}
PREPS = ["Raw (zeros kept)", "Median imputation", "KNN imputation", "Iterative (MICE-style)"]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED)
scoring = {"auc": "roc_auc", "acc": "accuracy", "prec": "precision", "rec": "recall", "f1": "f1",
           "brier": "neg_brier_score"}

@dask.delayed
def evaluate(prep, model):
    pipe = Pipeline(preprocessor(prep) + [("clf", MODELS[model]())])
    r = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    out = {"Preprocessing": prep, "Model": model}
    for k in scoring:
        v = r[f"test_{k}"] * (-1 if k == "brier" else 1)
        out[k + "_mean"], out[k + "_sd"] = v.mean(), v.std()
    out["auc_folds"] = r["test_auc"].tolist()
    return out

# Dask schedules the 12 independent preprocessing x model experiments in parallel
tasks = [evaluate(p, m) for p in PREPS for m in MODELS]
results = dask.compute(*tasks, scheduler="threads")
res = pd.DataFrame(results)
folds = res[["Preprocessing", "Model", "auc_folds"]]
res = res.drop(columns="auc_folds")
res.round(3).to_csv("table_cv_results.csv", index=False)
print(res.round(3).to_string())
json.dump(folds.to_dict(orient="records"), open("cv_auc_folds.json", "w"))

# heatmap of mean AUC
piv = res.pivot(index="Preprocessing", columns="Model", values="auc_mean").loc[PREPS]
plt.figure(figsize=(6.5, 3.6))
sns.heatmap(piv, annot=True, fmt=".3f", cmap="Blues", cbar_kws={"label": "Mean ROC-AUC"})
plt.title("Mean ROC-AUC (10-fold CV x 5 repeats)")
plt.ylabel(""); plt.xlabel("")
plt.tight_layout(); plt.savefig("fig_auc_heatmap.png", dpi=200); plt.close()

# ---------- hold-out evaluation for the chosen configuration ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
hold = {}
plt.figure(figsize=(5.2, 4.6))
for m in MODELS:
    pipe = Pipeline(preprocessor("Median imputation") + [("clf", MODELS[m]())]).fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]; yhat = (p >= 0.5).astype(int)
    hold[m] = {"auc": roc_auc_score(yte, p), "acc": accuracy_score(yte, yhat),
               "prec": precision_score(yte, yhat), "rec": recall_score(yte, yhat),
               "f1": f1_score(yte, yhat), "brier": brier_score_loss(yte, p),
               "cm": confusion_matrix(yte, yhat).tolist(), "pipe": pipe, "p": p}
    fpr, tpr, _ = roc_curve(yte, p)
    plt.plot(fpr, tpr, label=f"{m} (AUC = {hold[m]['auc']:.3f})")
plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
plt.xlabel("False positive rate (1 - specificity)"); plt.ylabel("True positive rate (sensitivity)")
plt.title("ROC curves on held-out test set (n = 192)"); plt.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig("fig_roc.png", dpi=200); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, m in zip(axes, MODELS):
    sns.heatmap(np.array(hold[m]["cm"]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    ax.set_title(m)
plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=200); plt.close()

# ---------- explainability: permutation importance (Logistic Regression) ----------
pi = permutation_importance(hold["Logistic Regression"]["pipe"], Xte, yte, scoring="roc_auc",
                            n_repeats=30, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
plt.figure(figsize=(5.5, 3.6))
imp.plot.barh(xerr=pd.Series(pi.importances_std, index=X.columns)[imp.index], color="#3b6ea5")
plt.xlabel("Mean decrease in ROC-AUC when permuted"); plt.title("Permutation importance (Logistic Regression)")
plt.tight_layout(); plt.savefig("fig_importance.png", dpi=200); plt.close()
coefs = pd.Series(hold["Logistic Regression"]["pipe"].named_steps["clf"].coef_[0], index=X.columns)
print("LR odds ratios per SD:\n", np.exp(coefs).round(2).sort_values())
print("perm importance:\n", imp.round(3))

# ---------- fairness audit by age group ----------
grp = np.where(Xte["Age"] < 30, "Age < 30", "Age >= 30")
rows = []
for m in MODELS:
    p = hold[m]["p"]; yhat = (p >= 0.5).astype(int)
    for g in ["Age < 30", "Age >= 30"]:
        k = grp == g
        rows.append({"Model": m, "Group": g, "n": int(k.sum()), "Prevalence": yte[k].mean(),
                     "Selection rate": yhat[k].mean(), "TPR (recall)": recall_score(yte[k], yhat[k]),
                     "FPR": ((yhat[k] == 1) & (yte[k] == 0)).sum() / (yte[k] == 0).sum(),
                     "AUC": roc_auc_score(yte[k], p[k])})
fair = pd.DataFrame(rows)
fair.round(3).to_csv("table_fairness.csv", index=False)
print(fair.round(3).to_string())

summary = {m: {k: v for k, v in d.items() if k not in ("pipe", "p")} for m, d in hold.items()}
json.dump(summary, open("holdout_results.json", "w"), indent=2)
print(json.dumps(summary, indent=1))
