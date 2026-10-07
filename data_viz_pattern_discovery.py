"""
Data Visualization for Pattern Discovery
----------------------------------------
Exploratory visual analysis of two Seaborn sample datasets (tips, iris).

Produces every figure used in the presentation plus the numbers quoted in it:
  1. Boxplots        - outliers in bills and tip percentage (tips)
  2. Violin plots    - distribution shape by day, split by meal time (tips)
  3. Scatter + hist  - bill/tip relationship and tip rounding (tips)
  4. Pairplot        - class separation across all feature pairs (iris)
  5. Heatmap+violin  - feature redundancy and Simpson's paradox (iris)
  6. 2x2 dashboard   - several plots combined on one GridSpec figure (tips)
Then checks the decisions the visuals suggested with quick cross-validated models.

Run:  python data_viz_pattern_discovery.py   (figures are written to figures/)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Palette shared with the slide deck
INK, TEAL, AMBER, CORAL, SLATE = "#1B2A41", "#0E9F8E", "#E89B2C", "#D9534F", "#6B7A90"
SPECIES = {"setosa": TEAL, "versicolor": AMBER, "virginica": "#5B5FC7"}

sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")
plt.rcParams.update({"axes.edgecolor": "#C9CED6", "axes.titleweight": "bold",
                     "axes.titlesize": 18, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": SLATE, "ytick.color": SLATE})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved figures/{name}")


def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return s[(s < lo) | (s > hi)], hi


tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
tips["tip_pct"] = tips["tip"] / tips["total_bill"] * 100
# Plain strings + explicit orders: seaborn 0.13 mis-maps hue colours on pandas>=3 categoricals
DAYS = ["Thur", "Fri", "Sat", "Sun"]
for col in ["sex", "smoker", "day", "time"]:
    tips[col] = tips[col].astype(str)

print("tips:", tips.shape, "| iris:", iris.shape)
print("missing values:", int(tips.isna().sum().sum()), int(iris.isna().sum().sum()))

# ---------------------------------------------------------------- 1. Boxplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(data=tips, x="day", y="total_bill", order=DAYS, color=TEAL, width=0.55,
            flierprops=dict(marker="o", markerfacecolor=CORAL, markeredgecolor=CORAL),
            ax=axes[0])
axes[0].set(title="Total bill by day", xlabel="", ylabel="Total bill ($)")
sns.boxplot(data=tips, x="smoker", y="tip_pct", hue="smoker", order=["Yes", "No"],
            palette={"Yes": AMBER, "No": TEAL}, width=0.5, legend=False,
            flierprops=dict(marker="o", markerfacecolor=CORAL, markeredgecolor=CORAL),
            ax=axes[1])
top = tips.loc[tips["tip_pct"].idxmax()]
axes[1].annotate(f"{top.tip_pct:.0f}% tip\n(\\${top.tip:.2f} on \\${top.total_bill:.2f})",
                 xy=(0, top.tip_pct), xytext=(0.35, 60), color=CORAL, fontsize=14,
                 arrowprops=dict(arrowstyle="->", color=CORAL))
axes[1].set(title="Tip % by smoker status", xlabel="Smoker", ylabel="Tip (% of bill)")
save(fig, "01_boxplots_tips.png")

bill_out, bill_fence = iqr_outliers(tips["total_bill"])
pct_out, _ = iqr_outliers(tips["tip_pct"])
print(f"bill outliers: {len(bill_out)} above ${bill_fence:.2f}; tip% outliers: {len(pct_out)}")
print(tips.groupby("smoker", observed=True)["tip_pct"].agg(["median", "std"]).round(2))

# ------------------------------------------------------------ 2. Violin plots
fig, ax = plt.subplots(figsize=(15, 6.5))
sns.violinplot(data=tips, x="day", y="total_bill", hue="time", split=True,
               order=DAYS, hue_order=["Lunch", "Dinner"], palette=[AMBER, TEAL], inner="quart",
               density_norm="count", cut=0, ax=ax)
ax.set(title="Bill distribution by day, split by meal time", xlabel="",
       ylabel="Total bill ($)")
ax.legend(title="", loc="upper left", frameon=False)
save(fig, "02_violin_tips.png")
print(pd.crosstab(tips["day"], tips["time"]).loc[DAYS])

# ------------------------------------------------ 3. Scatter + histogram grid
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.15, 1]})
sns.regplot(data=tips, x="total_bill", y="tip", ax=axes[0],
            scatter_kws=dict(color=TEAL, alpha=0.6, s=45),
            line_kws=dict(color=INK, lw=2.5))
r = tips["total_bill"].corr(tips["tip"])
slope, intercept = np.polyfit(tips["total_bill"], tips["tip"], 1)
axes[0].set(title=f"Tip vs. bill  (r = {r:.2f})", xlabel="Total bill ($)", ylabel="Tip ($)")
sns.histplot(data=tips, x="tip", binwidth=0.25, color=SLATE, ax=axes[1])
for p in axes[1].patches:  # highlight bars sitting on whole-dollar amounts
    if abs(p.get_x() - round(p.get_x())) < 1e-9:
        p.set_facecolor(CORAL)
axes[1].set(title="Tip amounts (whole dollars highlighted)", xlabel="Tip ($)", ylabel="Count")
save(fig, "03_scatter_hist_tips.png")
print(f"bill-tip r={r:.3f}, slope={slope:.4f}, intercept={intercept:.3f}")
print(f"whole-dollar tips: {(tips['tip'] % 1 == 0).mean():.1%}; "
      f"tip% vs bill r={tips['tip_pct'].corr(tips['total_bill']):.2f}")

# --------------------------------------------------------------- 4. Pairplot
g = sns.pairplot(iris, hue="species", palette=SPECIES, corner=True, height=2.4,
                 plot_kws=dict(s=28, alpha=0.75, edgecolor="none"),
                 diag_kws=dict(fill=True, alpha=0.5))
g.figure.suptitle("Iris pairplot - every feature pair, coloured by species",
                  y=1.01, fontweight="bold", fontsize=18)
sns.move_legend(g, "upper right", bbox_to_anchor=(0.92, 0.92), title="", frameon=False)
save(g.figure, "04_pairplot_iris.png")

# ---------------------------------------------------- 5. Heatmap + violin
fig, axes = plt.subplots(1, 2, figsize=(16, 6.3), gridspec_kw={"width_ratios": [1.05, 1]})
corr = iris.drop(columns="species").corr()
labels = [c.replace("_", "\n") for c in corr.columns]
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True,
            xticklabels=labels, yticklabels=labels, cbar_kws=dict(shrink=0.8),
            annot_kws=dict(size=15), ax=axes[0])
axes[0].set_title("Feature correlation (all flowers)")
sns.violinplot(data=iris, x="species", y="petal_length", hue="species",
               palette=SPECIES, inner="box", legend=False, ax=axes[1])
axes[1].set(title="Petal length by species", xlabel="", ylabel="Petal length (cm)")
save(fig, "05_heatmap_violin_iris.png")
print(corr.round(2))
within = iris.groupby("species", observed=True)[["sepal_length", "sepal_width"]].corr()
within = within.xs("sepal_length", level=1)["sepal_width"]
print("sepal length~width r overall:",
      round(iris["sepal_length"].corr(iris["sepal_width"]), 2),
      "| within species:", within.round(2).to_dict())
print(iris.groupby("species", observed=True)["petal_length"].agg(["min", "max"]))

# ----------------------------------------------- 6. Combined 2x2 dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22)
ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax3, ax4 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
sns.histplot(tips["total_bill"], kde=True, color=TEAL, ax=ax1)
ax1.set(title="A. Bill is right-skewed", xlabel="Total bill ($)")
sns.boxplot(data=tips, x="size", y="tip_pct", color=AMBER, ax=ax2,
            flierprops=dict(markerfacecolor=CORAL, markeredgecolor=CORAL))
ax2.set(title="B. Tip % by party size", xlabel="Party size", ylabel="Tip %")
sns.scatterplot(data=tips, x="total_bill", y="tip_pct", hue="time",
                hue_order=["Lunch", "Dinner"], palette=[AMBER, INK], alpha=0.7, ax=ax3)
ax3.set(title="C. Tip % falls as bill rises", xlabel="Total bill ($)", ylabel="Tip %")
ax3.legend(title="", frameon=False)
sns.countplot(data=tips, x="day", hue="sex", order=DAYS, hue_order=["Male", "Female"],
              palette=[INK, TEAL], ax=ax4)
ax4.set(title="D. Sample size is uneven", xlabel="", ylabel="Tables")
ax4.legend(title="", frameon=False)
fig.suptitle("Tips dataset at a glance - four views on one grid", fontsize=22,
             fontweight="bold", y=0.98)
save(fig, "06_dashboard_tips.png")
print("skew total_bill:", round(tips["total_bill"].skew(), 2),
      "| log skew:", round(np.log(tips["total_bill"]).skew(), 2))
print(tips.groupby("size")["tip_pct"].agg(["count", "median"]).round(1))

# ------------------------------------- 7. Validate the visual hunches
X_all = iris.drop(columns="species")
y = iris["species"]
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
for name, cols in {"all 4 features": list(X_all.columns),
                   "petal only": ["petal_length", "petal_width"],
                   "sepal only": ["sepal_length", "sepal_width"],
                   "petal length only": ["petal_length"]}.items():
    acc = cross_val_score(clf, iris[cols], y, cv=5).mean()
    print(f"iris logistic regression, {name:18s}: 5-fold accuracy = {acc:.3f}")

y_tip = tips["tip"]
X_bill = tips[["total_bill"]]
X_full = pd.get_dummies(tips[["total_bill", "size", "sex", "smoker", "day", "time"]],
                        drop_first=True)
for name, X in {"total_bill only": X_bill, "all features": X_full}.items():
    r2 = cross_val_score(LinearRegression(), X, y_tip, cv=5, scoring="r2").mean()
    print(f"tips linear regression, {name:16s}: 5-fold R^2 = {r2:.3f}")
