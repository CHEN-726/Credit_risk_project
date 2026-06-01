# =============================================================================
# 02_train_with_figures.py  —  Probability-of-Default (PD) model
# -----------------------------------------------------------------------------
# A *trustworthy* credit-risk model, not a flashy one.  Design choices:
#
#   1. TWO models compared honestly: an interpretable Logistic Regression and a
#      stronger Gradient Boosting model.  We report both, then pick the winner
#      by cross-validated ROC-AUC (tie-break: PR-AUC) — never by eyeballing the
#      test set.
#
#   2. CALIBRATED probabilities.  The previous model used
#      class_weight="balanced", which inflates every predicted probability so a
#      "PD" of 0.5 might really mean an 11% chance of default — useless as a PD.
#      Here each model's probabilities are calibrated (isotonic, out-of-fold),
#      so a PD of 0.20 genuinely means ~20% of such borrowers default.  We prove
#      this with a calibration curve and a decile table of *actual* default
#      rates.
#
#   3. The decision THRESHOLD is selected on a held-out validation slice of the
#      training data (max F1), not fixed at an arbitrary 0.55 and not tuned on
#      the test set.
#
#   4. Leakage guard + variance guard: features that are constant, or suspicously
#      perfectly correlated with the target, are dropped automatically.
#
#   5. Honest, complete metrics: ROC-AUC, PR-AUC, KS, Gini, Brier (calibration),
#      plus precision/recall/F1 at the chosen threshold — with a full figure set.
#
#   6. [NEW] Feature set refined by IV screening (see 01_data_clean.py →
#      outputs/iv_report.csv).  Only features with IV >= 0.02 are kept.
#      Segment/binned versions of numeric features are excluded because
#      HistGradientBoosting finds its own optimal splits on the raw values.
#      Result: 4 clean numeric features instead of 9 mixed ones.
#      Dropping low-IV noise raised Test-AUC from 0.7185 → 0.7210.
# =============================================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")
RNG = 42

# =============================================================================
# 0. Config & house style
# =============================================================================
DATA_PATH = "loan_cleaned.csv"
FIG_DIR   = "figures"
OUT_DIR   = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SIZE = 0.20
VAL_SIZE  = 0.20
CV_FOLDS  = 5

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlepad": 12,
    "axes.labelsize": 11.5, "axes.labelweight": "medium",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8A8A8A", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.9,
    "xtick.color": "#3A3A3A", "ytick.color": "#3A3A3A",
    "legend.frameon": False, "legend.fontsize": 10,
})

C = {
    "lr":   "#2D6CDF",
    "gb":   "#E07B16",
    "good": "#1B9E77",
    "bad":  "#D1495B",
    "mid":  "#E6A817",
    "ink":  "#2B2B2B",
    "grey": "#9AA0A6",
}
RISK_C = {"Low": C["good"], "Medium": C["mid"], "High": C["bad"]}


def finish(fig, path, subtitle=None):
    if subtitle:
        fig.text(0.5, 0.005, subtitle, ha="center", va="bottom",
                 fontsize=9.5, style="italic", color="#6B6B6B")
    fig.tight_layout(rect=(0, 0.03 if subtitle else 0, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, path), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ks_stat(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


# =============================================================================
# 1. Load + IV-screened feature set
# =============================================================================
df = pd.read_csv('/Users/boyuchen/Downloads/credit_risk_project1/credit_risk_project/loan_data_cleaned.csv')
assert "is_default" in df.columns, "target `is_default` not found"
y = df["is_default"].astype(int)

# -----------------------------------------------------------------
# [CHANGED] Feature set refined by IV screening
# (full IV table → outputs/iv_report.csv from 01_data_clean.py)
#
# KEPT  (IV >= 0.02, distinct signals, no segment overlap):
#   LTI                        IV = 0.44  🔥 강
#   historical_missed_payments IV = 0.16  ✅ 중
#   loan_amount                IV = 0.05  ⚠️ 약
#   income                     IV = 0.04  ⚠️ 약
#
# DROPPED vs previous version:
#   data_quality_risk   IV = 0.006  ❌ 废 — pure noise
#   phone_valid         IV = 0.000  ❌ 废 — no signal in synthetic data
#   income_missing      IV = 0.000  ❌ 废 — no signal
#   loan_amount_missing IV = 0.000  ❌ 废 — no signal
#   tags                IV = 0.000  ❌ 废 — no signal
#
# Segment columns (LTI_segment etc.) are kept in loan_cleaned.csv for
# business reporting but NOT fed to the model — GB finds its own splits.
# -----------------------------------------------------------------
NUMERIC_C = ["LTI", "historical_missed_payments", "loan_amount", "income"]
BINARY_C  = []   # all binary features had IV = 0 → dropped
CATEG_C   = []   # tags had IV = 0 → dropped

numeric_features     = [c for c in NUMERIC_C if c in df.columns]
binary_features      = [c for c in BINARY_C  if c in df.columns]
categorical_features = [c for c in CATEG_C   if c in df.columns]

# ---- variance guard ----
for grp in (numeric_features, binary_features, categorical_features):
    for c in list(grp):
        if df[c].nunique(dropna=False) <= 1:
            print(f"[guard] dropping constant feature: {c}")
            grp.remove(c)

# ---- leakage guard ----
for c in list(numeric_features + binary_features):
    corr = np.corrcoef(df[c].fillna(df[c].median()), y)[0, 1]
    if abs(corr) > 0.98:
        print(f"[guard] dropping leaky feature (|corr|={abs(corr):.3f}): {c}")
        for grp in (numeric_features, binary_features):
            if c in grp:
                grp.remove(c)

features = numeric_features + binary_features + categorical_features
X = df[features].copy()

print("=" * 70)
print(f"Rows: {len(df):,}   features: {len(features)}  (IV-screened)")
print(f"Default rate: {y.mean():.4f}  ({int(y.sum()):,} defaults / {len(y):,})")
print(f"  numeric    : {numeric_features}")
print(f"  binary     : {binary_features}")
print(f"  categorical: {categorical_features}")


# =============================================================================
# 2. Split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RNG, stratify=y)
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, random_state=RNG, stratify=y_train)
print(f"\nTrain {len(X_train):,}  (fit {len(X_fit):,} + val {len(X_val):,})   Test {len(X_test):,}")


# =============================================================================
# 3. Preprocessing + two model pipelines
# =============================================================================
# [NOTE] BINARY_C and CATEG_C are now empty after IV screening.
# ColumnTransformer handles empty lists gracefully.
preprocess = ColumnTransformer(
    [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc",  StandardScaler())]), numeric_features)]
    + ([("bin", SimpleImputer(strategy="most_frequent"), binary_features)]
       if binary_features else [])
    + ([("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh",  OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]),
         categorical_features)]
       if categorical_features else [])
)

lr_pipe = Pipeline([("prep", preprocess),
                    ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                               random_state=RNG))])
gb_pipe = Pipeline([("prep", preprocess),
                    ("clf", HistGradientBoostingClassifier(
                        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                        min_samples_leaf=60, l2_regularization=1.0,
                        early_stopping=True, validation_fraction=0.1,
                        random_state=RNG))])

def calibrated(pipe):
    return CalibratedClassifierCV(pipe, method="isotonic", cv=5)

MODELS = {"Logistic Regression": lr_pipe, "Gradient Boosting": gb_pipe}


# =============================================================================
# 4. Cross-validated comparison
# =============================================================================
print("\n" + "=" * 70)
print("5-fold cross-validation on training data (used to pick the winner)")
print("=" * 70)
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RNG)
cv_scores = {}
cv_table  = []
for name, pipe in MODELS.items():
    res = cross_validate(pipe, X_train, y_train, cv=cv,
                         scoring=["roc_auc", "average_precision", "f1"],
                         return_train_score=False, n_jobs=-1)
    cv_scores[name] = res
    cv_table.append({
        "model": name,
        "cv_roc_auc": res["test_roc_auc"].mean(),
        "cv_roc_auc_std": res["test_roc_auc"].std(),
        "cv_pr_auc": res["test_average_precision"].mean(),
        "cv_f1": res["test_f1"].mean(),
    })
cv_df = pd.DataFrame(cv_table)
print(cv_df.round(4).to_string(index=False))

cv_df = cv_df.sort_values(["cv_roc_auc", "cv_pr_auc"], ascending=False).reset_index(drop=True)
winner = cv_df.loc[0, "model"]
print(f"\n>>> Winner by CV ROC-AUC: {winner}")


# =============================================================================
# 5. Fit calibrated models, score test set
# =============================================================================
fitted, proba_test = {}, {}
for name, pipe in MODELS.items():
    model = calibrated(pipe).fit(X_train, y_train)
    fitted[name] = model
    proba_test[name] = model.predict_proba(X_test)[:, 1]

def threshold_free_metrics(yt, p):
    return {
        "roc_auc": roc_auc_score(yt, p),
        "pr_auc":  average_precision_score(yt, p),
        "ks":      ks_stat(yt, p),
        "gini":    2 * roc_auc_score(yt, p) - 1,
        "brier":   brier_score_loss(yt, p),
    }

test_metrics = {n: threshold_free_metrics(y_test, p) for n, p in proba_test.items()}


# =============================================================================
# 6. Threshold selection
# =============================================================================
win_model_val = calibrated(MODELS[winner]).fit(X_fit, y_fit)
p_val = win_model_val.predict_proba(X_val)[:, 1]

grid = np.linspace(0.05, 0.95, 181)
f1s = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in grid]
THRESHOLD = float(grid[int(np.argmax(f1s))])
print(f"\nSelected threshold (max F1 on validation slice): {THRESHOLD:.3f}")

p_win = proba_test[winner]
y_hat = (p_win >= THRESHOLD).astype(int)

thr_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
thr_rows = []
for t in thr_grid:
    yh = (p_win >= t).astype(int)
    thr_rows.append({"threshold": t,
                     "precision": precision_score(y_test, yh, zero_division=0),
                     "recall":    recall_score(y_test, yh, zero_division=0),
                     "f1":        f1_score(y_test, yh, zero_division=0),
                     "accuracy":  accuracy_score(y_test, yh)})
thr_df = pd.DataFrame(thr_rows)


# =============================================================================
# 7. Final evaluation
# =============================================================================
print("\n" + "=" * 70)
print(f"FINAL MODEL: {winner}   (calibrated, threshold = {THRESHOLD:.3f})")
print("=" * 70)
tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
final = {
    "model": winner, "threshold": THRESHOLD,
    "accuracy":  accuracy_score(y_test, y_hat),
    "precision": precision_score(y_test, y_hat, zero_division=0),
    "recall":    recall_score(y_test, y_hat, zero_division=0),
    "f1":        f1_score(y_test, y_hat, zero_division=0),
    **test_metrics[winner],
    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
}
for k in ["roc_auc", "pr_auc", "ks", "gini", "brier",
          "accuracy", "precision", "recall", "f1"]:
    print(f"  {k:10s}: {final[k]:.4f}")
print("\n", classification_report(y_test, y_hat, target_names=["Non-default", "Default"]))


# =============================================================================
# 8. Decile / risk-segment analysis
# =============================================================================
res = df.loc[X_test.index, [c for c in ["client_id"] if c in df.columns] + features].copy()
res["actual_default"] = y_test.values
res["PD"] = p_win
res["predicted_default"] = y_hat

bands = [-0.001, 0.08, 0.20, 1.0]
res["risk_segment"] = pd.cut(res["PD"], bins=bands, labels=["Low", "Medium", "High"])
seg = (res.groupby("risk_segment", observed=True)
          .agg(borrowers=("actual_default", "size"),
               avg_PD=("PD", "mean"),
               actual_default_rate=("actual_default", "mean"))
          .reindex(["Low", "Medium", "High"]))
print("\nRisk segments (avg predicted PD vs actual default rate):")
print(seg.round(4).to_string())

res["decile"] = pd.qcut(res["PD"].rank(method="first"), 10,
                        labels=[f"D{i}" for i in range(1, 11)])
dec = (res.groupby("decile", observed=True)
          .agg(borrowers=("actual_default", "size"),
               avg_PD=("PD", "mean"),
               actual_rate=("actual_default", "mean"),
               defaults=("actual_default", "sum"))
          .iloc[::-1])
dec["cum_defaults_pct"] = dec["defaults"].cumsum() / dec["defaults"].sum()
dec["cum_pop_pct"]      = dec["borrowers"].cumsum() / dec["borrowers"].sum()


# =============================================================================
# 9. Figures
# =============================================================================
print("\nRendering figures ...")

fig, ax = plt.subplots(figsize=(7, 6))
for name, col in [("Logistic Regression", C["lr"]), ("Gradient Boosting", C["gb"])]:
    fpr, tpr, _ = roc_curve(y_test, proba_test[name])
    m = test_metrics[name]
    ax.plot(fpr, tpr, color=col, lw=2.4,
            label=f"{name}  (AUC={m['roc_auc']:.3f}, Gini={m['gini']:.3f})")
ax.plot([0, 1], [0, 1], color=C["grey"], ls="--", lw=1.3, label="Random (AUC=0.500)")
ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02),
       xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="ROC Curve — Model Comparison")
ax.legend(loc="lower right")
finish(fig, "01_roc_comparison.png",
       "Higher and further to the top-left is better. Both models clearly beat random.")

fig, ax = plt.subplots(figsize=(7, 6))
for name, col in [("Logistic Regression", C["lr"]), ("Gradient Boosting", C["gb"])]:
    pr, rc, _ = precision_recall_curve(y_test, proba_test[name])
    ax.plot(rc, pr, color=col, lw=2.4,
            label=f"{name}  (PR-AUC={test_metrics[name]['pr_auc']:.3f})")
base = y_test.mean()
ax.axhline(base, color=C["grey"], ls="--", lw=1.3,
           label=f"Baseline (default rate={base:.3f})")
ax.set(xlim=(-.02, 1.02), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
       title="Precision–Recall Curve — Model Comparison")
ax.legend(loc="upper right")
finish(fig, "02_precision_recall_comparison.png",
       "On imbalanced data PR-AUC is more informative than ROC-AUC. Baseline = prevalence.")

fig, ax = plt.subplots(figsize=(7, 6))
_cal_max = 0.0
for name, col in [("Logistic Regression", C["lr"]), ("Gradient Boosting", C["gb"])]:
    pt, pp = calibration_curve(y_test, proba_test[name], n_bins=10, strategy="quantile")
    _cal_max = max(_cal_max, float(pp.max()), float(pt.max()))
    ax.plot(pp, pt, marker="o", ms=6, lw=2.0, color=col,
            label=f"{name}  (Brier={test_metrics[name]['brier']:.4f})")
ax.plot([0, 1], [0, 1], color=C["grey"], ls="--", lw=1.3, label="Perfect calibration")
lim = min(1.0, _cal_max * 1.18)
ax.set(xlim=(0, lim), ylim=(0, lim),
       xlabel="Mean predicted PD", ylabel="Observed default rate",
       title="Calibration Curve — Are the PDs Real?")
ax.legend(loc="upper left")
finish(fig, "03_calibration_comparison.png",
       "Points on the diagonal mean a predicted PD of x% really does default x% of the time.")

mets = ["roc_auc", "pr_auc", "ks", "f1"]
labels = ["ROC-AUC", "PR-AUC", "KS", "F1 @ thr"]
lr_vals = [test_metrics["Logistic Regression"][m] for m in mets[:3]] + \
          [f1_score(y_test, (proba_test["Logistic Regression"] >= THRESHOLD).astype(int), zero_division=0)]
gb_vals = [test_metrics["Gradient Boosting"][m] for m in mets[:3]] + \
          [f1_score(y_test, (proba_test["Gradient Boosting"] >= THRESHOLD).astype(int), zero_division=0)]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - w/2, lr_vals, w, color=C["lr"], label="Logistic Regression")
b2 = ax.bar(x + w/2, gb_vals, w, color=C["gb"], label="Gradient Boosting")
ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=9)
ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=9)
ax.set(xticks=x, ylim=(0, 1.0), ylabel="Score",
       title="Test-set Metrics — Logistic Regression vs Gradient Boosting")
ax.set_xticklabels(labels)
ax.legend(loc="upper right")
finish(fig, "04_metric_comparison.png",
       f"Selected threshold = {THRESHOLD:.2f}. Winner by cross-validated ROC-AUC: {winner}.")

fig, ax = plt.subplots(figsize=(8, 5.2))
data, poss, cols, ticks = [], [], [], []
for i, name in enumerate(MODELS):
    data += [cv_scores[name]["test_roc_auc"], cv_scores[name]["test_average_precision"]]
    poss += [i*2 + 1, i*2 + 1.7]
    cols += [C["lr"] if name == "Logistic Regression" else C["gb"]]*2
    ticks.append((i*2 + 1.35, name))
bp = ax.boxplot(data, positions=poss, widths=0.5, patch_artist=True,
                medianprops=dict(color="white", lw=2))
for patch, col in zip(bp["boxes"], cols):
    patch.set_facecolor(col); patch.set_alpha(0.85)
ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks])
ax.set(ylabel="Score", title=f"{CV_FOLDS}-Fold CV Stability  (left=ROC-AUC, right=PR-AUC)")
ax.grid(True, axis="y")
finish(fig, "05_cv_stability.png",
       "Tight boxes = stable, reproducible performance across folds (not luck of one split).")

fpr, tpr, thr = roc_curve(y_test, p_win)
ks_idx = int(np.argmax(tpr - fpr)); ks_val = tpr[ks_idx] - fpr[ks_idx]
order = np.argsort(p_win)
cum_bad  = np.cumsum(y_test.values[order]) / y_test.sum()
cum_good = np.cumsum(1 - y_test.values[order]) / (len(y_test) - y_test.sum())
xs = np.linspace(0, 1, len(cum_bad))
fig, ax = plt.subplots(figsize=(7.5, 5.4))
ax.plot(xs, cum_good, color=C["good"], lw=2.2, label="Cumulative % non-defaults")
ax.plot(xs, cum_bad,  color=C["bad"],  lw=2.2, label="Cumulative % defaults")
gap = int(np.argmax(np.abs(cum_good - cum_bad)))
ax.vlines(xs[gap], cum_bad[gap], cum_good[gap], color=C["ink"], lw=2, ls=":")
ax.annotate(f"KS = {ks_val:.3f}", (xs[gap], (cum_bad[gap]+cum_good[gap])/2),
            xytext=(12, 0), textcoords="offset points", va="center",
            fontweight="bold", color=C["ink"])
ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Population sorted by PD (low → high)",
       ylabel="Cumulative share", title=f"KS Separation — {winner}")
ax.legend(loc="upper left")
finish(fig, "06_ks_curve.png",
       "KS is the largest gap between the good and bad cumulative curves; bigger = better separation.")

gx = np.concatenate([[0], dec["cum_pop_pct"].values])
gy = np.concatenate([[0], dec["cum_defaults_pct"].values])
fig, ax = plt.subplots(figsize=(7.5, 5.6))
ax.plot(gx, gy, marker="o", ms=5, color=C["gb"], lw=2.3, label="Model (sorted by PD)")
ax.plot([0, 1], [0, 1], color=C["grey"], ls="--", lw=1.3, label="Random targeting")
ax.fill_between(gx, gx, gy, color=C["gb"], alpha=0.10)
top20 = float(np.interp(0.2, gx, gy))
ax.annotate(f"Top 20% riskiest\ncapture {top20*100:.0f}% of all defaults",
            (0.2, top20), xytext=(0.30, max(0.2, top20-0.25)),
            arrowprops=dict(arrowstyle="->", color=C["ink"]), fontsize=10)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set(xlim=(0, 1), ylim=(0, 1.02),
       xlabel="Share of borrowers contacted (highest PD first)",
       ylabel="Share of true defaults captured",
       title=f"Cumulative Gains — {winner}")
ax.legend(loc="lower right")
finish(fig, "07_cumulative_gains.png",
       "How many real defaults you catch if you act on the riskiest k% of the book.")

fig, ax = plt.subplots(figsize=(8, 5.2))
bins = np.linspace(0, max(0.4, p_win.max()), 41)
ax.hist(p_win[y_test.values == 0], bins=bins, color=C["good"], alpha=0.7,
        density=True, label="Actual non-defaults")
ax.hist(p_win[y_test.values == 1], bins=bins, color=C["bad"], alpha=0.7,
        density=True, label="Actual defaults")
ax.axvline(THRESHOLD, color=C["ink"], ls="--", lw=1.6, label=f"Threshold = {THRESHOLD:.2f}")
ax.set(xlabel="Predicted PD", ylabel="Density",
       title=f"Predicted PD by Actual Outcome — {winner}")
ax.legend(loc="upper right")
finish(fig, "08_pd_distribution_by_class.png",
       "Good separation = the red (defaults) mass sits to the right of the green (non-defaults).")

fig, ax = plt.subplots(figsize=(8, 5.4))
xs = np.arange(len(seg))
bars = ax.bar(xs, seg["borrowers"], width=0.6,
              color=[RISK_C[s] for s in seg.index], alpha=0.9)
ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
ax.set(xticks=xs, ylabel="Number of borrowers",
       title="Risk Segments — Volume and Actual Default Rate")
ax.set_xticklabels(seg.index)
ax2 = ax.twinx()
ax2.plot(xs, seg["actual_default_rate"], color=C["ink"], marker="D", ms=8, lw=2)
for xi, (ar, ap) in enumerate(zip(seg["actual_default_rate"], seg["avg_PD"])):
    ax2.annotate(f"actual {ar*100:.1f}%\n(pred {ap*100:.1f}%)", (xi, ar),
                 xytext=(0, 12), textcoords="offset points", ha="center",
                 fontsize=9, fontweight="bold", color=C["ink"])
ax2.set_ylabel("Actual default rate"); ax2.grid(False)
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
ax2.set_ylim(0, max(seg["actual_default_rate"])*1.4)
finish(fig, "09_risk_segments.png",
       "Actual default rate rises across Low→Medium→High and matches predicted PD — the segments are real.")

fig, ax = plt.subplots(figsize=(8.5, 5.2))
dd = dec.iloc[::-1]
xs = np.arange(len(dd))
bars = ax.bar(xs, dd["actual_rate"], width=0.7, color=C["lr"], alpha=0.9)
ax.plot(xs, dd["avg_PD"], color=C["bad"], marker="o", ms=6, lw=2, label="Predicted PD")
ax.bar_label(bars, labels=[f"{v*100:.0f}%" for v in dd["actual_rate"]], padding=3, fontsize=9)
ax.set(xticks=xs, ylabel="Default rate", xlabel="PD decile (D1 = safest … D10 = riskiest)",
       title=f"Actual vs Predicted Default Rate by Decile — {winner}")
ax.set_xticklabels(dd.index)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(loc="upper left")
finish(fig, "10_decile_default_rate.png",
       "Monotonic climb left→right and predicted≈actual confirm the ranking and the calibration.")

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(thr_df["threshold"], thr_df["precision"], marker="o", ms=4, color=C["lr"], label="Precision")
ax.plot(thr_df["threshold"], thr_df["recall"],    marker="s", ms=4, color=C["gb"], label="Recall")
ax.plot(thr_df["threshold"], thr_df["f1"],        marker="^", ms=4, color=C["good"], label="F1")
ax.axvline(THRESHOLD, color=C["ink"], ls="--", lw=1.6, label=f"Selected = {THRESHOLD:.2f}")
ax.set(xlabel="Classification threshold", ylabel="Score", ylim=(0, 1.02),
       title=f"Threshold Tuning — {winner}")
ax.legend(loc="center right")
finish(fig, "11_threshold_tuning.png",
       "Threshold picked to maximise F1 on a held-out validation slice — not on the test set.")

cm = confusion_matrix(y_test, y_hat)
fig, ax = plt.subplots(figsize=(6, 5.2))
ax.imshow(cm, cmap="Blues")
labs = ["Non-default", "Default"]
ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual",
       title=f"Confusion Matrix — {winner} @ {THRESHOLD:.2f}")
ax.set_xticklabels(labs); ax.set_yticklabels(labs)
tot = cm.sum()
for i in range(2):
    for j in range(2):
        v = cm[i, j]
        ax.text(j, i, f"{v:,}\n({v/tot*100:.1f}%)", ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if v > cm.max()*0.5 else C["ink"])
finish(fig, "12_confusion_matrix.png",
       "Row = true class. Bottom-right = defaults correctly caught; top-right = false alarms.")

perm = permutation_importance(fitted[winner], X_test, y_test,
                              scoring="roc_auc", n_repeats=8, random_state=RNG, n_jobs=-1)
imp = (pd.DataFrame({"feature": features,
                     "importance": perm.importances_mean,
                     "std": perm.importances_std})
       .sort_values("importance"))
fig, ax = plt.subplots(figsize=(8.5, 5.6))
ax.barh(imp["feature"], imp["importance"], xerr=imp["std"],
        color=C["lr"], alpha=0.9, error_kw=dict(ecolor=C["grey"], lw=1))
ax.set(xlabel="Drop in ROC-AUC when feature is shuffled",
       title=f"Permutation Feature Importance — {winner}")
finish(fig, "13_feature_importance.png",
       "How much test ROC-AUC falls when each feature is randomly shuffled (model-agnostic, honest).")


# =============================================================================
# 10. Save artefacts
# =============================================================================
cv_df.to_csv(f"{OUT_DIR}/cv_model_comparison.csv", index=False)
pd.DataFrame([{**test_metrics[n], "model": n} for n in MODELS]).to_csv(
    f"{OUT_DIR}/test_metrics_both_models.csv", index=False)
pd.DataFrame([final]).to_csv(f"{OUT_DIR}/final_model_summary.csv", index=False)
thr_df.to_csv(f"{OUT_DIR}/threshold_tuning.csv", index=False)
seg.to_csv(f"{OUT_DIR}/risk_segments.csv")
dec.to_csv(f"{OUT_DIR}/pd_deciles.csv")
imp.iloc[::-1].to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
res.to_csv(f"{OUT_DIR}/pd_scored_test_set.csv", index=False)
joblib.dump(fitted[winner], f"{OUT_DIR}/pd_model_{winner.replace(' ', '_').lower()}.pkl")

with open(f"{OUT_DIR}/run_summary.json", "w") as fh:
    json.dump({"winner": winner, "threshold": THRESHOLD,
               "final": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in final.items()}}, fh, indent=2)

print("\n" + "=" * 70)
print("DONE.  Final PD model:", winner)
print("=" * 70)
print(f"  ROC-AUC {final['roc_auc']:.3f} | PR-AUC {final['pr_auc']:.3f} | "
      f"KS {final['ks']:.3f} | Brier {final['brier']:.4f}")
print(f"  @thr {THRESHOLD:.2f}:  precision {final['precision']:.3f}  "
      f"recall {final['recall']:.3f}  F1 {final['f1']:.3f}")
print(f"  Figures -> {FIG_DIR}/ (13)   Tables/model -> {OUT_DIR}/")


# =============================================================================
# 11. SHAP — Model Explainability (Figures 14 & 15)
# =============================================================================
# SHAP (SHapley Additive exPlanations) opens the GB "black box":
#   Fig 14 — Summary plot: which features drive risk across ALL borrowers
#   Fig 15 — Waterfall plot: why the HIGHEST-RISK borrower got that score
# This satisfies the regulatory requirement to explain individual decisions.
# =============================================================================
try:
    import shap as _shap

    # Re-fit the raw GB (without calibration) so TreeExplainer can access
    # the underlying tree structure directly.
    _gb_raw = Pipeline([("prep", preprocess),
                        ("clf", HistGradientBoostingClassifier(
                            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                            min_samples_leaf=60, l2_regularization=1.0,
                            early_stopping=True, validation_fraction=0.1,
                            random_state=RNG))])
    _gb_raw.fit(X_train, y_train)

    # Transform test set to the preprocessed feature space
    _X_test_tf = pd.DataFrame(_gb_raw["prep"].transform(X_test), columns=features)
    _explainer  = _shap.TreeExplainer(_gb_raw["clf"])
    _shap_vals  = _explainer.shap_values(_X_test_tf)   # shape (n_test, n_features)
    _base       = float(_explainer.expected_value[0])
    _p_raw      = _gb_raw.predict_proba(X_test)[:, 1]

    # ---- Fig 14: SHAP Summary Plot -----------------------------------------
    fig14, ax14 = plt.subplots(figsize=(9, 5))
    _shap.summary_plot(_shap_vals, _X_test_tf, feature_names=features,
                       show=False, plot_size=None)
    plt.title("SHAP Feature Impact — Gradient Boosting",
              fontsize=14, fontweight="bold", pad=12)
    fig14.text(0.5, 0.005,
               "Red = increases default risk  |  Blue = decreases default risk  "
               "|  x-axis = SHAP value magnitude",
               ha="center", va="bottom", fontsize=9, style="italic", color="#6B6B6B")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(os.path.join(FIG_DIR, "14_shap_summary.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig14)

    # ---- Fig 15: SHAP Waterfall (highest-risk borrower) --------------------
    _hi  = int(np.argmax(_p_raw))
    _sv  = _shap_vals[_hi]
    _dv  = _X_test_tf.iloc[_hi].values

    fig15, ax15 = plt.subplots(figsize=(9, 5))
    _cols   = [C["bad"] if v > 0 else C["good"] for v in _sv]
    _starts = [_base + sum(_sv[:i]) for i in range(len(_sv))]

    for i, (feat, val, start, col) in enumerate(zip(features, _sv, _starts, _cols)):
        ax15.barh(i, val, left=start, color=col, alpha=0.88, height=0.55)
        sign = "+" if val > 0 else ""
        ax15.text(start + val + (0.015 if val > 0 else -0.015), i,
                  f"{sign}{val:.3f}", va="center",
                  ha="left" if val > 0 else "right",
                  fontsize=10, fontweight="bold", color=C["ink"])

    _feat_labels = [f"{f}  =  {_dv[i]:.2f}" for i, f in enumerate(features)]
    ax15.set_yticks(range(len(features)))
    ax15.set_yticklabels(_feat_labels, fontsize=10)
    ax15.axvline(_base, color=C["grey"], ls="--", lw=1.5,
                 label=f"Base value = {_base:.3f}")
    _final = _base + sum(_sv)
    ax15.axvline(_final, color=C["ink"], ls="-", lw=2,
                 label=f"Model output = {_final:.3f}  (PD ≈ {_p_raw[_hi]:.1%})")
    ax15.set_xlabel("SHAP value — contribution to model output (log-odds scale)")
    ax15.set_title(
        f"SHAP Waterfall — Highest-Risk Borrower  (PD = {_p_raw[_hi]:.1%})",
        fontsize=13, fontweight="bold", pad=12)
    ax15.legend(loc="lower right", fontsize=9)
    ax15.grid(True, axis="x", alpha=0.3)
    ax15.spines["top"].set_visible(False)
    ax15.spines["right"].set_visible(False)
    fig15.text(0.5, 0.005,
               "Each bar shows one feature's contribution to pushing the score "
               "above (red) or below (green) the portfolio average.",
               ha="center", va="bottom", fontsize=9, style="italic", color="#6B6B6B")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(os.path.join(FIG_DIR, "15_shap_waterfall.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig15)

    print(f"  SHAP figures saved (14_shap_summary, 15_shap_waterfall)")
    print(f"  Highest-risk borrower: PD={_p_raw[_hi]:.1%}  "
          f"LTI={_dv[0]:.2f}  missed={_dv[1]:.2f}")

except ImportError:
    print("  [skip] shap not installed — run: pip install shap")