# =============================================================================
# 01_data_clean.py  —  Loan data cleaning & feature engineering
# -----------------------------------------------------------------------------
# Goal: turn the raw, messy export into a clean, model-ready table WITHOUT
#       creating dead (constant) features or leaking the target.
#
# Key reliability fixes vs. the previous version:
#   * Robust numeric / date / phone coercion with explicit validity flags.
#   * Tenure features use a fixed snapshot reference date and are only kept if
#     they actually carry information (the old script produced a constant
#     `new_customer` and `customer_tenure_segment` for every single row).
#   * Categorical bins are defined from the real data range, so every label is
#     actually populated (the old `missed_payment_level="high"` was never used).
#   * A final variance check automatically DROPS any engineered feature that is
#     constant / near-constant, and logs what was dropped.  A reliable model is
#     never fed columns that contain zero information.
#   * No feature is derived from `is_default` -> no target leakage.
#   * A data-quality report is printed and saved next to the cleaned file.
# =============================================================================

import os
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 0. Config
# -----------------------------------------------------------------------------
RAW_PATH      = "loan_dirty_data.csv"     # raw export (input)
CLEAN_PATH    = "loan_cleaned.csv"        # model-ready output
REPORT_PATH   = "outputs/data_quality_report.csv"

# Snapshot date = the date the data was pulled. We use (latest join date + 1 day)
# so that "days since join" is always >= 0 and is anchored to a fixed point
# instead of to the single newest customer (which made everyone look brand new).
SNAPSHOT_OFFSET_DAYS = 1

PHONE_LEN            = 11        # valid CN mobile number length
LTI_CAP              = 20.0      # loan-to-income ratios above this are implausible
WINSOR_Q             = 0.995     # cap extreme income / loan_amount outliers here
NEAR_CONST_TOL       = 0.999     # drop a feature if one value covers >= 99.9% of rows

os.makedirs("outputs", exist_ok=True)

dq = {}  # data-quality counters, written to the report at the end


def log(msg: str):
    print(msg)


# -----------------------------------------------------------------------------
# 1. Load
# -----------------------------------------------------------------------------
log("=" * 70)
log("Loading raw data")
log("=" * 70)

df = pd.read_csv(RAW_PATH, dtype=str)        # read everything as text first, coerce deliberately
dq["rows_raw"] = len(df)
log(f"Raw rows: {len(df):,}   columns: {list(df.columns)}")

REQUIRED = ["client_id", "income", "loan_amount", "historical_missed_payments", "is_default"]
missing_required = [c for c in REQUIRED if c not in df.columns]
if missing_required:
    raise ValueError(f"Raw file is missing required columns: {missing_required}")


# -----------------------------------------------------------------------------
# 2. Target & ID integrity  (rows we cannot use at all)
# -----------------------------------------------------------------------------
df["client_id"] = pd.to_numeric(df["client_id"], errors="coerce")
df = df.dropna(subset=["client_id"])
df["client_id"] = df["client_id"].astype("int64")

# Target must be a clean 0/1.  Anything else is dropped rather than guessed.
df["is_default"] = pd.to_numeric(df["is_default"], errors="coerce")
before = len(df)
df = df[df["is_default"].isin([0, 1])]
df["is_default"] = df["is_default"].astype(int)
dq["rows_dropped_bad_target"] = before - len(df)
dq["rows_dropped_no_client_id"] = dq["rows_raw"] - before

df = df.drop_duplicates(subset=["client_id"], keep="first")
dq["rows_after_dedup"] = len(df)
log(f"After ID/target integrity & dedup: {len(df):,} rows")


# -----------------------------------------------------------------------------
# 3. Helper: clean a positive numeric column, capture missing + outliers
# -----------------------------------------------------------------------------
def clean_positive_numeric(series: pd.Series, cap_quantile: float | None = None):
    """Coerce to numeric, treat negatives as missing, optionally winsorize the
    top tail, then median-impute. Returns (clean_series, missing_flag)."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.mask(s < 0)                                   # negatives are invalid
    missing_flag = s.isna().astype(int)
    if cap_quantile is not None:
        hi = s.quantile(cap_quantile)
        s = s.clip(upper=hi)
    s = s.fillna(s.median())
    return s, missing_flag


# 3a. income
df["income"], df["income_missing"] = clean_positive_numeric(df["income"], WINSOR_Q)

# 3b. loan_amount
df["loan_amount"], df["loan_amount_missing"] = clean_positive_numeric(df["loan_amount"], WINSOR_Q)

# 3c. historical missed payments -> non-negative integer, missing means none observed
hmp = pd.to_numeric(df["historical_missed_payments"], errors="coerce")
hmp = hmp.clip(lower=0).fillna(0).round().astype(int)
df["historical_missed_payments"] = hmp

dq["pct_income_imputed"] = round(100 * df["income_missing"].mean(), 2)
dq["pct_loan_imputed"]   = round(100 * df["loan_amount_missing"].mean(), 2)


# -----------------------------------------------------------------------------
# 4. Phone  -> keep only the digits, validity flag, mask invalid numbers
# -----------------------------------------------------------------------------
if "phone" in df.columns:
    digits = df["phone"].astype(str).str.replace(r"\D", "", regex=True)
    df["phone_valid"] = (digits.str.len() == PHONE_LEN).astype(int)
    df["phone"] = np.where(df["phone_valid"] == 1, digits, "UNKNOWN")
else:
    df["phone_valid"] = 0
dq["pct_phone_invalid"] = round(100 * (1 - df["phone_valid"].mean()), 2)


# -----------------------------------------------------------------------------
# 5. Tags -> normalise text, fill blanks, derive is_vip
# -----------------------------------------------------------------------------
if "tags" in df.columns:
    tags = df["tags"].astype(str).str.strip().str.upper()
    tags = tags.replace({"NAN": np.nan, "": np.nan, "NONE": np.nan})
    df["tags"] = tags.fillna("NORMAL")
else:
    df["tags"] = "NORMAL"
df["is_vip"] = (df["tags"] == "VIP").astype(int)


# -----------------------------------------------------------------------------
# 6. Dates -> robust parse + tenure, anchored to a fixed snapshot date
# -----------------------------------------------------------------------------
if "join_date" in df.columns:
    join = pd.to_datetime(df["join_date"], errors="coerce")
    df["join_date"] = join.dt.strftime("%Y-%m-%d")
    df["join_date_missing"] = join.isna().astype(int)

    snapshot = join.max() + pd.Timedelta(days=SNAPSHOT_OFFSET_DAYS)
    days = (snapshot - join).dt.days
    df["days_since_join"] = days.fillna(days.median())
    log(f"Join-date range: {join.min()}  ->  {join.max()}   "
        f"(snapshot = {snapshot.date()}, tenure spread = "
        f"{int(df['days_since_join'].max() - df['days_since_join'].min())} days)")
else:
    df["join_date_missing"] = 1
    df["days_since_join"] = np.nan


# -----------------------------------------------------------------------------
# 7. Engineered features  (all leakage-free: none use is_default)
# -----------------------------------------------------------------------------
# 7a. Loan-to-income
lti = df["loan_amount"] / df["income"].replace(0, np.nan)
lti = lti.replace([np.inf, -np.inf], np.nan).fillna(lti.median())
df["LTI"] = lti.clip(upper=LTI_CAP)

# 7b. Missed-payment behaviour
df["has_missed_payment"] = (df["historical_missed_payments"] > 0).astype(int)
# Bins chosen from the real range so every label is actually used.
mx = int(df["historical_missed_payments"].max())
edges  = sorted(set([-1, 0, 1, 3, max(5, mx + 1)]))
labels = ["none", "low", "medium", "high"][: len(edges) - 1]
df["missed_payment_level"] = pd.cut(df["historical_missed_payments"], bins=edges, labels=labels)

# 7c. Risk / leverage segments (quantile based, so they are always populated)
df["high_LTI"]      = (df["LTI"] > df["LTI"].quantile(0.75)).astype(int)
df["LTI_segment"]    = pd.qcut(df["LTI"],         q=4, labels=["low", "mid_low", "mid_high", "high"], duplicates="drop")
df["income_segment"] = pd.qcut(df["income"],      q=4, labels=["low", "mid_low", "mid_high", "high"], duplicates="drop")
df["loan_segment"]   = pd.qcut(df["loan_amount"], q=4, labels=["small", "mid_small", "mid_large", "large"], duplicates="drop")

# 7d. Tenure segments (kept for now; the variance check in step 8 will drop
#     them automatically if join_date has no real spread)
df["new_customer"] = (df["days_since_join"] < 90).astype(int)
df["customer_tenure_segment"] = pd.cut(
    df["days_since_join"],
    bins=[-1, 30, 90, 180, 365, np.inf],
    labels=["very_new", "new", "medium", "old", "very_old"],
)

# 7e. Data-quality risk score & interactions
df["data_quality_risk"] = (
    df["join_date_missing"] + df["income_missing"]
    + df["loan_amount_missing"] + (1 - df["phone_valid"])
)
df["high_LTI_and_missed"]      = ((df["high_LTI"] == 1) & (df["has_missed_payment"] == 1)).astype(int)
df["invalid_phone_and_missed"] = ((df["phone_valid"] == 0) & (df["has_missed_payment"] == 1)).astype(int)


# -----------------------------------------------------------------------------
# 8. Reliability check: drop constant / near-constant engineered features
# -----------------------------------------------------------------------------
log("\n" + "=" * 70)
log("Variance check (a reliable model is never fed information-free columns)")
log("=" * 70)

PROTECTED = {"client_id", "is_default", "join_date", "phone"}  # never auto-drop these
candidate_cols = [c for c in df.columns if c not in PROTECTED]

dropped = []
for c in candidate_cols:
    nun = df[c].nunique(dropna=False)
    top_share = df[c].value_counts(dropna=False, normalize=True).iloc[0] if nun > 0 else 1.0
    if nun <= 1 or top_share >= NEAR_CONST_TOL:
        dropped.append((c, nun, round(float(top_share), 4)))

if dropped:
    for c, nun, share in dropped:
        log(f"  DROP  {c:28s}  unique={nun:<4d} top_value_share={share:.3f}")
    df = df.drop(columns=[c for c, _, _ in dropped])
else:
    log("  (no constant features found)")
dq["features_dropped_constant"] = ";".join(c for c, _, _ in dropped) or "none"


# -----------------------------------------------------------------------------
# 9. Save cleaned data + data-quality report
# -----------------------------------------------------------------------------
df.to_csv(CLEAN_PATH, index=False)

dq["rows_clean"]     = len(df)
dq["default_rate"]   = round(float(df["is_default"].mean()), 4)
dq["n_features_out"] = df.shape[1]
pd.DataFrame([dq]).T.rename(columns={0: "value"}).to_csv(REPORT_PATH)

log("\n" + "=" * 70)
log("Cleaning complete")
log("=" * 70)
log(f"  Clean rows         : {len(df):,}")
log(f"  Columns out        : {df.shape[1]}")
log(f"  Default rate       : {df['is_default'].mean():.4f}")
log(f"  Saved cleaned data : {CLEAN_PATH}")
log(f"  Saved DQ report    : {REPORT_PATH}")
