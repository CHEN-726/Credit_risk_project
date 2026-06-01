# Loan PD model — rewritten pipeline

Two rebuilt scripts that produce a **trustworthy** Probability-of-Default (PD) model
instead of one with meaningless probabilities.

```
01_data_clean.py            loan_dirty_data.csv  ->  loan_cleaned.csv
02_train_with_figures.py    loan_cleaned.csv     ->  figures/ + outputs/
```

Run:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
python 01_data_clean.py        # only if you have the raw file; otherwise use your existing loan_cleaned.csv
python 02_train_with_figures.py
```

## What was wrong before

| Problem in the old code | Why it made the model untrustworthy |
|---|---|
| `class_weight="balanced"` + `predict_proba` | Inflated every probability — a "PD" of 0.5 actually meant ~11% default. Useless as a PD. |
| Fixed threshold `0.55` | Arbitrary, never justified, gave precision **0.22** (78% false alarms). |
| `new_customer`, `customer_tenure_segment` | **Constant for all 93,114 rows** — zero information fed to the model. |
| `days_since_join` | Broken date logic (`reference = max(join_date)`) collapsed it to a 0–2 range. |
| `missed_payment_level="high"` | Bin never populated (max missed payments = 4, needed > 5). |
| Stacked `LTI` + `high_LTI` + `LTI_segment` | Same signal counted 3×, wrecking interpretability. |

## What makes the new model reliable

- **Calibrated probabilities (isotonic).** A predicted PD of *x%* really does default ~*x%* of the time — proven in `figures/03_calibration_comparison.png` and `figures/10_decile_default_rate.png`.
- **Two models compared honestly.** Logistic Regression (interpretable) vs Gradient Boosting; the winner is chosen by **cross-validated** ROC-AUC, never by peeking at the test set.
- **Threshold chosen on a held-out validation slice** (max F1), not on the test set and not arbitrarily.
- **Leakage guard + variance guard** automatically drop constant or target-leaking columns.
- **Honest, complete metrics:** ROC-AUC, PR-AUC, KS, Gini, Brier (calibration) — not just accuracy.

## Result (winner: Gradient Boosting, calibrated)

| Metric | Value | Reading |
|---|---|---|
| ROC-AUC | **0.718** | Solid, realistic credit-risk discrimination (not a fake 0.99, not a coin-flip 0.50). |
| KS | **0.418** | Strong separation (credit models usually want KS > 0.30). |
| PR-AUC | 0.244 | vs 0.108 baseline — 2.3× better than random on the rare class. |
| Brier | 0.088 | Low = well-calibrated probabilities. |
| Precision / Recall / F1 @ 0.165 | 0.245 / 0.663 / 0.358 | Catches 66% of defaulters; trade-off is explicit, not hidden. |

Calibration check (test set):

| Risk segment | Predicted PD | **Actual** default rate |
|---|---|---|
| Low (PD < 8%) | 4.9% | 5.1% |
| Medium (8–20%) | 18.4% | 19.2% |
| High (≥ 20%) | 25.4% | 25.1% |

Predicted ≈ actual in every band — that is the definition of a "real" PD model.

## Outputs

`figures/` — 13 publication-quality charts (ROC, PR, calibration, KS, gains/lift,
PD distribution, risk segments, deciles, threshold tuning, confusion matrix,
feature importance, CV stability).
`outputs/` — metric tables, scored test set, the saved model (`.pkl`), and a data-quality report.
