# Changelog

## v2.0

v2.0 retains the complete v1.0 synthetic parameter-recovery pipeline and adds an observational real-cascade analysis.

### Added

- Digg real-cascade data audit and preprocessing.
- A time-respecting baseline friendship network, Leiden communities, and leakage-safe exposure-table construction.
- M0/M1 observational logistic baselines and diagnostics for the limits of interpreting real-data coefficients as physical `a`, `b`, and `theta`.
- XGBoost context/full feature ablation, story-level grouped cross-validation, SHAP interpretation, and Platt probability calibration.
- Reproducibility tests, prepared-input checksums, and SNAP provenance documentation.

### Final real-data model

The final real-data model is Platt-calibrated XGB_full:

| Metric | Value |
|---|---:|
| Log loss | 0.00113033 |
| Brier | 0.000137271 |
| ROC-AUC | 0.90323 |
| PR-AUC | 0.0117310 |

Platt XGB_context has a slightly better Brier score of 0.000137012. User activity and cascade popularity provide most of the predictive gain; `m_in` and `m_out` add limited value for rare-adopter ranking. Digg results are observational predictions and associations, not causal or structural recovery of `a`, `b`, and `theta`.

## v1.0

- Synthetic cascades on empirical network topologies.
- Logistic Regression parameter recovery.
- Recovery of `a`, `b`, and `theta`.
- Parameter-grid validation.
- Identifiability diagnostics.
- Synthetic/model-based intervention experiments.

Synthetic parameter recovery belongs to v1.0 and remains unchanged in v2.0.
