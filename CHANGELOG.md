# Changelog

## v3.0

v3.0 retains the complete v1.0 synthetic and v2.0 observational XGBoost pipelines and adds a leakage-gated graph representation-learning stress test.

### Added

- A time-respecting two-layer mean GraphSAGE baseline for next-hour rare-node activation prediction.
- A five-step preflight covering environment, schemas, temporal edge cutoffs, and a one-percent forward/backward smoke run.
- Train-story-only early stopping using a fixed 72 fit / 8 internal-validation partition of the saved 80 outer-training stories; the 20 outer-test stories remain untouched until final evaluation.
- A `GNN+counts` fairness variant that adds strict-past `m_in` and `m_out` without changing architecture, seed, loss, split, or metrics.
- Sampling-weighted GNN metrics, a same-split comparison figure, and a report that retains the negative result.

### Same-split result

| Model | Weighted log loss | Weighted Brier | Weighted ROC-AUC | Weighted PR-AUC |
|---|---:|---:|---:|---:|
| XGB_full | 0.001245344393 | 0.0001522431093 | 0.9018507826 | 0.009664626652 |
| GNN | 0.001288529637 | 0.0001514369827 | 0.8825299028 | 0.003650674979 |
| GNN+counts | 0.001292044649 | 0.0001514114732 | 0.8801162830 | 0.003398611789 |

GraphSAGE does not outperform same-split XGB_full on weighted PR-AUC. Adding the handcrafted counts does not narrow the gap. The final real-data model remains Platt-calibrated XGB_full, whose mean nested-fold metrics are unchanged from v2.0. GNN outputs are predictive associations, not causal effects or recovery of physical `a`, `b`, or `theta`.

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
