# Changelog

## v3.1

v3.1 adds a Digg implicit-feedback recommendation task without changing the previously reported diffusion, XGBoost, or GraphSAGE outputs.

### Added

- Popularity and BPR-MF baselines.
- DeepFM_ID, DeepFM_context, and DeepFM_full feature ablations.
- A temporal leave-last-two-out split for users with at least five votes.
- Time-respecting shared candidate sampling with 99 sampled non-interactions per eligible validation/test target.
- Recall@10/20, NDCG@10/20, HitRate@10, MRR, sampled AUC, parameter counts, and measured training/inference time.
- User-level bootstrap intervals, paired model-delta intervals, candidate checksums, leakage documentation, and data-free unit tests.

### Final recommendation model

DeepFM_context is the selected v3.1 recommendation model:

| Model | NDCG@10 | Recall@10 | MRR | AUC |
|---|---:|---:|---:|---:|
| **DeepFM_context** | **0.727087** | **0.928401** | **0.665968** | **0.970299** |

DeepFM_context exceeds DeepFM_ID, while DeepFM_full is slightly lower than DeepFM_context. The result supports the value of the measured dynamic context features but does not support an additional recommendation gain from the baseline community feature in this experiment. It does not negate the role of community structure in the separate diffusion task. DIN is not implemented.

### Evaluation boundary

This is a warm-start, sampled-candidate offline evaluation, not full-catalog ranking or CTR prediction. Each evaluated target has one positive and 99 sampled non-interactions. Digg has no impression logs, sampled non-interactions are not confirmed negative feedback, cold-item rate is 0%, and the results do not establish online benefit. Recommendation metrics are not directly comparable with v2.0 diffusion PR-AUC.

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
