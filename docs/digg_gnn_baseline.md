# Digg GraphSAGE baseline

## Experimental gate and split

- Stage 1 preflight passed all five checks before formal training.
- Fit stories: 72; internal-validation stories: 8; test stories: 20.
- The internal validation set is a seed-42 sample of 10% (8/80) of the saved train stories.
- Test stories were used once, after early stopping and restoration of the best internal-validation checkpoint.
- Best internal-validation epoch: 5; maximum epochs: 50; patience: 5.
- Loss is sampling-weighted BCE. Metrics use the same `sampling_weight` convention as the existing baselines.

## Features

The GNN input is `degree`, a learned community embedding, `log_user_activity`, `log_cascade_size`, and `log_time`. Target-node controls are the exact leakage-audited values constructed by `construct_controlled_data`; sampled-neighbor degree and activity are computed strictly before the window start. `m_in` and `m_out` are not passed to GraphSAGE.

XGB_full uses `degree`, `log_time`, `log_user_activity`, `log_cascade_size`, `m_in`, and `m_out`. Thus the GNN replaces the two handcrafted exposure counts with learned message passing over time-respecting friendship snapshots.

## Same saved 80/20 story-split comparison

| model | weighted log loss | weighted Brier | weighted ROC-AUC | weighted PR-AUC | source |
|---|---:|---:|---:|---:|---|
| M0 | 0.001457248612 | 0.0001516466241 | 0.7062303748 | 0.0003183706267 | pilot_model_metrics.csv (saved 80/20 test stories) |
| M1 | 0.001454208096 | 0.0001516476799 | 0.7090371793 | 0.0003453067412 | pilot_model_metrics.csv (saved 80/20 test stories) |
| XGB_context | 0.001251234386 | 0.0001524467398 | 0.8989289792 | 0.005352667374 | xgb_ablation_metrics.csv (saved 80/20 test stories) |
| XGB_full | 0.001245344393 | 0.0001522431093 | 0.9018507826 | 0.009664626652 | xgb_ablation_metrics.csv (saved 80/20 test stories) |
| GNN | 0.001288529637 | 0.0001514369827 | 0.8825299028 | 0.003650674979 | gnn_metrics.csv (saved 80/20 test stories) |
| GNN+counts | 0.001292044649 | 0.0001514114732 | 0.880116283 | 0.003398611789 | gnn_metrics.csv (saved 80/20 test stories) |

The requested `final_model_metrics.csv` and `xgb_group_cv_metrics.csv` were read and schema-checked (32 and 21 rows). They contain grouped-CV and calibration aggregates, not M0/M1 rows on this single saved split, so those aggregates are not mixed into the table above. M0/M1 come from `pilot_model_metrics.csv`; same-split XGB_context/XGB_full come from `xgb_ablation_metrics.csv`.

## Result

**GNN lost to XGB_full on the primary ranking metric, weighted PR-AUC.** Its PR-AUC was lower by 0.006013951673 (62.23% relative).

- Weighted log loss was 0.001288529637 for GNN versus 0.001245344393 for XGB_full; GNN was worse because higher is worse.
- Weighted Brier was 0.0001514369827 for GNN versus 0.0001522431093 for XGB_full; GNN was slightly better on this probability-error metric because lower is better.
- Weighted ROC-AUC was 0.8825299028 for GNN versus 0.9018507826 for XGB_full.
- Weighted PR-AUC was 0.003650674979 for GNN versus 0.009664626652 for XGB_full.

### Plausible limitations behind the lower GNN ranking result

- **Limited training scale:** this is a 100-cascade pilot, with only 72 stories used for parameter fitting after the internal validation split.
- **Weak observed graph signal:** the existing ablation results already show that user activity and cascade popularity carry most of the predictive value, while community exposure adds only limited and cross-fold-unstable ranking value.
- **Restricted GNN feature set:** `m_in` and `m_out` were deliberately excluded, so the GNN had to infer exposure-related structure through message passing rather than receiving the handcrafted counts available to XGB_full.
- **Coarse snapshots:** one-hour windows may collapse activation order and short-lived diffusion signals that a snapshot GraphSAGE model could otherwise use.

These are plausible explanations grounded in the dataset and design, not excuses and not causes established by this single experiment. No number was changed and no post-test hyperparameter adjustment was performed.

Train weighted-BCE history: `0.0028273743, 0.0011316361, 0.0011280962, 0.0011290995, 0.00112715, 0.0011459691, 0.0011106188, 0.0011246901, 0.001124219, 0.0011302384`.

Internal-validation weighted PR-AUC history: `0.0046834782, 0.005190769, 0.0060136112, 0.0056403133, 0.0066414831, 0.0040415404, 0.0056166466, 0.0049157217, 0.0054912791, 0.0063117471`.

## GNN+counts fair-feature variant

This variant changes one feature condition only: it adds strict-past `m_in` and `m_out` to the existing GNN inputs. Architecture, story split, seed, weighted BCE, validation-only early stopping, and held-out metrics are unchanged.

**GNN+counts lost to XGB_full on weighted PR-AUC** by 0.006266014863 (64.83% relative). On this small pilot, the tree model's inductive bias remained more effective than GraphSAGE even when both received the count features.

- Weighted log loss: 0.001292044649.
- Weighted Brier: 0.0001514114732.
- Weighted ROC-AUC: 0.880116283.
- Weighted PR-AUC: 0.003398611789.
- Relative to the no-count GNN baseline, weighted PR-AUC decreased from 0.003650674979 to 0.003398611789: a decrease of 0.00025206319 (6.90% relative).
- These results are predictive associations, not causal effects.

## Leakage recheck

- Fit/validation/test story overlap: 0.
- Controlled-data user-history leakage flags: 0.
- Controlled-data cascade-size leakage flags: 0.
- Unsorted temporal adjacency lists: 0.
- Graph sampling uses `bisect_left(friend_date, window_start)`, so only strict-past edges are eligible.
- Scalers are fitted only on the 72 model-fit stories; neither validation nor test rows enter their estimates.

## Interpretation

This GNN result is a predictive association, not a causal effect. The real-data GraphSAGE, XGBoost, SHAP, and regression results do not recover physical `a`, `b`, or `theta`; parameter recovery remains a conclusion of the synthetic experiments only.
