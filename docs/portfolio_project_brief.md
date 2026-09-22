# Rare Node-Activation Prediction on Social Graphs

## Portfolio framing

This project studies a graph-based rare-event prediction problem with direct relevance to risk modeling, fraud detection, and graph-algorithm roles. In a social cascade, the task is to predict whether a node that is inactive at time (t) will become active during the next time window, using only graph structure and behavioral history available before that window.

Across all 100 Digg pilot stories, the sampling-weighted positive prevalence is approximately **0.014%** (`0.01410992%`). The saved 80-story training subset has prevalence `0.01382677%`, while the 20-story test subset has prevalence `0.01516731%`; the small difference is due to the story-level split. The unweighted row prevalence is exactly one sixth because the table retains all positives and samples up to five negatives per positive, so it is not an estimate of population prevalence. This extreme weighted class imbalance makes accuracy uninformative and places the problem in the same methodological family as rare fraud, abuse, default, or alert-prioritization tasks: the model must rank a very small number of positives while producing usable probabilities under biased negative sampling.

## Prediction target and available information

For each cascade, node, and one-hour prediction window:

- `y=1` means that a previously inactive node casts its first vote during the next hour;
- graph exposure is computed only from neighbors who adopted before the window starts;
- `m_in` counts prior adopters from the node's own community;
- `m_out` counts prior adopters from other communities;
- degree, elapsed cascade time, prior user activity, and cascade size so far provide structural and behavioral context; and
- current-window and future adopters are excluded from every feature.

The learning problem is evaluated across held-out stories rather than randomly split rows, so performance measures transfer to unseen cascades instead of benefiting from within-story leakage.

## Technical challenges

### 1. Theory-inspired linear features do not fully describe real diffusion

The count-threshold model is interpretable in synthetic experiments, where its assumptions and true parameters are known. On Digg, however, the fitted cross-community coefficient `m_out` is negative and the degree coefficient is positive. Consequently, `theta_eff = -coef_degree / beta` has no defensible physical threshold interpretation. These signs were reported without clipping or sign constraints and are retained as a model-stress-test result.

For an applied graph-learning project, this is a specification challenge rather than an optimization failure: a small collection of hand-engineered counts cannot be assumed to encode the heterogeneous network, behavioral, recommendation, and cascade-state mechanisms present in a real platform.

### 2. Community exposure adds limited and cross-fold-unstable ranking information

User activity and cascade popularity account for most of the predictive improvement. Adding `m_in` and `m_out` to XGB_context increases mean five-fold PR-AUC from `0.0113434` to `0.0130119`, a relative increase of approximately **14.7%**. The increment is not uniformly stable: XGB_full wins on PR-AUC in only **3 of 5 folds**, and the story-cluster bootstrap 95% interval for the pooled PR-AUC increment is **[-0.003924, 0.005473]**, which crosses zero.

The appropriate conclusion is therefore that community-aware count features provide **additional but limited rare-adopter ranking value**, not a stable improvement across cascades.

### 3. Extreme imbalance requires ranking, weighting, and calibration discipline

The pilot retains all positive observations and samples negatives within each cascade-time risk set. `sampling_weight` reconstructs the target risk-set contribution during model fitting and evaluation. The analysis reports weighted log loss and Brier score for probability quality, and weighted ROC-AUC and PR-AUC for ranking, with PR-AUC interpreted relative to the weighted positive prevalence.

This design separates two operational questions common in risk systems: whether a model can prioritize rare events and whether its scores behave as probabilities.

### 4. Structural signal may require learned graph representations

The observed limitations of scalar exposure counts motivate testing whether a graph representation learner can extract useful local structure that `m_in`, `m_out`, and degree omit. This is a hypothesis to evaluate under the existing leakage controls and grouped validation—not evidence that a GNN will necessarily outperform the current models.

## Existing modeling and validation stack

The repository already implements:

- time-respecting construction of the baseline friendship network;
- leakage-safe hourly cascade risk sets;
- inverse-probability weighting for sampled negatives;
- interpretable M0/M1 observational logistic baselines;
- XGBoost context/full feature ablation;
- five-fold grouped cross-validation with `group=story_id`;
- story-cluster bootstrap uncertainty for the exposure-feature PR-AUC increment;
- SHAP diagnostics on held-out predictions; and
- nested weighted Platt calibration using calibration stories drawn only from each training fold.

This stack provides a controlled benchmark for any additional graph model: the same story split, sampled observations, `sampling_weight`, and weighted evaluation metrics must be reused.

## Scientific interpretation boundary

All Digg findings are **predictive associations**, not causal effects. Neither regression coefficients nor SHAP values identify the causal effect of exposure. The negative `m_out` coefficient, positive degree coefficient, and nonphysical `theta_eff` are not hidden or reinterpreted as theoretical parameters.

Recovery of known `a`, `b`, and `theta` is supported only by the synthetic experiments, where cascades are generated from specified parameters. Synthetic parameter recovery is not equivalent to inverse estimation of physical or causal parameters from Digg.

## Next: GNN baseline

The next extension is a deliberately bounded GNN baseline for rare node-activation prediction. It should learn local structural representations while retaining the existing behavioral context features. To remain comparable with XGBoost, it must reuse the saved story-level train/test split, the five story-group folds, the existing exposure rows and labels, `sampling_weight`, and the same weighted log loss, Brier, ROC-AUC, and PR-AUC definitions.

The GNN experiment should be reported as an incremental predictive benchmark. Its purpose is to test whether learned graph structure adds held-out signal beyond the current context and count features; it must not be presented as causal identification or as real-data recovery of `a`, `b`, or `theta`.
