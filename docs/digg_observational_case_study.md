# Digg observational prediction and model stress test

## Repositioning and scope

The Digg analysis is an **observational prediction and model stress test**. It is not theory-constrained fitting, causal inference, or recovery of the paper's structural parameters.

Both earlier count-based results are retained:

- Original 7,277-community pilot M1: `coef_m_in=0.027108040`, `coef_m_out=-0.048634896`, `coef_degree=0.001986288`, `coef_log_time=-0.534487611`; converged in 24 iterations.
- Topology-selected two-community induced-subsystem M1: `coef_m_in=0.031146452`, `coef_m_out=-0.095500605`, `coef_degree=0.001099803`, `coef_log_time=-0.480611231`; converged in 7 iterations.
- The `m_in` point estimate is consistently positive across the original and strict two-community M1 fits. In the controlled diagnostics it also remains positive in D1, D2, D3, and the piecewise-time D3 point estimates; the D3 bootstrap shows 91% positive replicates, so 'stable' here means directional consistency of the fitted point estimates, not certainty of a structural effect.
- `m_out` remains negative in the strict two-community fit. Changing from 'own versus all other communities' to a genuine induced two-community subsystem does not restore the paper-expected positive sign.
- The degree coefficient is positive in both M1 fits. Therefore `theta_eff=-coef_degree/5` is negative and has no defensible interpretation as the paper's normalized adoption threshold.
- Both models converged normally with no optimizer warnings. The sign mismatch is not an optimization failure.
- Digg does not support interpreting these regression coefficients as true `a`, `b`, or `theta`. They remain observational associations under a misspecified probabilistic mapping.
- No community pair is changed, no coefficient sign is constrained, and the analysis is not expanded beyond the current 100 pilot cascades.

## XGBoost design

- Saved story-level split reused exactly: **80 train / 20 test cascades**, seed 42; **267,702 train / 78,636 test rows**.
- Sampling-weighted training prevalence: **0.01382677%**.
- Features: `m_in`, `m_out`, `degree`, `log1p(time_bin)`, `log_user_activity`, and `log_cascade_size`. No `frac_in` or `frac_out` is used.
- `log_user_activity` counts only the node's votes in other stories strictly before the current absolute window start. `log_cascade_size` counts network-covered adopters strictly before the current story-window start. Prior control audits found zero leakage violations.
- Training uses the original `sampling_weight`; no class weight is added.
- Tree parameters were fixed before test evaluation: 400 trees, learning rate 0.05, depth 4, minimum child weight 10, row subsampling 0.8, feature subsampling 0.9, L2 penalty 1, histogram trees, seed 42. The test set is not used for tuning or early stopping.
- Software: XGBoost `2.1.4`, SHAP `0.49.1`.

## Held-out prediction

All metrics use `sampling_weight` on the same 20 held-out cascades.

| model | weighted log loss | weighted Brier | weighted ROC-AUC | weighted PR-AUC |
|---|---:|---:|---:|---:|
| M0 | 0.001457248612 | 0.0001516466241 | 0.7062303748 | 0.0003183706267 |
| M1 | 0.001454208096 | 0.0001516476799 | 0.7090371793 | 0.0003453067412 |
| XGBoost | 0.001244566238 | 0.0001519312549 | 0.9008285685 | 0.01144090181 |

Relative to M1, XGBoost changes weighted log loss by **-0.0002096418579**, Brier by **+2.835750046e-07**, ROC-AUC by **+0.1917913891**, and PR-AUC by **+0.01109559507**.

M0 remains the non-network baseline, M1 tests the paper-inspired count rule in probabilistic form, and XGBoost asks whether nonlinearities and the two leakage-safe behavioral controls improve prediction. Better XGBoost performance would indicate predictive misspecification of the linear M1, not validation of a causal diffusion mechanism.

## SHAP interpretation

SHAP values were computed for a deterministic uniform sample of **20,000 held-out rows**. The summary plot ranks features by their contribution to this fitted predictor; the two dependence plots show how fitted contributions for `m_in` and `m_out` vary across observed rows and interactions.

SHAP values are **not causal effects**. They depend on the fitted XGBoost function, correlated inputs, sampled risk sets, inverse-sampling weights used in training, and the observational Digg data-generating process. No `a`, `b`, or `theta` is calculated from XGBoost or SHAP.

## Conclusion

The Digg case is retained as a model stress test: within-community count exposure has a directionally positive association, while cross-community and degree coefficients do not obey the structural signs required to interpret M1 as the original threshold mechanism. Predictive comparisons are valid for the held-out pilot cascades; structural or causal parameter claims are not.
