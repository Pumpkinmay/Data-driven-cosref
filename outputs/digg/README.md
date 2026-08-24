# Curated Digg outputs

The `outputs/digg/` working directory contains intermediate audit artifacts as well as final results. For a compact public release, the recommended Digg artifacts are:

- `controlled_model_coefficients.csv`, `controlled_model_report.md`, and `controlled_coefficient_plot.png` — linear count-model stress test;
- `leakage_audit.md` — temporal and story-split leakage checks;
- `xgb_group_cv_metrics.csv`, `xgb_group_cv_summary.md`, and `xgb_group_cv_plot.png` — feature ablation and story-group validation;
- `final_model_metrics.csv` and `final_calibration.png` — nested Platt calibration and final model comparison;
- `xgb_shap_summary.png` — held-out predictive-association summary.

The remaining CSVs, reports, and plots are useful local provenance from preprocessing, diagnostics, subgroup analysis, or superseded figure versions. They need not all be committed to the public repository. Raw data, processed data, compressed pilot exposure tables, full exposure tables, and serialized models must remain outside Git.
