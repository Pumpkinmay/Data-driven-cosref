#!/usr/bin/env python3
"""Fit and explain an observational XGBoost model on the 100-cascade pilot."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from diagnose_digg_pilot import DiagnosticError, atomic_csv, metrics, read_split
from fit_digg_controlled_models import construct_controlled_data


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "digg"
DOCS = ROOT / "docs"
SEED = 42
FEATURES = (
    "m_in",
    "m_out",
    "degree",
    "log_time",
    "log_user_activity",
    "log_cascade_size",
)
DISPLAY_NAMES = {
    "m_in": "m_in",
    "m_out": "m_out",
    "degree": "degree",
    "log_time": "log1p(time_bin)",
    "log_user_activity": "log_user_activity",
    "log_cascade_size": "log_cascade_size",
}


def require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def read_existing_metrics(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in ("M0", "M1") or row["subset"] != "all_test":
                continue
            rows.append(
                {
                    "model": row["model"],
                    "test_rows": int(row["rows"]),
                    "test_positives": int(row["positives"]),
                    "weighted_y_rate": float(row["weighted_positive_rate"]),
                    "weighted_log_loss": float(row["weighted_log_loss"]),
                    "weighted_brier": float(row["weighted_brier"]),
                    "weighted_roc_auc": float(row["weighted_roc_auc"]),
                    "weighted_pr_auc": float(row["weighted_pr_auc"]),
                    "feature_set": (
                        "degree;log1p(time_bin)" if row["model"] == "M0"
                        else "m_in;m_out;degree;log1p(time_bin)"
                    ),
                    "notes": "Existing weighted logistic-regression result on identical saved split",
                }
            )
    if {row["model"] for row in rows} != {"M0", "M1"}:
        raise DiagnosticError("Could not find existing M0/M1 all-test metrics")
    return sorted(rows, key=lambda row: str(row["model"]))


def xgboost_model(base_score: float) -> xgb.XGBClassifier:
    # Fixed before viewing test performance; no held-out-test tuning or early stopping.
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=10.0,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        max_delta_step=1.0,
        tree_method="hist",
        base_score=base_score,
        random_state=SEED,
        n_jobs=4,
        verbosity=0,
    )


def save_shap_plots(
    model: xgb.XGBClassifier,
    x_test: np.ndarray,
    feature_names: list[str],
) -> int:
    # A deterministic uniform held-out-row sample makes plotting tractable. It
    # describes model predictions for sampled test rows, not a causal estimand.
    rng = np.random.default_rng(SEED)
    sample_size = min(20_000, len(x_test))
    sample_index = np.sort(rng.choice(len(x_test), size=sample_size, replace=False))
    x_plot = x_test[sample_index]
    explainer = shap.TreeExplainer(model)
    explanation = explainer(x_plot)

    plt.figure(figsize=(9, 5.5))
    shap.summary_plot(
        explanation.values,
        x_plot,
        feature_names=feature_names,
        max_display=len(feature_names),
        show=False,
    )
    plt.title("Digg pilot XGBoost: held-out SHAP summary", pad=12)
    plt.tight_layout()
    plt.savefig(OUTPUT / "xgboost_shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    for feature, destination in (
        ("m_in", OUTPUT / "xgboost_shap_m_in.png"),
        ("m_out", OUTPUT / "xgboost_shap_m_out.png"),
    ):
        plt.figure(figsize=(8, 5.5))
        shap.dependence_plot(
            feature,
            explanation.values,
            x_plot,
            feature_names=feature_names,
            interaction_index="auto",
            show=False,
            alpha=0.35,
        )
        plt.title(f"Digg pilot XGBoost: SHAP dependence for {feature}", pad=12)
        plt.tight_layout()
        plt.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close()
    return sample_size


def fmt(value: object) -> str:
    return f"{float(value):.10g}"


def write_case_study(
    path: Path,
    metric_rows: list[dict[str, object]],
    shap_rows: int,
    train_rows: int,
    test_rows: int,
    train_stories: int,
    test_stories: int,
    weighted_train_rate: float,
    xgb_version: str,
    shap_version: str,
) -> None:
    lookup = {str(row["model"]): row for row in metric_rows}
    m0, m1, boosted = lookup["M0"], lookup["M1"], lookup["XGBoost"]
    lines = [
        "# Digg observational prediction and model stress test",
        "",
        "## Repositioning and scope",
        "",
        "The Digg analysis is an **observational prediction and model stress test**. It is not theory-constrained fitting, causal inference, or recovery of the paper's structural parameters.",
        "",
        "Both earlier count-based results are retained:",
        "",
        "- Original 7,277-community pilot M1: `coef_m_in=0.027108040`, `coef_m_out=-0.048634896`, `coef_degree=0.001986288`, `coef_log_time=-0.534487611`; converged in 24 iterations.",
        "- Topology-selected two-community induced-subsystem M1: `coef_m_in=0.031146452`, `coef_m_out=-0.095500605`, `coef_degree=0.001099803`, `coef_log_time=-0.480611231`; converged in 7 iterations.",
        "- The `m_in` point estimate is consistently positive across the original and strict two-community M1 fits. In the controlled diagnostics it also remains positive in D1, D2, D3, and the piecewise-time D3 point estimates; the D3 bootstrap shows 91% positive replicates, so 'stable' here means directional consistency of the fitted point estimates, not certainty of a structural effect.",
        "- `m_out` remains negative in the strict two-community fit. Changing from 'own versus all other communities' to a genuine induced two-community subsystem does not restore the paper-expected positive sign.",
        "- The degree coefficient is positive in both M1 fits. Therefore `theta_eff=-coef_degree/5` is negative and has no defensible interpretation as the paper's normalized adoption threshold.",
        "- Both models converged normally with no optimizer warnings. The sign mismatch is not an optimization failure.",
        "- Digg does not support interpreting these regression coefficients as true `a`, `b`, or `theta`. They remain observational associations under a misspecified probabilistic mapping.",
        "- No community pair is changed, no coefficient sign is constrained, and the analysis is not expanded beyond the current 100 pilot cascades.",
        "",
        "## XGBoost design",
        "",
        f"- Saved story-level split reused exactly: **{train_stories} train / {test_stories} test cascades**, seed 42; **{train_rows:,} train / {test_rows:,} test rows**.",
        f"- Sampling-weighted training prevalence: **{weighted_train_rate:.8%}**.",
        "- Features: `m_in`, `m_out`, `degree`, `log1p(time_bin)`, `log_user_activity`, and `log_cascade_size`. No `frac_in` or `frac_out` is used.",
        "- `log_user_activity` counts only the node's votes in other stories strictly before the current absolute window start. `log_cascade_size` counts network-covered adopters strictly before the current story-window start. Prior control audits found zero leakage violations.",
        "- Training uses the original `sampling_weight`; no class weight is added.",
        "- Tree parameters were fixed before test evaluation: 400 trees, learning rate 0.05, depth 4, minimum child weight 10, row subsampling 0.8, feature subsampling 0.9, L2 penalty 1, histogram trees, seed 42. The test set is not used for tuning or early stopping.",
        f"- Software: XGBoost `{xgb_version}`, SHAP `{shap_version}`.",
        "",
        "## Held-out prediction",
        "",
        "All metrics use `sampling_weight` on the same 20 held-out cascades.",
        "",
        "| model | weighted log loss | weighted Brier | weighted ROC-AUC | weighted PR-AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("M0", "M1", "XGBoost"):
        row = lookup[model]
        lines.append(
            f"| {model} | {fmt(row['weighted_log_loss'])} | {fmt(row['weighted_brier'])} | "
            f"{fmt(row['weighted_roc_auc'])} | {fmt(row['weighted_pr_auc'])} |"
        )
    lines.extend(
        [
            "",
            f"Relative to M1, XGBoost changes weighted log loss by **{float(boosted['weighted_log_loss']) - float(m1['weighted_log_loss']):+.10g}**, Brier by **{float(boosted['weighted_brier']) - float(m1['weighted_brier']):+.10g}**, ROC-AUC by **{float(boosted['weighted_roc_auc']) - float(m1['weighted_roc_auc']):+.10g}**, and PR-AUC by **{float(boosted['weighted_pr_auc']) - float(m1['weighted_pr_auc']):+.10g}**.",
            "",
            "M0 remains the non-network baseline, M1 tests the paper-inspired count rule in probabilistic form, and XGBoost asks whether nonlinearities and the two leakage-safe behavioral controls improve prediction. Better XGBoost performance would indicate predictive misspecification of the linear M1, not validation of a causal diffusion mechanism.",
            "",
            "## SHAP interpretation",
            "",
            f"SHAP values were computed for a deterministic uniform sample of **{shap_rows:,} held-out rows**. The summary plot ranks features by their contribution to this fitted predictor; the two dependence plots show how fitted contributions for `m_in` and `m_out` vary across observed rows and interactions.",
            "",
            "SHAP values are **not causal effects**. They depend on the fitted XGBoost function, correlated inputs, sampled risk sets, inverse-sampling weights used in training, and the observational Digg data-generating process. No `a`, `b`, or `theta` is calculated from XGBoost or SHAP.",
            "",
            "## Conclusion",
            "",
            "The Digg case is retained as a model stress test: within-community count exposure has a directionally positive association, while cross-community and degree coefficients do not obey the structural signs required to interpret M1 as the original threshold mechanism. Predictive comparisons are valid for the held-out pilot cascades; structural or causal parameter claims are not.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    exposure = PROCESSED / "digg_exposure_pilot.csv.gz"
    votes = PROCESSED / "digg_votes_clean.csv.gz"
    communities = PROCESSED / "digg_communities.csv"
    split_path = OUTPUT / "pilot_story_split.csv"
    old_metrics = OUTPUT / "pilot_model_metrics.csv"
    require([exposure, votes, communities, split_path, old_metrics])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    print("Constructing leakage-safe XGBoost features...", flush=True)
    data = construct_controlled_data(exposure, votes, communities)
    if data.user_control_leakage or data.cascade_control_leakage:
        raise DiagnosticError("Control-variable leakage audit failed")
    split = read_split(split_path)
    train_mask = np.asarray([split[int(story)] == "train" for story in data.story])
    test_mask = ~train_mask
    if set(np.unique(data.story[train_mask])) & set(np.unique(data.story[test_mask])):
        raise DiagnosticError("Story leakage between train and test")

    x = np.column_stack([data.columns[name] for name in FEATURES])
    if not np.all(np.isfinite(x)):
        raise DiagnosticError("Non-finite XGBoost feature")
    weighted_train_rate = float(
        np.sum(data.weight[train_mask] * data.y[train_mask]) / np.sum(data.weight[train_mask])
    )
    print("Fitting fixed-specification weighted XGBoost...", flush=True)
    model = xgboost_model(weighted_train_rate)
    model.fit(x[train_mask], data.y[train_mask], sample_weight=data.weight[train_mask])
    probability = model.predict_proba(x[test_mask])[:, 1]
    if not np.all(np.isfinite(probability)):
        raise DiagnosticError("Non-finite XGBoost probabilities")

    result = metrics(
        "XGBoost", "all_test", data.y[test_mask], probability, data.weight[test_mask]
    )
    metric_rows = read_existing_metrics(old_metrics)
    metric_rows.append(
        {
            "model": "XGBoost",
            "test_rows": result["rows"],
            "test_positives": result["positives"],
            "weighted_y_rate": result["weighted_y_rate"],
            "weighted_log_loss": result["weighted_log_loss"],
            "weighted_brier": result["weighted_brier"],
            "weighted_roc_auc": result["weighted_roc_auc"],
            "weighted_pr_auc": result["weighted_pr_auc"],
            "feature_set": ";".join(DISPLAY_NAMES[name] for name in FEATURES),
            "notes": "Fixed hyperparameters; sampling_weight; no test tuning or early stopping",
        }
    )
    atomic_csv(OUTPUT / "xgboost_metrics.csv", list(metric_rows[0]), metric_rows)

    print("Computing held-out SHAP values and plots...", flush=True)
    feature_names = [DISPLAY_NAMES[name] for name in FEATURES]
    shap_rows = save_shap_plots(model, x[test_mask], feature_names)
    write_case_study(
        DOCS / "digg_observational_case_study.md",
        metric_rows,
        shap_rows,
        int(np.sum(train_mask)),
        int(np.sum(test_mask)),
        sum(label == "train" for label in split.values()),
        sum(label == "test" for label in split.values()),
        weighted_train_rate,
        xgb.__version__,
        shap.__version__,
    )
    print("Completed XGBoost metrics, SHAP plots, and observational case study.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (DiagnosticError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
