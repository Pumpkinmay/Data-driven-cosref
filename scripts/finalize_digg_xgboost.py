#!/usr/bin/env python3
"""Nested story-only Platt calibration and final Digg XGBoost artifacts."""

from __future__ import annotations

import csv
import gzip
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GroupKFold

from diagnose_digg_pilot import (
    DiagnosticError, atomic_csv, metrics, sigmoid, weighted_logistic_irls,
)
from fit_digg_controlled_models import ControlledData, construct_controlled_data
from fit_digg_xgboost import DISPLAY_NAMES
from validate_digg_xgboost import (
    FEATURE_SETS, calibration_points, fit_xgb, matrix,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "digg"
SEED = 42
CLOSE_TOLERANCE = 0.01
MODELS = ("XGB_context", "XGB_full")


def internal_story_split(
    data: ControlledData,
    outer_train_stories: set[int],
    fold: int,
) -> tuple[set[int], set[int]]:
    positive_counts = Counter()
    for story, outcome in zip(data.story.astype(int), data.y.astype(int)):
        if story in outer_train_stories:
            positive_counts[story] += outcome
    ordered = sorted(outer_train_stories, key=lambda story: (positive_counts[story], story))
    strata = np.array_split(np.asarray(ordered, dtype=int), 5)
    quotas = (3, 3, 3, 3, 4)
    rng = random.Random(SEED * 100 + fold)
    calibration: set[int] = set()
    for stratum, quota in zip(strata, quotas):
        calibration.update(rng.sample(stratum.tolist(), quota))
    model_fit = outer_train_stories - calibration
    if len(model_fit) != 64 or len(calibration) != 16:
        raise DiagnosticError("Nested calibration split is not 64/16 stories")
    return model_fit, calibration


def fit_platt(
    probability: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, float, int, str]:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    logit = np.log(clipped / (1.0 - clipped))
    fitted = weighted_logistic_irls(
        logit.reshape(-1, 1), y, weight, 0.0, np.asarray([1.0]), 100, 1e-10
    )
    if not fitted.converged:
        raise DiagnosticError(f"Platt fit did not converge: {fitted.warning}")
    if fitted.coefficients[0] <= 0:
        raise DiagnosticError("Platt slope is non-positive and would change ranking direction")
    return fitted.intercept, float(fitted.coefficients[0]), fitted.iterations, fitted.warning


def row_metrics(
    model: str,
    calibration: str,
    fold: str,
    y: np.ndarray,
    probability: np.ndarray,
    weight: np.ndarray,
) -> dict[str, object]:
    result = metrics(model, calibration, y, probability, weight)
    return {
        "model": model,
        "calibration": calibration,
        "fold": fold,
        "model_fit_stories": "",
        "calibration_stories": "",
        "test_stories": "",
        "story_overlap": 0,
        "rows": result["rows"],
        "positives": result["positives"],
        "weighted_positive_prevalence": result["weighted_y_rate"],
        "weighted_log_loss": result["weighted_log_loss"],
        "weighted_brier": result["weighted_brier"],
        "weighted_roc_auc": result["weighted_roc_auc"],
        "weighted_pr_auc": result["weighted_pr_auc"],
        "platt_intercept": "",
        "platt_slope": "",
        "platt_converged": "",
        "platt_iterations": "",
        "platt_warning": "",
    }


def nested_calibration(
    data: ControlledData,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dummy = np.zeros((len(data.story), 1), dtype=float)
    oof = {
        (model, calibration): np.full(len(data.story), np.nan, dtype=float)
        for model in MODELS for calibration in ("uncalibrated", "platt")
    }
    rows: list[dict[str, object]] = []
    for fold, (outer_train_index, test_index) in enumerate(
        splitter.split(dummy, data.y, groups=data.story), start=1
    ):
        outer_train_stories = set(data.story[outer_train_index].astype(int))
        test_stories = set(data.story[test_index].astype(int))
        model_fit_stories, calibration_stories = internal_story_split(
            data, outer_train_stories, fold
        )
        overlap = (
            (model_fit_stories & calibration_stories)
            | (model_fit_stories & test_stories)
            | (calibration_stories & test_stories)
        )
        if overlap:
            raise DiagnosticError(f"Nested story leakage in fold {fold}")
        fit_mask = np.asarray([int(story) in model_fit_stories for story in data.story])
        calibration_mask = np.asarray(
            [int(story) in calibration_stories for story in data.story]
        )
        for model_name in MODELS:
            x = matrix(data, FEATURE_SETS[model_name])
            fitted = fit_xgb(x[fit_mask], data.y[fit_mask], data.weight[fit_mask])
            calibration_probability = fitted.predict_proba(x[calibration_mask])[:, 1]
            intercept, slope, iterations, warning = fit_platt(
                calibration_probability,
                data.y[calibration_mask],
                data.weight[calibration_mask],
            )
            raw = np.clip(fitted.predict_proba(x[test_index])[:, 1], 1e-12, 1 - 1e-12)
            raw_logit = np.log(raw / (1.0 - raw))
            calibrated = sigmoid(intercept + slope * raw_logit)
            oof[(model_name, "uncalibrated")][test_index] = raw
            oof[(model_name, "platt")][test_index] = calibrated
            for label, probability in (("uncalibrated", raw), ("platt", calibrated)):
                row = row_metrics(
                    model_name, label, str(fold), data.y[test_index], probability,
                    data.weight[test_index],
                )
                row.update(
                    {
                        "model_fit_stories": len(model_fit_stories),
                        "calibration_stories": len(calibration_stories),
                        "test_stories": len(test_stories),
                        "story_overlap": len(overlap),
                    }
                )
                if label == "platt":
                    row.update(
                        {
                            "platt_intercept": intercept,
                            "platt_slope": slope,
                            "platt_converged": 1,
                            "platt_iterations": iterations,
                            "platt_warning": warning,
                        }
                    )
                rows.append(row)
    if any(np.any(~np.isfinite(values)) for values in oof.values()):
        raise DiagnosticError("Missing nested out-of-fold calibrated predictions")
    return rows, oof


def add_summaries(
    rows: list[dict[str, object]],
    data: ControlledData,
    oof: dict[tuple[str, str], np.ndarray],
) -> None:
    metric_fields = (
        "weighted_positive_prevalence", "weighted_log_loss", "weighted_brier",
        "weighted_roc_auc", "weighted_pr_auc",
    )
    for model_name in MODELS:
        for calibration in ("uncalibrated", "platt"):
            fold_rows = [
                row for row in rows
                if row["model"] == model_name
                and row["calibration"] == calibration
                and str(row["fold"]).isdigit()
            ]
            for label, function in (
                ("mean", np.mean),
                ("std", lambda values: np.std(values, ddof=1)),
            ):
                summary = row_metrics(
                    model_name, calibration, label,
                    np.asarray([0.0, 1.0]), np.asarray([0.5, 0.5]), np.ones(2),
                )
                summary.update({"rows": "", "positives": ""})
                for field in metric_fields:
                    summary[field] = float(
                        function([float(row[field]) for row in fold_rows])
                    )
                rows.append(summary)
            pooled = row_metrics(
                model_name, calibration, "pooled_oof", data.y,
                oof[(model_name, calibration)], data.weight,
            )
            rows.append(pooled)


def final_decision(rows: list[dict[str, object]]) -> tuple[str, float, float]:
    lookup = {
        (str(row["model"]), str(row["calibration"]), str(row["fold"])): row
        for row in rows
    }
    context = lookup[("XGB_context", "platt", "mean")]
    full = lookup[("XGB_full", "platt", "mean")]
    log_loss_ratio = float(full["weighted_log_loss"]) / float(context["weighted_log_loss"])
    brier_ratio = float(full["weighted_brier"]) / float(context["weighted_brier"])
    if log_loss_ratio <= 1.0 + CLOSE_TOLERANCE and brier_ratio <= 1.0 + CLOSE_TOLERANCE:
        return "calibrated XGB_full", log_loss_ratio, brier_ratio
    return "dual: calibrated XGB_context for probability; XGB_full for ranking", log_loss_ratio, brier_ratio


def calibration_plot(
    path: Path,
    data: ControlledData,
    oof: dict[tuple[str, str], np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, model_name in zip(axes, MODELS):
        raw = oof[(model_name, "uncalibrated")]
        calibrated = oof[(model_name, "platt")]
        raw_x, raw_y = calibration_points(raw, raw, data.y, data.weight)
        calibrated_x, calibrated_y = calibration_points(
            raw, calibrated, data.y, data.weight
        )
        maximum = max(
            float(np.max(raw_x)), float(np.max(raw_y)),
            float(np.max(calibrated_x)), float(np.max(calibrated_y)),
        )
        axis.plot([0, maximum], [0, maximum], color="0.45", linestyle="--", label="ideal")
        axis.plot(raw_x, raw_y, marker="o", linewidth=1.6, label="uncalibrated")
        axis.plot(calibrated_x, calibrated_y, marker="s", linewidth=1.6, label="Platt")
        axis.set_title(model_name)
        axis.set_xlabel("Weighted mean predicted probability")
        axis.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))
        axis.legend(frameon=False)
    axes[0].set_ylabel("Weighted observed adoption rate")
    fig.suptitle("Nested story-only Platt calibration on out-of-fold predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def read_saved_split(path: Path) -> dict[int, str]:
    result = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["story_id"])] = row["split"]
    if Counter(result.values()) != {"train": 80, "test": 20}:
        raise DiagnosticError("Saved held-out split is not 80/20")
    return result


def update_shap_summary(data: ControlledData, split_path: Path) -> None:
    split = read_saved_split(split_path)
    train = np.asarray([split[int(story)] == "train" for story in data.story])
    test = ~train
    features = FEATURE_SETS["XGB_full"]
    x = matrix(data, features)
    fitted = fit_xgb(x[train], data.y[train], data.weight[train])
    rng = np.random.default_rng(SEED)
    test_indices = np.flatnonzero(test)
    sample = np.sort(rng.choice(test_indices, size=min(20_000, len(test_indices)), replace=False))
    explanation = shap.TreeExplainer(fitted)(x[sample])
    names = [DISPLAY_NAMES[name] for name in features]
    plt.figure(figsize=(9, 5.5))
    shap.summary_plot(explanation.values, x[sample], feature_names=names, show=False)
    plt.title("Digg XGB_full SHAP: predictive association, not causation", pad=12)
    plt.tight_layout()
    plt.savefig(OUTPUT / "xgb_shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    exposure = PROCESSED / "digg_exposure_pilot.csv.gz"
    votes = PROCESSED / "digg_votes_clean.csv.gz"
    communities = PROCESSED / "digg_communities.csv"
    split_path = OUTPUT / "pilot_story_split.csv"
    for path in (exposure, votes, communities, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Constructing leakage-safe context and exposure features...", flush=True)
    data = construct_controlled_data(exposure, votes, communities)
    print("Running nested story-only calibration for context and full...", flush=True)
    rows, oof = nested_calibration(data)
    add_summaries(rows, data, oof)
    decision, log_loss_ratio, brier_ratio = final_decision(rows)
    for row in rows:
        row["final_model_decision"] = decision if row["fold"] == "mean" else ""
        row["full_to_context_log_loss_ratio"] = log_loss_ratio if row["fold"] == "mean" else ""
        row["full_to_context_brier_ratio"] = brier_ratio if row["fold"] == "mean" else ""
    atomic_csv(OUTPUT / "final_model_metrics.csv", list(rows[0]), rows)
    calibration_plot(OUTPUT / "final_calibration.png", data, oof)
    print("Refreshing the held-out XGB_full SHAP summary...", flush=True)
    update_shap_summary(data, split_path)
    print(f"Final model decision: {decision}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (DiagnosticError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
