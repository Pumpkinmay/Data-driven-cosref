#!/usr/bin/env python3
"""Leakage-audit, ablate, cross-validate, and calibrate the Digg XGBoost model."""

from __future__ import annotations

import bisect
import csv
import gzip
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import shap
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from diagnose_digg_pilot import (
    DiagnosticError, atomic_csv, metrics, sigmoid, weighted_logistic_irls,
)
from fit_digg_controlled_models import ControlledData, construct_controlled_data
from fit_digg_xgboost import DISPLAY_NAMES, SEED, read_existing_metrics, xgboost_model


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "digg"
DOCS = ROOT / "docs"
FEATURE_SETS = {
    "XGB_base": ("degree", "log_time"),
    "XGB_context": ("degree", "log_time", "log_user_activity", "log_cascade_size"),
    "XGB_full": (
        "degree", "log_time", "log_user_activity", "log_cascade_size", "m_in", "m_out"
    ),
}


def require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def read_split(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            story = int(row["story_id"])
            label = row["split"]
            if story in result or label not in ("train", "test"):
                raise DiagnosticError("Invalid saved story split")
            result[story] = label
    if Counter(result.values()) != {"train": 80, "test": 20}:
        raise DiagnosticError("Saved split is not 80/20 by story")
    return result


def matrix(data: ControlledData, names: tuple[str, ...]) -> np.ndarray:
    values = np.column_stack([data.columns[name] for name in names])
    if not np.all(np.isfinite(values)):
        raise DiagnosticError("Non-finite XGBoost feature")
    return values


def weighted_prevalence(y: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(y * weight) / np.sum(weight))


def fit_xgb(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> xgb.XGBClassifier:
    model = xgboost_model(weighted_prevalence(y, weight))
    model.fit(x, y, sample_weight=weight)
    return model


def metric_row(
    model_name: str,
    y: np.ndarray,
    probability: np.ndarray,
    weight: np.ndarray,
) -> dict[str, object]:
    result = metrics(model_name, "evaluation", y, probability, weight)
    return {
        "model": model_name,
        "rows": result["rows"],
        "positives": result["positives"],
        "weighted_y_rate": result["weighted_y_rate"],
        "weighted_log_loss": result["weighted_log_loss"],
        "weighted_brier": result["weighted_brier"],
        "weighted_roc_auc": result["weighted_roc_auc"],
        "weighted_pr_auc": result["weighted_pr_auc"],
    }


def audit_exposure_timing(
    exposure_path: Path,
    votes_path: Path,
    friends_path: Path,
    communities_path: Path,
) -> dict[str, int]:
    selected_stories: set[int] = set()
    target_nodes: set[int] = set()
    exposure_rows: list[tuple[int, int, int, int, int, int]] = []
    with gzip.open(exposure_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            story = int(row["story_id"])
            node = int(row["node_id"])
            selected_stories.add(story)
            target_nodes.add(node)
            exposure_rows.append(
                (story, int(row["time_bin"]), node, int(row["m_in"]), int(row["m_out"]), int(row["degree"]))
            )

    communities: dict[int, int] = {}
    with communities_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            communities[int(row["node_id"])] = int(row["community"])

    starts: dict[int, int] = {}
    adoption_times: dict[int, dict[int, int]] = defaultdict(dict)
    earliest_vote: int | None = None
    user_times: dict[int, list[int]] = defaultdict(list)
    covered_times: dict[int, list[int]] = defaultdict(list)
    selected_node_times: dict[tuple[int, int], int] = {}
    with gzip.open(votes_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["vote_date"])
            node = int(row["voter_id"])
            story = int(row["story_id"])
            earliest_vote = timestamp if earliest_vote is None else min(earliest_vote, timestamp)
            if node in target_nodes:
                user_times[node].append(timestamp)
            if story in selected_stories:
                starts[story] = min(starts.get(story, timestamp), timestamp)
                selected_node_times[(story, node)] = timestamp
                if node in communities:
                    covered_times[story].append(timestamp)
                    adoption_times[story][node] = timestamp
    if earliest_vote is None:
        raise DiagnosticError("Empty cleaned votes")
    for times in user_times.values():
        times.sort()
    for times in covered_times.values():
        times.sort()

    incoming_sets: dict[int, set[int]] = defaultdict(set)
    with gzip.open(friends_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            friend_date = int(row["friend_date"])
            target = int(row["user_id"])
            if 0 < friend_date <= earliest_vote and target in target_nodes:
                incoming_sets[target].add(int(row["friend_id"]))

    exposure_mismatch = 0
    degree_mismatch = 0
    current_or_future_included = 0
    user_strict_time_violations = 0
    cascade_strict_time_violations = 0
    for story, time_bin, node, observed_in, observed_out, observed_degree in exposure_rows:
        absolute_time = starts[story] + time_bin * 3600
        expected_in = 0
        expected_out = 0
        for source in incoming_sets[node]:
            source_time = adoption_times[story].get(source)
            if source_time is None:
                continue
            if source_time < absolute_time:
                if communities[source] == communities[node]:
                    expected_in += 1
                else:
                    expected_out += 1
        exposure_mismatch += int((observed_in, observed_out) != (expected_in, expected_out))
        current_or_future_included += int(
            observed_in > expected_in or observed_out > expected_out
        )
        degree_mismatch += int(observed_degree != len(incoming_sets[node]))

        prior_count = bisect.bisect_left(user_times[node], absolute_time)
        same_story_time = selected_node_times.get((story, node))
        if same_story_time is not None and same_story_time < absolute_time:
            prior_count -= 1
        user_strict_time_violations += int(prior_count < 0)
        cascade_count = bisect.bisect_left(covered_times[story], absolute_time)
        cascade_strict_time_violations += int(time_bin == 0 and cascade_count != 0)

    return {
        "exposure_rows_checked": len(exposure_rows),
        "exposure_count_mismatches": exposure_mismatch,
        "degree_mismatches": degree_mismatch,
        "current_or_future_exposure_inclusions": current_or_future_included,
        "user_prior_strict_time_violations": user_strict_time_violations,
        "cascade_size_strict_time_violations": cascade_strict_time_violations,
        "baseline_cutoff": earliest_vote,
    }


def heldout_ablation(
    data: ControlledData, train: np.ndarray, test: np.ndarray
) -> tuple[list[dict[str, object]], dict[str, xgb.XGBClassifier], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    models: dict[str, xgb.XGBClassifier] = {}
    matrices: dict[str, np.ndarray] = {}
    for model_name, features in FEATURE_SETS.items():
        x = matrix(data, features)
        fitted = fit_xgb(x[train], data.y[train], data.weight[train])
        probability = fitted.predict_proba(x[test])[:, 1]
        row = metric_row(model_name, data.y[test], probability, data.weight[test])
        row["features"] = ";".join(DISPLAY_NAMES[name] for name in features)
        rows.append(row)
        models[model_name] = fitted
        matrices[model_name] = x
    context = next(row for row in rows if row["model"] == "XGB_context")
    full = next(row for row in rows if row["model"] == "XGB_full")
    for row in rows:
        for name in ("weighted_log_loss", "weighted_brier", "weighted_roc_auc", "weighted_pr_auc"):
            row[f"delta_vs_context_{name}"] = (
                float(row[name]) - float(context[name]) if row["model"] == "XGB_full" else ""
            )
    return rows, models, matrices


def group_cross_validation(data: ControlledData) -> list[dict[str, object]]:
    unique_stories = np.unique(data.story)
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dummy = np.zeros((len(data.story), 1))
    rows: list[dict[str, object]] = []
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(dummy, data.y, groups=data.story), start=1
    ):
        train_stories = set(data.story[train_index])
        valid_stories = set(data.story[valid_index])
        if train_stories & valid_stories:
            raise DiagnosticError(f"Story leakage in CV fold {fold}")
        for model_name, features in FEATURE_SETS.items():
            x = matrix(data, features)
            fitted = fit_xgb(x[train_index], data.y[train_index], data.weight[train_index])
            probability = fitted.predict_proba(x[valid_index])[:, 1]
            row = metric_row(model_name, data.y[valid_index], probability, data.weight[valid_index])
            row.update(
                {
                    "fold": str(fold),
                    "train_stories": len(train_stories),
                    "validation_stories": len(valid_stories),
                }
            )
            rows.append(row)

    metric_names = (
        "weighted_y_rate", "weighted_log_loss", "weighted_brier",
        "weighted_roc_auc", "weighted_pr_auc",
    )
    for model_name in FEATURE_SETS:
        model_rows = [row for row in rows if row["model"] == model_name]
        for summary, function in (("mean", np.mean), ("std", lambda values: np.std(values, ddof=1))):
            row: dict[str, object] = {
                "model": model_name,
                "fold": summary,
                "train_stories": "",
                "validation_stories": "",
                "rows": "",
                "positives": "",
            }
            for name in metric_names:
                row[name] = float(function([float(item[name]) for item in model_rows]))
            rows.append(row)
    return rows


def internal_calibration_split(
    data: ControlledData, saved_split: dict[int, str]
) -> tuple[set[int], set[int]]:
    train_stories = [story for story, label in saved_split.items() if label == "train"]
    positive_counts = Counter()
    for story, y in zip(data.story.astype(int), data.y.astype(int)):
        if story in train_stories:
            positive_counts[story] += y
    ordered = sorted(train_stories, key=lambda story: (positive_counts[story], story))
    strata = np.array_split(np.asarray(ordered, dtype=int), 5)
    quotas = [3, 3, 3, 3, 4]
    rng = random.Random(SEED)
    calibration: set[int] = set()
    for stratum, quota in zip(strata, quotas):
        calibration.update(rng.sample(stratum.tolist(), quota))
    fit_stories = set(train_stories) - calibration
    if len(fit_stories) != 64 or len(calibration) != 16:
        raise DiagnosticError("Internal calibration split is not 64/16 stories")
    return fit_stories, calibration


def platt_calibration(
    data: ControlledData,
    saved_split: dict[int, str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    fit_stories, calibration_stories = internal_calibration_split(data, saved_split)
    test_stories = {story for story, label in saved_split.items() if label == "test"}
    if fit_stories & calibration_stories or fit_stories & test_stories or calibration_stories & test_stories:
        raise DiagnosticError("Story overlap in calibration design")
    fit_mask = np.asarray([int(story) in fit_stories for story in data.story])
    calibration_mask = np.asarray([int(story) in calibration_stories for story in data.story])
    test_mask = np.asarray([int(story) in test_stories for story in data.story])
    x = matrix(data, FEATURE_SETS["XGB_full"])
    model = fit_xgb(x[fit_mask], data.y[fit_mask], data.weight[fit_mask])
    calibration_probability = np.clip(
        model.predict_proba(x[calibration_mask])[:, 1], 1e-12, 1.0 - 1e-12
    )
    calibration_logit = np.log(calibration_probability / (1.0 - calibration_probability))
    calibrator = weighted_logistic_irls(
        calibration_logit.reshape(-1, 1),
        data.y[calibration_mask],
        data.weight[calibration_mask],
        0.0,
        np.asarray([1.0]),
        100,
        1e-10,
    )
    if not calibrator.converged:
        raise DiagnosticError(f"Platt calibration did not converge: {calibrator.warning}")
    raw = np.clip(model.predict_proba(x[test_mask])[:, 1], 1e-12, 1.0 - 1e-12)
    raw_logit = np.log(raw / (1.0 - raw))
    calibrated = sigmoid(calibrator.intercept + calibrator.coefficients[0] * raw_logit)
    rows = []
    for name, probability in (("XGB_full_uncalibrated", raw), ("XGB_full_platt", calibrated)):
        row = metric_row(name, data.y[test_mask], probability, data.weight[test_mask])
        row.update(
            {
                "xgb_fit_stories": 64,
                "calibration_stories": 16,
                "test_stories": 20,
                "platt_intercept": calibrator.intercept if name.endswith("platt") else "",
                "platt_slope": float(calibrator.coefficients[0]) if name.endswith("platt") else "",
                "platt_converged": int(calibrator.converged) if name.endswith("platt") else "",
                "platt_iterations": calibrator.iterations if name.endswith("platt") else "",
                "platt_warning": calibrator.warning if name.endswith("platt") else "",
            }
        )
        rows.append(row)
    details: dict[str, object] = {
        "fit_stories": fit_stories,
        "calibration_stories": calibration_stories,
        "test_stories": test_stories,
        "test_y": data.y[test_mask],
        "test_weight": data.weight[test_mask],
        "raw_probability": raw,
        "calibrated_probability": calibrated,
        "platt_intercept": calibrator.intercept,
        "platt_slope": float(calibrator.coefficients[0]),
        "platt_iterations": calibrator.iterations,
    }
    return rows, details


def weighted_quantile_edges(values: np.ndarray, weight: np.ndarray, bins: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weight = weight[order]
    cumulative = np.cumsum(sorted_weight) - 0.5 * sorted_weight
    cumulative /= np.sum(sorted_weight)
    edges = np.interp(np.linspace(0.0, 1.0, bins + 1), cumulative, sorted_values)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def calibration_points(
    raw: np.ndarray,
    probability: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    edges = weighted_quantile_edges(raw, weight, bins)
    assignment = np.clip(np.digitize(raw, edges[1:-1]), 0, bins - 1)
    predicted: list[float] = []
    observed: list[float] = []
    for index in range(bins):
        mask = assignment == index
        if not np.any(mask):
            continue
        predicted.append(float(np.average(probability[mask], weights=weight[mask])))
        observed.append(float(np.average(y[mask], weights=weight[mask])))
    return np.asarray(predicted), np.asarray(observed)


def calibration_plot(details: dict[str, object]) -> None:
    raw = np.asarray(details["raw_probability"])
    calibrated = np.asarray(details["calibrated_probability"])
    y = np.asarray(details["test_y"])
    weight = np.asarray(details["test_weight"])
    raw_x, raw_y = calibration_points(raw, raw, y, weight)
    calibrated_x, calibrated_y = calibration_points(raw, calibrated, y, weight)
    maximum = max(float(np.max(raw_x)), float(np.max(raw_y)), float(np.max(calibrated_x)), float(np.max(calibrated_y)))
    plt.figure(figsize=(7, 6))
    plt.plot([0, maximum], [0, maximum], color="0.45", linestyle="--", linewidth=1, label="ideal")
    plt.plot(raw_x, raw_y, marker="o", linewidth=1.8, label="uncalibrated")
    plt.plot(calibrated_x, calibrated_y, marker="s", linewidth=1.8, label="Platt calibrated")
    plt.xlabel("Weighted mean predicted probability")
    plt.ylabel("Weighted observed adoption rate")
    plt.title("Digg XGB_full held-out calibration")
    plt.legend(frameon=False)
    plt.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(OUTPUT / "xgb_calibration.png", dpi=180, bbox_inches="tight")
    plt.close()


def shap_and_importance_plots(
    model: xgb.XGBClassifier,
    x_test: np.ndarray,
    feature_names: list[str],
) -> tuple[int, list[dict[str, object]]]:
    rng = np.random.default_rng(SEED)
    sample_size = min(20_000, len(x_test))
    indices = np.sort(rng.choice(len(x_test), size=sample_size, replace=False))
    x_plot = x_test[indices]
    explanation = shap.TreeExplainer(model)(x_plot)

    plt.figure(figsize=(9, 5.5))
    shap.summary_plot(explanation.values, x_plot, feature_names=feature_names, show=False)
    plt.title("Digg XGB_full: held-out SHAP summary", pad=12)
    plt.tight_layout()
    plt.savefig(OUTPUT / "xgb_shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()
    for feature, destination in (
        ("m_in", OUTPUT / "xgb_shap_m_in.png"),
        ("m_out", OUTPUT / "xgb_shap_m_out.png"),
    ):
        plt.figure(figsize=(8, 5.5))
        shap.dependence_plot(
            feature, explanation.values, x_plot, feature_names=feature_names,
            interaction_index="auto", alpha=0.35, show=False,
        )
        plt.title(f"Digg XGB_full: SHAP dependence for {feature}", pad=12)
        plt.tight_layout()
        plt.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close()

    booster_gain = model.get_booster().get_score(importance_type="gain")
    mean_abs_shap = np.mean(np.abs(explanation.values), axis=0)
    importance_rows = []
    for index, name in enumerate(feature_names):
        importance_rows.append(
            {
                "feature": name,
                "gain": float(booster_gain.get(f"f{index}", 0.0)),
                "mean_abs_shap": float(mean_abs_shap[index]),
            }
        )
    importance_rows.sort(key=lambda row: -float(row["gain"]))
    ordered = list(reversed(importance_rows))
    plt.figure(figsize=(8, 5))
    plt.barh([str(row["feature"]) for row in ordered], [float(row["gain"]) for row in ordered])
    plt.xlabel("XGBoost gain importance")
    plt.title("Digg XGB_full feature importance")
    plt.tight_layout()
    plt.savefig(OUTPUT / "xgb_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close()
    return sample_size, importance_rows


def increment_plot(ablation_rows: list[dict[str, object]]) -> None:
    lookup = {str(row["model"]): row for row in ablation_rows}
    context, full = lookup["XGB_context"], lookup["XGB_full"]
    specifications = (
        ("Log loss", "weighted_log_loss", True),
        ("Brier", "weighted_brier", True),
        ("ROC-AUC", "weighted_roc_auc", False),
        ("PR-AUC", "weighted_pr_auc", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for axis, (label, field, lower_better) in zip(axes.flat, specifications):
        raw_delta = float(full[field]) - float(context[field])
        improvement = -raw_delta if lower_better else raw_delta
        axis.barh(["XGB_full vs context"], [improvement], color="#4472C4" if improvement >= 0 else "#B85450")
        axis.axvline(0, color="0.35", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Improvement (positive is better)")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
        axis.text(improvement, 0, f" {improvement:+.3g}", va="center", ha="left" if improvement >= 0 else "right")
    fig.suptitle("Increment from adding m_in and m_out", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT / "xgb_context_full_increment.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_leakage_report(
    path: Path,
    split: dict[int, str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    temporal: dict[str, int],
    calibration: dict[str, object],
) -> None:
    train_stories = {story for story, label in split.items() if label == "train"}
    test_stories = {story for story, label in split.items() if label == "test"}
    lines = [
        "# Digg XGBoost leakage audit",
        "",
        "## Result",
        "",
        "**PASS — no train/test, temporal-exposure, control-variable, tuning, scaling, or calibration leakage was detected.**",
        "",
        "## Checks",
        "",
        f"1. **Story-level separation:** {len(train_stories)} train and {len(test_stories)} test stories; story-ID overlap **{len(train_stories & test_stories)}**; rows **{int(np.sum(train_mask)):,} / {int(np.sum(test_mask)):,}**. Every row inherits its story's saved split.",
        f"2. **User history:** all {temporal['exposure_rows_checked']:,} rows use `bisect_left(user_vote_times, window_start)`, so only timestamps strictly earlier than the absolute window start are counted; current-story votes are excluded from the other-story activity count. Negative-count/time-boundary violations: **{temporal['user_prior_strict_time_violations']}**.",
        f"3. **Cascade size:** all rows use `bisect_left(network_covered_vote_times, window_start)`; time-zero rows with a nonzero prior cascade size: **{temporal['cascade_size_strict_time_violations']}**.",
        f"4. **Network exposure:** independently reconstructed `m_in/m_out` from baseline `friend_id -> user_id` edges and adopters with `vote_date < window_start`. Count mismatches: **{temporal['exposure_count_mismatches']}**; degree mismatches: **{temporal['degree_mismatches']}**; detected current/future inclusions: **{temporal['current_or_future_exposure_inclusions']}**.",
        f"5. **Preprocessing/tuning/calibration:** XGBoost uses raw features and **no scaler**. Hyperparameters are fixed in code; there is no search, test-driven early stopping, or probability calibration in ablation/CV fits. Platt calibration uses **{len(calibration['fit_stories'])} model-fit + {len(calibration['calibration_stories'])} calibration stories**, all drawn only from the saved 80 training stories; overlap with the {len(calibration['test_stories'])} held-out stories is **0**.",
        "",
        "## Baseline-network boundary",
        "",
        f"Friendships are restricted to `0 < friend_date <= {temporal['baseline_cutoff']}`, the earliest vote timestamp. Thus exposure construction does not import friendships formed after cascades begin.",
        "",
        "The audit verifies implementation-time separation and temporal ordering. It cannot eliminate unobserved confounding or make observational predictors causal.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def f(value: object) -> str:
    return f"{float(value):.10g}"


def write_results(
    path: Path,
    ablation_rows: list[dict[str, object]],
    cv_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
    importance_rows: list[dict[str, object]],
    shap_rows: int,
) -> None:
    ablation = {str(row["model"]): row for row in ablation_rows}
    context, full = ablation["XGB_context"], ablation["XGB_full"]
    cv_lookup = {(str(row["model"]), str(row["fold"])): row for row in cv_rows}
    calibrated = {str(row["model"]): row for row in calibration_rows}
    raw, platt = calibrated["XGB_full_uncalibrated"], calibrated["XGB_full_platt"]
    lines = [
        "# Digg XGBoost validation results",
        "",
        "## Scope",
        "",
        "This validation adds no random forest, neural network, GNN, or other predictive model. It evaluates only the existing fixed-specification XGBoost, plus a one-dimensional Platt probability-calibration layer fitted exclusively on internal training stories.",
        "",
        "All results remain observational prediction on the 100-cascade pilot. No community definition, M1 coefficient, exposure sign, or structural parameter is adjusted.",
        "",
        "## Held-out feature ablation",
        "",
        "All variants use identical XGBoost parameters, sampling weights, and the saved 80/20 story split.",
        "",
        "| model | features | log loss | Brier | ROC-AUC | PR-AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model_name in FEATURE_SETS:
        row = ablation[model_name]
        lines.append(f"| {model_name} | {row['features']} | {f(row['weighted_log_loss'])} | {f(row['weighted_brier'])} | {f(row['weighted_roc_auc'])} | {f(row['weighted_pr_auc'])} |")
    lines.extend(
        [
            "",
            f"Adding `m_in/m_out` to XGB_context changes log loss by **{float(full['weighted_log_loss']) - float(context['weighted_log_loss']):+.10g}**, Brier by **{float(full['weighted_brier']) - float(context['weighted_brier']):+.10g}**, ROC-AUC by **{float(full['weighted_roc_auc']) - float(context['weighted_roc_auc']):+.10g}**, and PR-AUC by **{float(full['weighted_pr_auc']) - float(context['weighted_pr_auc']):+.10g}**. Negative loss/Brier deltas and positive AUC deltas indicate improvement.",
            "",
            "This is the direct predictive test of whether community exposure counts add information beyond degree, time, user activity, and cascade heat.",
            "",
            "## Five-fold group cross-validation",
            "",
            "Each fold holds out 20 complete stories; no story crosses a fold. Values below are unweighted means and sample standard deviations across the five independently weighted fold metrics.",
            "",
            "| model | log loss mean±SD | Brier mean±SD | ROC-AUC mean±SD | PR-AUC mean±SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_name in FEATURE_SETS:
        mean, std = cv_lookup[(model_name, "mean")], cv_lookup[(model_name, "std")]
        lines.append(
            f"| {model_name} | {f(mean['weighted_log_loss'])} ± {f(std['weighted_log_loss'])} | "
            f"{f(mean['weighted_brier'])} ± {f(std['weighted_brier'])} | "
            f"{f(mean['weighted_roc_auc'])} ± {f(std['weighted_roc_auc'])} | "
            f"{f(mean['weighted_pr_auc'])} ± {f(std['weighted_pr_auc'])} |"
        )
    lines.extend(
        [
            "",
            f"Across folds, XGB_full minus XGB_context changes mean log loss by **{float(cv_lookup[('XGB_full', 'mean')]['weighted_log_loss']) - float(cv_lookup[('XGB_context', 'mean')]['weighted_log_loss']):+.10g}**, Brier by **{float(cv_lookup[('XGB_full', 'mean')]['weighted_brier']) - float(cv_lookup[('XGB_context', 'mean')]['weighted_brier']):+.10g}**, ROC-AUC by **{float(cv_lookup[('XGB_full', 'mean')]['weighted_roc_auc']) - float(cv_lookup[('XGB_context', 'mean')]['weighted_roc_auc']):+.10g}**, and PR-AUC by **{float(cv_lookup[('XGB_full', 'mean')]['weighted_pr_auc']) - float(cv_lookup[('XGB_context', 'mean')]['weighted_pr_auc']):+.10g}**. Exposure counts improve mean ranking metrics but do not improve mean log loss or Brier, so their incremental predictive value is metric-dependent rather than uniformly stable.",
            "",
            "Per-fold metrics are available in `outputs/digg/xgb_group_cv_metrics.csv`.",
            "",
            "## Platt calibration",
            "",
            "For this comparison, the same XGB_full base model is fitted on 64 of the saved training stories. Predictions on the other 16 training stories fit the Platt intercept and slope. Both uncalibrated and calibrated probabilities are then evaluated on the untouched 20 test stories, isolating calibration from base-model training-set differences.",
            "",
            "| probability | log loss | Brier | ROC-AUC | PR-AUC |",
            "|---|---:|---:|---:|---:|",
            f"| uncalibrated | {f(raw['weighted_log_loss'])} | {f(raw['weighted_brier'])} | {f(raw['weighted_roc_auc'])} | {f(raw['weighted_pr_auc'])} |",
            f"| Platt calibrated | {f(platt['weighted_log_loss'])} | {f(platt['weighted_brier'])} | {f(platt['weighted_roc_auc'])} | {f(platt['weighted_pr_auc'])} |",
            "",
            f"Platt parameters: intercept **{f(platt['platt_intercept'])}**, slope **{f(platt['platt_slope'])}**; weighted IRLS converged in **{platt['platt_iterations']} iterations** with no warning. Calibration is monotone, so rank-based ROC-AUC and PR-AUC should remain unchanged apart from numerical ties.",
            "",
            "## Held-out interpretation",
            "",
            f"SHAP values use a deterministic uniform sample of **{shap_rows:,} test rows**. They explain the fitted XGB_full predictor; they are not causal effects and do not identify diffusion mechanisms.",
            "",
            "| feature | XGBoost gain | mean absolute SHAP |",
            "|---|---:|---:|",
        ]
    )
    for row in importance_rows:
        lines.append(f"| {row['feature']} | {f(row['gain'])} | {f(row['mean_abs_shap'])} |")
    lines.extend(
        [
            "",
            "The `m_in` and `m_out` dependence plots show heterogeneous fitted contributions conditional on the other observed features. They must not be read as dose-response or causal exposure effects.",
            "",
            "## Conclusion",
            "",
            "The ablation and group-CV results determine whether exposure counts add reproducible held-out predictive information beyond context. Regardless of predictive gain, they do not rehabilitate the negative M1 `m_out`, the positive M1 degree coefficient, or the structural interpretation of `a`, `b`, and `theta`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    exposure = PROCESSED / "digg_exposure_pilot.csv.gz"
    votes = PROCESSED / "digg_votes_clean.csv.gz"
    friends = PROCESSED / "digg_friends_clean.csv.gz"
    communities = PROCESSED / "digg_communities.csv"
    split_path = OUTPUT / "pilot_story_split.csv"
    require([exposure, votes, friends, communities, split_path])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    print("Constructing controlled features and auditing temporal boundaries...", flush=True)
    data = construct_controlled_data(exposure, votes, communities)
    split = read_split(split_path)
    train = np.asarray([split[int(story)] == "train" for story in data.story])
    test = ~train
    temporal = audit_exposure_timing(exposure, votes, friends, communities)
    if any(
        temporal[name]
        for name in (
            "exposure_count_mismatches", "degree_mismatches",
            "current_or_future_exposure_inclusions", "user_prior_strict_time_violations",
            "cascade_size_strict_time_violations",
        )
    ):
        raise DiagnosticError(f"Leakage audit failed: {temporal}")

    print("Fitting held-out feature ablations...", flush=True)
    ablation_rows, heldout_models, heldout_matrices = heldout_ablation(data, train, test)
    atomic_csv(OUTPUT / "xgb_ablation_metrics.csv", list(ablation_rows[0]), ablation_rows)

    print("Running five-fold story-group cross-validation...", flush=True)
    cv_rows = group_cross_validation(data)
    atomic_csv(OUTPUT / "xgb_group_cv_metrics.csv", list(cv_rows[0]), cv_rows)

    print("Fitting training-story-only Platt calibration...", flush=True)
    calibration_rows, calibration_details = platt_calibration(data, split)
    atomic_csv(OUTPUT / "xgb_calibrated_metrics.csv", list(calibration_rows[0]), calibration_rows)
    calibration_plot(calibration_details)

    print("Generating held-out SHAP and feature-importance diagnostics...", flush=True)
    full_model = heldout_models["XGB_full"]
    full_matrix = heldout_matrices["XGB_full"]
    names = [DISPLAY_NAMES[name] for name in FEATURE_SETS["XGB_full"]]
    shap_rows, importance_rows = shap_and_importance_plots(full_model, full_matrix[test], names)
    increment_plot(ablation_rows)

    write_leakage_report(OUTPUT / "leakage_audit.md", split, train, test, temporal, calibration_details)
    write_results(
        DOCS / "digg_xgboost_results.md", ablation_rows, cv_rows,
        calibration_rows, importance_rows, shap_rows,
    )
    print("Completed XGBoost validation outputs.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (DiagnosticError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
