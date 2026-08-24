#!/usr/bin/env python3
"""Focused five-fold story-group CV validation for the three existing XGBoost specifications."""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from diagnose_digg_pilot import DiagnosticError, atomic_csv, metrics
from fit_digg_controlled_models import ControlledData, construct_controlled_data
from fit_digg_xgboost import DISPLAY_NAMES
from validate_digg_xgboost import FEATURE_SETS, fit_xgb, matrix


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "digg"
SEED = 42
BOOTSTRAP_REPLICATES = 1000


def fold_validation(
    data: ControlledData,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dummy = np.zeros((len(data.story), 1), dtype=float)
    predictions = {
        model: np.full(len(data.story), np.nan, dtype=float) for model in FEATURE_SETS
    }
    fold_id = np.zeros(len(data.story), dtype=np.int8)
    rows: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(dummy, data.y, groups=data.story), start=1
    ):
        train_stories = set(data.story[train_index].astype(int))
        test_stories = set(data.story[test_index].astype(int))
        overlap = train_stories & test_stories
        if overlap:
            raise DiagnosticError(f"Fold {fold} has {len(overlap)} overlapping stories")
        fold_id[test_index] = fold
        prevalence = float(
            np.sum(data.weight[test_index] * data.y[test_index])
            / np.sum(data.weight[test_index])
        )
        for model_name, features in FEATURE_SETS.items():
            x = matrix(data, features)
            fitted = fit_xgb(x[train_index], data.y[train_index], data.weight[train_index])
            probability = fitted.predict_proba(x[test_index])[:, 1]
            predictions[model_name][test_index] = probability
            result = metrics(
                model_name, f"fold_{fold}", data.y[test_index], probability,
                data.weight[test_index],
            )
            rows.append(
                {
                    "model": model_name,
                    "fold": str(fold),
                    "train_stories": len(train_stories),
                    "test_stories": len(test_stories),
                    "story_overlap": len(overlap),
                    "rows": result["rows"],
                    "positives": result["positives"],
                    "weighted_positive_prevalence": prevalence,
                    "random_classifier_pr_auc": prevalence,
                    "weighted_log_loss": result["weighted_log_loss"],
                    "weighted_brier": result["weighted_brier"],
                    "weighted_roc_auc": result["weighted_roc_auc"],
                    "weighted_pr_auc": result["weighted_pr_auc"],
                    "pr_auc_delta_full_vs_context": "",
                    "full_beats_context_pr_auc": "",
                    "bootstrap_pr_delta_2_5pct": "",
                    "bootstrap_pr_delta_97_5pct": "",
                }
            )
    if np.any(fold_id == 0) or any(np.any(~np.isfinite(value)) for value in predictions.values()):
        raise DiagnosticError("Some rows lack out-of-fold predictions")
    predictions["fold_id"] = fold_id
    return rows, predictions


def add_fold_deltas(rows: list[dict[str, object]]) -> list[float]:
    lookup = {(str(row["model"]), str(row["fold"])): row for row in rows}
    deltas: list[float] = []
    for fold in range(1, 6):
        context = lookup[("XGB_context", str(fold))]
        full = lookup[("XGB_full", str(fold))]
        delta = float(full["weighted_pr_auc"]) - float(context["weighted_pr_auc"])
        full["pr_auc_delta_full_vs_context"] = delta
        full["full_beats_context_pr_auc"] = int(delta > 0)
        deltas.append(delta)
    return deltas


def cluster_bootstrap_pr_delta(
    data: ControlledData,
    context_probability: np.ndarray,
    full_probability: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    stories = np.unique(data.story.astype(int))
    story_to_index = {story: index for index, story in enumerate(stories)}
    row_story = np.fromiter(
        (story_to_index[int(story)] for story in data.story),
        dtype=np.int16,
        count=len(data.story),
    )

    def prepare(probability: np.ndarray) -> tuple[np.ndarray, ...]:
        order = np.argsort(-probability, kind="mergesort")
        score = probability[order]
        starts = np.r_[0, np.flatnonzero(score[1:] != score[:-1]) + 1]
        return (
            data.y[order], data.weight[order], row_story[order], starts,
        )

    def average_precision(prepared: tuple[np.ndarray, ...], multiplicity: np.ndarray) -> float:
        y, base_weight, sorted_story, starts = prepared
        weight = base_weight * multiplicity[sorted_story.astype(int)]
        group_positive = np.add.reduceat(weight * y, starts)
        group_total = np.add.reduceat(weight, starts)
        cumulative_positive = np.cumsum(group_positive)
        cumulative_total = np.cumsum(group_total)
        total_positive = cumulative_positive[-1]
        valid = (group_total > 0) & (cumulative_total > 0)
        return float(
            np.sum(
                (group_positive[valid] / total_positive)
                * (cumulative_positive[valid] / cumulative_total[valid])
            )
        )

    context_prepared = prepare(context_probability)
    full_prepared = prepare(full_probability)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(stories), size=len(stories))
        multiplicity = np.bincount(sampled, minlength=len(stories)).astype(float)
        context_ap = average_precision(context_prepared, multiplicity)
        full_ap = average_precision(full_prepared, multiplicity)
        deltas[replicate] = full_ap - context_ap
        if (replicate + 1) % 100 == 0:
            print(f"Story bootstrap {replicate + 1}/{replicates}", flush=True)
    return deltas


def add_summary_rows(
    rows: list[dict[str, object]],
    fold_deltas: list[float],
    bootstrap_deltas: np.ndarray,
) -> None:
    metric_fields = (
        "weighted_positive_prevalence",
        "random_classifier_pr_auc",
        "weighted_log_loss",
        "weighted_brier",
        "weighted_roc_auc",
        "weighted_pr_auc",
    )
    lower, upper = np.quantile(bootstrap_deltas, [0.025, 0.975])
    for model_name in FEATURE_SETS:
        model_rows = [row for row in rows if row["model"] == model_name and row["fold"].isdigit()]
        for label, function in (
            ("mean", np.mean),
            ("std", lambda values: np.std(values, ddof=1)),
        ):
            summary: dict[str, object] = {
                "model": model_name,
                "fold": label,
                "train_stories": "",
                "test_stories": "",
                "story_overlap": 0,
                "rows": "",
                "positives": "",
                "pr_auc_delta_full_vs_context": "",
                "full_beats_context_pr_auc": "",
                "bootstrap_pr_delta_2_5pct": "",
                "bootstrap_pr_delta_97_5pct": "",
            }
            for field in metric_fields:
                summary[field] = float(function([float(row[field]) for row in model_rows]))
            if model_name == "XGB_full":
                summary["pr_auc_delta_full_vs_context"] = float(function(fold_deltas))
                if label == "mean":
                    summary["full_beats_context_pr_auc"] = sum(delta > 0 for delta in fold_deltas)
                    summary["bootstrap_pr_delta_2_5pct"] = float(lower)
                    summary["bootstrap_pr_delta_97_5pct"] = float(upper)
            rows.append(summary)


def write_plot(
    path: Path,
    rows: list[dict[str, object]],
    fold_deltas: list[float],
    bootstrap_deltas: np.ndarray,
) -> None:
    lookup = {(str(row["model"]), str(row["fold"])): row for row in rows}
    folds = np.arange(1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for model_name, marker in (("XGB_base", "o"), ("XGB_context", "s"), ("XGB_full", "^")):
        values = [float(lookup[(model_name, str(fold))]["weighted_pr_auc"]) for fold in folds]
        axes[0].plot(folds, values, marker=marker, linewidth=1.8, label=model_name)
    prevalence = [float(lookup[("XGB_full", str(fold))]["weighted_positive_prevalence"]) for fold in folds]
    axes[0].plot(folds, prevalence, marker=".", linestyle="--", color="0.45", label="random baseline")
    axes[0].set_xticks(folds)
    axes[0].set_xlabel("Validation fold")
    axes[0].set_ylabel("Sampling-weighted PR-AUC")
    axes[0].set_title("Story-group CV PR-AUC")
    axes[0].legend(frameon=False, fontsize=9)

    colors = ["#4472C4" if delta > 0 else "#B85450" for delta in fold_deltas]
    axes[1].bar(folds, fold_deltas, color=colors)
    axes[1].axhline(0, color="0.35", linewidth=1)
    axes[1].set_xticks(folds)
    axes[1].set_xlabel("Validation fold")
    axes[1].set_ylabel("PR-AUC delta: full − context")
    lower, upper = np.quantile(bootstrap_deltas, [0.025, 0.975])
    axes[1].set_title(
        f"Exposure increment: {sum(delta > 0 for delta in fold_deltas)}/5 folds positive\n"
        f"story-bootstrap 95% CI [{lower:.4g}, {upper:.4g}]"
    )
    fig.suptitle("Digg XGBoost five-fold story-level validation", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def f(value: float) -> str:
    return f"{value:.10g}"


def write_summary(
    path: Path,
    rows: list[dict[str, object]],
    fold_deltas: list[float],
    bootstrap_deltas: np.ndarray,
) -> None:
    lookup = {(str(row["model"]), str(row["fold"])): row for row in rows}
    lower, median, upper = np.quantile(bootstrap_deltas, [0.025, 0.5, 0.975])
    wins = sum(delta > 0 for delta in fold_deltas)
    lines = [
        "# Digg XGBoost five-fold story-group cross-validation",
        "",
        "## Design and integrity",
        "",
        "- Models are limited to the existing `XGB_base`, `XGB_context`, and `XGB_full`; no model or feature was added or reselected.",
        "- Five-fold `GroupKFold`, shuffled with seed 42; `group=story_id`. Each fold contains 80 train and 20 validation stories, and every story is validation data exactly once.",
        "- Story overlap is zero in all five folds. All fits and metrics use the original `sampling_weight`.",
        "- XGBoost hyperparameters are unchanged and fixed before validation; no fold is used for tuning or feature selection.",
        f"- Cluster bootstrap: **{BOOTSTRAP_REPLICATES:,}** replicates, resampling the 100 out-of-fold validation stories with replacement and recomputing pooled weighted PR-AUC for context and full.",
        "",
        "## Per-fold results",
        "",
        "The sampling-weighted positive prevalence is the expected PR-AUC of a random ranking under the reconstructed target population.",
        "",
        "| fold | model | weighted prevalence / random PR baseline | log loss | Brier | ROC-AUC | PR-AUC | full−context PR-AUC |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in range(1, 6):
        for model_name in FEATURE_SETS:
            row = lookup[(model_name, str(fold))]
            delta = row["pr_auc_delta_full_vs_context"]
            lines.append(
                f"| {fold} | {model_name} | {f(float(row['weighted_positive_prevalence']))} | "
                f"{f(float(row['weighted_log_loss']))} | {f(float(row['weighted_brier']))} | "
                f"{f(float(row['weighted_roc_auc']))} | {f(float(row['weighted_pr_auc']))} | "
                f"{f(float(delta)) if delta != '' else '—'} |"
            )
    lines.extend(
        [
            "",
            "## Mean ± standard deviation across folds",
            "",
            "| model | prevalence | log loss | Brier | ROC-AUC | PR-AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in FEATURE_SETS:
        mean = lookup[(model_name, "mean")]
        std = lookup[(model_name, "std")]
        lines.append(
            f"| {model_name} | {f(float(mean['weighted_positive_prevalence']))} ± {f(float(std['weighted_positive_prevalence']))} | "
            f"{f(float(mean['weighted_log_loss']))} ± {f(float(std['weighted_log_loss']))} | "
            f"{f(float(mean['weighted_brier']))} ± {f(float(std['weighted_brier']))} | "
            f"{f(float(mean['weighted_roc_auc']))} ± {f(float(std['weighted_roc_auc']))} | "
            f"{f(float(mean['weighted_pr_auc']))} ± {f(float(std['weighted_pr_auc']))} |"
        )
    lines.extend(
        [
            "",
            "## Exposure increment and bootstrap",
            "",
            f"- Per-fold XGB_full minus XGB_context PR-AUC deltas: **{', '.join(f'{delta:+.8g}' for delta in fold_deltas)}**.",
            f"- XGB_full has higher PR-AUC in **{wins}/5 folds**. Therefore the requested **at least 4/5 folds** criterion is **{'met' if wins >= 4 else 'not met'}**.",
            f"- Mean fold delta: **{np.mean(fold_deltas):+.10g}**; fold SD: **{np.std(fold_deltas, ddof=1):.10g}**.",
            f"- Story-cluster bootstrap pooled PR-AUC delta: median **{median:+.10g}**, percentile 95% interval **[{lower:+.10g}, {upper:+.10g}]**.",
            f"- Bootstrap probability that the increment is positive: **{np.mean(bootstrap_deltas > 0):.2%}**.",
            "",
            "The exposure features improve mean PR-AUC, but the fold-win criterion and cluster interval determine whether that gain is stable across cascades. These are predictive, observational results and do not give `m_in` or `m_out` a causal interpretation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    exposure = PROCESSED / "digg_exposure_pilot.csv.gz"
    votes = PROCESSED / "digg_votes_clean.csv.gz"
    communities = PROCESSED / "digg_communities.csv"
    for path in (exposure, votes, communities):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Constructing the existing six leakage-safe features...", flush=True)
    data = construct_controlled_data(exposure, votes, communities)
    if len(np.unique(data.story)) != 100:
        raise DiagnosticError("Expected exactly 100 pilot stories")
    print("Fitting the three existing XGBoost specifications in five group folds...", flush=True)
    rows, predictions = fold_validation(data)
    fold_deltas = add_fold_deltas(rows)
    print("Bootstrapping out-of-fold test stories for the pooled PR-AUC delta...", flush=True)
    bootstrap_deltas = cluster_bootstrap_pr_delta(
        data, predictions["XGB_context"], predictions["XGB_full"],
        BOOTSTRAP_REPLICATES, SEED,
    )
    add_summary_rows(rows, fold_deltas, bootstrap_deltas)
    atomic_csv(OUTPUT / "xgb_group_cv_metrics.csv", list(rows[0]), rows)
    write_summary(OUTPUT / "xgb_group_cv_summary.md", rows, fold_deltas, bootstrap_deltas)
    write_plot(OUTPUT / "xgb_group_cv_plot.png", rows, fold_deltas, bootstrap_deltas)
    print("Completed focused XGBoost group-CV validation.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (DiagnosticError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
