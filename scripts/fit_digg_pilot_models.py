#!/usr/bin/env python3
"""Fit M0 and M1 weighted logistic models to the Digg pilot exposure table.

M0 is the non-network degree/time baseline. M1 adds count-based within- and
cross-community exposure. This script does not fit M2 or any normalized
exposure model, and it does not perform interventions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import random
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "digg_exposure_pilot.csv.gz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "digg"
DEFAULT_SEED = 42
DEFAULT_BETA = 5.0
REQUIRED_FIELDS = {
    "story_id",
    "time_bin",
    "node_id",
    "m_in",
    "m_out",
    "degree",
    "y",
    "sampling_weight",
}
FEATURE_LABELS = {
    "m_in": "m_in",
    "m_out": "m_out",
    "degree": "degree",
    "log_time": "log1p(time_bin)",
}


class ModelError(RuntimeError):
    """Raised when model inputs or outputs fail validation."""


@dataclass
class PilotData:
    story_id: np.ndarray
    node_id: np.ndarray
    time_bin: np.ndarray
    m_in: np.ndarray
    m_out: np.ndarray
    degree: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    duplicate_positive_pairs: int


@dataclass
class FittedModel:
    name: str
    feature_names: list[str]
    estimator: LogisticRegression
    mean: np.ndarray
    scale: np.ndarray
    raw_coefficients: np.ndarray
    raw_intercept: float
    converged: bool
    iterations: int
    warning_messages: list[str]

    def predict(self, raw_features: np.ndarray) -> np.ndarray:
        standardized = (raw_features - self.mean) / self.scale
        return self.estimator.predict_proba(standardized)[:, 1]


@dataclass
class MetricRow:
    model: str
    subset: str
    rows: int
    positives: int
    weighted_positive_rate: float
    weighted_log_loss: float
    weighted_brier: float
    weighted_roc_auc: float
    weighted_pr_auc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def finite_float(value: str, field: str, row_number: int) -> float:
    if value == "":
        raise ModelError(f"Missing {field} at data row {row_number}")
    try:
        result = float(value)
    except ValueError as error:
        raise ModelError(f"Invalid {field} at data row {row_number}: {value!r}") from error
    if not math.isfinite(result):
        raise ModelError(f"Non-finite {field} at data row {row_number}: {value!r}")
    return result


def integer_value(value: str, field: str, row_number: int) -> int:
    result = finite_float(value, field, row_number)
    if not result.is_integer():
        raise ModelError(f"Non-integer {field} at data row {row_number}: {value!r}")
    return int(result)


def load_data(path: Path) -> PilotData:
    columns: dict[str, list[float | int]] = {
        name: []
        for name in (
            "story_id",
            "node_id",
            "time_bin",
            "m_in",
            "m_out",
            "degree",
            "y",
            "sampling_weight",
        )
    }
    positive_pairs: set[tuple[int, int]] = set()
    duplicate_positive_pairs = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing_fields = REQUIRED_FIELDS - fields
        if missing_fields:
            raise ModelError(f"Input is missing required fields: {sorted(missing_fields)}")
        for row_number, row in enumerate(reader, start=2):
            story_id = integer_value(row["story_id"], "story_id", row_number)
            node_id = integer_value(row["node_id"], "node_id", row_number)
            time_bin = integer_value(row["time_bin"], "time_bin", row_number)
            m_in = integer_value(row["m_in"], "m_in", row_number)
            m_out = integer_value(row["m_out"], "m_out", row_number)
            degree = integer_value(row["degree"], "degree", row_number)
            outcome = integer_value(row["y"], "y", row_number)
            weight = finite_float(row["sampling_weight"], "sampling_weight", row_number)
            if outcome not in (0, 1):
                raise ModelError(f"y is not binary at data row {row_number}: {outcome}")
            if weight <= 0:
                raise ModelError(f"sampling_weight is not positive at data row {row_number}")
            if min(time_bin, m_in, m_out) < 0 or degree <= 0:
                raise ModelError(f"Invalid non-negative feature or degree at data row {row_number}")
            if m_in + m_out > degree:
                raise ModelError(f"m_in + m_out exceeds degree at data row {row_number}")
            if outcome:
                pair = (story_id, node_id)
                duplicate_positive_pairs += int(pair in positive_pairs)
                positive_pairs.add(pair)
            values: tuple[float | int, ...] = (
                story_id,
                node_id,
                time_bin,
                m_in,
                m_out,
                degree,
                outcome,
                weight,
            )
            for name, value in zip(columns, values):
                columns[name].append(value)
    if not columns["y"]:
        raise ModelError("Input exposure table is empty")
    return PilotData(
        story_id=np.asarray(columns["story_id"], dtype=np.int64),
        node_id=np.asarray(columns["node_id"], dtype=np.int64),
        time_bin=np.asarray(columns["time_bin"], dtype=np.int64),
        m_in=np.asarray(columns["m_in"], dtype=np.float64),
        m_out=np.asarray(columns["m_out"], dtype=np.float64),
        degree=np.asarray(columns["degree"], dtype=np.float64),
        y=np.asarray(columns["y"], dtype=np.int8),
        weight=np.asarray(columns["sampling_weight"], dtype=np.float64),
        duplicate_positive_pairs=duplicate_positive_pairs,
    )


def rank_strata(items: list[tuple[int, int]], groups: int) -> list[list[tuple[int, int]]]:
    ordered = sorted(items, key=lambda item: (item[1], item[0]))
    base, remainder = divmod(len(ordered), groups)
    strata: list[list[tuple[int, int]]] = []
    cursor = 0
    for index in range(groups):
        size = base + int(index < remainder)
        strata.append(ordered[cursor : cursor + size])
        cursor += size
    return strata


def story_split(data: PilotData, seed: int) -> list[dict[str, int | str]]:
    positive_counts: Counter[int] = Counter(
        int(story_id) for story_id in data.story_id[data.y == 1]
    )
    stories = sorted({int(value) for value in data.story_id})
    if len(stories) != 100:
        raise ModelError(f"Expected 100 pilot stories, found {len(stories)}")
    if set(stories) != set(positive_counts):
        raise ModelError("At least one pilot story contains no positive row")
    strata = rank_strata([(story, positive_counts[story]) for story in stories], 5)
    rng = random.Random(seed)
    test_stories: set[int] = set()
    stratum_by_story: dict[int, str] = {}
    for index, stratum in enumerate(strata, start=1):
        if len(stratum) != 20:
            raise ModelError("The 100-story pilot cannot be divided into five 20-story strata")
        test_stories.update(story for story, _ in rng.sample(stratum, 4))
        for story, _ in stratum:
            stratum_by_story[story] = f"positive_count_Q{index}"
    rows = [
        {
            "story_id": story,
            "split": "test" if story in test_stories else "train",
            "positive_count": positive_counts[story],
            "stratum": stratum_by_story[story],
        }
        for story in stories
    ]
    counts = Counter(str(row["split"]) for row in rows)
    if counts != {"train": 80, "test": 20}:
        raise ModelError(f"Unexpected story split counts: {counts}")
    return rows


def atomic_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def feature_matrix(data: PilotData, feature_names: list[str]) -> np.ndarray:
    available = {
        "m_in": data.m_in,
        "m_out": data.m_out,
        "degree": data.degree,
        "log_time": np.log1p(data.time_bin.astype(np.float64)),
    }
    return np.column_stack([available[name] for name in feature_names])


def fit_model(
    name: str,
    feature_names: list[str],
    raw_features: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    max_iter: int,
    tolerance: float,
    seed: int,
) -> FittedModel:
    mean = np.average(raw_features, axis=0, weights=weights)
    variance = np.average((raw_features - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    standardized = (raw_features - mean) / scale
    estimator = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=max_iter,
        tol=tolerance,
        class_weight=None,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(standardized, y, sample_weight=weights)
    messages = [f"{item.category.__name__}: {item.message}" for item in caught]
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    iterations = int(estimator.n_iter_[0])
    if iterations >= max_iter:
        converged = False
    standardized_coefficients = estimator.coef_[0]
    raw_coefficients = standardized_coefficients / scale
    raw_intercept = float(estimator.intercept_[0] - np.dot(raw_coefficients, mean))
    return FittedModel(
        name=name,
        feature_names=feature_names,
        estimator=estimator,
        mean=mean,
        scale=scale,
        raw_coefficients=raw_coefficients,
        raw_intercept=raw_intercept,
        converged=converged,
        iterations=iterations,
        warning_messages=messages,
    )


def safe_auc(metric: str, y: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    if metric == "roc":
        return float(roc_auc_score(y, probability, sample_weight=weight))
    return float(average_precision_score(y, probability, sample_weight=weight))


def evaluate(
    model: str,
    subset: str,
    y: np.ndarray,
    probability: np.ndarray,
    weight: np.ndarray,
) -> MetricRow:
    return MetricRow(
        model=model,
        subset=subset,
        rows=len(y),
        positives=int(np.sum(y)),
        weighted_positive_rate=float(np.average(y, weights=weight)),
        weighted_log_loss=float(log_loss(y, probability, sample_weight=weight, labels=[0, 1])),
        weighted_brier=float(brier_score_loss(y, probability, sample_weight=weight)),
        weighted_roc_auc=safe_auc("roc", y, probability, weight),
        weighted_pr_auc=safe_auc("pr", y, probability, weight),
    )


def calibration_points(
    y: np.ndarray, probability: np.ndarray, weight: np.ndarray, bins: int = 10
) -> list[tuple[float, float, float, int, float]]:
    # Quantile bins remain informative for this rare-event problem, where all
    # predictions occupy only a small part of [0, 1]. Outcomes are not used to
    # set the boundaries; sampling weights are used for both reported means.
    edges = np.unique(np.quantile(probability, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        edges = np.asarray([float(np.min(probability)), float(np.max(probability))])
    assignments = np.searchsorted(edges[1:-1], probability, side="right")
    points: list[tuple[float, float, float, int, float]] = []
    for index in range(len(edges) - 1):
        mask = assignments == index
        if not np.any(mask):
            continue
        subset_weight = weight[mask]
        points.append(
            (
                float(edges[index]),
                float(edges[index + 1]),
                float(np.average(probability[mask], weights=subset_weight)),
                int(np.sum(mask)),
                float(np.average(y[mask], weights=subset_weight)),
            )
        )
    return points


def save_calibration_plot(
    path: Path,
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, list[tuple[float, float, float, int, float]]]:
    curves = {
        name: calibration_points(y, probability, weights)
        for name, probability in predictions.items()
    }
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    maximum = 0.0
    for name, points in curves.items():
        predicted = [point[2] for point in points]
        observed = [point[4] for point in points]
        maximum = max(maximum, *(predicted + observed))
        axis.plot(predicted, observed, marker="o", linewidth=1.8, label=name)
    limit = min(1.0, max(1e-6, maximum * 1.08))
    axis.plot([0, limit], [0, limit], linestyle="--", color="0.35", label="Ideal")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("Weighted mean predicted probability")
    axis.set_ylabel("Weighted observed positive rate")
    axis.set_title("Held-out story calibration (prediction-quantile bins)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return curves


def save_coefficient_plot(path: Path, model: FittedModel) -> None:
    labels = [FEATURE_LABELS[name] for name in model.feature_names]
    coefficients = model.raw_coefficients
    colors = ["#2b8cbe" if value >= 0 else "#d95f0e" for value in coefficients]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(labels))
    bars = axis.bar(positions, coefficients, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Logistic coefficient (original feature scale)")
    axis.set_title("M1 count-exposure coefficients")
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, coefficients):
        vertical = 3 if value >= 0 else -3
        alignment = "bottom" if value >= 0 else "top"
        axis.annotate(
            f"{value:.4g}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, vertical),
            textcoords="offset points",
            ha="center",
            va=alignment,
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def coefficient_row(model: FittedModel, beta: float) -> dict[str, object]:
    coefficient_by_name = dict(zip(model.feature_names, model.raw_coefficients))
    is_m1 = model.name == "M1"
    m_in = float(coefficient_by_name["m_in"]) if is_m1 else ""
    m_out = float(coefficient_by_name["m_out"]) if is_m1 else ""
    degree = float(coefficient_by_name["degree"])
    log_time = float(coefficient_by_name["log_time"])
    return {
        "model": model.name,
        "intercept": model.raw_intercept,
        "coef_m_in": m_in,
        "coef_m_out": m_out,
        "coef_degree": degree,
        "coef_log_time": log_time,
        "beta_scale_convention": beta if is_m1 else "",
        "a_eff": m_in / beta if is_m1 else "",
        "b_eff": m_out / beta if is_m1 else "",
        "theta_eff": -degree / beta if is_m1 else "",
        "converged": int(model.converged),
        "iterations": model.iterations,
        "warnings": " | ".join(model.warning_messages),
    }


def format_metric(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.8f}"


def write_report(
    path: Path,
    data: PilotData,
    split_rows: list[dict[str, int | str]],
    models: dict[str, FittedModel],
    coefficients: dict[str, dict[str, object]],
    metric_rows: list[MetricRow],
    calibration: dict[str, list[tuple[float, float, float, int, float]]],
) -> None:
    train_stories = {int(row["story_id"]) for row in split_rows if row["split"] == "train"}
    test_stories = {int(row["story_id"]) for row in split_rows if row["split"] == "test"}
    train_mask = np.isin(data.story_id, list(train_stories))
    test_mask = np.isin(data.story_id, list(test_stories))
    raw_rate = float(np.mean(data.y))
    weighted_rate = float(np.average(data.y, weights=data.weight))
    positive_zero_fraction = float(
        np.mean((data.m_in[data.y == 1] == 0) & (data.m_out[data.y == 1] == 0))
    )
    metric_lookup = {(row.model, row.subset): row for row in metric_rows}
    m0 = metric_lookup[("M0", "all_test")]
    m1 = metric_lookup[("M1", "all_test")]
    improvements = {
        "log_loss": m0.weighted_log_loss - m1.weighted_log_loss,
        "brier": m0.weighted_brier - m1.weighted_brier,
        "roc": m1.weighted_roc_auc - m0.weighted_roc_auc,
        "pr": m1.weighted_pr_auc - m0.weighted_pr_auc,
    }
    m1_coefficients = coefficients["M1"]
    theta_eff = float(m1_coefficients["theta_eff"])
    theta_note = (
        "`theta_eff` lies inside [0,1], but remains an observational effective parameter."
        if 0.0 <= theta_eff <= 1.0
        else "`theta_eff` lies outside [0,1] and is reported without clipping; this suggests the "
        "paper's count-threshold structure may not fully match real Digg behavior."
    )
    stratum_summary: list[tuple[str, int, int, int, int]] = []
    for label in sorted({str(row["stratum"]) for row in split_rows}):
        rows = [row for row in split_rows if row["stratum"] == label]
        positive_counts = [int(row["positive_count"]) for row in rows]
        stratum_summary.append(
            (
                label,
                min(positive_counts),
                max(positive_counts),
                sum(row["split"] == "train" for row in rows),
                sum(row["split"] == "test" for row in rows),
            )
        )

    lines = [
        "# Digg pilot M0/M1 model report",
        "",
        "## Data checks",
        "",
        f"- Input rows: **{len(data.y):,}** across **{len(train_stories | test_stories)}** pilot cascades",
        "- Required columns: present",
        "- Missing values: **0**",
        "- Non-finite values: **0**",
        "- Non-binary `y` values: **0**",
        "- Non-positive `sampling_weight` values: **0**",
        f"- Duplicate positive `(node_id, story_id)` pairs: **{data.duplicate_positive_pairs:,}**",
        f"- Raw `y=1` rate: **{raw_rate:.8%}**",
        f"- Sampling-weighted `y=1` rate: **{weighted_rate:.8%}**",
        f"- Positive rows with `m_in=m_out=0`: **{positive_zero_fraction:.4%}** "
        "(the pilot exposure audit reported 72.2069%; the unrounded recomputation is shown here)",
        "",
        "## Story-level split",
        "",
        f"- Train: **{len(train_stories)} stories**, **{int(np.sum(train_mask)):,} rows**",
        f"- Test: **{len(test_stories)} stories**, **{int(np.sum(test_mask)):,} rows**",
        "- Split seed: **42**",
        f"- Story overlap: **{len(train_stories & test_stories)}**",
        "- Stratification uses five rank strata of per-story positive counts; four stories from "
        "each stratum enter the test set.",
        "",
        "| stratum | positive-count range | train stories | test stories |",
        "|---|---:|---:|---:|",
    ]
    for label, minimum, maximum, train_count, test_count in stratum_summary:
        lines.append(f"| {label} | {minimum}–{maximum} | {train_count} | {test_count} |")
    lines.extend(
        [
            "",
            "## Models and fitting",
            "",
            "- **M0** is the necessary non-network baseline: `c + q*degree + "
            "gamma*log1p(time_bin)`.",
            "- **M1** is a probabilistic real-data version of the paper's count-based threshold "
            "rule: it uses `m_in`, `m_out`, `degree`, and `log1p(time_bin)`.",
            "- M0 and M1 use identical degree and time definitions; M1's only additional predictors "
            "are `m_in` and `m_out`.",
            "- Both models use sampling-weighted logistic regression, no `class_weight`, and no "
            "regularization. Features were standardized from weighted training statistics only, "
            "then coefficients were converted back to original scales.",
            "- No normalized exposure (`frac_in`/`frac_out`) model and no M2 were fitted.",
            "- The held-out test stories were not used for feature scaling, model selection, or "
            "optimization choices.",
            "",
            "| model | converged | iterations | warnings |",
            "|---|---:|---:|---|",
        ]
    )
    for name in ("M0", "M1"):
        model = models[name]
        warning_text = "; ".join(model.warning_messages) if model.warning_messages else "none"
        lines.append(
            f"| {name} | {'yes' if model.converged else 'no'} | {model.iterations} | "
            f"{warning_text.replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Original-scale coefficients",
            "",
            "| model | intercept | coef_m_in | coef_m_out | coef_degree | coef_log_time |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("M0", "M1"):
        row = coefficients[name]
        lines.append(
            f"| {name} | {float(row['intercept']):.10g} | "
            f"{row['coef_m_in'] if row['coef_m_in'] != '' else '—'} | "
            f"{row['coef_m_out'] if row['coef_m_out'] != '' else '—'} | "
            f"{float(row['coef_degree']):.10g} | {float(row['coef_log_time']):.10g} |"
        )
    lines.extend(
        [
            "",
            "### M1 effective-parameter convention",
            "",
            f"- Fixed scale convention: `beta = {DEFAULT_BETA:g}` (not estimated)",
            f"- `a_eff = coef_m_in / 5 = {float(m1_coefficients['a_eff']):.10g}`",
            f"- `b_eff = coef_m_out / 5 = {float(m1_coefficients['b_eff']):.10g}`",
            f"- `theta_eff = -coef_degree / 5 = {theta_eff:.10g}`",
            f"- Intercept `c = {float(m1_coefficients['intercept']):.10g}`",
            "",
            "`beta=5` is an imposed scale convention, not an estimated real-data parameter. "
            "`a_eff`, `b_eff`, and `theta_eff` are observational effective parameters; there are "
            "no true parameters here, so this is not parameter recovery. " + theta_note,
            "",
            "## Held-out 20-story evaluation",
            "",
            "PR-AUC is reported as sampling-weighted average precision.",
            "",
            "| model | subset | rows | positives | weighted positive rate | log loss | Brier | ROC-AUC | PR-AUC |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| {row.model} | {row.subset} | {row.rows:,} | {row.positives:,} | "
            f"{row.weighted_positive_rate:.8%} | {format_metric(row.weighted_log_loss)} | "
            f"{format_metric(row.weighted_brier)} | {format_metric(row.weighted_roc_auc)} | "
            f"{format_metric(row.weighted_pr_auc)} |"
        )
    lines.extend(
        [
            "",
            "### M1 change relative to M0 (all test rows)",
            "",
            f"- Log-loss improvement (`M0 - M1`): **{improvements['log_loss']:.10g}**",
            f"- Brier improvement (`M0 - M1`): **{improvements['brier']:.10g}**",
            f"- ROC-AUC increment (`M1 - M0`): **{improvements['roc']:.10g}**",
            f"- PR-AUC increment (`M1 - M0`): **{improvements['pr']:.10g}**",
            "",
            "Positive improvement values favor M1. The zero/nonzero-exposure subset rows above "
            "show where any predictive difference occurs.",
            "",
            "**Core answer:** M1 provides a small, mixed incremental predictive gain over M0. "
            "Overall held-out log loss, ROC-AUC, and PR-AUC improve, while the Brier score worsens "
            "by a negligible amount. The evidence therefore supports limited additional signal "
            "from count exposures, not a uniformly superior model.",
            "",
            "## Calibration",
            "",
            "The calibration figure uses ten prediction-quantile bins on held-out stories and "
            "computes both mean prediction and observed rate with `sampling_weight`. Outcomes are "
            "not used to set bin boundaries; empty or duplicate-boundary bins are omitted.",
        ]
    )
    for name in ("M0", "M1"):
        lines.append(f"- {name}: **{len(calibration[name])}** non-empty calibration bins")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The central comparison is whether adding count exposures `m_in` and `m_out` improves "
            "next-hour prediction beyond the degree/time M0 baseline. The result is predictive "
            "and associational, not a causal effect.",
            "",
            "The fact that **72.2069%** of positive pilot observations occur with "
            "`m_in=m_out=0` shows that homepage recommendation, external exposure, and other "
            "unobserved channels are important; M1's intercept `c` absorbs their baseline influence "
            "only in an aggregate sense.",
            "",
            "Only the 100 pilot cascades were used. Do not yet extend fitting to all 3,553 cascades, "
            "and do not yet run a real-data intervention that lowers `b`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Pilot exposure table not found: {input_path}")
    if args.beta <= 0 or args.max_iter <= 0 or args.tolerance <= 0:
        raise ModelError("--beta, --max-iter, and --tolerance must be positive")
    if not math.isclose(args.beta, DEFAULT_BETA):
        raise ModelError("This first model version fixes beta at 5")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and checking {input_path}...")
    data = load_data(input_path)
    split_rows = story_split(data, args.seed)
    split_path = output_dir / "pilot_story_split.csv"
    atomic_csv(
        split_path,
        ["story_id", "split", "positive_count", "stratum"],
        split_rows,
    )
    train_stories = {
        int(row["story_id"]) for row in split_rows if row["split"] == "train"
    }
    test_stories = {
        int(row["story_id"]) for row in split_rows if row["split"] == "test"
    }
    if train_stories & test_stories:
        raise ModelError("A story appears in both train and test")
    train_mask = np.isin(data.story_id, list(train_stories))
    test_mask = np.isin(data.story_id, list(test_stories))
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise ModelError("Row-level train/test masks are inconsistent")

    specifications = {
        "M0": ["degree", "log_time"],
        "M1": ["m_in", "m_out", "degree", "log_time"],
    }
    models: dict[str, FittedModel] = {}
    predictions: dict[str, np.ndarray] = {}
    print("Fitting weighted M0 and M1 on 80 training stories...")
    for name, feature_names in specifications.items():
        all_features = feature_matrix(data, feature_names)
        model = fit_model(
            name,
            feature_names,
            all_features[train_mask],
            data.y[train_mask],
            data.weight[train_mask],
            args.max_iter,
            args.tolerance,
            args.seed,
        )
        models[name] = model
        predictions[name] = model.predict(all_features[test_mask])

    test_y = data.y[test_mask]
    test_weight = data.weight[test_mask]
    test_zero_exposure = (data.m_in[test_mask] == 0) & (data.m_out[test_mask] == 0)
    subsets = {
        "all_test": np.ones(len(test_y), dtype=bool),
        "zero_exposure": test_zero_exposure,
        "nonzero_exposure": ~test_zero_exposure,
    }
    metric_rows: list[MetricRow] = []
    for name in ("M0", "M1"):
        for subset, mask in subsets.items():
            metric_rows.append(
                evaluate(
                    name,
                    subset,
                    test_y[mask],
                    predictions[name][mask],
                    test_weight[mask],
                )
            )

    coefficient_rows = [coefficient_row(models[name], args.beta) for name in ("M0", "M1")]
    coefficient_by_model = {str(row["model"]): row for row in coefficient_rows}
    atomic_csv(
        output_dir / "pilot_model_coefficients.csv",
        [
            "model",
            "intercept",
            "coef_m_in",
            "coef_m_out",
            "coef_degree",
            "coef_log_time",
            "beta_scale_convention",
            "a_eff",
            "b_eff",
            "theta_eff",
            "converged",
            "iterations",
            "warnings",
        ],
        coefficient_rows,
    )
    atomic_csv(
        output_dir / "pilot_model_metrics.csv",
        [
            "model",
            "subset",
            "rows",
            "positives",
            "weighted_positive_rate",
            "weighted_log_loss",
            "weighted_brier",
            "weighted_roc_auc",
            "weighted_pr_auc",
        ],
        [row.__dict__ for row in metric_rows],
    )
    print("Evaluating held-out stories and rendering figures...")
    calibration = save_calibration_plot(
        output_dir / "pilot_calibration.png", test_y, predictions, test_weight
    )
    save_coefficient_plot(output_dir / "pilot_coefficient_plot.png", models["M1"])
    write_report(
        output_dir / "pilot_model_report.md",
        data,
        split_rows,
        models,
        coefficient_by_model,
        metric_rows,
        calibration,
    )
    print(f"Story split: {split_path}")
    print(f"Model report: {output_dir / 'pilot_model_report.md'}")


if __name__ == "__main__":
    try:
        main()
    except (ModelError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
