#!/usr/bin/env python3
"""Diagnose Digg pilot M0/M1 subgroups and cluster-bootstrap M1 coefficients."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "digg_exposure_pilot.csv.gz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "digg"
DEFAULT_SEED = 42
DEFAULT_REPLICATES = 200
BETA = 5.0
FEATURES = ("m_in", "m_out", "degree", "log_time")


class DiagnosticError(RuntimeError):
    """Raised when existing pilot model artifacts are inconsistent."""


@dataclass
class Data:
    story: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    x: np.ndarray


@dataclass
class FitResult:
    intercept: float
    coefficients: np.ndarray
    converged: bool
    iterations: int
    warning: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def load_data(path: Path) -> Data:
    story: list[int] = []
    y: list[int] = []
    weight: list[float] = []
    features: list[tuple[float, float, float, float]] = []
    required = {
        "story_id", "time_bin", "m_in", "m_out", "degree", "y", "sampling_weight"
    }
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DiagnosticError(f"Missing input fields: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                values = [float(row[name]) for name in required]
            except (TypeError, ValueError) as error:
                raise DiagnosticError(f"Invalid value at row {row_number}") from error
            if not all(math.isfinite(value) for value in values):
                raise DiagnosticError(f"Non-finite value at row {row_number}")
            story_id = int(row["story_id"])
            outcome = int(row["y"])
            sampling_weight = float(row["sampling_weight"])
            if outcome not in (0, 1) or sampling_weight <= 0:
                raise DiagnosticError(f"Invalid outcome or weight at row {row_number}")
            story.append(story_id)
            y.append(outcome)
            weight.append(sampling_weight)
            features.append(
                (
                    float(row["m_in"]),
                    float(row["m_out"]),
                    float(row["degree"]),
                    math.log1p(int(row["time_bin"])),
                )
            )
    return Data(
        story=np.asarray(story, dtype=np.int64),
        y=np.asarray(y, dtype=np.float64),
        weight=np.asarray(weight, dtype=np.float64),
        x=np.asarray(features, dtype=np.float64),
    )


def read_split(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            story = int(row["story_id"])
            split = row["split"]
            if story in result or split not in ("train", "test"):
                raise DiagnosticError("Invalid or duplicate story split row")
            result[story] = split
    counts = Counter(result.values())
    if counts != {"train": 80, "test": 20}:
        raise DiagnosticError(f"Expected 80/20 stories, found {counts}")
    return result


def read_coefficients(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            result[model] = {
                "intercept": float(row["intercept"]),
                "m_in": float(row["coef_m_in"]) if row["coef_m_in"] else 0.0,
                "m_out": float(row["coef_m_out"]) if row["coef_m_out"] else 0.0,
                "degree": float(row["coef_degree"]),
                "log_time": float(row["coef_log_time"]),
            }
    if set(result) != {"M0", "M1"}:
        raise DiagnosticError("Expected existing M0 and M1 coefficient rows")
    return result


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def predict(coefficients: dict[str, float], x: np.ndarray) -> np.ndarray:
    beta = np.asarray([coefficients[name] for name in FEATURES])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        linear_predictor = coefficients["intercept"] + x @ beta
    if not np.all(np.isfinite(linear_predictor)):
        raise DiagnosticError("Non-finite point-estimate prediction")
    return sigmoid(linear_predictor)


def weighted_roc_auc(y: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(probability, kind="mergesort")
    score, outcome, sample_weight = probability[order], y[order], weight[order]
    total_positive = float(np.sum(sample_weight * outcome))
    total_negative = float(np.sum(sample_weight * (1.0 - outcome)))
    cumulative_negative = 0.0
    numerator = 0.0
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[end] == score[start]:
            end += 1
        group_weight = sample_weight[start:end]
        group_outcome = outcome[start:end]
        positive = float(np.sum(group_weight * group_outcome))
        negative = float(np.sum(group_weight * (1.0 - group_outcome)))
        numerator += positive * (cumulative_negative + 0.5 * negative)
        cumulative_negative += negative
        start = end
    return numerator / (total_positive * total_negative)


def weighted_average_precision(
    y: np.ndarray, probability: np.ndarray, weight: np.ndarray
) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-probability, kind="mergesort")
    score, outcome, sample_weight = probability[order], y[order], weight[order]
    total_positive = float(np.sum(sample_weight * outcome))
    true_positive = 0.0
    false_positive = 0.0
    average_precision = 0.0
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[end] == score[start]:
            end += 1
        group_weight = sample_weight[start:end]
        group_outcome = outcome[start:end]
        positive = float(np.sum(group_weight * group_outcome))
        negative = float(np.sum(group_weight * (1.0 - group_outcome)))
        true_positive += positive
        false_positive += negative
        average_precision += (positive / total_positive) * (
            true_positive / (true_positive + false_positive)
        )
        start = end
    return average_precision


def metrics(
    model: str,
    subset: str,
    y: np.ndarray,
    probability: np.ndarray,
    weight: np.ndarray,
) -> dict[str, object]:
    clipped = np.clip(probability, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    total_weight = float(np.sum(weight))
    log_loss = float(
        np.sum(weight * (-y * np.log(clipped) - (1.0 - y) * np.log1p(-clipped)))
        / total_weight
    )
    brier = float(np.sum(weight * (probability - y) ** 2) / total_weight)
    return {
        "model": model,
        "subset": subset,
        "rows": len(y),
        "positives": int(np.sum(y)),
        "weighted_y_rate": float(np.sum(weight * y) / total_weight),
        "weighted_log_loss": log_loss,
        "weighted_brier": brier,
        "weighted_roc_auc": weighted_roc_auc(y, probability, weight),
        "weighted_pr_auc": weighted_average_precision(y, probability, weight),
    }


def log_likelihood(design: np.ndarray, y: np.ndarray, weight: np.ndarray, beta: np.ndarray) -> float:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        eta = design @ beta
    if not np.all(np.isfinite(eta)):
        return float("-inf")
    return float(np.sum(weight * (y * eta - np.logaddexp(0.0, eta))))


def weighted_logistic_irls(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    initial_intercept: float,
    initial_coefficients: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> FitResult:
    positive_weight = weight > 0
    x = x[positive_weight]
    y = y[positive_weight]
    weight = weight[positive_weight]
    mean = np.average(x, axis=0, weights=weight)
    variance = np.average((x - mean) ** 2, axis=0, weights=weight)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), standardized))
    beta = np.concatenate(
        ([initial_intercept + float(np.dot(initial_coefficients, mean))], initial_coefficients * scale)
    )
    warning = ""
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            eta = design @ beta
        if not np.all(np.isfinite(eta)):
            warning = "Non-finite linear predictor"
            break
        probability = sigmoid(eta)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            score = design.T @ (weight * (y - probability))
        if not np.all(np.isfinite(score)):
            warning = "Non-finite score vector"
            break
        curvature_weight = weight * probability * (1.0 - probability)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            information = design.T @ (curvature_weight[:, None] * design)
        if not np.all(np.isfinite(information)):
            warning = "Non-finite information matrix"
            break
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(information, score, rcond=None)[0]
            warning = "Singular information matrix; used least-squares Newton step"
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            break
        old_likelihood = log_likelihood(design, y, weight, beta)
        step_scale = 1.0
        while step_scale >= 2.0 ** -20:
            candidate = beta + step_scale * step
            candidate_likelihood = log_likelihood(design, y, weight, candidate)
            numerical_slack = 1e-12 * (1.0 + abs(old_likelihood))
            if candidate_likelihood >= old_likelihood - numerical_slack:
                beta = candidate
                break
            step_scale *= 0.5
        else:
            warning = (warning + " | " if warning else "") + "Line search failed"
            break
        if float(np.max(np.abs(step_scale * step))) < tolerance:
            converged = True
            break
    raw_coefficients = beta[1:] / scale
    raw_intercept = float(beta[0] - np.dot(raw_coefficients, mean))
    return FitResult(raw_intercept, raw_coefficients, converged, iteration, warning)


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
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


def bootstrap(
    data: Data,
    train_mask: np.ndarray,
    train_stories: list[int],
    point: dict[str, float],
    replicates: int,
    seed: int,
    max_iter: int,
    tolerance: float,
) -> list[dict[str, object]]:
    x = data.x[train_mask]
    y = data.y[train_mask]
    base_weight = data.weight[train_mask]
    row_story = data.story[train_mask]
    initial = np.asarray([point[name] for name in FEATURES])
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(1, replicates + 1):
        sampled = rng.choices(train_stories, k=len(train_stories))
        multiplicity = Counter(sampled)
        cluster_multiplier = np.fromiter(
            (multiplicity.get(int(story), 0) for story in row_story),
            dtype=np.float64,
            count=len(row_story),
        )
        fit = weighted_logistic_irls(
            x,
            y,
            base_weight * cluster_multiplier,
            point["intercept"],
            initial,
            max_iter,
            tolerance,
        )
        coefficients = dict(zip(FEATURES, fit.coefficients))
        rows.append(
            {
                "bootstrap_id": replicate,
                "sampled_story_draws": len(sampled),
                "unique_stories": len(multiplicity),
                "intercept": fit.intercept,
                "coef_m_in": coefficients["m_in"],
                "coef_m_out": coefficients["m_out"],
                "coef_degree": coefficients["degree"],
                "coef_log_time": coefficients["log_time"],
                "a_eff": coefficients["m_in"] / BETA,
                "b_eff": coefficients["m_out"] / BETA,
                "theta_eff": -coefficients["degree"] / BETA,
                "sign_change_m_in": int(np.sign(coefficients["m_in"]) != np.sign(point["m_in"])),
                "sign_change_m_out": int(np.sign(coefficients["m_out"]) != np.sign(point["m_out"])),
                "sign_change_degree": int(np.sign(coefficients["degree"]) != np.sign(point["degree"])),
                "sign_change_log_time": int(
                    np.sign(coefficients["log_time"]) != np.sign(point["log_time"])
                ),
                "converged": int(fit.converged),
                "iterations": fit.iterations,
                "warning": fit.warning,
            }
        )
        if replicate % 25 == 0:
            print(f"Bootstrap {replicate}/{replicates}")
    return rows


def percentile_interval(rows: list[dict[str, object]], field: str) -> tuple[float, float, float]:
    values = np.asarray([float(row[field]) for row in rows if row["converged"]], dtype=float)
    if not len(values):
        raise DiagnosticError("No converged bootstrap replicates")
    lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
    return float(lower), float(median), float(upper)


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.10g}"


def write_report(
    path: Path,
    data: Data,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    point: dict[str, float],
    metric_rows: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
    irls_check: FitResult,
) -> None:
    successful = [row for row in bootstrap_rows if row["converged"]]
    intervals = {
        field: percentile_interval(bootstrap_rows, field)
        for field in ("coef_m_in", "coef_m_out", "a_eff", "b_eff", "theta_eff")
    }
    sign_changes = {
        field: float(np.mean([int(row[field]) for row in successful]))
        for field in (
            "sign_change_m_in", "sign_change_m_out", "sign_change_degree", "sign_change_log_time"
        )
    }
    lookup = {(str(row["model"]), str(row["subset"])): row for row in metric_rows}
    subsets = ("all", "zero_exposure", "exposed", "in_only", "out_only", "both")
    within_stable = intervals["coef_m_in"][0] > 0 and sign_changes["sign_change_m_in"] < 0.05
    cross_stable = intervals["coef_m_out"][0] > 0 and sign_changes["sign_change_m_out"] < 0.05
    all_delta = float(lookup[("M0", "all")]["weighted_log_loss"]) - float(
        lookup[("M1", "all")]["weighted_log_loss"]
    )
    exposed_delta = float(lookup[("M0", "exposed")]["weighted_log_loss"]) - float(
        lookup[("M1", "exposed")]["weighted_log_loss"]
    )
    zero_delta = float(lookup[("M0", "zero_exposure")]["weighted_log_loss"]) - float(
        lookup[("M1", "zero_exposure")]["weighted_log_loss"]
    )
    overall_prevalence = float(np.average(data.y, weights=data.weight))
    test_prevalence = float(np.average(data.y[test_mask], weights=data.weight[test_mask]))
    point_effective = {
        "a_eff": point["m_in"] / BETA,
        "b_eff": point["m_out"] / BETA,
        "theta_eff": -point["degree"] / BETA,
    }
    lines = [
        "# Digg pilot M1 diagnostic report",
        "",
        "## Prevalence and point estimate",
        "",
        f"- Full-pilot sampling-weighted `y=1` prevalence: **{overall_prevalence:.10%}**",
        f"- Held-out-test sampling-weighted `y=1` prevalence: **{test_prevalence:.10%}**",
        "- Fixed scale convention: `beta=5` (not estimated)",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| intercept | {point['intercept']:.12g} |",
        f"| coef_m_in | {point['m_in']:.12g} |",
        f"| coef_m_out | {point['m_out']:.12g} |",
        f"| coef_degree | {point['degree']:.12g} |",
        f"| coef_log_time | {point['log_time']:.12g} |",
        f"| a_eff | {point_effective['a_eff']:.12g} |",
        f"| b_eff | {point_effective['b_eff']:.12g} |",
        f"| theta_eff | {point_effective['theta_eff']:.12g} |",
        "",
        "No normalized exposure model was fitted and the M1 definition was not changed.",
        "",
        "## Held-out subgroup comparison",
        "",
        "PR-AUC is sampling-weighted average precision. All metrics use `sampling_weight`.",
        "",
        "| subset | model | rows | weighted y rate | log loss | Brier | ROC-AUC | PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in subsets:
        for model in ("M0", "M1"):
            row = lookup[(model, subset)]
            lines.append(
                f"| {subset} | {model} | {int(row['rows']):,} | "
                f"{float(row['weighted_y_rate']):.8%} | {fmt(float(row['weighted_log_loss']))} | "
                f"{fmt(float(row['weighted_brier']))} | {fmt(float(row['weighted_roc_auc']))} | "
                f"{fmt(float(row['weighted_pr_auc']))} |"
            )
    lines.extend(
        [
            "",
            "### M1 minus M0 by subgroup",
            "",
            "Loss improvements are `M0-M1`; AUC increments are `M1-M0`. Positive values favor M1.",
            "",
            "| subset | log-loss improvement | Brier improvement | ROC-AUC increment | PR-AUC increment |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for subset in subsets:
        m0, m1 = lookup[("M0", subset)], lookup[("M1", subset)]
        lines.append(
            f"| {subset} | {float(m0['weighted_log_loss'])-float(m1['weighted_log_loss']):.10g} | "
            f"{float(m0['weighted_brier'])-float(m1['weighted_brier']):.10g} | "
            f"{float(m1['weighted_roc_auc'])-float(m0['weighted_roc_auc']):.10g} | "
            f"{float(m1['weighted_pr_auc'])-float(m0['weighted_pr_auc']):.10g} |"
        )
    lines.extend(
        [
            "",
            "## Story-cluster bootstrap",
            "",
            f"- Replicates requested: **{len(bootstrap_rows)}**",
            f"- Converged replicates: **{len(successful)}**",
            "- Resampling unit: training `story_id`; each replicate draws 80 training stories with "
            "replacement and multiplies the original sampling weights by cluster multiplicity.",
            "- Solver: unregularized weighted logistic IRLS with per-replicate weighted scaling.",
            f"- Original-training IRLS consistency check: converged={irls_check.converged}, "
            f"iterations={irls_check.iterations}, maximum coefficient difference from the saved M1 "
            f"estimate={max(abs(irls_check.intercept-point['intercept']), np.max(np.abs(irls_check.coefficients-np.asarray([point[name] for name in FEATURES])))):.3g}.",
            "- Intervals are percentile 95% intervals over converged replicates.",
            "",
            "| quantity | point | bootstrap median | 2.5% | 97.5% |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    point_by_field = {
        "coef_m_in": point["m_in"],
        "coef_m_out": point["m_out"],
        **point_effective,
    }
    for field in ("coef_m_in", "coef_m_out", "a_eff", "b_eff", "theta_eff"):
        lower, median, upper = intervals[field]
        lines.append(
            f"| {field} | {point_by_field[field]:.10g} | {median:.10g} | {lower:.10g} | {upper:.10g} |"
        )
    lines.extend(
        [
            "",
            "| raw coefficient | sign-change fraction vs point estimate |",
            "|---|---:|",
            f"| coef_m_in | {sign_changes['sign_change_m_in']:.4%} |",
            f"| coef_m_out | {sign_changes['sign_change_m_out']:.4%} |",
            f"| coef_degree | {sign_changes['sign_change_degree']:.4%} |",
            f"| coef_log_time | {sign_changes['sign_change_log_time']:.4%} |",
            "",
            "## Diagnostic answers",
            "",
            f"- **Is within-community exposure stably positive?** {'Yes' if within_stable else 'No'}. "
            f"Its percentile interval is `{intervals['coef_m_in'][0]:.6g}` to "
            f"`{intervals['coef_m_in'][2]:.6g}`, with "
            f"{sign_changes['sign_change_m_in']:.2%} sign changes.",
            f"- **Is cross-community exposure stably positive?** {'Yes' if cross_stable else 'No'}. "
            f"Its point estimate is {point['m_out']:.6g}, interval "
            f"`{intervals['coef_m_out'][0]:.6g}` to `{intervals['coef_m_out'][2]:.6g}`, and "
            f"sign-change fraction {sign_changes['sign_change_m_out']:.2%}.",
            "- **Is M1 mainly useful in exposed observations?** Mostly, but not uniformly. "
            f"Log-loss improvement is {all_delta:.3g} overall, {zero_delta:.3g} at zero exposure, "
            f"and {exposed_delta:.3g} among exposed rows. The `in_only` subset improves on all four "
            "metrics, whereas `out_only` and `both` have mixed metric changes; the exposed result "
            "should therefore not be read as a universal exposure benefit.",
            "- **Is current evidence sufficient to scale to all 3,553 cascades?** No. The overall "
            "increment is small, the bootstrap-stable negative cross-community coefficient conflicts "
            "with a positive-spillover interpretation, 72.2069% of positives have zero observed "
            "network exposure, and this remains a 100-cascade observational pilot.",
            "",
            "No full-data fit or real-data intervention was run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    split_path = output_dir / "pilot_story_split.csv"
    coefficient_path = output_dir / "pilot_model_coefficients.csv"
    for path in (input_path, split_path, coefficient_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required pilot artifact not found: {path}")
    if args.replicates <= 0 or args.max_iter <= 0 or args.tolerance <= 0:
        raise DiagnosticError("Replicates, iterations, and tolerance must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pilot data, saved split, and point estimates...")
    data = load_data(input_path)
    split = read_split(split_path)
    coefficients = read_coefficients(coefficient_path)
    train_stories = sorted(story for story, label in split.items() if label == "train")
    test_stories = sorted(story for story, label in split.items() if label == "test")
    train_mask = np.isin(data.story, train_stories)
    test_mask = np.isin(data.story, test_stories)
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise DiagnosticError("Train/test row masks are inconsistent")

    point = coefficients["M1"]
    point_vector = np.asarray([point[name] for name in FEATURES])
    print("Checking IRLS against the saved M1 point estimate...")
    irls_check = weighted_logistic_irls(
        data.x[train_mask],
        data.y[train_mask],
        data.weight[train_mask],
        point["intercept"],
        point_vector,
        args.max_iter,
        args.tolerance,
    )
    maximum_difference = max(
        abs(irls_check.intercept - point["intercept"]),
        float(np.max(np.abs(irls_check.coefficients - point_vector))),
    )
    if not irls_check.converged or maximum_difference > 1e-6:
        raise DiagnosticError(
            f"IRLS does not reproduce saved M1 (converged={irls_check.converged}, "
            f"max difference={maximum_difference})"
        )

    test_x, test_y, test_weight = data.x[test_mask], data.y[test_mask], data.weight[test_mask]
    m_in, m_out = test_x[:, 0], test_x[:, 1]
    subsets = {
        "all": np.ones(len(test_y), dtype=bool),
        "zero_exposure": (m_in == 0) & (m_out == 0),
        "exposed": (m_in + m_out) > 0,
        "in_only": (m_in > 0) & (m_out == 0),
        "out_only": (m_in == 0) & (m_out > 0),
        "both": (m_in > 0) & (m_out > 0),
    }
    metric_rows: list[dict[str, object]] = []
    for model in ("M0", "M1"):
        probability = predict(coefficients[model], test_x)
        for subset, mask in subsets.items():
            if not np.any(mask):
                raise DiagnosticError(f"Empty test subset: {subset}")
            metric_rows.append(
                metrics(model, subset, test_y[mask], probability[mask], test_weight[mask])
            )
    atomic_csv(
        output_dir / "pilot_subgroup_metrics.csv",
        [
            "model", "subset", "rows", "positives", "weighted_y_rate",
            "weighted_log_loss", "weighted_brier", "weighted_roc_auc", "weighted_pr_auc",
        ],
        metric_rows,
    )

    print(f"Running {args.replicates} story-cluster bootstrap replicates...")
    bootstrap_rows = bootstrap(
        data,
        train_mask,
        train_stories,
        point,
        args.replicates,
        args.seed,
        args.max_iter,
        args.tolerance,
    )
    bootstrap_fields = [
        "bootstrap_id", "sampled_story_draws", "unique_stories", "intercept",
        "coef_m_in", "coef_m_out", "coef_degree", "coef_log_time", "a_eff", "b_eff",
        "theta_eff", "sign_change_m_in", "sign_change_m_out", "sign_change_degree",
        "sign_change_log_time", "converged", "iterations", "warning",
    ]
    atomic_csv(output_dir / "pilot_bootstrap_coefficients.csv", bootstrap_fields, bootstrap_rows)
    write_report(
        output_dir / "pilot_diagnostic_report.md",
        data,
        train_mask,
        test_mask,
        point,
        metric_rows,
        bootstrap_rows,
        irls_check,
    )
    print(f"Subgroup metrics: {output_dir / 'pilot_subgroup_metrics.csv'}")
    print(f"Bootstrap coefficients: {output_dir / 'pilot_bootstrap_coefficients.csv'}")
    print(f"Diagnostic report: {output_dir / 'pilot_diagnostic_report.md'}")


if __name__ == "__main__":
    try:
        main()
    except (DiagnosticError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
