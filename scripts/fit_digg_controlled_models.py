#!/usr/bin/env python3
"""Fit leakage-safe controlled diagnostic models for the Digg pilot."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose_digg_pilot import (
    DiagnosticError,
    FitResult,
    atomic_csv,
    metrics,
    read_coefficients,
    read_split,
    sigmoid,
    weighted_logistic_irls,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPOSURE = REPO_ROOT / "data" / "processed" / "digg_exposure_pilot.csv.gz"
DEFAULT_VOTES = REPO_ROOT / "data" / "processed" / "digg_votes_clean.csv.gz"
DEFAULT_COMMUNITIES = REPO_ROOT / "data" / "processed" / "digg_communities.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "digg"
DEFAULT_SEED = 42
DEFAULT_REPLICATES = 200
BETA_SCALE = 5.0


@dataclass
class ControlledData:
    story: np.ndarray
    node: np.ndarray
    time_bin: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    columns: dict[str, np.ndarray]
    user_control_leakage: int
    cascade_control_leakage: int


@dataclass(frozen=True)
class Specification:
    name: str
    features: tuple[str, ...]
    time_control: str


CONTINUOUS_SPECS = {
    "D0": Specification("D0", ("degree", "log_time"), "log1p_time_bin"),
    "D1": Specification(
        "D1", ("m_in", "m_out", "degree", "log_time"), "log1p_time_bin"
    ),
    "D2": Specification(
        "D2",
        ("m_in", "m_out", "degree", "log_time", "log_user_activity"),
        "log1p_time_bin",
    ),
    "D3": Specification(
        "D3",
        (
            "m_in",
            "m_out",
            "degree",
            "log_time",
            "log_user_activity",
            "log_cascade_size",
        ),
        "log1p_time_bin",
    ),
}
PIECEWISE_FEATURES = (
    "m_in",
    "m_out",
    "degree",
    "log_user_activity",
    "log_cascade_size",
    "time_1_3h",
    "time_3_6h",
    "time_6_12h",
    "time_12_24h",
    "time_24h_plus",
)
PIECEWISE_SPEC = Specification("D3", PIECEWISE_FEATURES, "piecewise_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure", type=Path, default=DEFAULT_EXPOSURE)
    parser.add_argument("--votes", type=Path, default=DEFAULT_VOTES)
    parser.add_argument("--communities", type=Path, default=DEFAULT_COMMUNITIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def read_exposure(path: Path) -> tuple[dict[str, list[float | int]], set[int], set[int]]:
    fields = (
        "story_id", "time_bin", "node_id", "m_in", "m_out", "degree", "y",
        "sampling_weight",
    )
    columns: dict[str, list[float | int]] = {field: [] for field in fields}
    stories: set[int] = set()
    nodes: set[int] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(fields) - set(reader.fieldnames or [])
        if missing:
            raise DiagnosticError(f"Exposure table is missing fields: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                story = int(row["story_id"])
                time_bin = int(row["time_bin"])
                node = int(row["node_id"])
                m_in = float(row["m_in"])
                m_out = float(row["m_out"])
                degree = float(row["degree"])
                outcome = int(row["y"])
                weight = float(row["sampling_weight"])
            except ValueError as error:
                raise DiagnosticError(f"Invalid exposure value at row {row_number}") from error
            values = (m_in, m_out, degree, weight)
            if not all(math.isfinite(value) for value in values):
                raise DiagnosticError(f"Non-finite exposure value at row {row_number}")
            if time_bin < 0 or min(m_in, m_out) < 0 or degree <= 0 or outcome not in (0, 1):
                raise DiagnosticError(f"Invalid exposure feature at row {row_number}")
            if weight <= 0:
                raise DiagnosticError(f"Non-positive sampling weight at row {row_number}")
            for field, value in zip(
                fields, (story, time_bin, node, m_in, m_out, degree, outcome, weight)
            ):
                columns[field].append(value)
            stories.add(story)
            nodes.add(node)
    return columns, stories, nodes


def read_community_nodes(path: Path) -> set[int]:
    nodes: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["node_id", "community"]:
            raise DiagnosticError(f"Unexpected community schema: {reader.fieldnames}")
        for row in reader:
            nodes.add(int(row["node_id"]))
    return nodes


def build_temporal_indexes(
    votes_path: Path,
    target_nodes: set[int],
    selected_stories: set[int],
    community_nodes: set[int],
) -> tuple[
    dict[int, list[int]],
    dict[int, int],
    dict[int, list[int]],
    dict[tuple[int, int], int],
]:
    user_vote_times: dict[int, list[int]] = defaultdict(list)
    story_start: dict[int, int] = {}
    covered_story_times: dict[int, list[int]] = defaultdict(list)
    selected_story_user_time: dict[tuple[int, int], int] = {}
    with gzip.open(votes_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["vote_date", "voter_id", "story_id"]
        if reader.fieldnames != expected:
            raise DiagnosticError(f"Unexpected cleaned-vote schema: {reader.fieldnames}")
        for row in reader:
            vote_date = int(row["vote_date"])
            voter = int(row["voter_id"])
            story = int(row["story_id"])
            if voter in target_nodes:
                user_vote_times[voter].append(vote_date)
            if story in selected_stories:
                story_start[story] = min(story_start.get(story, vote_date), vote_date)
                selected_story_user_time[(story, voter)] = vote_date
                if voter in community_nodes:
                    covered_story_times[story].append(vote_date)
    missing_starts = selected_stories - set(story_start)
    if missing_starts:
        raise DiagnosticError(f"Missing start times for stories: {sorted(missing_starts)}")
    for times in user_vote_times.values():
        times.sort()
    for times in covered_story_times.values():
        times.sort()
    return user_vote_times, story_start, covered_story_times, selected_story_user_time


def time_category_columns(time_bin: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "time_1_3h": ((time_bin >= 1) & (time_bin < 3)).astype(float),
        "time_3_6h": ((time_bin >= 3) & (time_bin < 6)).astype(float),
        "time_6_12h": ((time_bin >= 6) & (time_bin < 12)).astype(float),
        "time_12_24h": ((time_bin >= 12) & (time_bin < 24)).astype(float),
        "time_24h_plus": (time_bin >= 24).astype(float),
    }


def construct_controlled_data(
    exposure_path: Path, votes_path: Path, communities_path: Path
) -> ControlledData:
    raw, selected_stories, target_nodes = read_exposure(exposure_path)
    community_nodes = read_community_nodes(communities_path)
    indexes = build_temporal_indexes(
        votes_path, target_nodes, selected_stories, community_nodes
    )
    user_times, story_start, covered_times, selected_story_user_time = indexes
    story = np.asarray(raw["story_id"], dtype=np.int64)
    node = np.asarray(raw["node_id"], dtype=np.int64)
    time_bin = np.asarray(raw["time_bin"], dtype=np.int64)
    user_prior_votes = np.empty(len(story), dtype=np.float64)
    cascade_size_so_far = np.empty(len(story), dtype=np.float64)
    user_leakage = 0
    cascade_leakage = 0
    for index, (story_id, node_id, bin_id) in enumerate(zip(story, node, time_bin)):
        absolute_time = story_start[int(story_id)] + int(bin_id) * 3600
        prior = bisect.bisect_left(user_times[int(node_id)], absolute_time)
        same_story_time = selected_story_user_time.get((int(story_id), int(node_id)))
        if same_story_time is not None and same_story_time < absolute_time:
            prior -= 1
            user_leakage += 1
        if prior < 0:
            raise DiagnosticError("Negative prior-vote count")
        user_prior_votes[index] = prior
        prior_covered = bisect.bisect_left(covered_times[int(story_id)], absolute_time)
        cascade_size_so_far[index] = prior_covered
        if bin_id == 0 and prior_covered != 0:
            cascade_leakage += 1
    if user_leakage:
        raise DiagnosticError(
            f"{user_leakage} exposure rows refer to nodes already adopted in the current story"
        )
    if cascade_leakage:
        raise DiagnosticError(f"{cascade_leakage} time-zero rows have prior cascade adopters")

    columns = {
        "m_in": np.asarray(raw["m_in"], dtype=np.float64),
        "m_out": np.asarray(raw["m_out"], dtype=np.float64),
        "degree": np.asarray(raw["degree"], dtype=np.float64),
        "log_time": np.log1p(time_bin.astype(np.float64)),
        "log_user_activity": np.log1p(user_prior_votes),
        "log_cascade_size": np.log1p(cascade_size_so_far),
    }
    columns.update(time_category_columns(time_bin))
    return ControlledData(
        story=story,
        node=node,
        time_bin=time_bin,
        y=np.asarray(raw["y"], dtype=np.float64),
        weight=np.asarray(raw["sampling_weight"], dtype=np.float64),
        columns=columns,
        user_control_leakage=user_leakage,
        cascade_control_leakage=cascade_leakage,
    )


def matrix(data: ControlledData, spec: Specification) -> np.ndarray:
    return np.column_stack([data.columns[name] for name in spec.features])


def initial_for_spec(
    spec: Specification,
    previous: tuple[Specification, FitResult] | None,
    saved: dict[str, float] | None = None,
) -> tuple[float, np.ndarray]:
    coefficients = np.zeros(len(spec.features), dtype=float)
    intercept = 0.0
    if saved is not None:
        intercept = saved["intercept"]
        for index, feature in enumerate(spec.features):
            if feature in saved:
                coefficients[index] = saved[feature]
    if previous is not None:
        old_spec, old_fit = previous
        intercept = old_fit.intercept
        old = dict(zip(old_spec.features, old_fit.coefficients))
        for index, feature in enumerate(spec.features):
            coefficients[index] = old.get(feature, coefficients[index])
    return intercept, coefficients


def fit_spec(
    data: ControlledData,
    spec: Specification,
    mask: np.ndarray,
    initial_intercept: float,
    initial_coefficients: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> FitResult:
    fit = weighted_logistic_irls(
        matrix(data, spec)[mask],
        data.y[mask],
        data.weight[mask],
        initial_intercept,
        initial_coefficients,
        max_iter,
        tolerance,
    )
    if not fit.converged:
        raise DiagnosticError(
            f"{spec.name}/{spec.time_control} failed to converge: {fit.warning}"
        )
    return fit


def fit_models(
    data: ControlledData,
    train_mask: np.ndarray,
    exposed_mask: np.ndarray,
    saved: dict[str, dict[str, float]],
    max_iter: int,
    tolerance: float,
) -> dict[tuple[str, str, str], tuple[Specification, FitResult]]:
    fits: dict[tuple[str, str, str], tuple[Specification, FitResult]] = {}
    previous: tuple[Specification, FitResult] | None = None
    for name in ("D0", "D1", "D2", "D3"):
        spec = CONTINUOUS_SPECS[name]
        saved_start = saved["M0"] if name == "D0" else saved["M1"] if name == "D1" else None
        start = initial_for_spec(spec, previous, saved_start)
        fit = fit_spec(data, spec, train_mask, *start, max_iter, tolerance)
        fits[("full", name, spec.time_control)] = (spec, fit)
        previous = (spec, fit)

    full_d3 = fits[("full", "D3", "log1p_time_bin")]
    start = initial_for_spec(PIECEWISE_SPEC, full_d3)
    piece_fit = fit_spec(data, PIECEWISE_SPEC, train_mask, *start, max_iter, tolerance)
    fits[("full", "D3", "piecewise_time")] = (PIECEWISE_SPEC, piece_fit)

    previous = fits[("full", "D1", "log1p_time_bin")]
    for name in ("D1", "D2", "D3"):
        spec = CONTINUOUS_SPECS[name]
        start = initial_for_spec(spec, previous)
        fit = fit_spec(data, spec, train_mask & exposed_mask, *start, max_iter, tolerance)
        fits[("exposed_only", name, spec.time_control)] = (spec, fit)
        previous = (spec, fit)
    return fits


def fit_to_dict(spec: Specification, fit: FitResult) -> dict[str, float]:
    result = {name: float(value) for name, value in zip(spec.features, fit.coefficients)}
    result["intercept"] = fit.intercept
    return result


def coefficient_rows(
    fits: dict[tuple[str, str, str], tuple[Specification, FitResult]]
) -> list[dict[str, object]]:
    coefficient_fields = (
        "m_in", "m_out", "degree", "log_time", "log_user_activity", "log_cascade_size",
        "time_1_3h", "time_3_6h", "time_6_12h", "time_12_24h", "time_24h_plus",
    )
    rows: list[dict[str, object]] = []
    for (scope, model, time_control), (spec, fit) in fits.items():
        values = fit_to_dict(spec, fit)
        row: dict[str, object] = {
            "fit_scope": scope,
            "model": model,
            "time_control": time_control,
            "intercept": fit.intercept,
        }
        for field in coefficient_fields:
            row[f"coef_{field}"] = values.get(field, "")
        row["a_eff"] = values["m_in"] / BETA_SCALE if "m_in" in values else ""
        row["b_eff"] = values["m_out"] / BETA_SCALE if "m_out" in values else ""
        row["theta_eff"] = -values["degree"] / BETA_SCALE
        row["converged"] = int(fit.converged)
        row["iterations"] = fit.iterations
        row["warning"] = fit.warning
        rows.append(row)
    return rows


def model_probability(
    data: ControlledData, spec: Specification, fit: FitResult, mask: np.ndarray
) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        linear_predictor = fit.intercept + matrix(data, spec)[mask] @ fit.coefficients
    if not np.all(np.isfinite(linear_predictor)):
        raise DiagnosticError(f"Non-finite predictions for {spec.name}/{spec.time_control}")
    return sigmoid(linear_predictor)


def metric_rows(
    data: ControlledData,
    fits: dict[tuple[str, str, str], tuple[Specification, FitResult]],
    test_mask: np.ndarray,
    exposed_mask: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (scope, model, time_control), (spec, fit) in fits.items():
        evaluation_masks: dict[str, np.ndarray]
        if scope == "full":
            evaluation_masks = {
                "all_test": test_mask,
                "zero_exposure_test": test_mask & ~exposed_mask,
                "exposed_test": test_mask & exposed_mask,
            }
        else:
            evaluation_masks = {"exposed_test": test_mask & exposed_mask}
        for subset, mask in evaluation_masks.items():
            probability = model_probability(data, spec, fit, mask)
            row = metrics(
                model,
                subset,
                data.y[mask],
                probability,
                data.weight[mask],
            )
            row["fit_scope"] = scope
            row["time_control"] = time_control
            rows.append(row)
    return rows


def bootstrap_models(
    data: ControlledData,
    train_mask: np.ndarray,
    train_stories: list[int],
    fits: dict[tuple[str, str, str], tuple[Specification, FitResult]],
    replicates: int,
    seed: int,
    max_iter: int,
    tolerance: float,
) -> list[dict[str, object]]:
    row_stories = data.story[train_mask]
    base_weight = data.weight[train_mask]
    rng = random.Random(seed)
    models = ["D1", "D2", "D3"]
    rows: list[dict[str, object]] = []
    matrices = {
        name: matrix(data, CONTINUOUS_SPECS[name])[train_mask] for name in models
    }
    for replicate in range(1, replicates + 1):
        sampled = rng.choices(train_stories, k=len(train_stories))
        multiplicity = Counter(sampled)
        multiplier = np.fromiter(
            (multiplicity.get(int(story), 0) for story in row_stories),
            dtype=np.float64,
            count=len(row_stories),
        )
        bootstrap_weight = base_weight * multiplier
        for model in models:
            spec = CONTINUOUS_SPECS[model]
            point_fit = fits[("full", model, "log1p_time_bin")][1]
            fit = weighted_logistic_irls(
                matrices[model],
                data.y[train_mask],
                bootstrap_weight,
                point_fit.intercept,
                point_fit.coefficients,
                max_iter,
                tolerance,
            )
            values = fit_to_dict(spec, fit)
            rows.append(
                {
                    "bootstrap_id": replicate,
                    "model": model,
                    "sampled_story_draws": len(sampled),
                    "unique_stories": len(multiplicity),
                    "coef_m_in": values["m_in"],
                    "coef_m_out": values["m_out"],
                    "coef_degree": values["degree"],
                    "a_eff": values["m_in"] / BETA_SCALE,
                    "b_eff": values["m_out"] / BETA_SCALE,
                    "theta_eff": -values["degree"] / BETA_SCALE,
                    "m_in_positive": int(values["m_in"] > 0),
                    "m_in_negative": int(values["m_in"] < 0),
                    "m_out_positive": int(values["m_out"] > 0),
                    "m_out_negative": int(values["m_out"] < 0),
                    "degree_positive": int(values["degree"] > 0),
                    "degree_negative": int(values["degree"] < 0),
                    "converged": int(fit.converged),
                    "iterations": fit.iterations,
                    "warning": fit.warning,
                }
            )
        if replicate % 25 == 0:
            print(f"Bootstrap {replicate}/{replicates}")
    return rows


def bootstrap_summary(
    rows: list[dict[str, object]], model: str, field: str
) -> tuple[float, float, float, float, float, int]:
    selected = [row for row in rows if row["model"] == model and row["converged"]]
    values = np.asarray([float(row[field]) for row in selected], dtype=float)
    lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
    positive = float(np.mean(values > 0))
    negative = float(np.mean(values < 0))
    return float(lower), float(median), float(upper), positive, negative, len(values)


def save_coefficient_plot(
    path: Path,
    fits: dict[tuple[str, str, str], tuple[Specification, FitResult]],
) -> None:
    order = [
        ("full", "D1", "log1p_time_bin", "Full D1"),
        ("full", "D2", "log1p_time_bin", "Full D2"),
        ("full", "D3", "log1p_time_bin", "Full D3"),
        ("full", "D3", "piecewise_time", "Full D3 piecewise"),
        ("exposed_only", "D1", "log1p_time_bin", "Exposed D1"),
        ("exposed_only", "D2", "log1p_time_bin", "Exposed D2"),
        ("exposed_only", "D3", "log1p_time_bin", "Exposed D3"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for axis, feature, title in zip(
        axes, ("m_in", "m_out", "degree"), ("coef_m_in", "coef_m_out", "coef_degree")
    ):
        labels: list[str] = []
        values: list[float] = []
        for scope, model, time_control, label in order:
            spec, fit = fits[(scope, model, time_control)]
            coefficient = fit_to_dict(spec, fit)[feature]
            labels.append(label)
            values.append(coefficient)
        colors = ["#2b8cbe" if value >= 0 else "#d95f0e" for value in values]
        positions = np.arange(len(labels))
        axis.bar(positions, values, color=colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, labels, rotation=55, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Original-scale logistic coefficient")
    figure.suptitle("Digg controlled count-exposure coefficients")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.10g}"


def write_report(
    path: Path,
    data: ControlledData,
    fits: dict[tuple[str, str, str], tuple[Specification, FitResult]],
    metrics_rows: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    full_keys = [
        ("full", "D0", "log1p_time_bin"),
        ("full", "D1", "log1p_time_bin"),
        ("full", "D2", "log1p_time_bin"),
        ("full", "D3", "log1p_time_bin"),
        ("full", "D3", "piecewise_time"),
    ]
    exposed_keys = [
        ("exposed_only", name, "log1p_time_bin") for name in ("D1", "D2", "D3")
    ]
    lookup = {
        (str(row["fit_scope"]), str(row["model"]), str(row["time_control"]), str(row["subset"])): row
        for row in metrics_rows
    }
    d2_values = fit_to_dict(*fits[("full", "D2", "log1p_time_bin")])
    d3_values = fit_to_dict(*fits[("full", "D3", "log1p_time_bin")])
    piece_values = fit_to_dict(*fits[("full", "D3", "piecewise_time")])
    bootstrap_tables = {
        (model, field): bootstrap_summary(bootstrap_rows, model, field)
        for model in ("D1", "D2", "D3")
        for field in ("coef_m_in", "coef_m_out", "coef_degree", "a_eff", "b_eff", "theta_eff")
    }
    all_bootstrap_converged = sum(int(row["converged"]) for row in bootstrap_rows)
    exposed_test = test_mask & ((data.columns["m_in"] + data.columns["m_out"]) > 0)
    d0_exposed = lookup[("full", "D0", "log1p_time_bin", "exposed_test")]
    d1_exposed = lookup[("full", "D1", "log1p_time_bin", "exposed_test")]
    lines = [
        "# Digg controlled-model diagnostic report",
        "",
        "## Leakage-safe controls",
        "",
        "- `user_prior_votes`: votes by the same node in other stories with timestamp strictly "
        "earlier than the current window's absolute start.",
        "- `cascade_size_so_far`: community-labelled adopters in the current story strictly before "
        "the current window start.",
        "- Transformations: `log_user_activity=log1p(user_prior_votes)` and "
        "`log_cascade_size=log1p(cascade_size_so_far)`.",
        f"- User-control leakage violations: **{data.user_control_leakage}**",
        f"- Cascade-control leakage violations: **{data.cascade_control_leakage}**",
        f"- Rows: **{len(data.y):,}**; train rows: **{int(np.sum(train_mask)):,}**; "
        f"test rows: **{int(np.sum(test_mask)):,}**",
        "- The existing 80/20 story split and original sampling weights were reused.",
        "- No `frac_in`/`frac_out`, coefficient constraints, class weights, or interventions were used.",
        "",
        "## Full-population coefficients",
        "",
        "All coefficients below are converted back to their original feature scales.",
        "",
        "| model | time control | coef_m_in | coef_m_out | coef_degree | log_user_activity | log_cascade_size | converged/iterations |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for key in full_keys:
        spec, fit = fits[key]
        values = fit_to_dict(spec, fit)
        lines.append(
            f"| {key[1]} | {key[2]} | {fmt(values.get('m_in', float('nan')))} | "
            f"{fmt(values.get('m_out', float('nan')))} | {fmt(values['degree'])} | "
            f"{fmt(values.get('log_user_activity', float('nan')))} | "
            f"{fmt(values.get('log_cascade_size', float('nan')))} | "
            f"{'yes' if fit.converged else 'no'}/{fit.iterations} |"
        )
    lines.extend(
        [
            "",
            "## Time-control sensitivity",
            "",
            "The piecewise D3 uses 0–1h as the reference and indicators for 1–3h, 3–6h, "
            "6–12h, 12–24h, and 24h+.",
            "",
            "| D3 time control | coef_m_in | coef_m_out | coef_degree |",
            "|---|---:|---:|---:|",
            f"| log1p(time_bin) | {d3_values['m_in']:.10g} | {d3_values['m_out']:.10g} | {d3_values['degree']:.10g} |",
            f"| piecewise categories | {piece_values['m_in']:.10g} | {piece_values['m_out']:.10g} | {piece_values['degree']:.10g} |",
            "",
            "## Held-out metrics",
            "",
            "PR-AUC is sampling-weighted average precision. Full models are evaluated on all, "
            "zero-exposure, and exposed test rows; exposed-only fits are evaluated only on exposed rows.",
            "",
            "| fit scope | model | time control | test subset | rows | weighted y rate | log loss | Brier | ROC-AUC | PR-AUC |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics_rows:
        lines.append(
            f"| {row['fit_scope']} | {row['model']} | {row['time_control']} | {row['subset']} | "
            f"{int(row['rows']):,} | {float(row['weighted_y_rate']):.8%} | "
            f"{fmt(float(row['weighted_log_loss']))} | {fmt(float(row['weighted_brier']))} | "
            f"{fmt(float(row['weighted_roc_auc']))} | {fmt(float(row['weighted_pr_auc']))} |"
        )
    lines.extend(
        [
            "",
            "## Exposed-only association fits",
            "",
            "These coefficients condition on `m_in+m_out>0`; they are exposed-only associations "
            "and are not models for the full user population.",
            "",
            "| model | coef_m_in | coef_m_out | coef_degree | log_user_activity | log_cascade_size |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in exposed_keys:
        spec, fit = fits[key]
        values = fit_to_dict(spec, fit)
        lines.append(
            f"| {key[1]} | {values['m_in']:.10g} | {values['m_out']:.10g} | "
            f"{values['degree']:.10g} | {fmt(values.get('log_user_activity', float('nan')))} | "
            f"{fmt(values.get('log_cascade_size', float('nan')))} |"
        )
    lines.extend(
        [
            "",
            "## Story-cluster bootstrap",
            "",
            f"- Requested fits: **{len(bootstrap_rows):,}** (200 replicates × D1/D2/D3)",
            f"- Converged fits: **{all_bootstrap_converged:,}**",
            "- Each replicate resamples the same 80 training stories with replacement and uses "
            "cluster multiplicity times the original sampling weight.",
            "- Intervals are percentile 95% intervals over converged fits. Positive/negative shares "
            "are computed without imposing signs.",
            "",
            "| model | quantity | point | 2.5% | median | 97.5% | positive | negative | n |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ("D1", "D2", "D3"):
        spec, fit = fits[("full", model, "log1p_time_bin")]
        values = fit_to_dict(spec, fit)
        points = {
            "coef_m_in": values["m_in"],
            "coef_m_out": values["m_out"],
            "coef_degree": values["degree"],
            "a_eff": values["m_in"] / BETA_SCALE,
            "b_eff": values["m_out"] / BETA_SCALE,
            "theta_eff": -values["degree"] / BETA_SCALE,
        }
        for field in ("coef_m_in", "coef_m_out", "coef_degree", "a_eff", "b_eff", "theta_eff"):
            lower, median, upper, positive, negative, count = bootstrap_tables[(model, field)]
            lines.append(
                f"| {model} | {field} | {points[field]:.10g} | {lower:.10g} | {median:.10g} | "
                f"{upper:.10g} | {positive:.2%} | {negative:.2%} | {count} |"
            )
    degree_still_positive = d2_values["degree"] > 0
    m_out_log_negative = d3_values["m_out"] < 0
    m_out_piece_negative = piece_values["m_out"] < 0
    d1_in = bootstrap_tables[("D1", "coef_m_in")]
    d2_in = bootstrap_tables[("D2", "coef_m_in")]
    d3_in = bootstrap_tables[("D3", "coef_m_in")]
    d1_out = bootstrap_tables[("D1", "coef_m_out")]
    d2_out = bootstrap_tables[("D2", "coef_m_out")]
    d3_out = bootstrap_tables[("D3", "coef_m_out")]
    logloss_gain = float(d0_exposed["weighted_log_loss"]) - float(d1_exposed["weighted_log_loss"])
    roc_gain = float(d1_exposed["weighted_roc_auc"]) - float(d0_exposed["weighted_roc_auc"])
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. **After controlling user history, is degree still positive?** The D2 point estimate "
            f"is positive (`{d2_values['degree']:.6g}`), but its 95% interval "
            f"`[{bootstrap_tables[('D2','coef_degree')][0]:.6g}, "
            f"{bootstrap_tables[('D2','coef_degree')][2]:.6g}]` crosses zero and only "
            f"{bootstrap_tables[('D2','coef_degree')][3]:.2%} of replicates are positive. In D3, "
            f"degree is more stable: {bootstrap_tables[('D3','coef_degree')][3]:.2%} positive.",
            f"2. **After flexible time and cascade-heat controls, is m_out still negative?** Both D3 "
            f"point estimates are negative: log-time `{d3_values['m_out']:.6g}` and piecewise time "
            f"`{piece_values['m_out']:.6g}`. This is not bootstrap-stable evidence of a negative "
            f"effect: the log-time D3 interval `[{d3_out[0]:.6g}, {d3_out[2]:.6g}]` crosses zero "
            f"and {d3_out[4]:.2%} of replicates are negative.",
            "3. **Are exposure signs bootstrap-stable?** Only in the uncontrolled D1: `m_in` is "
            f"positive in {d1_in[3]:.1%} and `m_out` negative in {d1_out[4]:.1%}. After controls, "
            f"`m_in` is positive in D2/D3 at {d2_in[3]:.1%}/{d3_in[3]:.1%}; `m_out` is negative "
            f"at only {d2_out[4]:.1%}/{d3_out[4]:.1%}. D2's `m_out` point estimate reverses "
            "positive, and both D2/D3 `m_out` intervals cross zero. No theoretical sign was imposed.",
            "4. **Is M1 clearly better for exposed users?** It has more signal there, but not a "
            f"uniformly decisive gain: full-fit D1 improves exposed-test log loss by {logloss_gain:.3g} "
            f"and ROC-AUC by {roc_gain:.4g}, while Brier/PR changes are mixed. In exposed-only D2/D3, "
            "`m_out` becomes positive and degree becomes negative, further showing sensitivity to "
            "conditioning and controls. Exposed-only fits are associations conditional on exposure.",
            "5. **Can Digg estimate effective paper a, b, theta?** Not with a defensible structural "
            "interpretation in this pilot. The signs of `b_eff` and `theta_eff` remain contrary to "
            "the paper-parameter interpretation after controls. Digg currently supports predictive "
            "and associational validation only, not recovery of causal or structural a/b/theta.",
            "",
            "This remains a 100-cascade pilot. No fit was expanded to all 3,553 cascades, no "
            "coefficient was clipped, and no real-data intervention was run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    exposure_path = args.exposure.expanduser().resolve()
    votes_path = args.votes.expanduser().resolve()
    communities_path = args.communities.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    split_path = output_dir / "pilot_story_split.csv"
    saved_coefficient_path = output_dir / "pilot_model_coefficients.csv"
    for path in (exposure_path, votes_path, communities_path, split_path, saved_coefficient_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")
    if args.replicates != 200:
        raise DiagnosticError("This diagnostic requires exactly 200 bootstrap replicates")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Constructing strict-past user activity and cascade-size controls...")
    data = construct_controlled_data(exposure_path, votes_path, communities_path)
    split = read_split(split_path)
    saved = read_coefficients(saved_coefficient_path)
    train_stories = sorted(story for story, label in split.items() if label == "train")
    test_stories = sorted(story for story, label in split.items() if label == "test")
    train_mask = np.isin(data.story, train_stories)
    test_mask = np.isin(data.story, test_stories)
    exposed_mask = (data.columns["m_in"] + data.columns["m_out"]) > 0
    if not np.all(train_mask | test_mask) or np.any(train_mask & test_mask):
        raise DiagnosticError("Controlled rows do not respect the saved story split")

    print("Fitting full, time-sensitive, and exposed-only diagnostic models...")
    fits = fit_models(
        data, train_mask, exposed_mask, saved, args.max_iter, args.tolerance
    )
    for model, saved_name in (("D0", "M0"), ("D1", "M1")):
        spec, fit = fits[("full", model, "log1p_time_bin")]
        values = fit_to_dict(spec, fit)
        differences = [abs(values["intercept"] - saved[saved_name]["intercept"])]
        for feature in spec.features:
            differences.append(abs(values[feature] - saved[saved_name][feature]))
        maximum_difference = max(differences)
        if maximum_difference > 5e-6:
            raise DiagnosticError(
                f"{model} does not reproduce saved {saved_name}; maximum difference="
                f"{maximum_difference:.12g}"
            )

    coefficients = coefficient_rows(fits)
    coefficient_fields = list(coefficients[0])
    atomic_csv(output_dir / "controlled_model_coefficients.csv", coefficient_fields, coefficients)
    evaluation = metric_rows(data, fits, test_mask, exposed_mask)
    metric_fields = [
        "fit_scope", "model", "time_control", "subset", "rows", "positives",
        "weighted_y_rate", "weighted_log_loss", "weighted_brier", "weighted_roc_auc",
        "weighted_pr_auc",
    ]
    atomic_csv(output_dir / "controlled_model_metrics.csv", metric_fields, evaluation)

    print("Running 200 shared story-cluster bootstrap replicates for D1/D2/D3...")
    bootstrap_rows = bootstrap_models(
        data,
        train_mask,
        train_stories,
        fits,
        args.replicates,
        args.seed,
        args.max_iter,
        args.tolerance,
    )
    bootstrap_fields = list(bootstrap_rows[0])
    atomic_csv(output_dir / "controlled_model_bootstrap.csv", bootstrap_fields, bootstrap_rows)
    save_coefficient_plot(output_dir / "controlled_coefficient_plot.png", fits)
    write_report(
        output_dir / "controlled_model_report.md",
        data,
        fits,
        evaluation,
        bootstrap_rows,
        train_mask,
        test_mask,
    )
    print(f"Coefficients: {output_dir / 'controlled_model_coefficients.csv'}")
    print(f"Metrics: {output_dir / 'controlled_model_metrics.csv'}")
    print(f"Bootstrap: {output_dir / 'controlled_model_bootstrap.csv'}")
    print(f"Report: {output_dir / 'controlled_model_report.md'}")


if __name__ == "__main__":
    try:
        main()
    except (DiagnosticError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
