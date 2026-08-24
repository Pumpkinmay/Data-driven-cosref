#!/usr/bin/env python3
"""Audit and model a topology-selected two-community Digg subsystem."""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from diagnose_digg_pilot import metrics, sigmoid, weighted_logistic_irls


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "digg"
SEED = 42
WINDOW_SECONDS = 3600
NEGATIVE_RATIO = 5
MIN_COMMUNITY_SIZE = 1000
MIN_CASCADE_SIZE = 20
MAX_CASCADES = 100
BETA = 5.0


class AnalysisError(RuntimeError):
    pass


class DeterministicGzipWriter:
    def __init__(self, path: Path):
        self.raw = path.open("wb")
        self.gz = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)
        self.text = io.TextIOWrapper(self.gz, encoding="utf-8", newline="")

    def __enter__(self) -> TextIO:
        return self.text

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.text.close()


@dataclass(frozen=True)
class Choice:
    story_id: int
    size: int
    community_0_adopters: int
    community_1_adopters: int
    stratum: str


@dataclass
class CascadeAudit:
    rows: int = 0
    positives: int = 0
    negatives: int = 0
    positive_windows: int = 0


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


def load_communities(path: Path) -> tuple[dict[int, int], Counter[int]]:
    labels: dict[int, int] = {}
    sizes: Counter[int] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["node_id", "community"]:
            raise AnalysisError(f"Unexpected community schema: {reader.fieldnames}")
        for row in reader:
            node = int(row["node_id"])
            community = int(row["community"])
            if node in labels:
                raise AnalysisError(f"Duplicate community label for node {node}")
            labels[node] = community
            sizes[community] += 1
    if len(sizes) != 7277:
        raise AnalysisError(f"Expected 7277 communities, found {len(sizes)}")
    return labels, sizes


def earliest_vote(path: Path) -> int:
    earliest: int | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = int(row["vote_date"])
            earliest = value if earliest is None else min(earliest, value)
    if earliest is None:
        raise AnalysisError("Cleaned vote table is empty")
    return earliest


def audit_baseline(
    path: Path,
    cutoff: int,
    labels: dict[int, int],
    large: set[int],
) -> tuple[Counter[int], Counter[tuple[int, int]], int]:
    internal: Counter[int] = Counter()
    large_edges: Counter[tuple[int, int]] = Counter()
    baseline_rows = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["friend_date", "user_id", "friend_id", "mutual"]
        if reader.fieldnames != expected:
            raise AnalysisError(f"Unexpected friendship schema: {reader.fieldnames}")
        for row in reader:
            friend_date = int(row["friend_date"])
            if friend_date <= 0 or friend_date > cutoff:
                continue
            source = int(row["friend_id"])
            target = int(row["user_id"])
            source_community = labels[source]
            target_community = labels[target]
            baseline_rows += 1
            if source_community == target_community:
                internal[source_community] += 1
            elif source_community in large and target_community in large:
                large_edges[(source_community, target_community)] += 1
    return internal, large_edges, baseline_rows


def choose_pair(
    large: set[int], large_edges: Counter[tuple[int, int]]
) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []
    ordered = sorted(large)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            forward = large_edges[(first, second)]
            reverse = large_edges[(second, first)]
            candidates.append((forward + reverse, first, second, forward, reverse))
    if not candidates:
        raise AnalysisError("No eligible community pairs")
    _, first, second, forward, reverse = min(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    )
    return first, second, forward, reverse


def build_induced_network(
    path: Path,
    cutoff: int,
    pair_nodes: set[int],
) -> tuple[dict[int, tuple[int, ...]], set[tuple[int, int]]]:
    edges: set[tuple[int, int]] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            friend_date = int(row["friend_date"])
            if friend_date <= 0 or friend_date > cutoff:
                continue
            source = int(row["friend_id"])
            target = int(row["user_id"])
            if source in pair_nodes and target in pair_nodes:
                edges.add((source, target))
    incoming_sets: dict[int, set[int]] = defaultdict(set)
    for source, target in edges:
        incoming_sets[target].add(source)
    incoming = {node: tuple(sorted(values)) for node, values in incoming_sets.items()}
    return incoming, edges


def write_community_outputs(
    sizes: Counter[int],
    internal: Counter[int],
    large_edges: Counter[tuple[int, int]],
    cutoff: int,
    baseline_rows: int,
    pair: tuple[int, int, int, int],
) -> None:
    ranked = sorted(sizes, key=lambda community: (-sizes[community], community))
    rows = [
        {
            "size_rank": rank,
            "community": community,
            "node_count": sizes[community],
            "node_share": sizes[community] / sum(sizes.values()),
            "internal_influence_edges": internal[community],
        }
        for rank, community in enumerate(ranked, start=1)
    ]
    atomic_csv(
        OUTPUT / "community_size_diagnostics.csv",
        ["size_rank", "community", "node_count", "node_share", "internal_influence_edges"],
        rows,
    )

    first, second, forward, reverse = pair
    payload = {
        "selection_uses_vote_outcomes": False,
        "selection_rule": (
            "Among original Leiden communities with at least 1000 nodes, maximize the sum "
            "of baseline directed influence edges in both directions; break exact ties "
            "lexicographically by the two ascending original community IDs."
        ),
        "baseline_cutoff_earliest_vote_date": cutoff,
        "minimum_community_size": MIN_COMMUNITY_SIZE,
        "eligible_community_count": sum(value >= MIN_COMMUNITY_SIZE for value in sizes.values()),
        "community_0": {"original_community": first, "nodes": sizes[first]},
        "community_1": {"original_community": second, "nodes": sizes[second]},
        "directed_edges_community_0_to_1": forward,
        "directed_edges_community_1_to_0": reverse,
        "bidirectional_total_cross_edges": forward + reverse,
        "pair_will_not_be_changed_based_on_model_results": True,
        "random_seed_for_downstream_sampling": SEED,
    }
    (OUTPUT / "selected_community_pair.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    size_values = np.asarray(list(sizes.values()), dtype=float)
    top_pairs = []
    large = sorted(community for community, size in sizes.items() if size >= MIN_COMMUNITY_SIZE)
    for i, source in enumerate(large):
        for target in large[i + 1 :]:
            st = large_edges[(source, target)]
            ts = large_edges[(target, source)]
            top_pairs.append((st + ts, source, target, st, ts))
    top_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    lines = [
        "# Digg community-size and baseline-edge audit",
        "",
        "## Scope",
        "",
        f"- Leiden communities: **{len(sizes):,}**; labeled nodes: **{sum(sizes.values()):,}**.",
        f"- Baseline cutoff: `0 < friend_date <= {cutoff}`; influence direction: `friend_id -> user_id`.",
        f"- Baseline directed edge rows audited: **{baseline_rows:,}**.",
        "- Community selection uses only community labels, node counts, and baseline topology; votes, outcomes, coefficients, and parameter signs are excluded.",
        "",
        "## Size distribution",
        "",
        f"- min / median / mean / max: **{int(size_values.min())} / {np.median(size_values):.0f} / {size_values.mean():.2f} / {int(size_values.max()):,}** nodes.",
    ]
    for threshold in (10, 50, 100, 500, 1000):
        lines.append(f"- Communities with `size < {threshold}`: **{sum(value < threshold for value in sizes.values()):,}**.")
    lines.extend(["", "## Largest 20 communities", "", "| rank | original community | nodes | internal directed edges |", "|---:|---:|---:|---:|"])
    for rank, community in enumerate(ranked[:20], start=1):
        lines.append(f"| {rank} | {community} | {sizes[community]:,} | {internal[community]:,} |")
    lines.extend([
        "",
        "The CSV contains the node count and internal directed-edge count for every one of the 7,277 communities.",
        "",
        "## Cross-community connections among communities with at least 1,000 nodes",
        "",
        "| rank | community A | community B | A→B | B→A | both directions |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, (total, source, target, st, ts) in enumerate(top_pairs, start=1):
        lines.append(f"| {rank} | {source} | {target} | {st:,} | {ts:,} | {total:,} |")
    lines.extend([
        "",
        "## Primary pair",
        "",
        f"- Selected original communities: **{first}** and **{second}**.",
        f"- Directed cross edges: `{first} -> {second}` **{forward:,}**; `{second} -> {first}` **{reverse:,}**; total **{forward + reverse:,}**.",
        "- Exact-tie rule: ascending original community IDs. This pair is fixed before reading model outcomes and will not be replaced based on the sign of `b`.",
        "",
    ])
    (OUTPUT / "community_size_report.md").write_text("\n".join(lines), encoding="utf-8")


def scan_votes_for_eligibility(
    path: Path, remapped: dict[int, int]
) -> tuple[set[int], dict[int, int], dict[int, Counter[int]]]:
    voters: set[int] = set()
    starts: dict[int, int] = {}
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            story = int(row["story_id"])
            timestamp = int(row["vote_date"])
            node = int(row["voter_id"])
            if story not in starts:
                starts[story] = timestamp
            if node in remapped:
                voters.add(node)
                counts[story][remapped[node]] += 1
    return voters, starts, counts


def stratified_choices(counts: dict[int, Counter[int]]) -> tuple[list[Choice], list[tuple[int, int, int, int]]]:
    eligible = sorted(
        (
            (story, count[0] + count[1], count[0], count[1])
            for story, count in counts.items()
            if count[0] + count[1] >= MIN_CASCADE_SIZE and count[0] > 0 and count[1] > 0
        ),
        key=lambda item: (item[1], item[0]),
    )
    if not eligible:
        raise AnalysisError("No cascades satisfy the two-community eligibility rule")
    n = min(MAX_CASCADES, len(eligible))
    base, remainder = divmod(len(eligible), 3)
    sizes = [base + int(index < remainder) for index in range(3)]
    strata: list[list[tuple[int, int, int, int]]] = []
    cursor = 0
    for size in sizes:
        strata.append(eligible[cursor : cursor + size])
        cursor += size
    quota_base, quota_remainder = divmod(n, 3)
    quotas = [quota_base + int(index < quota_remainder) for index in range(3)]
    labels = ("small", "medium", "large")
    rng = random.Random(SEED)
    choices: list[Choice] = []
    for label, stratum, quota in zip(labels, strata, quotas):
        quota = min(quota, len(stratum))
        for story, total, count0, count1 in rng.sample(stratum, quota):
            choices.append(Choice(story, total, count0, count1, label))
    # Fill any quota lost to a tiny stratum using the remaining eligible stories.
    chosen = {item.story_id for item in choices}
    if len(choices) < n:
        remaining = [item for item in eligible if item[0] not in chosen]
        for story, total, count0, count1 in rng.sample(remaining, n - len(choices)):
            rank = eligible.index((story, total, count0, count1))
            label = labels[min(2, 3 * rank // len(eligible))]
            choices.append(Choice(story, total, count0, count1, label))
    return sorted(choices, key=lambda item: item.story_id), eligible


def load_selected_votes(path: Path, selected: set[int]) -> dict[int, list[tuple[int, int]]]:
    cascades = {story: [] for story in selected}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            story = int(row["story_id"])
            if story in cascades:
                cascades[story].append((int(row["vote_date"]), int(row["voter_id"])))
    return cascades


def sample_negatives(
    population: list[int], forbidden: set[int], sample_size: int, rng: random.Random
) -> tuple[list[int], int]:
    available = len(population) - len(forbidden)
    sample_size = min(sample_size, available)
    if sample_size <= 0:
        return [], available
    if sample_size * 4 >= available:
        candidates = [node for node in population if node not in forbidden]
        return rng.sample(candidates, sample_size), available
    sampled: set[int] = set()
    while len(sampled) < sample_size:
        node = population[rng.randrange(len(population))]
        if node not in forbidden:
            sampled.add(node)
    return sorted(sampled), available


def build_exposure(
    path: Path,
    choices: list[Choice],
    cascades: dict[int, list[tuple[int, int]]],
    starts: dict[int, int],
    population: set[int],
    remapped: dict[int, int],
    incoming: dict[int, tuple[int, ...]],
) -> tuple[dict[int, CascadeAudit], dict[str, int], float]:
    population_list = sorted(population)
    rng = random.Random(SEED + 1)
    audits: dict[int, CascadeAudit] = {}
    totals: Counter[str] = Counter()
    seen_positive: set[tuple[int, int]] = set()
    temporary = path.with_suffix(path.suffix + ".tmp")
    started = time.perf_counter()
    try:
        with DeterministicGzipWriter(temporary) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["story_id", "time_bin", "node_id", "community", "m_in", "m_out", "degree", "y", "sampling_weight"])
            for choice in choices:
                story = choice.story_id
                by_bin: dict[int, set[int]] = defaultdict(set)
                for timestamp, node in cascades[story]:
                    if node in population:
                        by_bin[(timestamp - starts[story]) // WINDOW_SECONDS].add(node)
                adopted_before: set[int] = set()
                audit = CascadeAudit()
                audits[story] = audit
                for time_bin in sorted(by_bin):
                    positives = sorted(by_bin[time_bin] - adopted_before)
                    if not positives:
                        adopted_before.update(by_bin[time_bin])
                        continue
                    audit.positive_windows += 1
                    forbidden = adopted_before | set(positives)
                    negatives, all_negative_count = sample_negatives(
                        population_list, forbidden, NEGATIVE_RATIO * len(positives), rng
                    )
                    negative_weight = all_negative_count / len(negatives) if negatives else 0.0
                    for y, nodes, weight in ((1, positives, 1.0), (0, negatives, negative_weight)):
                        for node in nodes:
                            same = 0
                            other = 0
                            for source in incoming.get(node, ()):
                                if source not in adopted_before:
                                    continue
                                if source in by_bin[time_bin]:
                                    totals["leakage_violations"] += 1
                                if remapped[source] == remapped[node]:
                                    same += 1
                                else:
                                    other += 1
                            degree = len(incoming.get(node, ()))
                            writer.writerow([story, time_bin, node, remapped[node], same, other, degree, y, "1" if y else f"{negative_weight:.12g}"])
                            totals["rows"] += 1
                            totals["positives"] += y
                            totals["negatives"] += 1 - y
                            totals["m_out_positive"] += int(other > 0)
                            if y:
                                totals["positive_m_out_positive"] += int(other > 0)
                                totals["positive_zero_exposure"] += int(same == 0 and other == 0)
                                key = (story, node)
                                totals["duplicate_positive"] += int(key in seen_positive)
                                seen_positive.add(key)
                            audit.rows += 1
                            audit.positives += y
                            audit.negatives += 1 - y
                    adopted_before.update(by_bin[time_bin])
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return audits, dict(totals), time.perf_counter() - started


def load_exposure(path: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[float]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for name in ("story_id", "time_bin", "node_id", "m_in", "m_out", "degree", "y", "sampling_weight"):
                columns[name].append(float(row[name]))
    return {name: np.asarray(values) for name, values in columns.items()}


def story_split(stories: np.ndarray, y: np.ndarray) -> dict[int, str]:
    positive_counts = Counter()
    for story, outcome in zip(stories.astype(int), y.astype(int)):
        positive_counts[int(story)] += int(outcome)
    ordered = sorted(positive_counts, key=lambda story: (positive_counts[story], story))
    n_test = max(1, round(len(ordered) * 0.2))
    groups = np.array_split(np.asarray(ordered, dtype=int), 5)
    quotas = [n_test // 5 + int(index < n_test % 5) for index in range(5)]
    rng = random.Random(SEED)
    test: set[int] = set()
    for group, quota in zip(groups, quotas):
        quota = min(quota, len(group))
        test.update(rng.sample(group.tolist(), quota))
    if len(test) < n_test:
        test.update(rng.sample([story for story in ordered if story not in test], n_test - len(test)))
    return {story: ("test" if story in test else "train") for story in ordered}


def fit_models(data: dict[str, np.ndarray], split: dict[int, str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stories = data["story_id"].astype(int)
    y = data["y"]
    weight = data["sampling_weight"]
    train = np.asarray([split[int(story)] == "train" for story in stories])
    test = ~train
    log_time = np.log1p(data["time_bin"])
    designs = {
        "M0": np.column_stack((data["degree"], log_time)),
        "M1": np.column_stack((data["m_in"], data["m_out"], data["degree"], log_time)),
    }
    feature_names = {"M0": ("degree", "log_time"), "M1": ("m_in", "m_out", "degree", "log_time")}
    coefficients: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    prevalence = float(np.sum(weight[train] * y[train]) / np.sum(weight[train]))
    initial_intercept = math.log(prevalence / (1.0 - prevalence))
    for model in ("M0", "M1"):
        x = designs[model]
        fit = weighted_logistic_irls(x[train], y[train], weight[train], initial_intercept, np.zeros(x.shape[1]), 150, 1e-9)
        lookup = dict(zip(feature_names[model], fit.coefficients))
        row = {
            "model": model,
            "intercept": fit.intercept,
            "coef_m_in": lookup.get("m_in", ""),
            "coef_m_out": lookup.get("m_out", ""),
            "coef_degree": lookup["degree"],
            "coef_log_time": lookup["log_time"],
            "beta_scale_convention": BETA if model == "M1" else "",
            "a_eff": lookup.get("m_in", float("nan")) / BETA if model == "M1" else "",
            "b_eff": lookup.get("m_out", float("nan")) / BETA if model == "M1" else "",
            "theta_eff": -lookup["degree"] / BETA if model == "M1" else "",
            "converged": int(fit.converged),
            "iterations": fit.iterations,
            "warning": fit.warning,
        }
        coefficients.append(row)
        # Some macOS Accelerate builds emit spurious overflow warnings for
        # otherwise finite BLAS dot products. Validate the result explicitly.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            linear_predictor = fit.intercept + x[test] @ fit.coefficients
        if not np.all(np.isfinite(linear_predictor)):
            raise AnalysisError(f"{model} produced non-finite test predictions")
        probability = sigmoid(linear_predictor)
        exposed = data["m_in"][test] + data["m_out"][test] > 0
        for subset_name, subset_mask in (("all_test", np.ones(np.sum(test), dtype=bool)), ("zero_exposure", ~exposed), ("exposed", exposed)):
            result = metrics(model, subset_name, y[test][subset_mask], probability[subset_mask], weight[test][subset_mask])
            metric_rows.append(result)
    return coefficients, metric_rows


def write_exposure_report(
    path: Path,
    first: int,
    second: int,
    sizes: Counter[int],
    edges: set[tuple[int, int]],
    remapped: dict[int, int],
    voter_population: set[int],
    eligible: list[tuple[int, int, int, int]],
    choices: list[Choice],
    audits: dict[int, CascadeAudit],
    totals: dict[str, int],
    runtime: float,
) -> None:
    internal0 = sum(remapped[source] == remapped[target] == 0 for source, target in edges)
    internal1 = sum(remapped[source] == remapped[target] == 1 for source, target in edges)
    cross = len(edges) - internal0 - internal1
    lines = [
        "# Digg strict two-community pilot exposure report",
        "",
        "## Induced subsystem",
        "",
        f"- Original Leiden community **{first}** is remapped to `community=0`: **{sizes[first]:,}** nodes.",
        f"- Original Leiden community **{second}** is remapped to `community=1`: **{sizes[second]:,}** nodes.",
        f"- Induced baseline influence edges: **{len(edges):,}**; within community 0: **{internal0:,}**; within community 1: **{internal1:,}**; cross-community: **{cross:,}** (**{cross / len(edges):.4%}**).",
        f"- Nodes in the pair with cleaned vote records: **{len(voter_population):,}**.",
        f"- Cascades satisfying ≥{MIN_CASCADE_SIZE} subsystem adopters and at least one adopter in each community: **{len(eligible):,}**.",
        "- This is a **two-community induced subsystem within the Digg network**. It does not merge or force the full Digg network into two communities.",
        "",
        "## Sampling and leakage controls",
        "",
        f"- Selected cascades: **{len(choices)}** (maximum 100), rank-tertile stratified by subsystem adopter count; seed **{SEED}**.",
        f"- Window: **1 hour**; negative sampling: up to **{NEGATIVE_RATIO}:1** within each cascade-window; positives weight 1 and negatives use inverse sampling probability.",
        "- Exposure uses only adopters in strictly earlier time bins. Current-window adopters are added only after every row in the window is written.",
        "- `degree` is induced-subnetwork influence in-degree; degree-zero pair voters remain valid risk-set members.",
        f"- Total rows: **{totals.get('rows', 0):,}**; positives: **{totals.get('positives', 0):,}**; negatives: **{totals.get('negatives', 0):,}**.",
        f"- Rows with `m_out>0`: **{totals.get('m_out_positive', 0) / totals['rows']:.4%}**.",
        f"- Positive rows with `m_out>0`: **{totals.get('positive_m_out_positive', 0) / totals['positives']:.4%}**.",
        f"- Zero-exposure positives: **{totals.get('positive_zero_exposure', 0):,}** (**{totals.get('positive_zero_exposure', 0) / totals['positives']:.4%}**).",
        f"- Duplicate positive node-story pairs: **{totals.get('duplicate_positive', 0):,}**; current/future exposure violations: **{totals.get('leakage_violations', 0):,}**.",
        f"- Construction runtime: **{runtime:.1f} seconds**.",
        "",
        "## Selected cascades",
        "",
        "| story_id | stratum | subsystem adopters | community 0 | community 1 | rows | y=1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in choices:
        audit = audits[item.story_id]
        lines.append(f"| {item.story_id} | {item.stratum} | {item.size} | {item.community_0_adopters} | {item.community_1_adopters} | {audit.rows} | {audit.positives} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_model_report(
    path: Path,
    coefficients: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    totals: dict[str, int],
    split: dict[int, str],
) -> None:
    coefs = {str(row["model"]): row for row in coefficients}
    metric = {(str(row["model"]), str(row["subset"])): row for row in metric_rows}
    m0 = metric[("M0", "all_test")]
    m1 = metric[("M1", "all_test")]
    old = {"m_in": 0.027108040, "m_out": -0.048634896, "degree": 0.001986288, "log_time": -0.534487611}
    new = coefs["M1"]
    expected_old = old["m_in"] > 0 and old["m_out"] > 0 and old["degree"] < 0
    expected_new = float(new["coef_m_in"]) > 0 and float(new["coef_m_out"]) > 0 and float(new["coef_degree"]) < 0
    comparison = "more consistent" if expected_new and not expected_old else ("not more consistent" if not expected_new else "equally sign-consistent")
    lines = [
        "# Digg two-community M0/M1 exploratory model report",
        "",
        "## Design",
        "",
        f"- Story-level split: **{sum(value == 'train' for value in split.values())} train / {sum(value == 'test' for value in split.values())} test** cascades, stratified by positive count, seed **{SEED}**; no story appears in both sets.",
        "- M0: `intercept + degree + log1p(time_bin)`.",
        "- M1 is the unconstrained probabilistic count-threshold form: `intercept + beta*(a*m_in + b*m_out - theta*degree) - gamma*log1p(time_bin)`, with beta fixed at 5 only as a scale convention.",
        "- Both models use sampling weights, no class weights, no normalized exposures, no sign constraints, and no test-set tuning.",
        "",
        "## Raw-scale coefficients",
        "",
        "| model | intercept | m_in | m_out | degree | log_time | converged | iterations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("M0", "M1"):
        row = coefs[model]
        lines.append(f"| {model} | {float(row['intercept']):.9g} | {row['coef_m_in'] or '—'} | {row['coef_m_out'] or '—'} | {float(row['coef_degree']):.9g} | {float(row['coef_log_time']):.9g} | {row['converged']} | {row['iterations']} |")
    lines.extend([
        "",
        f"For M1 under beta=5: `a_eff={float(new['a_eff']):.9g}`, `b_eff={float(new['b_eff']):.9g}`, `theta_eff={float(new['theta_eff']):.9g}`. These are observational effective coefficients, not recovered true parameters, and theta is not clipped.",
        "",
        "## Held-out cascade prediction",
        "",
        "| model | subset | rows | weighted y rate | log loss | Brier | ROC-AUC | PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in metric_rows:
        lines.append(f"| {row['model']} | {row['subset']} | {row['rows']} | {float(row['weighted_y_rate']):.6g} | {float(row['weighted_log_loss']):.9g} | {float(row['weighted_brier']):.9g} | {float(row['weighted_roc_auc']):.6g} | {float(row['weighted_pr_auc']):.6g} |")
    lines.extend([
        "",
        f"- M1 minus M0 held-out changes: log loss **{float(m1['weighted_log_loss']) - float(m0['weighted_log_loss']):+.9g}** (negative is better), Brier **{float(m1['weighted_brier']) - float(m0['weighted_brier']):+.9g}**, ROC-AUC **{float(m1['weighted_roc_auc']) - float(m0['weighted_roc_auc']):+.6g}**, PR-AUC **{float(m1['weighted_pr_auc']) - float(m0['weighted_pr_auc']):+.6g}**.",
        f"- Whole pilot rows with `m_out>0`: **{totals.get('m_out_positive', 0) / totals['rows']:.4%}**; positive rows with `m_out>0`: **{totals.get('positive_m_out_positive', 0) / totals['positives']:.4%}**; zero-exposure positive rows: **{totals.get('positive_zero_exposure', 0) / totals['positives']:.4%}**.",
        "",
        "## Comparison with the full-network multi-community pilot",
        "",
        f"- Previous full-network coefficients: `m_in={old['m_in']:.9g}`, `m_out={old['m_out']:.9g}`, `degree={old['degree']:.9g}`, `log_time={old['log_time']:.9g}`.",
        f"- Under the paper-sign heuristic (`m_in>0`, `m_out>0`, `degree<0`), the fixed two-community result is **{comparison}** than the previous full-network result.",
        "- This comparison is exploratory association/prediction, not causal identification or parameter recovery. The community pair was fixed by topology before model fitting and was not changed after inspecting these coefficients.",
        "- The analysis remains a maximum-100-cascade pilot. It is not expanded to all cascades, and no real-data intervention is run.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    votes = PROCESSED / "digg_votes_clean.csv.gz"
    friends = PROCESSED / "digg_friends_clean.csv.gz"
    communities_path = PROCESSED / "digg_communities.csv"
    for path in (votes, friends, communities_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading Leiden labels and auditing baseline topology...", flush=True)
    labels, sizes = load_communities(communities_path)
    cutoff = earliest_vote(votes)
    large = {community for community, size in sizes.items() if size >= MIN_COMMUNITY_SIZE}
    internal, large_edges, baseline_rows = audit_baseline(friends, cutoff, labels, large)
    pair = choose_pair(large, large_edges)
    write_community_outputs(sizes, internal, large_edges, cutoff, baseline_rows, pair)
    first, second, _, _ = pair
    print(f"Fixed primary pair: {first}, {second}", flush=True)

    pair_nodes = {node for node, community in labels.items() if community in (first, second)}
    remapped = {node: int(labels[node] == second) for node in pair_nodes}
    incoming, induced_edges = build_induced_network(friends, cutoff, pair_nodes)
    print("Scanning votes and selecting eligible cascades...", flush=True)
    voter_population, starts, cascade_counts = scan_votes_for_eligibility(votes, remapped)
    choices, eligible = stratified_choices(cascade_counts)
    cascades = load_selected_votes(votes, {item.story_id for item in choices})

    exposure_path = PROCESSED / "digg_two_community_exposure_pilot.csv.gz"
    print(f"Constructing exposure table for {len(choices)} cascades...", flush=True)
    audits, totals, runtime = build_exposure(exposure_path, choices, cascades, starts, voter_population, remapped, incoming)
    if totals.get("leakage_violations", 0) or totals.get("duplicate_positive", 0):
        raise AnalysisError(f"Exposure audit failed: {totals}")
    write_exposure_report(OUTPUT / "two_community_exposure_report.md", first, second, sizes, induced_edges, remapped, voter_population, eligible, choices, audits, totals, runtime)

    data = load_exposure(exposure_path)
    split = story_split(data["story_id"], data["y"])
    coefficients, metric_rows = fit_models(data, split)
    atomic_csv(OUTPUT / "two_community_coefficients.csv", list(coefficients[0]), coefficients)
    atomic_csv(OUTPUT / "two_community_metrics.csv", list(metric_rows[0]), metric_rows)
    write_model_report(OUTPUT / "two_community_model_report.md", coefficients, metric_rows, totals, split)
    print("Completed two-community audit, exposure construction, and M0/M1 fitting.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (AnalysisError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
