#!/usr/bin/env python3
"""Build and audit a sampled Digg 2009 pilot exposure table.

The pilot uses leakage-safe baseline friendships and Leiden communities from
the data-preparation stage. It intentionally does not fit any model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
import random
import sys
import time
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "digg"
DEFAULT_SEED = 42
DEFAULT_N_CASCADES = 100
DEFAULT_WINDOW_SECONDS = 3600
DEFAULT_NEGATIVE_RATIO = 5


class PilotError(RuntimeError):
    """Raised when required prepared data are missing or inconsistent."""


@dataclass(frozen=True)
class CascadeChoice:
    story_id: int
    covered_size: int
    stratum: str


@dataclass
class CascadeAudit:
    rows: int = 0
    positives: int = 0
    negatives: int = 0
    windows_with_positives: int = 0
    eligible_positive_adopters: int = 0
    degree_zero_positive_adopters_excluded: int = 0


@dataclass
class ExposureAudit:
    rows: int = 0
    positives: int = 0
    negatives: int = 0
    m_in_positive: int = 0
    m_out_positive: int = 0
    positive_m_in_positive: int = 0
    positive_m_out_positive: int = 0
    positive_zero_exposure: int = 0
    duplicate_positive_node_story: int = 0
    leakage_violations: int = 0
    correlation_count: int = 0
    sum_m_in: float = 0.0
    sum_m_out: float = 0.0
    sum_m_in_sq: float = 0.0
    sum_m_out_sq: float = 0.0
    sum_m_in_m_out: float = 0.0
    values: dict[str, array] = field(
        default_factory=lambda: {
            name: array("d")
            for name in ("m_in", "m_out", "degree", "frac_in", "frac_out")
        }
    )
    seen_positive_pairs: set[tuple[int, int]] = field(default_factory=set)

    def add(
        self,
        story_id: int,
        node_id: int,
        m_in: int,
        m_out: int,
        degree: int,
        frac_in: float,
        frac_out: float,
        outcome: int,
    ) -> None:
        self.rows += 1
        self.positives += outcome
        self.negatives += 1 - outcome
        self.m_in_positive += int(m_in > 0)
        self.m_out_positive += int(m_out > 0)
        if outcome:
            self.positive_m_in_positive += int(m_in > 0)
            self.positive_m_out_positive += int(m_out > 0)
            self.positive_zero_exposure += int(m_in == 0 and m_out == 0)
            pair = (story_id, node_id)
            self.duplicate_positive_node_story += int(pair in self.seen_positive_pairs)
            self.seen_positive_pairs.add(pair)
        self.correlation_count += 1
        self.sum_m_in += m_in
        self.sum_m_out += m_out
        self.sum_m_in_sq += m_in * m_in
        self.sum_m_out_sq += m_out * m_out
        self.sum_m_in_m_out += m_in * m_out
        self.values["m_in"].append(float(m_in))
        self.values["m_out"].append(float(m_out))
        self.values["degree"].append(float(degree))
        self.values["frac_in"].append(frac_in)
        self.values["frac_out"].append(frac_out)

    def correlation(self) -> float | None:
        count = self.correlation_count
        if count < 2:
            return None
        numerator = count * self.sum_m_in_m_out - self.sum_m_in * self.sum_m_out
        left = count * self.sum_m_in_sq - self.sum_m_in * self.sum_m_in
        right = count * self.sum_m_out_sq - self.sum_m_out * self.sum_m_out
        denominator = math.sqrt(max(0.0, left) * max(0.0, right))
        return numerator / denominator if denominator else None


class DeterministicGzipTextWriter:
    def __init__(self, path: Path):
        self.raw = path.open("wb")
        self.compressed = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)
        self.text = io.TextIOWrapper(self.compressed, encoding="utf-8", newline="")

    def __enter__(self) -> TextIO:
        return self.text

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.text.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-cascades", type=int, default=DEFAULT_N_CASCADES)
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--negative-ratio", type=int, default=DEFAULT_NEGATIVE_RATIO)
    return parser.parse_args()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required prepared file(s) not found: " + ", ".join(missing))


def read_communities(path: Path) -> dict[int, int]:
    communities: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["node_id", "community"]:
            raise PilotError(f"Unexpected community schema in {path}: {reader.fieldnames}")
        for row in reader:
            node_id = int(row["node_id"])
            if node_id in communities:
                raise PilotError(f"Duplicate community label for node {node_id}")
            communities[node_id] = int(row["community"])
    if not communities:
        raise PilotError("Community file is empty")
    return communities


def read_vote_population_and_start(path: Path) -> tuple[set[int], int]:
    voters: set[int] = set()
    earliest: int | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["vote_date", "voter_id", "story_id"]:
            raise PilotError(f"Unexpected cleaned-vote schema in {path}: {reader.fieldnames}")
        for row in reader:
            vote_date = int(row["vote_date"])
            voters.add(int(row["voter_id"]))
            earliest = vote_date if earliest is None else min(earliest, vote_date)
    if earliest is None:
        raise PilotError("Cleaned votes are empty")
    return voters, earliest


def build_baseline_influencers(
    path: Path, cutoff: int, communities: dict[int, int]
) -> dict[int, tuple[int, ...]]:
    # Cleaned row user_id -> friend_id means user follows friend. Information
    # exposure is reversed: friend_id (source) -> user_id (target).
    incoming_sets: dict[int, set[int]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["friend_date", "user_id", "friend_id", "mutual"]
        if reader.fieldnames != expected:
            raise PilotError(f"Unexpected cleaned-friend schema in {path}: {reader.fieldnames}")
        for row in reader:
            friend_date = int(row["friend_date"])
            if friend_date <= 0 or friend_date > cutoff:
                continue
            target = int(row["user_id"])
            source = int(row["friend_id"])
            if source == target:
                raise PilotError("Self-edge found in cleaned friendships")
            if source not in communities or target not in communities:
                raise PilotError("Baseline edge endpoint lacks a community label")
            incoming_sets[target].add(source)
    return {target: tuple(sorted(sources)) for target, sources in incoming_sets.items()}


def read_eligibility(path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"story_id", "network_covered_adopters"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise PilotError(f"Unexpected eligibility schema in {path}: {reader.fieldnames}")
        for row in reader:
            rows.append((int(row["story_id"]), int(row["network_covered_adopters"])))
    if not rows:
        raise PilotError("Cascade eligibility file is empty")
    return rows


def rank_tertiles(
    eligibility: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    ordered = sorted(eligibility, key=lambda item: (item[1], item[0]))
    base, remainder = divmod(len(ordered), 3)
    stratum_sizes = [base + int(index < remainder) for index in range(3)]
    strata: list[list[tuple[int, int]]] = []
    cursor = 0
    for size in stratum_sizes:
        strata.append(ordered[cursor : cursor + size])
        cursor += size
    return strata


def stratified_sample(
    eligibility: list[tuple[int, int]], n_cascades: int, seed: int
) -> list[CascadeChoice]:
    if n_cascades <= 0 or n_cascades > len(eligibility):
        raise PilotError(f"--n-cascades must be between 1 and {len(eligibility)}")
    strata = rank_tertiles(eligibility)

    quota_base, quota_remainder = divmod(n_cascades, 3)
    quotas = [quota_base + int(index < quota_remainder) for index in range(3)]
    labels = ["small", "medium", "large"]
    rng = random.Random(seed)
    choices: list[CascadeChoice] = []
    for label, stratum, quota in zip(labels, strata, quotas):
        if quota > len(stratum):
            raise PilotError(f"Not enough cascades in {label} stratum")
        for story_id, covered_size in rng.sample(stratum, quota):
            choices.append(CascadeChoice(story_id, covered_size, label))
    return sorted(choices, key=lambda choice: choice.story_id)


def write_choices(path: Path, choices: list[CascadeChoice]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["story_id", "stratum", "network_covered_cascade_size"])
            writer.writerows(
                (choice.story_id, choice.stratum, choice.covered_size) for choice in choices
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_selected_votes(
    path: Path, selected_story_ids: set[int]
) -> dict[int, list[tuple[int, int]]]:
    cascades: dict[int, list[tuple[int, int]]] = {
        story_id: [] for story_id in selected_story_ids
    }
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            story_id = int(row["story_id"])
            if story_id in cascades:
                cascades[story_id].append((int(row["vote_date"]), int(row["voter_id"])))
    empty = [story_id for story_id, votes in cascades.items() if not votes]
    if empty:
        raise PilotError(f"Selected stories have no cleaned votes: {empty}")
    return cascades


def uniform_negative_sample(
    eligible_nodes: list[int],
    forbidden: set[int],
    negative_count: int,
    sample_size: int,
    rng: random.Random,
) -> list[int]:
    if sample_size <= 0 or negative_count <= 0:
        return []
    sample_size = min(sample_size, negative_count)
    if sample_size * 4 >= negative_count:
        candidates = [node for node in eligible_nodes if node not in forbidden]
        return rng.sample(candidates, sample_size)
    sampled: set[int] = set()
    while len(sampled) < sample_size:
        node = eligible_nodes[rng.randrange(len(eligible_nodes))]
        if node not in forbidden:
            sampled.add(node)
    return sorted(sampled)


def exposure_values(
    node_id: int,
    adopted_before: set[int],
    adoption_bins: dict[int, int],
    time_bin: int,
    incoming: dict[int, tuple[int, ...]],
    communities: dict[int, int],
) -> tuple[int, int, int, float, float, int]:
    influencers = incoming[node_id]
    node_community = communities[node_id]
    m_in = 0
    m_out = 0
    leakage_violations = 0
    for source in influencers:
        if source not in adopted_before:
            continue
        if adoption_bins[source] >= time_bin:
            leakage_violations += 1
            continue
        if communities[source] == node_community:
            m_in += 1
        else:
            m_out += 1
    degree = len(influencers)
    return m_in, m_out, degree, m_in / degree, m_out / degree, leakage_violations


def build_exposure_table(
    destination: Path,
    choices: list[CascadeChoice],
    cascades: dict[int, list[tuple[int, int]]],
    population: set[int],
    eligible_nodes: list[int],
    incoming: dict[int, tuple[int, ...]],
    communities: dict[int, int],
    seed: int,
    window_seconds: int,
    negative_ratio: int,
) -> tuple[ExposureAudit, dict[int, CascadeAudit], float]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    audit = ExposureAudit()
    cascade_audits: dict[int, CascadeAudit] = {}
    eligible_set = set(eligible_nodes)
    rng = random.Random(seed)
    started = time.perf_counter()
    try:
        with DeterministicGzipTextWriter(temporary) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "story_id",
                    "time_bin",
                    "node_id",
                    "community",
                    "m_in",
                    "m_out",
                    "degree",
                    "frac_in",
                    "frac_out",
                    "y",
                    "sampling_weight",
                ]
            )
            for choice in choices:
                story_id = choice.story_id
                raw_votes = cascades[story_id]
                cascade_start = min(vote_date for vote_date, _ in raw_votes)
                adopters_by_bin: dict[int, list[int]] = defaultdict(list)
                adoption_bins: dict[int, int] = {}
                for vote_date, node_id in raw_votes:
                    if node_id not in population:
                        continue
                    time_bin = (vote_date - cascade_start) // window_seconds
                    adopters_by_bin[time_bin].append(node_id)
                    adoption_bins[node_id] = time_bin

                adopted_before: set[int] = set()
                adopted_eligible: set[int] = set()
                cascade_audit = CascadeAudit()
                cascade_audits[story_id] = cascade_audit
                for time_bin in sorted(adopters_by_bin):
                    current_adopters = set(adopters_by_bin[time_bin])
                    positives = sorted(current_adopters & eligible_set)
                    cascade_audit.eligible_positive_adopters += len(positives)
                    cascade_audit.degree_zero_positive_adopters_excluded += len(
                        current_adopters - eligible_set
                    )
                    if positives:
                        cascade_audit.windows_with_positives += 1
                        forbidden = adopted_eligible | set(positives)
                        all_negative_count = len(eligible_nodes) - len(forbidden)
                        negatives = uniform_negative_sample(
                            eligible_nodes,
                            forbidden,
                            all_negative_count,
                            negative_ratio * len(positives),
                            rng,
                        )
                        negative_weight = (
                            all_negative_count / len(negatives) if negatives else 0.0
                        )
                        for outcome, nodes, weight in (
                            (1, positives, 1.0),
                            (0, negatives, negative_weight),
                        ):
                            for node_id in nodes:
                                values = exposure_values(
                                    node_id,
                                    adopted_before,
                                    adoption_bins,
                                    time_bin,
                                    incoming,
                                    communities,
                                )
                                m_in, m_out, degree, frac_in, frac_out, violations = values
                                if degree <= 0:
                                    raise PilotError("Degree-zero node entered the exposure table")
                                audit.leakage_violations += violations
                                writer.writerow(
                                    [
                                        story_id,
                                        time_bin,
                                        node_id,
                                        communities[node_id],
                                        m_in,
                                        m_out,
                                        degree,
                                        f"{frac_in:.10f}",
                                        f"{frac_out:.10f}",
                                        outcome,
                                        "1" if outcome else f"{weight:.10f}",
                                    ]
                                )
                                audit.add(
                                    story_id,
                                    node_id,
                                    m_in,
                                    m_out,
                                    degree,
                                    frac_in,
                                    frac_out,
                                    outcome,
                                )
                                cascade_audit.rows += 1
                                cascade_audit.positives += outcome
                                cascade_audit.negatives += 1 - outcome
                    adopted_before.update(current_adopters)
                    adopted_eligible.update(current_adopters & eligible_set)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return audit, cascade_audits, time.perf_counter() - started


def quantiles(values: array) -> dict[str, float]:
    view = np.frombuffer(values, dtype=np.float64)
    probabilities = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    results = np.quantile(view, probabilities)
    return {
        label: float(value)
        for label, value in zip(
            ["min", "p25", "median", "p75", "p90", "p99", "max"], results
        )
    }


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    return f"{seconds / 3600:.2f} hours"


def write_report(
    path: Path,
    choices: list[CascadeChoice],
    cascade_audits: dict[int, CascadeAudit],
    audit: ExposureAudit,
    population_size: int,
    eligible_population_size: int,
    window_seconds: int,
    negative_ratio: int,
    seed: int,
    runtime_seconds: float,
    exposure_size_bytes: int,
    eligibility: list[tuple[int, int]],
) -> None:
    if audit.rows == 0 or audit.positives == 0:
        raise PilotError("Pilot exposure table contains no usable observations")
    total_cascades = len(eligibility)
    scale = total_cascades / len(choices)
    estimated_seconds = runtime_seconds * scale
    estimated_bytes = exposure_size_bytes * scale
    correlation = audit.correlation()
    stratum_counts = Counter(choice.stratum for choice in choices)
    stratum_ranges: dict[str, tuple[int, int]] = {}
    for label in ("small", "medium", "large"):
        sizes = [choice.covered_size for choice in choices if choice.stratum == label]
        stratum_ranges[label] = (min(sizes), max(sizes))
    population_strata = rank_tertiles(eligibility)
    population_ranges = [
        (len(stratum), min(size for _, size in stratum), max(size for _, size in stratum))
        for stratum in population_strata
    ]

    lines = [
        "# Digg pilot exposure-table report",
        "",
        "## Design",
        "",
        f"- Pilot cascades: **{len(choices):,}** of **{total_cascades:,}**",
        "- Population strata are rank tertiles sorted by network-covered size (ties may cross "
        "a rank boundary): "
        f"small **{population_ranges[0][0]:,}** cascades / size "
        f"`{population_ranges[0][1]}–{population_ranges[0][2]}`; medium "
        f"**{population_ranges[1][0]:,}** / `{population_ranges[1][1]}–"
        f"{population_ranges[1][2]}`; large **{population_ranges[2][0]:,}** / "
        f"`{population_ranges[2][1]}–{population_ranges[2][2]}`.",
        f"- Stratified sample: small **{stratum_counts['small']}**, medium "
        f"**{stratum_counts['medium']}**, large **{stratum_counts['large']}**",
        f"- Selected covered-size ranges: small `{stratum_ranges['small'][0]}–"
        f"{stratum_ranges['small'][1]}`, medium `{stratum_ranges['medium'][0]}–"
        f"{stratum_ranges['medium'][1]}`, large `{stratum_ranges['large'][0]}–"
        f"{stratum_ranges['large'][1]}`",
        f"- Random seed: **{seed}** (separate deterministic streams for cascade and negative sampling)",
        f"- Window width: **{window_seconds:,} seconds**",
        "- Window convention: `[start, start + width)`; exposure includes only adopters from "
        "strictly earlier time bins.",
        f"- Negative sampling ratio: up to **{negative_ratio}:1** per cascade-time window",
        f"- User population: **{population_size:,}** cleaned voters in the baseline network with "
        "a Leiden label",
        f"- Main-table population after excluding degree-zero targets: **{eligible_population_size:,}**",
        "- Influence direction: `friend_id -> user_id`; degree is target in-degree in that network.",
        "",
        "## Overall audit",
        "",
        f"- Total rows: **{audit.rows:,}**",
        f"- Positive rows (`y=1`): **{audit.positives:,}**",
        f"- Negative rows (`y=0`): **{audit.negatives:,}**",
        f"- Rows with `m_in > 0`: **{audit.m_in_positive / audit.rows:.4%}**",
        f"- Rows with `m_out > 0`: **{audit.m_out_positive / audit.rows:.4%}**",
        f"- Positive rows with `m_in > 0`: **{audit.positive_m_in_positive / audit.positives:.4%}**",
        f"- Positive rows with `m_out > 0`: **{audit.positive_m_out_positive / audit.positives:.4%}**",
        f"- Positive rows with `m_in = m_out = 0`: **{audit.positive_zero_exposure:,}** "
        f"({audit.positive_zero_exposure / audit.positives:.4%})",
        f"- Duplicate positive `(node_id, story_id)` pairs: **{audit.duplicate_positive_node_story:,}**",
        f"- Current/future-adopter exposure violations: **{audit.leakage_violations:,}**",
        f"- Pearson correlation of `m_in` and `m_out`: **{correlation:.8f}**"
        if correlation is not None
        else "- Pearson correlation of `m_in` and `m_out`: undefined (zero variance)",
        "",
        "## Exposure distributions",
        "",
        "| variable | min | p25 | median | p75 | p90 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("m_in", "m_out", "degree", "frac_in", "frac_out"):
        values = quantiles(audit.values[name])
        lines.append(
            f"| {name} | {values['min']:.6g} | {values['p25']:.6g} | "
            f"{values['median']:.6g} | {values['p75']:.6g} | {values['p90']:.6g} | "
            f"{values['p99']:.6g} | {values['max']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Per-cascade counts",
            "",
            "| story_id | stratum | covered size | rows | y=1 | y=0 | positive windows | "
            "degree-zero positives excluded |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    choice_by_story = {choice.story_id: choice for choice in choices}
    for story_id in sorted(cascade_audits):
        item = cascade_audits[story_id]
        choice = choice_by_story[story_id]
        lines.append(
            f"| {story_id} | {choice.stratum} | {choice.covered_size} | {item.rows} | "
            f"{item.positives} | {item.negatives} | {item.windows_with_positives} | "
            f"{item.degree_zero_positive_adopters_excluded} |"
        )
    lines.extend(
        [
            "",
            "## Full-data scale estimate",
            "",
            f"- Measured pilot exposure construction time: **{format_duration(runtime_seconds)}**",
            f"- Pilot compressed table size: **{exposure_size_bytes / (1024 ** 2):.2f} MiB**",
            f"- Linear estimate for all {total_cascades:,} cascades: "
            f"**{format_duration(estimated_seconds)}**, **{estimated_bytes / (1024 ** 3):.2f} GiB** compressed.",
            "- This estimate scales the stratified pilot linearly and excludes one-time input loading; "
            "actual runtime depends on storage and compression throughput.",
            "",
            "> This stage only constructs and audits the pilot exposure table. It does **not** fit "
            "logistic regression or any other parameters.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.window_seconds <= 0 or args.negative_ratio <= 0:
        raise PilotError("--window-seconds and --negative-ratio must be positive")
    processed_dir = args.processed_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    votes_path = processed_dir / "digg_votes_clean.csv.gz"
    friends_path = processed_dir / "digg_friends_clean.csv.gz"
    communities_path = processed_dir / "digg_communities.csv"
    eligibility_path = output_dir / "cascade_eligibility.csv"
    require_files([votes_path, friends_path, communities_path, eligibility_path])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading communities and cleaned-vote population...")
    communities = read_communities(communities_path)
    voters, earliest_vote_date = read_vote_population_and_start(votes_path)
    population = voters & set(communities)
    print("Building directed baseline influencer lists...")
    incoming = build_baseline_influencers(
        friends_path, earliest_vote_date, communities
    )
    eligible_nodes = sorted(population & set(incoming))
    if not eligible_nodes:
        raise PilotError("No population nodes have positive baseline in-degree")

    eligibility = read_eligibility(eligibility_path)
    choices = stratified_sample(eligibility, args.n_cascades, args.seed)
    pilot_path = output_dir / "pilot_cascades.csv"
    write_choices(pilot_path, choices)
    print("Loading selected cascades...")
    cascades = read_selected_votes(votes_path, {choice.story_id for choice in choices})
    print("Constructing sampled pilot exposure rows...")
    exposure_path = processed_dir / "digg_exposure_pilot.csv.gz"
    audit, cascade_audits, runtime_seconds = build_exposure_table(
        exposure_path,
        choices,
        cascades,
        population,
        eligible_nodes,
        incoming,
        communities,
        args.seed,
        args.window_seconds,
        args.negative_ratio,
    )
    if audit.leakage_violations:
        raise PilotError(f"Detected {audit.leakage_violations} exposure leakage violations")
    report_path = output_dir / "exposure_pilot_report.md"
    write_report(
        report_path,
        choices,
        cascade_audits,
        audit,
        len(population),
        len(eligible_nodes),
        args.window_seconds,
        args.negative_ratio,
        args.seed,
        runtime_seconds,
        exposure_path.stat().st_size,
        eligibility,
    )
    print(f"Selected cascades: {pilot_path}")
    print(f"Exposure table: {exposure_path}")
    print(f"Audit report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except (PilotError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
