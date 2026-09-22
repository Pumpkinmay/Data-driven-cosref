#!/usr/bin/env python3
"""Fit a leakage-aware two-layer GraphSAGE baseline on the Digg pilot.

The model reuses the saved 80/20 story split, sampled exposure rows,
sampling_weight, and weighted metrics used by the existing XGBoost analysis.
It does not tune on or early-stop against test stories.
"""

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
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from diagnose_digg_pilot import DiagnosticError, atomic_csv, metrics
from fit_digg_controlled_models import ControlledData, construct_controlled_data
from validate_digg_xgboost import FEATURE_SETS, fit_xgb, matrix, read_split


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "digg"
DOCS = ROOT / "docs"
SEED = 42
WINDOW_SECONDS = 3600


@dataclass(frozen=True)
class Scale:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


@dataclass
class VoteIndex:
    story_start: dict[int, int]
    user_times: dict[int, list[int]]
    story_user_time: dict[tuple[int, int], int]

    def absolute_time(self, story: int, time_bin: int) -> int:
        return self.story_start[story] + time_bin * WINDOW_SECONDS

    def node_history(self, node: int, story: int, timestamp: int) -> tuple[int, int, int]:
        total_prior = bisect.bisect_left(self.user_times.get(node, ()), timestamp)
        same_story_time = self.story_user_time.get((story, node))
        adopted_before = int(same_story_time is not None and same_story_time < timestamp)
        other_story_prior = total_prior - adopted_before
        if other_story_prior < 0:
            raise DiagnosticError("Negative other-story prior activity")
        return total_prior, other_story_prior, adopted_before


@dataclass
class TemporalGraph:
    node_ids: list[int]
    node_to_index: dict[int, int]
    community_index: np.ndarray
    incoming_sources: list[list[int]]
    incoming_dates: list[list[int]]
    unknown_date_edges_excluded: int

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_communities(self) -> int:
        return int(np.max(self.community_index)) + 1

    def indegree_before(self, node_index: int, timestamp: int) -> int:
        return bisect.bisect_left(self.incoming_dates[node_index], timestamp)

    def sampled_subgraph(
        self,
        target_indices: list[int],
        timestamp: int,
        fanouts: tuple[int, int],
        rng: random.Random,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        nodes = set(target_indices)
        edges: set[tuple[int, int]] = set()
        frontier = set(target_indices)
        for fanout in fanouts:
            next_frontier: set[int] = set()
            for target in sorted(frontier):
                cutoff = bisect.bisect_left(self.incoming_dates[target], timestamp)
                if cutoff <= fanout:
                    positions = range(cutoff)
                else:
                    positions = sorted(rng.sample(range(cutoff), fanout))
                for position in positions:
                    source = self.incoming_sources[target][position]
                    edges.add((source, target))
                    nodes.add(source)
                    next_frontier.add(source)
            frontier = next_frontier

        ordered_nodes = sorted(nodes)
        local = {node: index for index, node in enumerate(ordered_nodes)}
        if edges:
            ordered_edges = sorted(edges)
            edge_index = torch.tensor(
                [[local[source] for source, _ in ordered_edges],
                 [local[target] for _, target in ordered_edges]],
                dtype=torch.long,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        target_local = torch.tensor([local[node] for node in target_indices], dtype=torch.long)
        return ordered_nodes, edge_index, target_local


class TemporalGraphSAGE(nn.Module):
    """Two-layer GraphSAGE with a learned community embedding."""

    def __init__(
        self,
        num_communities: int,
        numeric_features: int = 4,
        context_features: int = 2,
        community_dim: int = 16,
        hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.community_embedding = nn.Embedding(num_communities, community_dim)
        self.conv1 = SAGEConv(numeric_features + community_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.classifier = nn.Linear(hidden + context_features, 1)
        self.dropout = dropout

    def forward(
        self,
        numeric_x: torch.Tensor,
        community: torch.Tensor,
        edge_index: torch.Tensor,
        target_index: torch.Tensor,
        target_context: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.community_embedding(community)
        hidden = torch.cat((numeric_x, embedded), dim=1)
        hidden = F.relu(self.conv1(hidden, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = F.relu(self.conv2(hidden, edge_index))
        selected = torch.cat((hidden[target_index], target_context), dim=1)
        return self.classifier(selected).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--friends", type=Path, default=PROCESSED / "digg_friends_clean.csv.gz")
    parser.add_argument("--communities", type=Path, default=PROCESSED / "digg_communities.csv")
    parser.add_argument("--exposure", type=Path, default=PROCESSED / "digg_exposure_pilot.csv.gz")
    parser.add_argument("--votes", type=Path, default=PROCESSED / "digg_votes_clean.csv.gz")
    parser.add_argument("--split", type=Path, default=OUTPUT / "pilot_story_split.csv")
    parser.add_argument("--xgb-metrics", type=Path, default=OUTPUT / "xgboost_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=DOCS / "digg_gnn_baseline.md")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--community-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fanout", type=int, nargs=2, default=(15, 10), metavar=("L1", "L2"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate schemas, split, temporal indexes, and weighted prevalence without training.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace only the three GNN-specific outputs.")
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_communities(path: Path) -> tuple[list[int], dict[int, int], np.ndarray]:
    labels: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["node_id", "community"]:
            raise DiagnosticError(f"Unexpected community schema: {reader.fieldnames}")
        for row in reader:
            node = int(row["node_id"])
            if node in labels:
                raise DiagnosticError(f"Duplicate community label for node {node}")
            labels[node] = int(row["community"])
    node_ids = sorted(labels)
    node_to_index = {node: index for index, node in enumerate(node_ids)}
    community_values = sorted(set(labels.values()))
    remap = {community: index for index, community in enumerate(community_values)}
    communities = np.asarray([remap[labels[node]] for node in node_ids], dtype=np.int64)
    return node_ids, node_to_index, communities


def read_temporal_graph(friends_path: Path, communities_path: Path) -> TemporalGraph:
    node_ids, node_to_index, communities = read_communities(communities_path)
    earliest_by_target: list[dict[int, int]] = [dict() for _ in node_ids]
    unknown = 0
    with gzip.open(friends_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["friend_date", "user_id", "friend_id", "mutual"]
        if reader.fieldnames != expected:
            raise DiagnosticError(f"Unexpected friendship schema: {reader.fieldnames}")
        for row in reader:
            date = int(row["friend_date"])
            target_id = int(row["user_id"])
            source_id = int(row["friend_id"])
            if date <= 0:
                unknown += 1
                continue
            if source_id == target_id:
                raise DiagnosticError("Self-edge found in cleaned friendship input")
            source = node_to_index.get(source_id)
            target = node_to_index.get(target_id)
            if source is None or target is None:
                continue
            previous = earliest_by_target[target].get(source)
            if previous is None or date < previous:
                earliest_by_target[target][source] = date

    incoming_sources: list[list[int]] = []
    incoming_dates: list[list[int]] = []
    for mapping in earliest_by_target:
        ordered = sorted((date, source) for source, date in mapping.items())
        incoming_dates.append([date for date, _ in ordered])
        incoming_sources.append([source for _, source in ordered])
    return TemporalGraph(
        node_ids, node_to_index, communities, incoming_sources, incoming_dates, unknown
    )


def read_vote_index(path: Path, selected_stories: set[int], graph_nodes: set[int]) -> VoteIndex:
    starts: dict[int, int] = {}
    user_times: dict[int, list[int]] = defaultdict(list)
    story_user_time: dict[tuple[int, int], int] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["vote_date", "voter_id", "story_id"]:
            raise DiagnosticError(f"Unexpected cleaned-vote schema: {reader.fieldnames}")
        for row in reader:
            timestamp = int(row["vote_date"])
            node = int(row["voter_id"])
            story = int(row["story_id"])
            if node in graph_nodes:
                user_times[node].append(timestamp)
            if story in selected_stories:
                starts[story] = min(starts.get(story, timestamp), timestamp)
                story_user_time[(story, node)] = timestamp
    missing = selected_stories - set(starts)
    if missing:
        raise DiagnosticError(f"Missing vote start times for stories: {sorted(missing)}")
    for times in user_times.values():
        times.sort()
    return VoteIndex(starts, dict(user_times), story_user_time)


def group_rows(data: ControlledData, mask: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row_index in np.flatnonzero(mask):
        groups[(int(data.story[row_index]), int(data.time_bin[row_index]))].append(int(row_index))
    return [
        (story, time_bin, np.asarray(indices, dtype=np.int64))
        for (story, time_bin), indices in sorted(groups.items())
    ]


def raw_target_features(
    data: ControlledData,
    rows: np.ndarray,
    graph: TemporalGraph,
    votes: VoteIndex,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = np.empty((len(rows), 3), dtype=np.float64)
    context = np.column_stack(
        (data.columns["log_time"][rows], data.columns["log_cascade_size"][rows])
    )
    for output_index, row in enumerate(rows):
        story = int(data.story[row])
        time_bin = int(data.time_bin[row])
        timestamp = votes.absolute_time(story, time_bin)
        node_id = int(data.node[row])
        graph_index = graph.node_to_index[node_id]
        history, other_story, _ = votes.node_history(node_id, story, timestamp)
        numeric[output_index] = (
            graph.indegree_before(graph_index, timestamp),
            history,
            math.log1p(other_story),
        )
    return numeric, context


def fit_scalers(
    data: ControlledData,
    train_mask: np.ndarray,
    graph: TemporalGraph,
    votes: VoteIndex,
) -> tuple[Scale, Scale]:
    rows = np.flatnonzero(train_mask)
    numeric, context = raw_target_features(data, rows, graph, votes)
    numeric_std = np.std(numeric, axis=0)
    context_std = np.std(context, axis=0)
    return (
        Scale(np.mean(numeric, axis=0), np.where(numeric_std > 0, numeric_std, 1.0)),
        Scale(np.mean(context, axis=0), np.where(context_std > 0, context_std, 1.0)),
    )


def node_features(
    global_indices: list[int],
    story: int,
    timestamp: int,
    graph: TemporalGraph,
    votes: VoteIndex,
    scale: Scale,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.empty((len(global_indices), 4), dtype=np.float32)
    community = np.empty(len(global_indices), dtype=np.int64)
    for local, global_index in enumerate(global_indices):
        node_id = graph.node_ids[global_index]
        history, other_story, adopted = votes.node_history(node_id, story, timestamp)
        continuous = np.asarray(
            [graph.indegree_before(global_index, timestamp), history, math.log1p(other_story)],
            dtype=np.float64,
        )
        values[local, :3] = scale.apply(continuous).astype(np.float32)
        values[local, 3] = adopted
        community[local] = graph.community_index[global_index]
    return torch.from_numpy(values), torch.from_numpy(community)


def row_batches(rows: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def forward_batch(
    model: TemporalGraphSAGE,
    data: ControlledData,
    rows: np.ndarray,
    story: int,
    time_bin: int,
    graph: TemporalGraph,
    votes: VoteIndex,
    node_scale: Scale,
    context_scale: Scale,
    fanouts: tuple[int, int],
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    timestamp = votes.absolute_time(story, time_bin)
    target_global = [graph.node_to_index[int(data.node[row])] for row in rows]
    local_nodes, edge_index, target_local = graph.sampled_subgraph(
        target_global, timestamp, fanouts, rng
    )
    numeric_x, community = node_features(
        local_nodes, story, timestamp, graph, votes, node_scale
    )
    raw_context = np.column_stack(
        (data.columns["log_time"][rows], data.columns["log_cascade_size"][rows])
    )
    context = torch.from_numpy(context_scale.apply(raw_context).astype(np.float32))
    return model(
        numeric_x.to(device),
        community.to(device),
        edge_index.to(device),
        target_local.to(device),
        context.to(device),
    )


def train_model(
    model: TemporalGraphSAGE,
    data: ControlledData,
    train_groups: list[tuple[int, int, np.ndarray]],
    graph: TemporalGraph,
    votes: VoteIndex,
    node_scale: Scale,
    context_scale: Scale,
    args: argparse.Namespace,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        group_order = list(range(len(train_groups)))
        random.Random(args.seed + epoch).shuffle(group_order)
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for group_position in group_order:
            story, time_bin, group = train_groups[group_position]
            shuffled = group.copy()
            np.random.default_rng(args.seed * 1000 + epoch * 100 + group_position).shuffle(shuffled)
            for batch_number, rows in enumerate(row_batches(shuffled, args.batch_size)):
                optimizer.zero_grad(set_to_none=True)
                logits = forward_batch(
                    model, data, rows, story, time_bin, graph, votes,
                    node_scale, context_scale, tuple(args.fanout),
                    random.Random(args.seed * 1_000_000 + epoch * 10_000 + group_position * 10 + batch_number),
                    device,
                )
                target = torch.from_numpy(data.y[rows].astype(np.float32)).to(device)
                weight = torch.from_numpy(data.weight[rows].astype(np.float32)).to(device)
                loss = F.binary_cross_entropy_with_logits(
                    logits, target, weight=weight, reduction="sum"
                ) / torch.sum(weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite GraphSAGE training loss")
                loss.backward()
                optimizer.step()
                batch_weight = float(torch.sum(weight).detach().cpu())
                weighted_loss_sum += float(loss.detach().cpu()) * batch_weight
                weight_sum += batch_weight
        epoch_loss = weighted_loss_sum / weight_sum
        history.append(epoch_loss)
        print(f"epoch {epoch + 1}/{args.epochs}: weighted BCE={epoch_loss:.10g}", flush=True)
    return history


def predict(
    model: TemporalGraphSAGE,
    data: ControlledData,
    groups: list[tuple[int, int, np.ndarray]],
    graph: TemporalGraph,
    votes: VoteIndex,
    node_scale: Scale,
    context_scale: Scale,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    print(">>> PREDICT ENTERED", flush=True)
    rows_out: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    first_batch = True
    model.eval()
    with torch.no_grad():
        for group_position, (story, time_bin, group) in enumerate(groups):
            for batch_number, rows in enumerate(row_batches(group, args.batch_size)):
                if first_batch:
                    print(">>> PREDICT FIRST BATCH START", flush=True)
                logits = forward_batch(
                    model, data, rows, story, time_bin, graph, votes,
                    node_scale, context_scale, tuple(args.fanout),
                    random.Random(args.seed * 2_000_000 + group_position * 10 + batch_number),
                    device,
                )
                if first_batch:
                    print(">>> PREDICT FIRST BATCH FORWARD FINISHED", flush=True)
                    first_batch = False
                rows_out.append(rows)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
    indices = np.concatenate(rows_out)
    prediction = np.concatenate(probabilities)
    order = np.argsort(indices)
    print(">>> PREDICT INTERNAL FINISHED", flush=True)
    return indices[order], prediction[order]


def read_reference_metrics(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
    required = {"M0", "M1", "XGBoost"}
    if {str(row["model"]) for row in rows} != required:
        raise DiagnosticError("Expected M0, M1, and XGBoost reference metrics")
    return rows


def metric_output_row(result: dict[str, object], feature_set: str, notes: str) -> dict[str, object]:
    return {
        "model": result["model"],
        "test_rows": result["rows"],
        "test_positives": result["positives"],
        "weighted_y_rate": result["weighted_y_rate"],
        "weighted_log_loss": result["weighted_log_loss"],
        "weighted_brier": result["weighted_brier"],
        "weighted_roc_auc": result["weighted_roc_auc"],
        "weighted_pr_auc": result["weighted_pr_auc"],
        "feature_set": feature_set,
        "notes": notes,
    }


def plot_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    names = [str(row["model"]) for row in rows]
    roc = [float(row["weighted_roc_auc"]) for row in rows]
    pr = [float(row["weighted_pr_auc"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = ["#7f8c8d", "#95a5a6", "#1f77b4", "#d62728"]
    axes[0].bar(names, roc, color=colors)
    axes[0].set_ylabel("Weighted ROC-AUC")
    axes[0].set_ylim(0, 1)
    axes[1].bar(names, pr, color=colors)
    axes[1].set_ylabel("Weighted PR-AUC")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Held-out Digg story split: GraphSAGE vs existing baselines")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def subgroup_metrics(
    data: ControlledData,
    test_rows: np.ndarray,
    gnn_probability: np.ndarray,
    xgb_probability: np.ndarray,
) -> list[dict[str, object]]:
    exposure = data.columns["m_in"][test_rows] + data.columns["m_out"][test_rows]
    degree = data.columns["degree"][test_rows]
    median_degree = float(np.median(degree))
    subsets = {
        "all": np.ones(len(test_rows), dtype=bool),
        "zero_exposure": exposure == 0,
        "exposed": exposure > 0,
        "degree_at_or_below_test_median": degree <= median_degree,
        "degree_above_test_median": degree > median_degree,
    }
    rows: list[dict[str, object]] = []
    for subset, mask in subsets.items():
        if not np.any(mask) or len(np.unique(data.y[test_rows][mask])) < 2:
            continue
        for model, probability in (("XGBoost", xgb_probability), ("GraphSAGE", gnn_probability)):
            result = metrics(
                model, subset, data.y[test_rows][mask], probability[mask], data.weight[test_rows][mask]
            )
            rows.append(result)
    return rows


def write_report(
    path: Path,
    args: argparse.Namespace,
    graph: TemporalGraph,
    training_loss: list[float],
    overall: list[dict[str, object]],
    subgroups: list[dict[str, object]],
) -> None:
    lookup = {str(row["model"]): row for row in overall}
    gnn = lookup["GraphSAGE"]
    xgb = lookup["XGBoost"]
    pr_delta = float(gnn["weighted_pr_auc"]) - float(xgb["weighted_pr_auc"])
    if pr_delta > 0:
        conclusion = (
            "GraphSAGE has a higher held-out PR-AUC point estimate than XGBoost on this saved split. "
            "No story-bootstrap interval is computed here, so this is not evidence of a stable or "
            "statistically significant improvement across cascades."
        )
    else:
        conclusion = (
            "Graph representation learning on this small pilot did not outperform the handcrafted-feature "
            "XGBoost baseline on held-out PR-AUC. Possible contributors include the small number of training "
            "cascades, sparse observed network exposure, limited neighborhood sampling, and strong behavioral "
            "context features. These are hypotheses, not established causes."
        )

    lines = [
        "# Digg GraphSAGE baseline",
        "",
        "## Scope and split integrity",
        "",
        "- Task: predict next-hour first adoption for rows in the existing pilot exposure table.",
        "- Split: the saved 80/20 `story_id` split from `pilot_story_split.csv`; story overlap is prohibited.",
        "- Loss: weighted binary cross-entropy using the existing `sampling_weight` and no class weight.",
        "- Test stories are evaluated only after the fixed epoch budget; they are not used for tuning or early stopping.",
        f"- Architecture: two-layer GraphSAGE, hidden={args.hidden}, community embedding={args.community_dim}, fanouts={tuple(args.fanout)}, epochs={args.epochs}, seed={args.seed}.",
        f"- Temporal graph: `friend_id -> user_id`; each sampled edge satisfies `0 < friend_date < window_start`. Unknown-date edges excluded: {graph.unknown_date_edges_excluded:,}.",
        "- Node histories and activity counts use timestamps strictly earlier than each window start.",
        "",
        "## Held-out metrics",
        "",
        "| model | log loss | Brier | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['model']} | {float(row['weighted_log_loss']):.10g} | "
            f"{float(row['weighted_brier']):.10g} | {float(row['weighted_roc_auc']):.10g} | "
            f"{float(row['weighted_pr_auc']):.10g} |"
        )
    lines.extend(["", "## Subgroup diagnostics", "", "| subset | model | rows | weighted y rate | ROC-AUC | PR-AUC |", "|---|---|---:|---:|---:|---:|"])
    for row in subgroups:
        lines.append(
            f"| {row['subset']} | {row['model']} | {row['rows']} | "
            f"{float(row['weighted_y_rate']):.10g} | {float(row['weighted_roc_auc']):.10g} | "
            f"{float(row['weighted_pr_auc']):.10g} |"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            conclusion,
            "",
            f"Final training weighted-BCE history: `{', '.join(f'{value:.8g}' for value in training_loss)}`.",
            "",
            "## Leakage and interpretation cautions",
            "",
            "- Friendship edges with `friend_date=0` are excluded because their temporal order is unknown.",
            "- Graph snapshots, node histories, activity, and within-story adoption indicators are all left-truncated at the window start.",
            "- The story split is grouped but not a chronological platform split; this limits claims about deployment under temporal distribution shift.",
            "- Shared graph nodes may appear in different stories, as in the existing transductive pilot, but no test-story outcome is used for fitting.",
            "- Metrics are predictive. GraphSAGE outputs, regression coefficients, and SHAP values are not causal effects and do not recover physical `a`, `b`, or `theta` from Digg.",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.epochs <= 50:
        raise ValueError("--epochs must be between 1 and 50")
    if min(args.hidden, args.community_dim, args.batch_size, *args.fanout) <= 0:
        raise ValueError("Hidden size, embedding size, batch size, and fanouts must be positive")
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimization hyperparameter")


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_files(
        [args.friends, args.communities, args.exposure, args.votes, args.split, args.xgb_metrics]
    )
    metrics_path = args.output_dir / "gnn_metrics.csv"
    plot_path = args.output_dir / "gnn_vs_xgb.png"
    destinations = [metrics_path, plot_path, args.report]
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not args.overwrite and not args.preflight_only:
        raise FileExistsError(
            "GNN output(s) already exist; pass --overwrite to replace only these files: "
            + ", ".join(existing)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)

    data = construct_controlled_data(args.exposure, args.votes, args.communities)
    split = read_split(args.split)
    if set(int(value) for value in np.unique(data.story)) != set(split):
        raise DiagnosticError("Exposure stories do not exactly match the saved story split")
    train_mask = np.asarray([split[int(story)] == "train" for story in data.story])
    test_mask = ~train_mask
    if set(data.story[train_mask].astype(int)) & set(data.story[test_mask].astype(int)):
        raise DiagnosticError("A story appears in both train and test")

    graph = read_temporal_graph(args.friends, args.communities)
    missing_nodes = sorted(set(data.node.astype(int)) - set(graph.node_to_index))
    if missing_nodes:
        raise DiagnosticError(f"Exposure nodes missing from graph: {missing_nodes[:10]}")
    votes = read_vote_index(
        args.votes, set(data.story.astype(int)), set(graph.node_to_index)
    )
    if args.preflight_only:
        for label, mask in (
            ("all", np.ones(len(data.y), dtype=bool)),
            ("train", train_mask),
            ("test", test_mask),
        ):
            prevalence = float(np.sum(data.weight[mask] * data.y[mask]) / np.sum(data.weight[mask]))
            print(
                f"{label}: stories={len(set(data.story[mask].astype(int)))}, "
                f"rows={int(np.sum(mask))}, positives={int(np.sum(data.y[mask]))}, "
                f"weighted_prevalence={prevalence:.12%}"
            )
        print(
            f"graph: nodes={graph.num_nodes}, communities={graph.num_communities}, "
            f"unknown_date_edges_excluded={graph.unknown_date_edges_excluded}"
        )
        print("Preflight passed; no model was trained and no output was written.")
        return
    node_scale, context_scale = fit_scalers(data, train_mask, graph, votes)
    train_groups = group_rows(data, train_mask)
    test_groups = group_rows(data, test_mask)

    model = TemporalGraphSAGE(
        graph.num_communities,
        community_dim=args.community_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    training_loss = train_model(
        model, data, train_groups, graph, votes, node_scale, context_scale, args, device
    )
    print(">>> 1 TRAIN FINISHED", flush=True)
    print(">>> 2 START GNN PREDICT", flush=True)
    test_rows, gnn_probability = predict(
        model, data, test_groups, graph, votes, node_scale, context_scale, args, device
    )
    print(">>> 3 GNN PREDICT FINISHED", flush=True)
    expected_test_rows = np.flatnonzero(test_mask)
    if not np.array_equal(test_rows, expected_test_rows):
        raise DiagnosticError("GraphSAGE predictions do not cover the exact saved test rows")

    gnn_result = metrics(
        "GraphSAGE", "test", data.y[test_rows], gnn_probability, data.weight[test_rows]
    )
    print(">>> 4 GNN METRICS FINISHED", flush=True)
    reference_rows = read_reference_metrics(args.xgb_metrics)
    output_rows = reference_rows + [
        metric_output_row(
            gnn_result,
            "temporal GraphSAGE;degree;community embedding;history;activity;time;cascade size",
            "Fixed hyperparameters; saved story split; sampling_weight; no test tuning or early stopping",
        )
    ]

    x_train = matrix(data, FEATURE_SETS["XGB_full"])[train_mask]
    x_test = matrix(data, FEATURE_SETS["XGB_full"])[test_mask]
    print(">>> 5 START XGBOOST FIT", flush=True)
    xgb = fit_xgb(x_train, data.y[train_mask], data.weight[train_mask])
    print(">>> 6 XGBOOST FIT FINISHED", flush=True)
    print(">>> 7 START XGBOOST PREDICT", flush=True)
    xgb_probability = xgb.predict_proba(x_test)[:, 1]
    print(">>> 8 XGBOOST PREDICT FINISHED", flush=True)
    subgroups = subgroup_metrics(data, test_rows, gnn_probability, xgb_probability)
    print(">>> 9 SUBGROUP METRICS FINISHED", flush=True)

    fields = [
        "model", "test_rows", "test_positives", "weighted_y_rate",
        "weighted_log_loss", "weighted_brier", "weighted_roc_auc", "weighted_pr_auc",
        "feature_set", "notes",
    ]
    print(">>> START CSV WRITE", flush=True)
    atomic_csv(metrics_path, fields, output_rows)
    print(">>> CSV WRITE FINISHED", flush=True)
    print(">>> START PLOT WRITE", flush=True)
    plot_comparison(plot_path, output_rows)
    print(">>> PLOT WRITE FINISHED", flush=True)
    print(">>> START REPORT WRITE", flush=True)
    write_report(args.report, args, graph, training_loss, output_rows, subgroups)
    print(">>> REPORT WRITE FINISHED", flush=True)
    print(f"Metrics: {metrics_path}")
    print(f"Plot: {plot_path}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
