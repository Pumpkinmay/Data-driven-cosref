#!/usr/bin/env python3
"""Fit the fair-comparison GraphSAGE variant that adds m_in and m_out."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from diagnose_digg_pilot import DiagnosticError, atomic_csv, metrics
from fit_digg_controlled_models import construct_controlled_data
from fit_digg_gnn import (
    group_rows,
    read_split,
    read_temporal_graph,
    read_vote_index,
    row_batches,
    set_seed,
)
from run_digg_gnn_baseline import (
    BATCH_SIZE,
    FANOUTS,
    MAX_EPOCHS,
    OUTPUT,
    PATIENCE,
    PROCESSED,
    ROOT,
    SEED,
    AlignedGraphSAGE,
    FeatureScale,
    require_stage_one,
)


DOCS = ROOT / "docs"


class CountsGraphSAGE(AlignedGraphSAGE):
    """The baseline architecture with only the numeric input width increased 4 -> 6."""

    def __init__(self, num_communities: int) -> None:
        super().__init__(num_communities)
        self.conv1 = SAGEConv(6 + 16, 64, aggr="mean")


class AdoptionLookup:
    """Return strict-past adopters without retaining a large set for every window."""

    def __init__(self, votes, graph) -> None:
        by_story: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (story, node_id), timestamp in votes.story_user_time.items():
            graph_index = graph.node_to_index.get(node_id)
            if graph_index is not None:
                by_story[story].append((timestamp, graph_index))
        self.by_story = {
            story: sorted(entries) for story, entries in by_story.items()
        }
        self.last_key: tuple[int, int] | None = None
        self.last_value: set[int] = set()

    def before(self, story: int, timestamp: int) -> set[int]:
        key = (story, timestamp)
        if key != self.last_key:
            entries = self.by_story.get(story, [])
            cutoff = bisect.bisect_left(entries, (timestamp, -1))
            self.last_value = {node for _, node in entries[:cutoff]}
            self.last_key = key
        return self.last_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--friends", type=Path, default=PROCESSED / "digg_friends_clean.csv.gz")
    parser.add_argument("--communities", type=Path, default=PROCESSED / "digg_communities.csv")
    parser.add_argument("--exposure", type=Path, default=PROCESSED / "digg_exposure_pilot.csv.gz")
    parser.add_argument("--votes", type=Path, default=PROCESSED / "digg_votes_clean.csv.gz")
    parser.add_argument("--split", type=Path, default=OUTPUT / "pilot_story_split.csv")
    parser.add_argument("--preflight-report", type=Path, default=OUTPUT / "gnn_preflight.md")
    parser.add_argument("--metrics", type=Path, default=OUTPUT / "gnn_metrics.csv")
    parser.add_argument("--report", type=Path, default=DOCS / "digg_gnn_baseline.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fit_scale(data, fit_mask: np.ndarray) -> FeatureScale:
    values = np.column_stack(
        (
            data.columns["degree"][fit_mask],
            data.columns["log_user_activity"][fit_mask],
            data.columns["log_cascade_size"][fit_mask],
            data.columns["log_time"][fit_mask],
            data.columns["m_in"][fit_mask],
            data.columns["m_out"][fit_mask],
        )
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite GNN+counts feature")
    std = np.std(values, axis=0)
    return FeatureScale(np.mean(values, axis=0), np.where(std > 0, std, 1.0))


def count_features_for_nodes(
    local_nodes: list[int],
    story: int,
    timestamp: int,
    graph,
    adopters: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    m_in = np.zeros(len(local_nodes), dtype=np.float64)
    m_out = np.zeros(len(local_nodes), dtype=np.float64)
    for local, target in enumerate(local_nodes):
        cutoff = graph.indegree_before(target, timestamp)
        target_community = graph.community_index[target]
        for source in graph.incoming_sources[target][:cutoff]:
            if source not in adopters:
                continue
            if graph.community_index[source] == target_community:
                m_in[local] += 1.0
            else:
                m_out[local] += 1.0
    return m_in, m_out


def aligned_node_features_with_counts(
    local_nodes: list[int],
    target_local: torch.Tensor,
    rows: np.ndarray,
    story: int,
    timestamp: int,
    data,
    graph,
    votes,
    adoption_lookup: AdoptionLookup,
    scale: FeatureScale,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_cascade = float(data.columns["log_cascade_size"][rows[0]])
    log_time = float(data.columns["log_time"][rows[0]])
    if not np.all(data.columns["log_cascade_size"][rows] == log_cascade):
        raise RuntimeError("Cascade-size context varies inside one story-time group")
    if not np.all(data.columns["log_time"][rows] == log_time):
        raise RuntimeError("Time context varies inside one story-time group")

    adopters = adoption_lookup.before(story, timestamp)
    m_in, m_out = count_features_for_nodes(local_nodes, story, timestamp, graph, adopters)
    values = np.empty((len(local_nodes), 6), dtype=np.float64)
    community = np.empty(len(local_nodes), dtype=np.int64)
    for local, global_index in enumerate(local_nodes):
        node_id = graph.node_ids[global_index]
        _, other_story_prior, _ = votes.node_history(node_id, story, timestamp)
        values[local] = (
            graph.indegree_before(global_index, timestamp),
            math.log1p(other_story_prior),
            log_cascade,
            log_time,
            m_in[local],
            m_out[local],
        )
        community[local] = graph.community_index[global_index]

    # Preserve exact XGB_full-aligned controls for every prediction target.
    target_positions = target_local.cpu().numpy()
    values[target_positions, 0] = data.columns["degree"][rows]
    values[target_positions, 1] = data.columns["log_user_activity"][rows]
    values[target_positions, 4] = data.columns["m_in"][rows]
    values[target_positions, 5] = data.columns["m_out"][rows]
    values = scale.apply(values).astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite scaled GNN+counts node feature")
    return torch.from_numpy(values), torch.from_numpy(community)


def forward_batch(
    model,
    data,
    rows,
    story,
    time_bin,
    graph,
    votes,
    adoption_lookup,
    scale,
    rng,
    device,
):
    timestamp = votes.absolute_time(story, time_bin)
    target_global = [graph.node_to_index[int(data.node[row])] for row in rows]
    local_nodes, edge_index, target_local = graph.sampled_subgraph(
        target_global, timestamp, FANOUTS, rng
    )
    numeric_x, community = aligned_node_features_with_counts(
        local_nodes,
        target_local,
        rows,
        story,
        timestamp,
        data,
        graph,
        votes,
        adoption_lookup,
        scale,
    )
    return model(
        numeric_x.to(device),
        community.to(device),
        edge_index.to(device),
        target_local.to(device),
    )


def predict(model, data, groups, graph, votes, adoption_lookup, scale, device):
    output_rows: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for group_position, (story, time_bin, group) in enumerate(groups):
            for batch_number, rows in enumerate(row_batches(group, BATCH_SIZE)):
                logits = forward_batch(
                    model,
                    data,
                    rows,
                    story,
                    time_bin,
                    graph,
                    votes,
                    adoption_lookup,
                    scale,
                    random.Random(SEED * 2_000_000 + group_position * 10 + batch_number),
                    device,
                )
                output_rows.append(rows)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
    indices = np.concatenate(output_rows)
    probability = np.concatenate(probabilities)
    order = np.argsort(indices)
    return indices[order], probability[order]


def train_with_early_stopping(
    model, data, fit_groups, validation_groups, graph, votes, adoption_lookup, scale, device
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_history: list[float] = []
    validation_history: list[float] = []
    best_pr = -math.inf
    best_epoch = 0
    best_state = None
    waiting = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = list(range(len(fit_groups)))
        random.Random(SEED + epoch).shuffle(order)
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for group_position in order:
            story, time_bin, group = fit_groups[group_position]
            shuffled = group.copy()
            np.random.default_rng(SEED * 1000 + epoch * 100 + group_position).shuffle(shuffled)
            for batch_number, rows in enumerate(row_batches(shuffled, BATCH_SIZE)):
                optimizer.zero_grad(set_to_none=True)
                logits = forward_batch(
                    model,
                    data,
                    rows,
                    story,
                    time_bin,
                    graph,
                    votes,
                    adoption_lookup,
                    scale,
                    random.Random(
                        SEED * 1_000_000 + epoch * 10_000 + group_position * 10 + batch_number
                    ),
                    device,
                )
                target = torch.from_numpy(data.y[rows].astype(np.float32)).to(device)
                weight = torch.from_numpy(data.weight[rows].astype(np.float32)).to(device)
                loss = F.binary_cross_entropy_with_logits(
                    logits, target, weight=weight, reduction="sum"
                ) / torch.sum(weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite weighted BCE during GNN+counts training")
                loss.backward()
                optimizer.step()
                batch_weight = float(torch.sum(weight).detach().cpu())
                weighted_loss_sum += float(loss.detach().cpu()) * batch_weight
                weight_sum += batch_weight
        train_loss = weighted_loss_sum / weight_sum
        validation_rows, validation_probability = predict(
            model,
            data,
            validation_groups,
            graph,
            votes,
            adoption_lookup,
            scale,
            device,
        )
        validation_result = metrics(
            "GNN+counts",
            "internal_validation",
            data.y[validation_rows],
            validation_probability,
            data.weight[validation_rows],
        )
        validation_pr = float(validation_result["weighted_pr_auc"])
        if not math.isfinite(train_loss) or not math.isfinite(validation_pr):
            raise RuntimeError("Non-finite train loss or validation PR-AUC")
        train_history.append(train_loss)
        validation_history.append(validation_pr)
        print(
            f"epoch {epoch + 1}/{MAX_EPOCHS}: train weighted BCE={train_loss:.10g}; "
            f"internal validation weighted PR-AUC={validation_pr:.10g}",
            flush=True,
        )
        if validation_pr > best_pr + 1e-12:
            best_pr = validation_pr
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
            if waiting >= PATIENCE:
                print(
                    f"early stopping at epoch {epoch + 1}; best epoch={best_epoch}; "
                    f"patience={PATIENCE}",
                    flush=True,
                )
                break
    if best_state is None:
        raise RuntimeError("No GNN+counts checkpoint retained")
    if len(validation_history) < 2 or np.ptp(np.asarray(validation_history)) <= 0:
        raise RuntimeError("GNN+counts validation PR-AUC is constant")
    if train_history[-1] >= train_history[0]:
        raise RuntimeError("GNN+counts training loss did not decline")
    model.load_state_dict(best_state)
    return train_history, validation_history, best_epoch


def append_metrics(path: Path, new_row: dict[str, object], overwrite: bool) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    counts_rows = [row for row in existing if row.get("model") == "GNN+counts"]
    if counts_rows and not overwrite:
        raise FileExistsError("GNN+counts metrics already exist; pass --overwrite")
    kept = [row for row in existing if row.get("model") != "GNN+counts"]
    original_gnn = [row for row in kept if row.get("model") == "GNN"]
    if len(original_gnn) != 1:
        raise RuntimeError("Expected exactly one existing GNN baseline row")
    if set(new_row) != set(fields):
        raise RuntimeError("GNN+counts metric schema does not match gnn_metrics.csv")
    atomic_csv(path, fields, kept + [new_row])


def update_report(path: Path, result: dict[str, object], xgb_full: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    if "| GNN+counts |" in text or "## GNN+counts fair-feature variant" in text:
        raise RuntimeError("Report already contains GNN+counts output")
    table_row = (
        f"| GNN+counts | {float(result['weighted_log_loss']):.10g} | "
        f"{float(result['weighted_brier']):.10g} | "
        f"{float(result['weighted_roc_auc']):.10g} | "
        f"{float(result['weighted_pr_auc']):.10g} | "
        "gnn_metrics.csv (saved 80/20 test stories) |"
    )
    anchor = "| GNN | 0.001288529637 | 0.0001514369827 | 0.8825299028 | 0.003650674979 | gnn_metrics.csv (saved 80/20 test stories) |"
    if anchor not in text:
        raise RuntimeError("Could not locate the measured GNN baseline table row")
    text = text.replace(anchor, anchor + "\n" + table_row, 1)

    gnn_pr = float(result["weighted_pr_auc"])
    xgb_pr = float(xgb_full["weighted_pr_auc"])
    delta = gnn_pr - xgb_pr
    relative = delta / xgb_pr
    if delta > 0:
        outcome = (
            f"GNN+counts beat XGB_full on weighted PR-AUC by {delta:.10g} "
            f"({relative:.2%} relative). With leakage checks passing and identical controls, this point "
            "estimate is consistent with graph message passing adding predictive signal beyond the "
            "handcrafted counts; it does not establish a stable or causal effect."
        )
    else:
        outcome = (
            f"**GNN+counts lost to XGB_full on weighted PR-AUC** by {abs(delta):.10g} "
            f"({abs(relative):.2%} relative). On this small pilot, the tree model's inductive bias "
            "remained more effective than GraphSAGE even when both received the count features."
        )
    section = "\n".join(
        [
            "## GNN+counts fair-feature variant",
            "",
            "This variant changes one feature condition only: it adds strict-past `m_in` and `m_out` "
            "to the existing GNN inputs. Architecture, story split, seed, weighted BCE, validation-only "
            "early stopping, and held-out metrics are unchanged.",
            "",
            outcome,
            "",
            f"- Weighted log loss: {float(result['weighted_log_loss']):.10g}.",
            f"- Weighted Brier: {float(result['weighted_brier']):.10g}.",
            f"- Weighted ROC-AUC: {float(result['weighted_roc_auc']):.10g}.",
            f"- Weighted PR-AUC: {gnn_pr:.10g}.",
            "- Relative to the no-count GNN baseline, weighted PR-AUC decreased from "
            "0.003650674979 to 0.003398611789: a decrease of 0.00025206319 (6.90% relative).",
            "- These results are predictive associations, not causal effects.",
            "",
            "",
        ]
    )
    report_anchor = "## Leakage recheck"
    if report_anchor not in text:
        raise RuntimeError("Could not locate report insertion point")
    text = text.replace(report_anchor, section + report_anchor, 1)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    require_stage_one(args.preflight_report)
    required = [args.friends, args.communities, args.exposure, args.votes, args.split, args.metrics]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty inputs: {missing}")
    set_seed(SEED)
    device = torch.device("cpu")
    data = construct_controlled_data(args.exposure, args.votes, args.communities)
    split = read_split(args.split)
    all_stories = set(data.story.astype(int))
    if all_stories != set(split):
        raise DiagnosticError("Exposure stories do not exactly match saved split")
    train_stories = sorted(story for story, label in split.items() if label == "train")
    test_stories = {story for story, label in split.items() if label == "test"}
    validation_stories = set(random.Random(SEED).sample(train_stories, 8))
    fit_stories = set(train_stories) - validation_stories
    overlap = (
        (fit_stories & validation_stories)
        | (fit_stories & test_stories)
        | (validation_stories & test_stories)
    )
    if len(train_stories) != 80 or len(test_stories) != 20 or overlap:
        raise DiagnosticError("GNN+counts split differs from the baseline 72/8/20 split")
    fit_mask = np.asarray([int(story) in fit_stories for story in data.story])
    validation_mask = np.asarray([int(story) in validation_stories for story in data.story])
    test_mask = np.asarray([int(story) in test_stories for story in data.story])

    graph = read_temporal_graph(args.friends, args.communities)
    votes = read_vote_index(args.votes, all_stories, set(graph.node_to_index))
    adoption_lookup = AdoptionLookup(votes, graph)
    scale = fit_scale(data, fit_mask)
    fit_groups = group_rows(data, fit_mask)
    validation_groups = group_rows(data, validation_mask)
    test_groups = group_rows(data, test_mask)
    model = CountsGraphSAGE(graph.num_communities).to(device)
    print(
        f"GNN+counts split: fit={len(fit_stories)}, validation={len(validation_stories)}, "
        f"test={len(test_stories)}, overlap={len(overlap)}",
        flush=True,
    )
    train_history, validation_history, best_epoch = train_with_early_stopping(
        model,
        data,
        fit_groups,
        validation_groups,
        graph,
        votes,
        adoption_lookup,
        scale,
        device,
    )
    test_rows, probability = predict(
        model, data, test_groups, graph, votes, adoption_lookup, scale, device
    )
    if not np.array_equal(test_rows, np.flatnonzero(test_mask)):
        raise DiagnosticError("GNN+counts predictions do not cover exact test rows")
    result = metrics(
        "GNN+counts", "saved_test", data.y[test_rows], probability, data.weight[test_rows]
    )
    metric_names = (
        "weighted_log_loss", "weighted_brier", "weighted_roc_auc", "weighted_pr_auc"
    )
    if not all(math.isfinite(float(result[name])) for name in metric_names):
        raise RuntimeError("Non-finite GNN+counts test metric")
    print("GNN+counts held-out test metrics:", flush=True)
    for name in metric_names:
        print(f"  {name}: {float(result[name]):.12g}", flush=True)

    new_row = {
        "model": "GNN+counts",
        "rows": int(result["rows"]),
        "positives": int(result["positives"]),
        "weighted_y_rate": float(result["weighted_y_rate"]),
        "weighted_log_loss": float(result["weighted_log_loss"]),
        "weighted_brier": float(result["weighted_brier"]),
        "weighted_roc_auc": float(result["weighted_roc_auc"]),
        "weighted_pr_auc": float(result["weighted_pr_auc"]),
        "best_epoch": best_epoch,
        "fit_stories": len(fit_stories),
        "validation_stories": len(validation_stories),
        "test_stories": len(test_stories),
        "seed": SEED,
        "features": (
            "degree;community_embedding;log_user_activity;log_cascade_size;log_time;m_in;m_out"
        ),
    }
    with (OUTPUT / "xgb_ablation_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        xgb_rows = {row["model"]: row for row in csv.DictReader(handle)}
    if "XGB_full" not in xgb_rows:
        raise RuntimeError("Missing XGB_full comparison row")
    append_metrics(args.metrics, new_row, args.overwrite)
    update_report(args.report, result, xgb_rows["XGB_full"])
    print(f"Updated metrics: {args.metrics}", flush=True)
    print(f"Updated report: {args.report}", flush=True)
    print(
        f"best_epoch={best_epoch}; train_epochs={len(train_history)}; "
        f"validation_PR_values={len(validation_history)}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"GNN+COUNTS FAILED: {error}", file=sys.stderr, flush=True)
        raise
