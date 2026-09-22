#!/usr/bin/env python3
"""Run the gated, no-full-training preflight for the Digg GraphSAGE baseline."""

from __future__ import annotations

import argparse
import bisect
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.nn import SAGEConv

from fit_digg_controlled_models import construct_controlled_data
from fit_digg_gnn import (
    TemporalGraphSAGE,
    fit_scalers,
    group_rows,
    predict,
    read_split,
    read_temporal_graph,
    read_vote_index,
    set_seed,
    train_model,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "digg"
SEED = 42


def emit(lines: list[str], text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def main() -> None:
    friends = PROCESSED / "digg_friends_clean.csv.gz"
    communities = PROCESSED / "digg_communities.csv"
    exposure = PROCESSED / "digg_exposure_pilot.csv.gz"
    votes = PROCESSED / "digg_votes_clean.csv.gz"
    split_path = OUTPUT / "pilot_story_split.csv"
    report_path = OUTPUT / "gnn_preflight.md"
    required_files = [friends, communities, exposure, split_path]
    lines: list[str] = ["# Digg GraphSAGE preflight", ""]

    emit(lines, "## 1. Environment")
    emit(lines, "")
    emit(lines, f"- torch: `{torch.__version__}`")
    emit(lines, f"- torch_geometric: `{torch_geometric.__version__}`")
    emit(lines, f"- `from torch_geometric.nn import SAGEConv`: PASS (`{SAGEConv.__name__}`)")

    emit(lines, "")
    emit(lines, "## 2. Required data files")
    emit(lines, "")
    for path in required_files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Required file is missing or empty: {path}")
        emit(lines, f"- PASS: `{path.relative_to(ROOT)}` ({path.stat().st_size:,} bytes)")
    if not votes.is_file() or votes.stat().st_size <= 0:
        raise RuntimeError(f"Auxiliary vote file required for absolute time is missing or empty: {votes}")
    emit(lines, f"- Auxiliary temporal input: `{votes.relative_to(ROOT)}` ({votes.stat().st_size:,} bytes)")

    emit(lines, "")
    emit(lines, "## 3. Exposure schema")
    emit(lines, "")
    frame = pd.read_csv(exposure)
    required_columns = {
        "story_id", "time_bin", "node_id", "m_in", "m_out", "degree", "y", "sampling_weight"
    }
    missing = sorted(required_columns - set(frame.columns))
    emit(lines, f"- Shape: `{frame.shape}`")
    emit(lines, f"- Columns: `{list(frame.columns)}`")
    emit(lines, "")
    emit(lines, "First five rows (`pandas.DataFrame.head()`):")
    emit(lines, "")
    emit(lines, "```text")
    for row in frame.head().to_string(index=False).splitlines():
        emit(lines, row)
    emit(lines, "```")
    if missing:
        raise RuntimeError(f"Missing required exposure columns: {missing}")
    emit(lines, "- Required actual columns: PASS")

    data = construct_controlled_data(exposure, votes, communities)
    split = read_split(split_path)
    train_mask = np.asarray([split[int(story)] == "train" for story in data.story])
    test_mask = ~train_mask
    if set(data.story[train_mask].astype(int)) & set(data.story[test_mask].astype(int)):
        raise RuntimeError("A story appears in both train and test")

    graph = read_temporal_graph(friends, communities)
    vote_index = read_vote_index(votes, set(data.story.astype(int)), set(graph.node_to_index))
    first_story = int(np.min(data.story))
    story_rows = data.story == first_story
    first_time_bin = int(np.min(data.time_bin[story_rows]))
    snapshot_rows = np.flatnonzero(story_rows & (data.time_bin == first_time_bin))
    snapshot_start = vote_index.absolute_time(first_story, first_time_bin)

    edge_count = 0
    future_edge_violations = 0
    missed_past_edges = 0
    latest_included: int | None = None
    for dates in graph.incoming_dates:
        cutoff = bisect.bisect_left(dates, snapshot_start)
        edge_count += cutoff
        if cutoff:
            latest_included = dates[cutoff - 1] if latest_included is None else max(
                latest_included, dates[cutoff - 1]
            )
            future_edge_violations += sum(date >= snapshot_start for date in dates[:cutoff])
        missed_past_edges += sum(date < snapshot_start for date in dates[cutoff:])

    emit(lines, "")
    emit(lines, "## 4. First temporal snapshot dry-run")
    emit(lines, "")
    emit(lines, f"- Story: `{first_story}`")
    emit(lines, f"- Time bin: `{first_time_bin}`")
    emit(lines, f"- Absolute window start: `{snapshot_start}`")
    emit(lines, f"- Snapshot node universe: `{graph.num_nodes:,}`")
    emit(lines, f"- Directed edges with `friend_date < window_start`: `{edge_count:,}`")
    emit(lines, f"- Exposure rows in snapshot: `{len(snapshot_rows):,}`")
    emit(lines, f"- Positive samples in snapshot: `{int(np.sum(data.y[snapshot_rows])):,}`")
    emit(lines, f"- Latest included edge time: `{latest_included}`")
    emit(lines, f"- Included edges at/current-after window start: `{future_edge_violations}`")
    emit(lines, f"- Past edges incorrectly excluded by cutoff: `{missed_past_edges}`")
    if future_edge_violations or missed_past_edges:
        raise RuntimeError("Temporal snapshot boundary check failed")
    emit(lines, "- Strict past-only edge check: PASS")

    emit(lines, "")
    emit(lines, "## 5. One-percent training smoke run")
    emit(lines, "")
    train_rows = np.flatnonzero(train_mask)
    smoke_size = max(1, math.ceil(0.01 * len(train_rows)))
    rng = np.random.default_rng(SEED)
    smoke_rows = np.sort(rng.choice(train_rows, size=smoke_size, replace=False))
    smoke_mask = np.zeros(len(data.y), dtype=bool)
    smoke_mask[smoke_rows] = True
    smoke_groups = group_rows(data, smoke_mask)
    node_scale, context_scale = fit_scalers(data, train_mask, graph, vote_index)
    args = argparse.Namespace(
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=1,
        seed=SEED,
        batch_size=512,
        fanout=(15, 10),
    )
    set_seed(SEED)
    device = torch.device("cpu")
    model = TemporalGraphSAGE(
        graph.num_communities, community_dim=16, hidden=64, dropout=0.2
    ).to(device)
    history = train_model(
        model, data, smoke_groups, graph, vote_index, node_scale, context_scale, args, device
    )
    predicted_rows, probability = predict(
        model, data, smoke_groups, graph, vote_index, node_scale, context_scale, args, device
    )
    if len(history) != 1 or not np.isfinite(history[0]):
        raise RuntimeError(f"Smoke loss is not finite: {history}")
    if probability.shape != (smoke_size,) or not np.all(np.isfinite(probability)):
        raise RuntimeError(f"Unexpected or non-finite probability output: {probability.shape}")
    if not np.array_equal(predicted_rows, smoke_rows):
        raise RuntimeError("Smoke predictions do not cover the sampled train rows exactly")
    emit(lines, f"- Full train rows: `{len(train_rows):,}`")
    emit(lines, f"- Smoke rows (ceil 1%, seed 42): `{smoke_size:,}`")
    emit(lines, f"- Smoke story-time groups: `{len(smoke_groups):,}`")
    emit(lines, f"- Epochs: `1`")
    emit(lines, f"- Weighted BCE: `{history[0]:.12g}`")
    emit(lines, f"- Output probability shape: `{probability.shape}`")
    emit(lines, "- Finite forward/backward loss and predictions: PASS")

    emit(lines, "")
    emit(lines, "## Gate result")
    emit(lines, "")
    emit(lines, "**PASS — all five Stage 1 checks completed. No full model training was run.**")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Preflight report: {report_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PREFLIGHT FAILED: {error}", file=sys.stderr, flush=True)
        raise
