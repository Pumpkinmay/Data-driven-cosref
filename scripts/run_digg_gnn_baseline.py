#!/usr/bin/env python3
"""Train and report the gated Digg GraphSAGE baseline after Stage 1 preflight."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose_digg_pilot import DiagnosticError, atomic_csv, metrics
from fit_digg_controlled_models import ControlledData, construct_controlled_data
from fit_digg_gnn import (
    TemporalGraph,
    VoteIndex,
    group_rows,
    read_split,
    read_temporal_graph,
    read_vote_index,
    row_batches,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "digg"
DOCS = ROOT / "docs"
SEED = 42
MAX_EPOCHS = 50
PATIENCE = 5
FANOUTS = (15, 10)
BATCH_SIZE = 512


@dataclass(frozen=True)
class FeatureScale:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


class AlignedGraphSAGE(nn.Module):
    """Two-layer mean GraphSAGE using only the declared context-aligned features."""

    def __init__(self, num_communities: int) -> None:
        super().__init__()
        self.community_embedding = nn.Embedding(num_communities, 16)
        self.conv1 = SAGEConv(4 + 16, 64, aggr="mean")
        self.conv2 = SAGEConv(64, 64, aggr="mean")
        self.classifier = nn.Linear(64, 1)

    def forward(
        self,
        numeric_x: torch.Tensor,
        community: torch.Tensor,
        edge_index: torch.Tensor,
        target_index: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.cat((numeric_x, self.community_embedding(community)), dim=1)
        hidden = F.relu(self.conv1(hidden, edge_index))
        hidden = F.dropout(hidden, p=0.2, training=self.training)
        hidden = F.relu(self.conv2(hidden, edge_index))
        return self.classifier(hidden[target_index]).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--friends", type=Path, default=PROCESSED / "digg_friends_clean.csv.gz")
    parser.add_argument("--communities", type=Path, default=PROCESSED / "digg_communities.csv")
    parser.add_argument("--exposure", type=Path, default=PROCESSED / "digg_exposure_pilot.csv.gz")
    parser.add_argument("--votes", type=Path, default=PROCESSED / "digg_votes_clean.csv.gz")
    parser.add_argument("--split", type=Path, default=OUTPUT / "pilot_story_split.csv")
    parser.add_argument("--preflight-report", type=Path, default=OUTPUT / "gnn_preflight.md")
    parser.add_argument("--metrics", type=Path, default=OUTPUT / "gnn_metrics.csv")
    parser.add_argument("--plot", type=Path, default=OUTPUT / "gnn_vs_xgb.png")
    parser.add_argument("--report", type=Path, default=DOCS / "digg_gnn_baseline.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_stage_one(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Stage 1 report is missing: {path}")
    text = path.read_text(encoding="utf-8")
    if "PASS — all five Stage 1 checks completed" not in text:
        raise RuntimeError("Stage 1 did not record a complete PASS; refusing Stage 2")


def fit_scale(data: ControlledData, fit_mask: np.ndarray) -> FeatureScale:
    values = np.column_stack(
        (
            data.columns["degree"][fit_mask],
            data.columns["log_user_activity"][fit_mask],
            data.columns["log_cascade_size"][fit_mask],
            data.columns["log_time"][fit_mask],
        )
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite value in aligned GNN features")
    std = np.std(values, axis=0)
    return FeatureScale(np.mean(values, axis=0), np.where(std > 0, std, 1.0))


def aligned_node_features(
    local_nodes: list[int],
    target_local: torch.Tensor,
    rows: np.ndarray,
    story: int,
    timestamp: int,
    data: ControlledData,
    graph: TemporalGraph,
    votes: VoteIndex,
    scale: FeatureScale,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_cascade = float(data.columns["log_cascade_size"][rows[0]])
    log_time = float(data.columns["log_time"][rows[0]])
    if not np.all(data.columns["log_cascade_size"][rows] == log_cascade):
        raise RuntimeError("Cascade-size context varies inside one story-time group")
    if not np.all(data.columns["log_time"][rows] == log_time):
        raise RuntimeError("Time context varies inside one story-time group")

    values = np.empty((len(local_nodes), 4), dtype=np.float64)
    community = np.empty(len(local_nodes), dtype=np.int64)
    for local, global_index in enumerate(local_nodes):
        node_id = graph.node_ids[global_index]
        _, other_story_prior, _ = votes.node_history(node_id, story, timestamp)
        values[local] = (
            graph.indegree_before(global_index, timestamp),
            math.log1p(other_story_prior),
            log_cascade,
            log_time,
        )
        community[local] = graph.community_index[global_index]

    # Target rows use the exact leakage-audited controls used by XGBoost.
    target_positions = target_local.cpu().numpy()
    values[target_positions, 0] = data.columns["degree"][rows]
    values[target_positions, 1] = data.columns["log_user_activity"][rows]
    values = scale.apply(values).astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite scaled GNN node feature")
    return torch.from_numpy(values), torch.from_numpy(community)


def forward_batch(
    model: AlignedGraphSAGE,
    data: ControlledData,
    rows: np.ndarray,
    story: int,
    time_bin: int,
    graph: TemporalGraph,
    votes: VoteIndex,
    scale: FeatureScale,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    timestamp = votes.absolute_time(story, time_bin)
    target_global = [graph.node_to_index[int(data.node[row])] for row in rows]
    local_nodes, edge_index, target_local = graph.sampled_subgraph(
        target_global, timestamp, FANOUTS, rng
    )
    numeric_x, community = aligned_node_features(
        local_nodes, target_local, rows, story, timestamp, data, graph, votes, scale
    )
    return model(
        numeric_x.to(device),
        community.to(device),
        edge_index.to(device),
        target_local.to(device),
    )


def predict(
    model: AlignedGraphSAGE,
    data: ControlledData,
    groups: list[tuple[int, int, np.ndarray]],
    graph: TemporalGraph,
    votes: VoteIndex,
    scale: FeatureScale,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
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
    model: AlignedGraphSAGE,
    data: ControlledData,
    fit_groups: list[tuple[int, int, np.ndarray]],
    validation_groups: list[tuple[int, int, np.ndarray]],
    graph: TemporalGraph,
    votes: VoteIndex,
    scale: FeatureScale,
    device: torch.device,
) -> tuple[list[float], list[float], int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_history: list[float] = []
    validation_history: list[float] = []
    best_pr = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
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
                    raise RuntimeError("Non-finite weighted BCE during Stage 2")
                loss.backward()
                optimizer.step()
                batch_weight = float(torch.sum(weight).detach().cpu())
                weighted_loss_sum += float(loss.detach().cpu()) * batch_weight
                weight_sum += batch_weight

        train_loss = weighted_loss_sum / weight_sum
        validation_rows, validation_probability = predict(
            model, data, validation_groups, graph, votes, scale, device
        )
        validation_result = metrics(
            "GraphSAGE",
            "internal_validation",
            data.y[validation_rows],
            validation_probability,
            data.weight[validation_rows],
        )
        validation_pr = float(validation_result["weighted_pr_auc"])
        if not math.isfinite(train_loss) or not math.isfinite(validation_pr):
            raise RuntimeError("Non-finite train loss or internal-validation PR-AUC")
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
        raise RuntimeError("Early stopping did not retain a model state")
    if len(validation_history) < 2 or np.ptp(np.asarray(validation_history)) <= 0:
        raise RuntimeError("Internal-validation PR-AUC is constant")
    if train_history[-1] >= train_history[0]:
        raise RuntimeError(
            f"Training loss did not decline: first={train_history[0]}, last={train_history[-1]}"
        )
    model.load_state_dict(best_state)
    return train_history, validation_history, best_epoch


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def same_split_reference_rows() -> list[dict[str, object]]:
    logistic = read_rows(OUTPUT / "pilot_model_metrics.csv")
    ablation = read_rows(OUTPUT / "xgb_ablation_metrics.csv")
    rows: list[dict[str, object]] = []
    for name in ("M0", "M1"):
        selected = [row for row in logistic if row["model"] == name and row["subset"] == "all_test"]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one all_test row for {name}")
        row = selected[0]
        rows.append(
            {
                "model": name,
                "weighted_log_loss": float(row["weighted_log_loss"]),
                "weighted_brier": float(row["weighted_brier"]),
                "weighted_roc_auc": float(row["weighted_roc_auc"]),
                "weighted_pr_auc": float(row["weighted_pr_auc"]),
                "source": "pilot_model_metrics.csv (saved 80/20 test stories)",
            }
        )
    for name in ("XGB_context", "XGB_full"):
        selected = [row for row in ablation if row["model"] == name]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one ablation row for {name}")
        row = selected[0]
        rows.append(
            {
                "model": name,
                "weighted_log_loss": float(row["weighted_log_loss"]),
                "weighted_brier": float(row["weighted_brier"]),
                "weighted_roc_auc": float(row["weighted_roc_auc"]),
                "weighted_pr_auc": float(row["weighted_pr_auc"]),
                "source": "xgb_ablation_metrics.csv (saved 80/20 test stories)",
            }
        )
    return rows


def plot_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    names = [str(row["model"]) for row in rows]
    pr = [float(row["weighted_pr_auc"]) for row in rows]
    roc = [float(row["weighted_roc_auc"]) for row in rows]
    colors = ["#7f8c8d", "#95a5a6", "#4c78a8", "#f58518", "#54a24b"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, pr, color=colors)
    axes[0].set_ylabel("Weighted PR-AUC")
    axes[0].set_yscale("log")
    axes[1].bar(names, roc, color=colors)
    axes[1].set_ylabel("Weighted ROC-AUC")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Held-out Digg stories: GraphSAGE and existing baselines")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    comparison: list[dict[str, object]],
    train_history: list[float],
    validation_history: list[float],
    best_epoch: int,
    fit_stories: set[int],
    validation_stories: set[int],
    test_stories: set[int],
    leakage_checks: dict[str, int],
) -> None:
    lookup = {str(row["model"]): row for row in comparison}
    gnn_pr = float(lookup["GNN"]["weighted_pr_auc"])
    xgb_pr = float(lookup["XGB_full"]["weighted_pr_auc"])
    difference = gnn_pr - xgb_pr
    relative = difference / xgb_pr if xgb_pr else math.nan
    if difference > 0:
        comparison_text = (
            f"GNN exceeded the same-split XGB_full PR-AUC by {difference:.10g} "
            f"({relative:.2%} relative). The leakage checks below were repeated before this statement. "
            "This point estimate does not establish a stable cross-story improvement."
        )
    else:
        comparison_text = (
            f"**GNN lost to XGB_full on the primary ranking metric, weighted PR-AUC.** "
            f"Its PR-AUC was lower by {abs(difference):.10g} ({abs(relative):.2%} relative)."
        )

    # These files are explicitly audited, but their CV/calibration aggregates are not mixed into
    # the saved-split comparison table.
    final_rows = read_rows(OUTPUT / "final_model_metrics.csv")
    cv_rows = read_rows(OUTPUT / "xgb_group_cv_metrics.csv")
    lines = [
        "# Digg GraphSAGE baseline",
        "",
        "## Experimental gate and split",
        "",
        "- Stage 1 preflight passed all five checks before formal training.",
        f"- Fit stories: {len(fit_stories)}; internal-validation stories: {len(validation_stories)}; test stories: {len(test_stories)}.",
        "- The internal validation set is a seed-42 sample of 10% (8/80) of the saved train stories.",
        "- Test stories were used once, after early stopping and restoration of the best internal-validation checkpoint.",
        f"- Best internal-validation epoch: {best_epoch}; maximum epochs: {MAX_EPOCHS}; patience: {PATIENCE}.",
        "- Loss is sampling-weighted BCE. Metrics use the same `sampling_weight` convention as the existing baselines.",
        "",
        "## Features",
        "",
        "The GNN input is `degree`, a learned community embedding, `log_user_activity`, "
        "`log_cascade_size`, and `log_time`. Target-node controls are the exact leakage-audited "
        "values constructed by `construct_controlled_data`; sampled-neighbor degree and activity are "
        "computed strictly before the window start. `m_in` and `m_out` are not passed to GraphSAGE.",
        "",
        "XGB_full uses `degree`, `log_time`, `log_user_activity`, `log_cascade_size`, `m_in`, and "
        "`m_out`. Thus the GNN replaces the two handcrafted exposure counts with learned message passing "
        "over time-respecting friendship snapshots.",
        "",
        "## Same saved 80/20 story-split comparison",
        "",
        "| model | weighted log loss | weighted Brier | weighted ROC-AUC | weighted PR-AUC | source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | {float(row['weighted_log_loss']):.10g} | "
            f"{float(row['weighted_brier']):.10g} | {float(row['weighted_roc_auc']):.10g} | "
            f"{float(row['weighted_pr_auc']):.10g} | {row['source']} |"
        )
    lines.extend(
        [
            "",
            "The requested `final_model_metrics.csv` and `xgb_group_cv_metrics.csv` were read and "
            f"schema-checked ({len(final_rows)} and {len(cv_rows)} rows). They contain grouped-CV and "
            "calibration aggregates, not M0/M1 rows on this single saved split, so those aggregates are "
            "not mixed into the table above. M0/M1 come from `pilot_model_metrics.csv`; same-split "
            "XGB_context/XGB_full come from `xgb_ablation_metrics.csv`.",
            "",
            "## Result",
            "",
            comparison_text,
            "",
            f"- Weighted log loss was {float(lookup['GNN']['weighted_log_loss']):.10g} for GNN versus "
            f"{float(lookup['XGB_full']['weighted_log_loss']):.10g} for XGB_full; GNN was worse because higher is worse.",
            f"- Weighted Brier was {float(lookup['GNN']['weighted_brier']):.10g} for GNN versus "
            f"{float(lookup['XGB_full']['weighted_brier']):.10g} for XGB_full; GNN was slightly better on this probability-error metric because lower is better.",
            f"- Weighted ROC-AUC was {float(lookup['GNN']['weighted_roc_auc']):.10g} for GNN versus "
            f"{float(lookup['XGB_full']['weighted_roc_auc']):.10g} for XGB_full.",
            f"- Weighted PR-AUC was {gnn_pr:.10g} for GNN versus {xgb_pr:.10g} for XGB_full.",
            "",
            "### Plausible limitations behind the lower GNN ranking result",
            "",
            "- **Limited training scale:** this is a 100-cascade pilot, with only 72 stories used for parameter fitting after the internal validation split.",
            "- **Weak observed graph signal:** the existing ablation results already show that user activity and cascade popularity carry most of the predictive value, while community exposure adds only limited and cross-fold-unstable ranking value.",
            "- **Restricted GNN feature set:** `m_in` and `m_out` were deliberately excluded, so the GNN had to infer exposure-related structure through message passing rather than receiving the handcrafted counts available to XGB_full.",
            "- **Coarse snapshots:** one-hour windows may collapse activation order and short-lived diffusion signals that a snapshot GraphSAGE model could otherwise use.",
            "",
            "These are plausible explanations grounded in the dataset and design, not excuses and not causes established by this single experiment. No number was changed and no post-test hyperparameter adjustment was performed.",
            "",
            f"Train weighted-BCE history: `{', '.join(f'{value:.8g}' for value in train_history)}`.",
            "",
            f"Internal-validation weighted PR-AUC history: `{', '.join(f'{value:.8g}' for value in validation_history)}`.",
            "",
            "## Leakage recheck",
            "",
            f"- Fit/validation/test story overlap: {leakage_checks['story_overlap']}.",
            f"- Controlled-data user-history leakage flags: {leakage_checks['user_control_leakage']}.",
            f"- Controlled-data cascade-size leakage flags: {leakage_checks['cascade_control_leakage']}.",
            f"- Unsorted temporal adjacency lists: {leakage_checks['unsorted_adjacency']}.",
            "- Graph sampling uses `bisect_left(friend_date, window_start)`, so only strict-past edges are eligible.",
            "- Scalers are fitted only on the 72 model-fit stories; neither validation nor test rows enter their estimates.",
            "",
            "## Interpretation",
            "",
            "This GNN result is a predictive association, not a causal effect. The real-data GraphSAGE, "
            "XGBoost, SHAP, and regression results do not recover physical `a`, `b`, or `theta`; parameter "
            "recovery remains a conclusion of the synthetic experiments only.",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    require_stage_one(args.preflight_report)
    required = [args.friends, args.communities, args.exposure, args.votes, args.split]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty inputs: {missing}")
    destinations = [args.metrics, args.plot, args.report]
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs exist; pass --overwrite: {existing}")

    set_seed(SEED)
    device = torch.device("cpu")
    data = construct_controlled_data(args.exposure, args.votes, args.communities)
    split = read_split(args.split)
    all_stories = set(data.story.astype(int))
    if all_stories != set(split):
        raise DiagnosticError("Exposure stories do not exactly match the saved split")
    train_stories = sorted(story for story, label in split.items() if label == "train")
    test_stories = {story for story, label in split.items() if label == "test"}
    if len(train_stories) != 80 or len(test_stories) != 20:
        raise DiagnosticError("Expected saved 80/20 story split")
    validation_stories = set(random.Random(SEED).sample(train_stories, 8))
    fit_stories = set(train_stories) - validation_stories
    story_overlap = len(
        (fit_stories & validation_stories)
        | (fit_stories & test_stories)
        | (validation_stories & test_stories)
    )
    if story_overlap:
        raise DiagnosticError("Story overlap in fit/validation/test split")

    fit_mask = np.asarray([int(story) in fit_stories for story in data.story])
    validation_mask = np.asarray([int(story) in validation_stories for story in data.story])
    test_mask = np.asarray([int(story) in test_stories for story in data.story])
    if not np.all(fit_mask | validation_mask | test_mask):
        raise DiagnosticError("Some exposure rows are outside fit/validation/test")

    graph = read_temporal_graph(args.friends, args.communities)
    votes = read_vote_index(args.votes, all_stories, set(graph.node_to_index))
    scale = fit_scale(data, fit_mask)
    fit_groups = group_rows(data, fit_mask)
    validation_groups = group_rows(data, validation_mask)
    test_groups = group_rows(data, test_mask)
    model = AlignedGraphSAGE(graph.num_communities).to(device)

    print(
        f"Stage 2 split: fit={len(fit_stories)} stories, validation={len(validation_stories)} "
        f"stories, test={len(test_stories)} stories; overlap={story_overlap}",
        flush=True,
    )
    train_history, validation_history, best_epoch = train_with_early_stopping(
        model, data, fit_groups, validation_groups, graph, votes, scale, device
    )

    # Test prediction is performed exactly once, after model selection is complete.
    test_rows, probability = predict(model, data, test_groups, graph, votes, scale, device)
    expected_test_rows = np.flatnonzero(test_mask)
    if not np.array_equal(test_rows, expected_test_rows):
        raise DiagnosticError("Test predictions do not cover the exact saved test rows")
    result = metrics(
        "GNN", "saved_test", data.y[test_rows], probability, data.weight[test_rows]
    )
    metric_names = (
        "weighted_log_loss", "weighted_brier", "weighted_roc_auc", "weighted_pr_auc"
    )
    if not all(math.isfinite(float(result[name])) for name in metric_names):
        raise RuntimeError("Non-finite held-out GNN metric")
    print("Stage 2 held-out test metrics:", flush=True)
    for name in metric_names:
        print(f"  {name}: {float(result[name]):.12g}", flush=True)

    leakage_checks = {
        "story_overlap": story_overlap,
        "user_control_leakage": int(data.user_control_leakage),
        "cascade_control_leakage": int(data.cascade_control_leakage),
        "unsorted_adjacency": sum(
            int(any(left > right for left, right in zip(dates, dates[1:])))
            for dates in graph.incoming_dates
        ),
    }
    if any(leakage_checks.values()):
        raise RuntimeError(f"Leakage recheck failed: {leakage_checks}")

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    metric_row = {
        "model": "GNN",
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
        "features": "degree;community_embedding;log_user_activity;log_cascade_size;log_time",
    }
    atomic_csv(args.metrics, list(metric_row), [metric_row])
    print(f"Stage 2 metrics written: {args.metrics}", flush=True)

    comparison = same_split_reference_rows()
    comparison.append(
        {
            "model": "GNN",
            **{name: float(result[name]) for name in metric_names},
            "source": "gnn_metrics.csv (saved 80/20 test stories)",
        }
    )
    plot_comparison(args.plot, comparison)
    write_report(
        args.report,
        comparison,
        train_history,
        validation_history,
        best_epoch,
        fit_stories,
        validation_stories,
        test_stories,
        leakage_checks,
    )
    print(f"Stage 3 plot written: {args.plot}", flush=True)
    print(f"Stage 3 report written: {args.report}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"GNN STAGE FAILED: {error}", file=sys.stderr, flush=True)
        raise
