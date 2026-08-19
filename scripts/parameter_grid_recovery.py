#!/usr/bin/env python3
"""Parameter-grid recovery on multiple remapped SNAP community networks.

For every available network and every Cartesian-product parameter setting, the
script simulates synchronous irreversible cascades with

    sigmoid(beta * (a * m_in + b * m_out - theta * degree)),

then recovers a, b, and theta using logistic regression on m_in, m_out, and
degree. Missing or invalid networks are recorded and skipped without stopping
the remaining experiment.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from cosref_core import make_demo_graph


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PIPELINE_ROOT / "data"
DEFAULT_OUTPUTS_DIR = PIPELINE_ROOT / "outputs"
NETWORKS = ("friendster", "youtube", "orkut")
DEFAULT_A_VALUES = (0.6, 0.8, 1.0)
DEFAULT_B_VALUES = (0.2, 0.4, 0.6)
DEFAULT_THETA_VALUES = (0.05, 0.10, 0.15)

SUMMARY_FIELDS = [
    "network",
    "status",
    "skip_reason",
    "num_nodes",
    "num_edges",
    "num_communities",
    "original_num_nodes",
    "original_num_edges",
    "sampled",
    "sampling_method",
    "a_true",
    "b_true",
    "theta_true",
    "beta",
    "a_hat",
    "b_hat",
    "theta_hat",
    "a_abs_error",
    "b_abs_error",
    "theta_abs_error",
    "a_rel_error",
    "b_rel_error",
    "theta_rel_error",
    "num_exposure_rows",
    "positive_y_rate",
    "valid_exposure_rate",
    "n_cascades",
    "max_steps",
    "seed_size",
    "zero_exposure_rate",
    "m_in_std",
    "m_out_std",
    "degree_std",
    "design_condition_number",
    "fit_iterations",
]


@dataclass
class NetworkData:
    name: str
    node_ids: np.ndarray
    adjacency: list[np.ndarray]
    labels: np.ndarray
    num_edges: int
    original_num_nodes: int
    original_num_edges: int
    sampled: bool
    sampling_method: str

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_communities(self) -> int:
        return len(np.unique(self.labels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run normalized-threshold parameter recovery across a grid and SNAP networks."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--networks", nargs="+", default=list(NETWORKS))
    parser.add_argument(
        "--a-values", nargs="+", type=float, default=list(DEFAULT_A_VALUES)
    )
    parser.add_argument(
        "--b-values", nargs="+", type=float, default=list(DEFAULT_B_VALUES)
    )
    parser.add_argument(
        "--theta-values", nargs="+", type=float, default=list(DEFAULT_THETA_VALUES)
    )
    parser.add_argument("--n-cascades", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--seed-size",
        type=int,
        default=5,
        help="Number of initial seeds sampled from community 0 per cascade.",
    )
    parser.add_argument(
        "--beta", type=float, default=5.0, help="Fixed sigmoid steepness."
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=5000,
        help=(
            "Maximum analyzed nodes per network. Larger graphs are reduced to a "
            "connected BFS sample from their largest component; use 0 to disable."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--demo", action="store_true", help="Use a built-in small graph")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_cascades <= 0:
        raise ValueError("--n-cascades must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.seed_size <= 0:
        raise ValueError("--seed-size must be positive")
    if args.beta <= 0.0:
        raise ValueError("--beta must be positive")
    if args.max_nodes < 0:
        raise ValueError("--max-nodes must be nonnegative")
    if not args.a_values or not args.b_values or not args.theta_values:
        raise ValueError("Parameter value lists cannot be empty")
    invalid_theta = [value for value in args.theta_values if not 0.0 < value < 1.0]
    if invalid_theta:
        raise ValueError(f"All theta values must be in (0, 1); invalid: {invalid_theta}")


def read_communities(path: Path) -> dict[int, int]:
    communities: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "community"]:
            raise ValueError(
                f"expected community columns ['id', 'community'], got {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                node = int(row["id"])
                community = int(row["community"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid community record at line {line_number}: {row}"
                ) from exc
            if node in communities:
                raise ValueError(f"duplicate community label for node {node}")
            communities[node] = community
    if not communities:
        raise ValueError("community file is empty")
    return communities


def read_edges(path: Path) -> tuple[list[tuple[int, int]], set[int]]:
    edges: list[tuple[int, int]] = []
    edge_nodes: set[int] = set()
    seen_edges: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "target"]:
            raise ValueError(
                f"expected edge columns ['source', 'target'], got {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                source = int(row["source"])
                target = int(row["target"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid edge record at line {line_number}: {row}") from exc
            if source == target:
                raise ValueError(f"self-loop at edge line {line_number}: node {source}")
            edge = (min(source, target), max(source, target))
            if edge in seen_edges:
                raise ValueError(f"duplicate undirected edge at line {line_number}: {edge}")
            seen_edges.add(edge)
            edges.append((source, target))
            edge_nodes.update((source, target))
    if not edges:
        raise ValueError("edge file is empty")
    return edges, edge_nodes


def build_adjacency(
    node_ids: np.ndarray, edges: Sequence[tuple[int, int]]
) -> list[np.ndarray]:
    node_to_index = {int(node): index for index, node in enumerate(node_ids)}
    adjacency_lists: list[list[int]] = [[] for _ in node_ids]
    for source, target in edges:
        source_index = node_to_index[source]
        target_index = node_to_index[target]
        adjacency_lists[source_index].append(target_index)
        adjacency_lists[target_index].append(source_index)
    return [np.asarray(neighbors, dtype=np.int64) for neighbors in adjacency_lists]


def largest_component(adjacency: Sequence[np.ndarray]) -> list[int]:
    visited = np.zeros(len(adjacency), dtype=bool)
    largest: list[int] = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        component: list[int] = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                neighbor_index = int(neighbor)
                if not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    queue.append(neighbor_index)
        if len(component) > len(largest):
            largest = component
    return largest


def connected_bfs_sample(
    component: Sequence[int],
    adjacency: Sequence[np.ndarray],
    sample_size: int,
    rng: np.random.Generator,
) -> list[int]:
    """Select a connected node set by randomized BFS inside one component."""
    allowed = np.zeros(len(adjacency), dtype=bool)
    allowed[np.asarray(component, dtype=np.int64)] = True
    start = int(rng.choice(np.asarray(component, dtype=np.int64)))
    selected: list[int] = []
    seen = {start}
    queue: deque[int] = deque([start])
    while queue and len(selected) < sample_size:
        node = queue.popleft()
        selected.append(node)
        neighbors = [int(value) for value in adjacency[node] if allowed[int(value)]]
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if len(selected) != sample_size:
        raise ValueError(
            f"connected sampler selected {len(selected)} of {sample_size} requested nodes"
        )
    return selected


def induce_subgraph(
    node_ids: np.ndarray,
    adjacency: Sequence[np.ndarray],
    labels: np.ndarray,
    selected_indices: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, int]:
    selected = np.asarray(sorted(selected_indices), dtype=np.int64)
    old_to_new = np.full(len(node_ids), -1, dtype=np.int64)
    old_to_new[selected] = np.arange(len(selected), dtype=np.int64)
    new_adjacency: list[np.ndarray] = []
    for old_index in selected:
        retained = old_to_new[adjacency[int(old_index)]]
        new_adjacency.append(retained[retained >= 0])
    num_edges = sum(len(neighbors) for neighbors in new_adjacency) // 2
    return node_ids[selected], new_adjacency, labels[selected], num_edges


def load_network(
    network: str,
    data_dir: Path,
    max_nodes: int,
    sampling_rng: np.random.Generator,
) -> NetworkData:
    edge_path = data_dir / f"edges_{network}_remapped.csv"
    community_path = data_dir / f"community_{network}_remapped.csv"
    missing_files = [
        str(path.name) for path in (edge_path, community_path) if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError("missing file(s): " + ", ".join(missing_files))

    communities = read_communities(community_path)
    edges, edge_nodes = read_edges(edge_path)
    missing_labels = sorted(edge_nodes - set(communities))
    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} edge nodes lack community labels; "
            f"examples: {missing_labels[:10]}"
        )

    node_ids = np.asarray(sorted(edge_nodes), dtype=np.int64)
    labels = np.asarray([communities[int(node)] for node in node_ids], dtype=np.int64)
    adjacency = build_adjacency(node_ids, edges)
    original_num_nodes = len(node_ids)
    original_num_edges = len(edges)
    sampled = False
    sampling_method = "none"
    num_edges = original_num_edges

    if max_nodes > 0 and original_num_nodes > max_nodes:
        component = largest_component(adjacency)
        target_size = min(max_nodes, len(component))
        selected = connected_bfs_sample(
            component, adjacency, target_size, sampling_rng
        )
        node_ids, adjacency, labels, num_edges = induce_subgraph(
            node_ids, adjacency, labels, selected
        )
        sampled = True
        sampling_method = "randomized_bfs_from_largest_component"

    if len(np.unique(labels)) < 2:
        raise ValueError("analyzed graph has fewer than two represented communities")

    return NetworkData(
        name=network,
        node_ids=node_ids,
        adjacency=adjacency,
        labels=labels,
        num_edges=num_edges,
        original_num_nodes=original_num_nodes,
        original_num_edges=original_num_edges,
        sampled=sampled,
        sampling_method=sampling_method,
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def add_active_neighbor_exposure(
    newly_active: np.ndarray,
    adjacency: Sequence[np.ndarray],
    labels: np.ndarray,
    m_in: np.ndarray,
    m_out: np.ndarray,
) -> None:
    for active_node in newly_active:
        neighbors = adjacency[int(active_node)]
        same_community = labels[neighbors] == labels[int(active_node)]
        m_in[neighbors[same_community]] += 1
        m_out[neighbors[~same_community]] += 1


def simulate_exposure_table(
    graph: NetworkData,
    *,
    a: float,
    b: float,
    theta: float,
    beta: float,
    n_cascades: int,
    max_steps: int,
    seed_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Construct the exposure table in memory for one grid setting."""
    seed_candidates = np.flatnonzero(graph.labels == 0)
    if len(seed_candidates) < seed_size:
        raise ValueError(
            f"community 0 has {len(seed_candidates)} nodes, fewer than seed_size={seed_size}"
        )

    degrees = np.asarray(
        [len(neighbors) for neighbors in graph.adjacency], dtype=np.int64
    )
    feature_batches: list[np.ndarray] = []
    outcome_batches: list[np.ndarray] = []

    for _ in range(n_cascades):
        active = np.zeros(graph.num_nodes, dtype=bool)
        initial_seeds = rng.choice(seed_candidates, size=seed_size, replace=False)
        active[initial_seeds] = True
        m_in = np.zeros(graph.num_nodes, dtype=np.int64)
        m_out = np.zeros(graph.num_nodes, dtype=np.int64)
        add_active_neighbor_exposure(
            initial_seeds, graph.adjacency, graph.labels, m_in, m_out
        )

        for _time in range(max_steps):
            susceptible = np.flatnonzero(~active)
            if not len(susceptible):
                break
            features = np.column_stack(
                (m_in[susceptible], m_out[susceptible], degrees[susceptible])
            ).astype(np.float64)
            probabilities = sigmoid(
                beta
                * (
                    a * features[:, 0]
                    + b * features[:, 1]
                    - theta * features[:, 2]
                )
            )
            outcomes = (rng.random(len(susceptible)) < probabilities).astype(np.int8)
            feature_batches.append(features)
            outcome_batches.append(outcomes)

            newly_active = susceptible[outcomes == 1]
            if not len(newly_active):
                break
            active[newly_active] = True
            add_active_neighbor_exposure(
                newly_active, graph.adjacency, graph.labels, m_in, m_out
            )

    features = np.vstack(feature_batches)
    outcomes = np.concatenate(outcome_batches)
    valid = (
        np.isfinite(features).all(axis=1)
        & (features >= 0).all(axis=1)
        & (features[:, 0] + features[:, 1] <= features[:, 2])
        & np.isin(outcomes, (0, 1))
    )
    valid_exposure_rate = float(np.mean(valid))
    features = features[valid]
    outcomes = outcomes[valid]
    if not len(outcomes):
        raise ValueError("no valid exposure rows were generated")

    standard_deviations = np.std(features, axis=0)
    if np.any(standard_deviations == 0):
        condition_number = float("inf")
    else:
        standardized = (features - np.mean(features, axis=0)) / standard_deviations
        condition_number = float(np.linalg.cond(standardized))
    diagnostics = {
        "zero_exposure_rate": float(np.mean(features[:, 0] + features[:, 1] == 0)),
        "m_in_std": float(standard_deviations[0]),
        "m_out_std": float(standard_deviations[1]),
        "degree_std": float(standard_deviations[2]),
        "design_condition_number": condition_number,
    }
    return features, outcomes, valid_exposure_rate, diagnostics


def fit_parameters(
    features: np.ndarray, outcomes: np.ndarray, beta: float
) -> tuple[float, float, float, int]:
    if len(np.unique(outcomes)) != 2:
        positive_rate = float(np.mean(outcomes))
        raise ValueError(f"only one outcome class; positive_y_rate={positive_rate:.6g}")
    model = LogisticRegression(
        C=1.0e6,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2000,
    )
    model.fit(features, outcomes)
    coef_m_in, coef_m_out, coef_degree = model.coef_[0]
    return (
        float(coef_m_in / beta),
        float(coef_m_out / beta),
        float(-coef_degree / beta),
        int(model.n_iter_[0]),
    )


def empty_summary_row(network: str, status: str, reason: str) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in SUMMARY_FIELDS}
    row.update({"network": network, "status": status, "skip_reason": reason})
    return row


def graph_metadata(graph: NetworkData) -> dict[str, object]:
    return {
        "network": graph.name,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "num_communities": graph.num_communities,
        "original_num_nodes": graph.original_num_nodes,
        "original_num_edges": graph.original_num_edges,
        "sampled": graph.sampled,
        "sampling_method": graph.sampling_method,
    }


def run_grid_for_network(
    graph: NetworkData,
    parameter_grid: Sequence[tuple[float, float, float]],
    args: argparse.Namespace,
    network_index: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    metadata = graph_metadata(graph)
    for grid_index, (a_true, b_true, theta_true) in enumerate(parameter_grid):
        row = empty_summary_row(graph.name, "ok", "")
        row.update(metadata)
        row.update(
            {
                "a_true": a_true,
                "b_true": b_true,
                "theta_true": theta_true,
                "beta": args.beta,
                "n_cascades": args.n_cascades,
                "max_steps": args.max_steps,
                "seed_size": args.seed_size,
            }
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([args.random_seed, network_index, grid_index])
        )
        try:
            features, outcomes, valid_rate, diagnostics = simulate_exposure_table(
                graph,
                a=a_true,
                b=b_true,
                theta=theta_true,
                beta=args.beta,
                n_cascades=args.n_cascades,
                max_steps=args.max_steps,
                seed_size=args.seed_size,
                rng=rng,
            )
            a_hat, b_hat, theta_hat, fit_iterations = fit_parameters(
                features, outcomes, args.beta
            )
            estimates = {"a": a_hat, "b": b_hat, "theta": theta_hat}
            truths = {"a": a_true, "b": b_true, "theta": theta_true}
            row.update(
                {
                    "a_hat": a_hat,
                    "b_hat": b_hat,
                    "theta_hat": theta_hat,
                    "num_exposure_rows": len(outcomes),
                    "positive_y_rate": float(np.mean(outcomes)),
                    "valid_exposure_rate": valid_rate,
                    "fit_iterations": fit_iterations,
                    **diagnostics,
                }
            )
            for parameter in ("a", "b", "theta"):
                absolute_error = abs(estimates[parameter] - truths[parameter])
                row[f"{parameter}_abs_error"] = absolute_error
                row[f"{parameter}_rel_error"] = absolute_error / abs(
                    truths[parameter]
                )
        except Exception as exc:
            row["status"] = "fit_failed"
            row["skip_reason"] = str(exc)
        results.append(row)
        print(
            f"  {graph.name} grid {grid_index + 1:02d}/{len(parameter_grid)} "
            f"(a={a_true:g}, b={b_true:g}, theta={theta_true:g}): {row['status']}"
        )
    return results


def write_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def successful_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["status"] == "ok"]


def plot_parameter_recovery(
    path: Path, rows: Sequence[dict[str, object]]
) -> None:
    valid_rows = successful_rows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    if not valid_rows:
        for axis in axes:
            axis.text(0.5, 0.5, "No successful fits", ha="center", va="center")
            axis.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    networks = sorted({str(row["network"]) for row in valid_rows})
    colors = plt.get_cmap("tab10")
    color_by_network = {name: colors(index) for index, name in enumerate(networks)}
    marker_sequence = ("o", "s", "^")
    marker_by_network = {
        name: marker_sequence[index % len(marker_sequence)]
        for index, name in enumerate(networks)
    }
    for axis, parameter in zip(axes, ("a", "b", "theta")):
        all_values: list[float] = []
        for network in networks:
            network_rows = [row for row in valid_rows if row["network"] == network]
            true_values = np.asarray(
                [float(row[f"{parameter}_true"]) for row in network_rows]
            )
            estimates = np.asarray(
                [float(row[f"{parameter}_hat"]) for row in network_rows]
            )
            all_values.extend(true_values.tolist())
            all_values.extend(estimates.tolist())
            axis.scatter(
                true_values,
                estimates,
                s=48,
                alpha=0.90,
                color=color_by_network[network],
                marker=marker_by_network[network],
                edgecolors="white",
                linewidths=0.55,
                zorder=3 + networks.index(network),
                label=network.title(),
            )
        lower = min(all_values)
        upper = max(all_values)
        padding = max(0.02 * (upper - lower), 0.005)
        axis.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            linestyle="--",
            color="black",
            linewidth=1,
            label="y = x" if parameter == "a" else None,
        )
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xlabel(f"{parameter}_true")
        axis.set_ylabel(f"{parameter}_hat")
        axis.set_title(f"{parameter}: estimated vs true")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Normalized-threshold parameter-grid recovery")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def network_mean_relative_errors(
    rows: Sequence[dict[str, object]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for network in sorted({str(row["network"]) for row in successful_rows(rows)}):
        network_rows = [
            row
            for row in successful_rows(rows)
            if row["network"] == network
        ]
        errors = [
            float(row[f"{parameter}_rel_error"])
            for row in network_rows
            for parameter in ("a", "b", "theta")
        ]
        result[network] = float(np.mean(errors))
    return result


def plot_network_error_summary(
    path: Path, mean_errors: dict[str, float]
) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    if mean_errors:
        networks = list(mean_errors)
        values = [100.0 * mean_errors[network] for network in networks]
        bars = axis.bar([name.title() for name in networks], values)
        axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=3)
        axis.set_ylabel("Mean relative error (%)")
        axis.set_title("Mean recovery error by network")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0.0, max(values) * 1.2 if max(values) > 0 else 1.0)
    else:
        axis.text(0.5, 0.5, "No successful fits", ha="center", va="center")
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_diagnostics(rows: Sequence[dict[str, object]]) -> None:
    valid_rows = successful_rows(rows)
    poor_rows = [
        row
        for row in valid_rows
        if np.mean(
            [float(row[f"{parameter}_rel_error"]) for parameter in ("a", "b", "theta")]
        )
        > 0.25
    ]
    if not poor_rows:
        print("No grid setting has mean relative error above 25%.")
        return
    print(f"Poor-recovery settings (>25% mean relative error): {len(poor_rows)}")
    reason_counts = {"class_imbalance": 0, "limited_exposure": 0, "collinearity": 0}
    for row in poor_rows:
        positive_rate = float(row["positive_y_rate"])
        if positive_rate < 0.01 or positive_rate > 0.99:
            reason_counts["class_imbalance"] += 1
        if (
            float(row["zero_exposure_rate"]) > 0.95
            or float(row["m_in_std"]) < 0.1
            or float(row["m_out_std"]) < 0.1
        ):
            reason_counts["limited_exposure"] += 1
        if float(row["design_condition_number"]) > 100.0:
            reason_counts["collinearity"] += 1
    print("Diagnostic flags among poor settings: " + str(reason_counts))


def main() -> None:
    args = parse_args()
    validate_args(args)
    data_dir = args.data_dir.expanduser().resolve()
    outputs_dir = args.outputs_dir.expanduser().resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    max_nodes = args.max_nodes if args.max_nodes > 0 else 0
    parameter_grid = list(
        itertools.product(args.a_values, args.b_values, args.theta_values)
    )
    requested_networks = ["demo"] if args.demo else args.networks
    print(
        f"Parameter grid: {len(parameter_grid)} settings; "
        f"networks requested: {', '.join(requested_networks)}"
    )

    rows: list[dict[str, object]] = []
    successful_networks: list[str] = []
    skipped_networks: dict[str, str] = {}
    for network_index, network in enumerate(requested_networks):
        sampling_rng = np.random.default_rng(
            np.random.SeedSequence([args.random_seed, network_index, 2_147_483_647])
        )
        try:
            graph = (
                make_demo_graph(args.random_seed)
                if args.demo
                else load_network(network, data_dir, max_nodes, sampling_rng)
            )
        except Exception as exc:
            reason = str(exc)
            rows.append(empty_summary_row(network, "network_skipped", reason))
            skipped_networks[network] = reason
            print(f"Skipping {network}: {reason}")
            continue
        successful_networks.append(network)
        print(
            f"Loaded {network}: nodes={graph.num_nodes}, edges={graph.num_edges}, "
            f"communities={graph.num_communities}, sampled={graph.sampled}"
        )
        rows.extend(run_grid_for_network(graph, parameter_grid, args, network_index))

    summary_path = outputs_dir / "parameter_grid_recovery.csv"
    recovery_plot_path = outputs_dir / "parameter_grid_recovery.png"
    error_plot_path = outputs_dir / "network_error_summary.png"
    write_summary(summary_path, rows)
    plot_parameter_recovery(recovery_plot_path, rows)
    mean_errors = network_mean_relative_errors(rows)
    plot_network_error_summary(error_plot_path, mean_errors)

    print("Successful networks: " + (", ".join(successful_networks) or "none"))
    if skipped_networks:
        for network, reason in skipped_networks.items():
            print(f"Skipped network {network}: {reason}")
    else:
        print("Skipped networks: none")
    for network, error in mean_errors.items():
        print(f"Mean relative error for {network}: {error:.3%}")
    print_diagnostics(rows)
    print(f"Grid summary: {summary_path}")
    print(f"Recovery scatter: {recovery_plot_path}")
    print(f"Network error summary: {error_plot_path}")


if __name__ == "__main__":
    main()
