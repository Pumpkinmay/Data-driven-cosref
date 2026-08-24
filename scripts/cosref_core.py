"""Shared graph, cascade, and estimation utilities for data-driven COSREF."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_NETWORKS = ("friendster", "youtube", "orkut")


@dataclass
class GraphData:
    name: str
    node_ids: np.ndarray
    adjacency: list[np.ndarray]
    labels: np.ndarray
    num_edges: int
    original_num_nodes: int
    original_num_edges: int
    sampled: bool = False
    sampling_method: str = "none"

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_communities(self) -> int:
        return len(np.unique(self.labels))


@dataclass
class FitResult:
    raw_coef_m_in: float
    raw_coef_m_out: float
    raw_coef_degree: float
    intercept: float
    a_hat: float
    b_hat: float
    theta_hat: float
    iterations: int


def effective_parameters(
    coef_m_in: float, coef_m_out: float, coef_degree: float, beta: float
) -> tuple[float, float, float]:
    """Map raw count-logit coefficients to the fixed-beta parameter scale."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    return coef_m_in / beta, coef_m_out / beta, -coef_degree / beta


def read_communities(path: Path) -> dict[int, int]:
    labels: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "community"]:
            raise ValueError(
                f"Expected columns ['id', 'community']; got {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                node = int(row["id"])
                community = int(row["community"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid community row at line {line_number}: {row}"
                ) from exc
            if node in labels:
                raise ValueError(f"Duplicate community label for node {node}")
            labels[node] = community
    if not labels:
        raise ValueError("Community file is empty")
    return labels


def read_edges(path: Path) -> tuple[list[tuple[int, int]], set[int]]:
    edges: list[tuple[int, int]] = []
    nodes: set[int] = set()
    seen: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "target"]:
            raise ValueError(
                f"Expected columns ['source', 'target']; got {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                source = int(row["source"])
                target = int(row["target"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid edge at line {line_number}: {row}") from exc
            if source == target:
                raise ValueError(f"Self-loop at line {line_number}: {source}")
            canonical = (min(source, target), max(source, target))
            if canonical in seen:
                raise ValueError(f"Duplicate undirected edge at line {line_number}")
            seen.add(canonical)
            edges.append((source, target))
            nodes.update((source, target))
    if not edges:
        raise ValueError("Edge file is empty")
    return edges, nodes


def build_adjacency(
    node_ids: np.ndarray, edges: Sequence[tuple[int, int]]
) -> list[np.ndarray]:
    lookup = {int(node): index for index, node in enumerate(node_ids)}
    neighbors: list[list[int]] = [[] for _ in node_ids]
    for source, target in edges:
        u = lookup[source]
        v = lookup[target]
        neighbors[u].append(v)
        neighbors[v].append(u)
    return [np.asarray(values, dtype=np.int64) for values in neighbors]


def largest_component(adjacency: Sequence[np.ndarray]) -> list[int]:
    visited = np.zeros(len(adjacency), dtype=bool)
    largest: list[int] = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        component: list[int] = []
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                index = int(neighbor)
                if not visited[index]:
                    visited[index] = True
                    stack.append(index)
        if len(component) > len(largest):
            largest = component
    return largest


def connected_bfs_sample(
    component: Sequence[int],
    adjacency: Sequence[np.ndarray],
    size: int,
    rng: np.random.Generator,
) -> list[int]:
    allowed = np.zeros(len(adjacency), dtype=bool)
    allowed[np.asarray(component)] = True
    start = int(rng.choice(np.asarray(component)))
    selected: list[int] = []
    seen = {start}
    queue: deque[int] = deque([start])
    while queue and len(selected) < size:
        node = queue.popleft()
        selected.append(node)
        candidates = [int(value) for value in adjacency[node] if allowed[int(value)]]
        rng.shuffle(candidates)
        for neighbor in candidates:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if len(selected) != size:
        raise ValueError(f"Could sample only {len(selected)} of {size} requested nodes")
    return selected


def induce_subgraph(graph: GraphData, selected_indices: Sequence[int]) -> GraphData:
    selected = np.asarray(sorted(selected_indices), dtype=np.int64)
    old_to_new = np.full(graph.num_nodes, -1, dtype=np.int64)
    old_to_new[selected] = np.arange(len(selected))
    adjacency: list[np.ndarray] = []
    for old_index in selected:
        retained = old_to_new[graph.adjacency[int(old_index)]]
        adjacency.append(retained[retained >= 0])
    return GraphData(
        name=graph.name,
        node_ids=graph.node_ids[selected],
        adjacency=adjacency,
        labels=graph.labels[selected],
        num_edges=sum(map(len, adjacency)) // 2,
        original_num_nodes=graph.original_num_nodes,
        original_num_edges=graph.original_num_edges,
        sampled=True,
        sampling_method="randomized_bfs_from_largest_component",
    )


def load_network(
    network: str,
    data_dir: Path,
    max_nodes: int = 5000,
    rng: np.random.Generator | None = None,
) -> GraphData:
    edge_path = data_dir / f"edges_{network}_remapped.csv"
    community_path = data_dir / f"community_{network}_remapped.csv"
    missing = [path.name for path in (edge_path, community_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing file(s): " + ", ".join(missing))
    communities = read_communities(community_path)
    edges, edge_nodes = read_edges(edge_path)
    missing_labels = sorted(edge_nodes - set(communities))
    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} edge-list nodes lack community labels; "
            f"examples: {missing_labels[:10]}"
        )
    node_ids = np.asarray(sorted(edge_nodes), dtype=np.int64)
    graph = GraphData(
        name=network,
        node_ids=node_ids,
        adjacency=build_adjacency(node_ids, edges),
        labels=np.asarray([communities[int(node)] for node in node_ids]),
        num_edges=len(edges),
        original_num_nodes=len(node_ids),
        original_num_edges=len(edges),
    )
    if max_nodes > 0 and graph.num_nodes > max_nodes:
        rng = rng or np.random.default_rng(0)
        component = largest_component(graph.adjacency)
        size = min(max_nodes, len(component))
        graph = induce_subgraph(
            graph, connected_bfs_sample(component, graph.adjacency, size, rng)
        )
    if graph.num_communities < 2:
        raise ValueError("Analyzed graph contains fewer than two communities")
    return graph


def make_demo_graph(random_seed: int = 12345) -> GraphData:
    """Create a small deterministic two-community graph for smoke tests."""
    rng = np.random.default_rng(random_seed)
    n_per_community = 60
    n = 2 * n_per_community
    labels = np.repeat([0, 1], n_per_community)
    edges: set[tuple[int, int]] = set()
    for offset in (0, n_per_community):
        for local in range(n_per_community):
            for step in (1, 2, 3):
                target = offset + (local + step) % n_per_community
                source = offset + local
                edges.add((min(source, target), max(source, target)))
    while len(edges) < 440:
        source = int(rng.integers(0, n_per_community))
        target = int(rng.integers(n_per_community, n))
        edges.add((source, target))
    edge_list = sorted(edges)
    node_ids = np.arange(1, n + 1, dtype=np.int64)
    return GraphData(
        name="demo",
        node_ids=node_ids,
        adjacency=build_adjacency(np.arange(n), edge_list),
        labels=labels,
        num_edges=len(edge_list),
        original_num_nodes=n,
        original_num_edges=len(edge_list),
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def add_active_exposure(
    newly_active: np.ndarray,
    graph: GraphData,
    m_in: np.ndarray,
    m_out: np.ndarray,
) -> None:
    for active_node in newly_active:
        neighbors = graph.adjacency[int(active_node)]
        same = graph.labels[neighbors] == graph.labels[int(active_node)]
        m_in[neighbors[same]] += 1
        m_out[neighbors[~same]] += 1


def simulate_cascades(
    graph: GraphData,
    *,
    a: float,
    b: float,
    theta: float,
    beta: float,
    n_cascades: int,
    max_steps: int,
    seed_size: int,
    rng: np.random.Generator,
    collect_exposures: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < theta < 1 or min(a, beta) <= 0 or b < 0:
        raise ValueError("Require a>0, b>=0, beta>0, and theta in (0, 1)")
    candidates = np.flatnonzero(graph.labels == 0)
    if len(candidates) < seed_size:
        raise ValueError("Community 0 is smaller than seed_size")
    degrees = np.asarray([len(values) for values in graph.adjacency])
    feature_batches: list[np.ndarray] = []
    outcome_batches: list[np.ndarray] = []
    final_sizes = np.empty(n_cascades, dtype=np.float64)
    for cascade_id in range(n_cascades):
        active = np.zeros(graph.num_nodes, dtype=bool)
        seeds = rng.choice(candidates, size=seed_size, replace=False)
        active[seeds] = True
        m_in = np.zeros(graph.num_nodes, dtype=np.int64)
        m_out = np.zeros(graph.num_nodes, dtype=np.int64)
        add_active_exposure(seeds, graph, m_in, m_out)
        for _ in range(max_steps):
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
            if collect_exposures:
                feature_batches.append(features)
                outcome_batches.append(outcomes)
            newly_active = susceptible[outcomes == 1]
            if not len(newly_active):
                break
            active[newly_active] = True
            add_active_exposure(newly_active, graph, m_in, m_out)
        final_sizes[cascade_id] = np.mean(active)
    if collect_exposures:
        return np.vstack(feature_batches), np.concatenate(outcome_batches), final_sizes
    return np.empty((0, 3)), np.empty(0, dtype=np.int8), final_sizes


def fit_parameters(features: np.ndarray, outcomes: np.ndarray, beta: float) -> FitResult:
    if len(np.unique(outcomes)) != 2:
        raise ValueError("Logistic regression requires both y classes")
    model = LogisticRegression(
        C=1.0e6, solver="lbfgs", fit_intercept=True, max_iter=2000
    )
    model.fit(features, outcomes)
    coef_m_in, coef_m_out, coef_degree = (float(value) for value in model.coef_[0])
    a_hat, b_hat, theta_hat = effective_parameters(
        coef_m_in, coef_m_out, coef_degree, beta
    )
    return FitResult(
        raw_coef_m_in=coef_m_in,
        raw_coef_m_out=coef_m_out,
        raw_coef_degree=coef_degree,
        intercept=float(model.intercept_[0]),
        a_hat=a_hat,
        b_hat=b_hat,
        theta_hat=theta_hat,
        iterations=int(model.n_iter_[0]),
    )


def confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    half_width = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, max(0.0, mean - half_width), min(1.0, mean + half_width)


def write_rows(path: Path, fields: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
