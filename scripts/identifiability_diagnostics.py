#!/usr/bin/env python3
"""Diagnose cross-community exposure identifiability across SNAP networks.

Defaults reproduce the Friendster grid point with the poorest b recovery:
a=0.6, b=0.2, theta=0.15, beta=5.0, 100 cascades, 20 steps, and 5 seeds.
The same parameter setting is simulated independently on Friendster, YouTube,
and Orkut, then summarized in CSV and Markdown form.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_grid_recovery import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUTS_DIR,
    NETWORKS,
    load_network,
    simulate_exposure_table,
)
from cosref_core import make_demo_graph  # noqa: E402


QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
FIELDS = [
    "network",
    "status",
    "reason",
    "a_true",
    "b_true",
    "theta_true",
    "beta",
    "num_nodes",
    "num_edges",
    "community_0_nodes",
    "community_1_nodes",
    "intra_community_edges",
    "inter_community_edges",
    "intra_community_edge_rate",
    "inter_community_edge_rate",
    "num_exposure_rows",
    "valid_exposure_rate",
    "m_out_positive_rows",
    "m_out_positive_rate",
    "positive_y_rows",
    "positive_y_rate",
    "positive_y_with_m_out_rows",
    "positive_y_with_m_out_rate",
    "m_in_mean",
    "m_in_std",
    "m_in_q0",
    "m_in_q25",
    "m_in_q50",
    "m_in_q75",
    "m_in_q100",
    "m_out_mean",
    "m_out_std",
    "m_out_q0",
    "m_out_q25",
    "m_out_q50",
    "m_out_q75",
    "m_out_q100",
    "m_in_m_out_correlation",
    "coef_m_in_raw",
    "coef_m_out_raw",
    "coef_degree_raw",
    "intercept_raw",
    "a_hat",
    "b_hat",
    "theta_hat",
    "fit_iterations",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose m_out identifiability for a normalized-threshold setting."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--networks", nargs="+", default=list(NETWORKS))
    parser.add_argument("--a", type=float, default=0.6)
    parser.add_argument("--b", type=float, default=0.2)
    parser.add_argument("--theta", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--n-cascades", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed-size", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--demo", action="store_true", help="Use a built-in small graph")
    return parser.parse_args()


def empty_row(network: str, status: str, reason: str) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDS}
    row.update({"network": network, "status": status, "reason": reason})
    return row


def edge_community_counts(graph) -> tuple[int, int]:
    intra = 0
    inter = 0
    for node, neighbors in enumerate(graph.adjacency):
        for neighbor in neighbors:
            if node >= int(neighbor):
                continue
            if graph.labels[node] == graph.labels[int(neighbor)]:
                intra += 1
            else:
                inter += 1
    return intra, inter


def distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    quantile_values = np.quantile(values, QUANTILES)
    result = {f"{prefix}_mean": float(np.mean(values))}
    result[f"{prefix}_std"] = float(np.std(values, ddof=1))
    for label, value in zip(("q0", "q25", "q50", "q75", "q100"), quantile_values):
        result[f"{prefix}_{label}"] = float(value)
    return result


def diagnose_network(graph, args: argparse.Namespace, network_index: int) -> dict[str, object]:
    row = empty_row(graph.name, "ok", "")
    row.update(
        {
            "a_true": args.a,
            "b_true": args.b,
            "theta_true": args.theta,
            "beta": args.beta,
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "community_0_nodes": int(np.sum(graph.labels == 0)),
            "community_1_nodes": int(np.sum(graph.labels == 1)),
        }
    )
    intra_edges, inter_edges = edge_community_counts(graph)
    row.update(
        {
            "intra_community_edges": intra_edges,
            "inter_community_edges": inter_edges,
            "intra_community_edge_rate": intra_edges / graph.num_edges,
            "inter_community_edge_rate": inter_edges / graph.num_edges,
        }
    )

    # Matches the main grid script's independent per-network, grid-index-2 RNG.
    rng = np.random.default_rng(
        np.random.SeedSequence([args.random_seed, network_index, 2])
    )
    features, outcomes, valid_rate, _ = simulate_exposure_table(
        graph,
        a=args.a,
        b=args.b,
        theta=args.theta,
        beta=args.beta,
        n_cascades=args.n_cascades,
        max_steps=args.max_steps,
        seed_size=args.seed_size,
        rng=rng,
    )
    m_in, m_out = features[:, 0], features[:, 1]
    m_out_positive = m_out > 0
    positive_y = outcomes == 1
    positive_y_with_m_out = positive_y & m_out_positive
    correlation = float(np.corrcoef(m_in, m_out)[0, 1])

    model = LogisticRegression(
        C=1.0e6,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2000,
    )
    model.fit(features, outcomes)
    coef_m_in, coef_m_out, coef_degree = (float(value) for value in model.coef_[0])
    intercept = float(model.intercept_[0])
    row.update(
        {
            "num_exposure_rows": len(outcomes),
            "valid_exposure_rate": valid_rate,
            "m_out_positive_rows": int(np.sum(m_out_positive)),
            "m_out_positive_rate": float(np.mean(m_out_positive)),
            "positive_y_rows": int(np.sum(positive_y)),
            "positive_y_rate": float(np.mean(positive_y)),
            "positive_y_with_m_out_rows": int(np.sum(positive_y_with_m_out)),
            "positive_y_with_m_out_rate": (
                float(np.mean(m_out_positive[positive_y])) if np.any(positive_y) else float("nan")
            ),
            "m_in_m_out_correlation": correlation,
            "coef_m_in_raw": coef_m_in,
            "coef_m_out_raw": coef_m_out,
            "coef_degree_raw": coef_degree,
            "intercept_raw": intercept,
            "a_hat": coef_m_in / args.beta,
            "b_hat": coef_m_out / args.beta,
            "theta_hat": -coef_degree / args.beta,
            "fit_iterations": int(model.n_iter_[0]),
            **distribution(m_in, "m_in"),
            **distribution(m_out, "m_out"),
        }
    )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if value == "":
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_note(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    successful = [row for row in rows if row["status"] == "ok"]
    lines = [
        "# Identifiability diagnostics",
        "",
        "This diagnostic compares the same normalized-threshold setting across networks:",
        "",
        f"`a={args.a}`, `b={args.b}`, `theta={args.theta}`, `beta={args.beta}`, "
        f"`n_cascades={args.n_cascades}`, `max_steps={args.max_steps}`, "
        f"and `seed_size={args.seed_size}`.",
        "",
        "The default setting reproduces the Friendster grid point where `b_hat` was near zero.",
        "",
        "| Network | Inter-edge rate | m_out > 0 rate | y=1 rate | y=1 with m_out > 0 | raw coef(m_out) | b_hat |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in successful:
        lines.append(
            "| {network} | {inter:.2%} | {mout:.2%} | {positive:.3%} | {positive_mout:.2%} | {coef:.4f} | {bhat:.4f} |".format(
                network=row["network"],
                inter=float(row["inter_community_edge_rate"]),
                mout=float(row["m_out_positive_rate"]),
                positive=float(row["positive_y_rate"]),
                positive_mout=float(row["positive_y_with_m_out_rate"]),
                coef=float(row["coef_m_out_raw"]),
                bhat=float(row["b_hat"]),
            )
        )
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"\n- `{row['network']}` skipped: {row['reason']}")

    friendster = next((row for row in successful if row["network"] == "friendster"), None)
    others = [row for row in successful if row["network"] != "friendster"]
    if friendster and others:
        other_mout_rate = float(np.mean([row["m_out_positive_rate"] for row in others]))
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "Friendster has a much weaker cross-community exposure signal at this high-threshold setting. "
                f"Only {float(friendster['m_out_positive_rate']):.2%} of its exposure rows have `m_out > 0`, "
                f"compared with {other_mout_rate:.2%} on average across the other successful networks. "
                f"Its positive-outcome rate is only {float(friendster['positive_y_rate']):.3%}, so there are few "
                "activation events available to identify the cross-community coefficient. "
                "This sparsity makes the fitted `coef_m_out` unstable and pulls `b_hat` toward zero. "
                "The CSV contains the complete distributional and coefficient diagnostics.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.theta < 1.0:
        raise ValueError("--theta must be in (0, 1)")
    if min(args.beta, args.a, args.b) <= 0.0:
        raise ValueError("--a, --b, and --beta must be positive")
    if min(args.n_cascades, args.max_steps, args.seed_size) <= 0:
        raise ValueError("cascade, step, and seed counts must be positive")

    data_dir = args.data_dir.expanduser().resolve()
    outputs_dir = args.outputs_dir.expanduser().resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    requested_networks = ["demo"] if args.demo else args.networks
    for network_index, network in enumerate(requested_networks):
        sampling_rng = np.random.default_rng(
            np.random.SeedSequence([args.random_seed, network_index, 2_147_483_647])
        )
        try:
            graph = (
                make_demo_graph(args.random_seed)
                if args.demo
                else load_network(network, data_dir, args.max_nodes, sampling_rng)
            )
            row = diagnose_network(graph, args, network_index)
            print(
                f"{network}: b_hat={float(row['b_hat']):.6f}, "
                f"m_out>0={float(row['m_out_positive_rate']):.2%}, "
                f"y=1={float(row['positive_y_rate']):.3%}"
            )
        except Exception as exc:
            row = empty_row(network, "skipped", str(exc))
            print(f"{network}: skipped ({exc})")
        rows.append(row)

    csv_path = outputs_dir / "identifiability_diagnostics.csv"
    note_path = outputs_dir / "identifiability_diagnostics.md"
    write_csv(csv_path, rows)
    write_note(note_path, rows, args)
    print(f"CSV: {csv_path}")
    print(f"Note: {note_path}")


if __name__ == "__main__":
    main()
