#!/usr/bin/env python3
"""Plot cross-network cascade response to scaling the inter-community weight b."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_grid_recovery import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUTS_DIR,
    NETWORKS,
    add_active_neighbor_exposure,
    load_network,
    sigmoid,
)
from cosref_core import make_demo_graph  # noqa: E402


DEFAULT_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final cascade response as b is scaled around a baseline."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--networks", nargs="+", default=list(NETWORKS))
    parser.add_argument("--b-ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--a", type=float, default=0.6)
    parser.add_argument("--b-baseline", type=float, default=0.2)
    parser.add_argument("--theta", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Independent random repeats used to estimate confidence intervals.",
    )
    parser.add_argument(
        "--n-cascades",
        type=int,
        default=50,
        help="Cascades per random repeat.",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed-size", type=int, default=5)
    parser.add_argument(
        "--global-threshold",
        type=float,
        default=0.5,
        help="Minimum final active fraction defining a global cascade.",
    )
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--demo", action="store_true", help="Use a built-in small graph")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(args.a, args.b_baseline, args.beta) <= 0:
        raise ValueError("--a, --b-baseline, and --beta must be positive")
    if not 0 < args.theta < 1:
        raise ValueError("--theta must be in (0, 1)")
    if not 0 < args.global_threshold <= 1:
        raise ValueError("--global-threshold must be in (0, 1]")
    if min(args.n_repeats, args.n_cascades, args.max_steps, args.seed_size) <= 0:
        raise ValueError("repeat, cascade, step, and seed counts must be positive")
    if args.n_repeats < 2:
        raise ValueError("--n-repeats must be at least 2 to estimate a confidence interval")
    if any(ratio < 0 for ratio in args.b_ratios):
        raise ValueError("--b-ratios cannot contain negative values")


def simulate_final_sizes(
    graph,
    *,
    a: float,
    b: float,
    theta: float,
    beta: float,
    n_cascades: int,
    max_steps: int,
    seed_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seed_candidates = np.flatnonzero(graph.labels == 0)
    if len(seed_candidates) < seed_size:
        raise ValueError(
            f"community 0 has {len(seed_candidates)} nodes, fewer than seed_size={seed_size}"
        )
    degrees = np.asarray([len(neighbors) for neighbors in graph.adjacency])
    final_sizes = np.empty(n_cascades, dtype=np.float64)

    for cascade_id in range(n_cascades):
        active = np.zeros(graph.num_nodes, dtype=bool)
        seeds = rng.choice(seed_candidates, size=seed_size, replace=False)
        active[seeds] = True
        m_in = np.zeros(graph.num_nodes, dtype=np.int64)
        m_out = np.zeros(graph.num_nodes, dtype=np.int64)
        add_active_neighbor_exposure(seeds, graph.adjacency, graph.labels, m_in, m_out)

        for _ in range(max_steps):
            susceptible = np.flatnonzero(~active)
            if not len(susceptible):
                break
            probabilities = sigmoid(
                beta
                * (
                    a * m_in[susceptible]
                    + b * m_out[susceptible]
                    - theta * degrees[susceptible]
                )
            )
            newly_active = susceptible[rng.random(len(susceptible)) < probabilities]
            if not len(newly_active):
                break
            active[newly_active] = True
            add_active_neighbor_exposure(
                newly_active, graph.adjacency, graph.labels, m_in, m_out
            )
        final_sizes[cascade_id] = np.mean(active)
    return final_sizes


def confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean and normal-approximation 95% CI across random repeats."""
    mean = float(np.mean(values))
    half_width = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, max(0.0, mean - half_width), min(1.0, mean + half_width)


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "network",
        "b_ratio",
        "b_value",
        "num_nodes",
        "num_edges",
        "n_repeats",
        "cascades_per_repeat",
        "total_cascades",
        "mean_final_cascade_size",
        "final_cascade_size_ci_low",
        "final_cascade_size_ci_high",
        "global_cascade_probability",
        "global_cascade_probability_ci_low",
        "global_cascade_probability_ci_high",
        "global_threshold",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)
    colors = {
        "friendster": "#1f77b4",
        "youtube": "#2ca02c",
        "orkut": "#ff7f0e",
    }
    available_networks = list(dict.fromkeys(str(row["network"]) for row in rows))
    for network in available_networks:
        network_rows = [row for row in rows if row["network"] == network]
        if not network_rows:
            continue
        network_rows.sort(key=lambda row: float(row["b_ratio"]))
        ratios = [float(row["b_ratio"]) for row in network_rows]
        final_sizes = [float(row["mean_final_cascade_size"]) for row in network_rows]
        final_ci_low = [
            float(row["final_cascade_size_ci_low"]) for row in network_rows
        ]
        final_ci_high = [
            float(row["final_cascade_size_ci_high"]) for row in network_rows
        ]
        global_probabilities = [
            float(row["global_cascade_probability"]) for row in network_rows
        ]
        global_ci_low = [
            float(row["global_cascade_probability_ci_low"]) for row in network_rows
        ]
        global_ci_high = [
            float(row["global_cascade_probability_ci_high"]) for row in network_rows
        ]
        color = colors.get(network)
        axes[0].plot(
            ratios,
            final_sizes,
            marker="o",
            linewidth=2,
            markersize=5,
            label=network.title(),
            color=color,
        )
        axes[1].plot(
            ratios,
            global_probabilities,
            marker="o",
            linewidth=2,
            markersize=5,
            label=network.title(),
            color=color,
        )
        axes[0].fill_between(
            ratios, final_ci_low, final_ci_high, color=color, alpha=0.16, linewidth=0
        )
        axes[1].fill_between(
            ratios, global_ci_low, global_ci_high, color=color, alpha=0.16, linewidth=0
        )

    axes[0].set_ylabel("Mean final cascade size\n(active fraction)")
    axes[1].set_ylabel("Global cascade probability")
    axes[1].set_xlabel(r"$b / b_{\mathrm{baseline}}$")
    axes[0].set_title(
        "Cross-network response to inter-community coupling intervention\n"
        rf"$a={args.a:g}$, $b_{{baseline}}={args.b_baseline:g}$, "
        rf"$\theta={args.theta:g}$, $\beta={args.beta:g}$"
    )
    axes[1].set_title(
        f"Global cascade: final active fraction ≥ {args.global_threshold:.0%}; "
        f"shading = 95% CI across {args.n_repeats} repeats"
    )
    for axis in axes:
        axis.axvline(1.0, color="0.35", linestyle="--", linewidth=1, alpha=0.8)
        axis.set_ylim(-0.025, 1.025)
        axis.grid(alpha=0.25)
    if available_networks:
        axes[0].legend(ncol=min(3, len(available_networks)), frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_orkut_main(
    path: Path, rows: list[dict[str, object]], args: argparse.Namespace
) -> None:
    orkut_rows = [row for row in rows if row["network"] == "orkut"]
    if not orkut_rows:
        return
    orkut_rows.sort(key=lambda row: float(row["b_ratio"]))
    ratios = np.asarray([float(row["b_ratio"]) for row in orkut_rows])
    probabilities = np.asarray(
        [float(row["global_cascade_probability"]) for row in orkut_rows]
    )
    ci_low = np.asarray(
        [float(row["global_cascade_probability_ci_low"]) for row in orkut_rows]
    )
    ci_high = np.asarray(
        [float(row["global_cascade_probability_ci_high"]) for row in orkut_rows]
    )

    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    color = "#d95f02"
    axis.fill_between(ratios, ci_low, ci_high, color=color, alpha=0.18, linewidth=0)
    axis.plot(
        ratios,
        probabilities,
        color=color,
        linewidth=2.6,
        marker="o",
        markersize=6,
        label="Mean across repeats",
    )
    axis.axvline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    axis.text(1.0, 0.035, "baseline", ha="center", va="bottom", color="0.3")
    axis.set_xlabel(r"Intervention strength  $b / b_{\mathrm{baseline}}$")
    axis.set_ylabel("Global cascade probability")
    axis.set_title("Orkut: inter-community coupling controls global diffusion")
    axis.set_ylim(-0.025, 1.025)
    axis.set_xlim(min(ratios) - 0.04, max(ratios) + 0.04)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="upper left")
    axis.text(
        0.99,
        0.03,
        f"95% CI across {args.n_repeats} repeats\n"
        f"{args.n_cascades} cascades per repeat",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
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
        except Exception as exc:
            print(f"Skipping {network}: {exc}")
            continue
        for ratio_index, ratio in enumerate(args.b_ratios):
            repeat_final_means = np.empty(args.n_repeats, dtype=np.float64)
            repeat_global_probabilities = np.empty(args.n_repeats, dtype=np.float64)
            for repeat_index in range(args.n_repeats):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            args.random_seed,
                            network_index,
                            ratio_index,
                            repeat_index,
                            91,
                        ]
                    )
                )
                final_sizes = simulate_final_sizes(
                    graph,
                    a=args.a,
                    b=args.b_baseline * ratio,
                    theta=args.theta,
                    beta=args.beta,
                    n_cascades=args.n_cascades,
                    max_steps=args.max_steps,
                    seed_size=args.seed_size,
                    rng=rng,
                )
                repeat_final_means[repeat_index] = np.mean(final_sizes)
                repeat_global_probabilities[repeat_index] = np.mean(
                    final_sizes >= args.global_threshold
                )
            final_mean, final_ci_low, final_ci_high = confidence_interval(
                repeat_final_means
            )
            global_mean, global_ci_low, global_ci_high = confidence_interval(
                repeat_global_probabilities
            )
            rows.append(
                {
                    "network": network,
                    "b_ratio": ratio,
                    "b_value": args.b_baseline * ratio,
                    "num_nodes": graph.num_nodes,
                    "num_edges": graph.num_edges,
                    "n_repeats": args.n_repeats,
                    "cascades_per_repeat": args.n_cascades,
                    "total_cascades": args.n_repeats * args.n_cascades,
                    "mean_final_cascade_size": final_mean,
                    "final_cascade_size_ci_low": final_ci_low,
                    "final_cascade_size_ci_high": final_ci_high,
                    "global_cascade_probability": global_mean,
                    "global_cascade_probability_ci_low": global_ci_low,
                    "global_cascade_probability_ci_high": global_ci_high,
                    "global_threshold": args.global_threshold,
                }
            )
            print(
                f"{network}: b/b0={ratio:g}, mean final={final_mean:.4f} "
                f"[{final_ci_low:.4f}, {final_ci_high:.4f}], "
                f"P(global)={global_mean:.3f} [{global_ci_low:.3f}, {global_ci_high:.3f}]"
            )

    csv_path = outputs_dir / "unified_b_intervention.csv"
    plot_path = outputs_dir / "unified_b_intervention.png"
    orkut_plot_path = outputs_dir / "orkut_intervention_global_probability.png"
    write_results(csv_path, rows)
    plot_results(plot_path, rows, args)
    plot_orkut_main(orkut_plot_path, rows, args)
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    if orkut_plot_path.exists():
        print(f"Orkut main plot: {orkut_plot_path}")


if __name__ == "__main__":
    main()
