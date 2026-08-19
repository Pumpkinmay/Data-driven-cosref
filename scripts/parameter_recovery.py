#!/usr/bin/env python3
"""Recover one normalized-threshold parameter setting on one network."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cosref_core import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUTS_DIR,
    fit_parameters,
    load_network,
    make_demo_graph,
    simulate_cascades,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="friendster")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--a", type=float, default=0.8)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--n-cascades", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed-size", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--demo", action="store_true", help="Use a built-in small graph")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir.expanduser().resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    graph = (
        make_demo_graph(args.random_seed)
        if args.demo
        else load_network(
            args.network,
            args.data_dir.expanduser().resolve(),
            args.max_nodes,
            np.random.default_rng(args.random_seed),
        )
    )
    features, outcomes, _ = simulate_cascades(
        graph,
        a=args.a,
        b=args.b,
        theta=args.theta,
        beta=args.beta,
        n_cascades=args.n_cascades,
        max_steps=args.max_steps,
        seed_size=args.seed_size,
        rng=np.random.default_rng(args.random_seed),
    )
    fit = fit_parameters(features, outcomes, args.beta)
    truths = {"a": args.a, "b": args.b, "theta": args.theta}
    estimates = {"a": fit.a_hat, "b": fit.b_hat, "theta": fit.theta_hat}
    row: dict[str, object] = {
        "network": graph.name,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "a_true": args.a,
        "b_true": args.b,
        "theta_true": args.theta,
        "beta": args.beta,
        "a_hat": fit.a_hat,
        "b_hat": fit.b_hat,
        "theta_hat": fit.theta_hat,
        "num_exposure_rows": len(outcomes),
        "positive_y_rate": float(np.mean(outcomes)),
    }
    for name in truths:
        error = abs(estimates[name] - truths[name])
        row[f"{name}_abs_error"] = error
        row[f"{name}_rel_error"] = error / abs(truths[name])
    fields = list(row)
    write_rows(outputs_dir / "parameter_recovery.csv", fields, [row])

    x = np.arange(3)
    fig, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.bar(x - 0.18, list(truths.values()), 0.36, label="True")
    axis.bar(x + 0.18, list(estimates.values()), 0.36, label="Estimated")
    axis.set_xticks(x, ["a", "b", "theta"])
    axis.set_ylabel("Parameter value")
    axis.set_title(f"Parameter recovery on {graph.name.title()}")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outputs_dir / "parameter_recovery.png", dpi=180)
    plt.close(fig)
    print(
        f"{graph.name}: a_hat={fit.a_hat:.6f}, b_hat={fit.b_hat:.6f}, "
        f"theta_hat={fit.theta_hat:.6f}; rows={len(outcomes)}"
    )


if __name__ == "__main__":
    main()
