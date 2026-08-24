#!/usr/bin/env python3
"""Deterministically build a two-community induced SNAP subgraph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_rows(path: Path, kind: str) -> Iterator[tuple[int, int]]:
    """Read a two-column comma- or whitespace-delimited file with an optional header."""
    header_words = {
        "edge": {"source", "src", "from", "node1", "target", "dst", "to", "node2"},
        "community": {"id", "node", "node_id", "community", "community_id"},
    }[kind]
    header_seen = False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            fields = [value.strip() for value in line.split(",")] if "," in line else line.split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected at least two columns")
            try:
                yield int(fields[0]), int(fields[1])
            except ValueError as exc:
                normalized = {fields[0].lower(), fields[1].lower()}
                if not header_seen and normalized <= header_words:
                    header_seen = True
                    continue
                raise ValueError(
                    f"{path}:{line_number}: first two columns must be integer IDs"
                ) from exc


def prepare_two_community(
    edge_list: Path,
    community_mapping: Path,
    community_a: int,
    community_b: int,
    network: str,
    output_dir: Path,
) -> dict[str, object]:
    if community_a == community_b:
        raise ValueError("The two community IDs must be different")
    if not network or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in network):
        raise ValueError("--network may contain only letters, digits, '-' and '_'")
    for path in (edge_list, community_mapping):
        if not path.is_file():
            raise FileNotFoundError(path)

    selected: dict[int, int] = {}
    selected_source = {community_a: 0, community_b: 1}
    for node_id, community_id in pair_rows(community_mapping, "community"):
        if community_id not in selected_source:
            continue
        output_label = selected_source[community_id]
        previous = selected.get(node_id)
        if previous is not None and previous != output_label:
            raise ValueError(
                f"Node {node_id} belongs to both selected communities; assignment is ambiguous"
            )
        selected[node_id] = output_label
    if not selected:
        raise ValueError("Neither selected community contains any mapped nodes")
    if set(selected.values()) != {0, 1}:
        raise ValueError("Both selected communities must contain at least one node")

    original_nodes = sorted(selected)
    remap = {node_id: index for index, node_id in enumerate(original_nodes, start=1)}
    induced_edges: set[tuple[int, int]] = set()
    for source, target in pair_rows(edge_list, "edge"):
        if source == target or source not in selected or target not in selected:
            continue
        remapped_source = remap[source]
        remapped_target = remap[target]
        induced_edges.add(
            (min(remapped_source, remapped_target), max(remapped_source, remapped_target))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    edge_output = output_dir / f"edges_{network}_remapped.csv"
    community_output = output_dir / f"community_{network}_remapped.csv"
    manifest_output = output_dir / f"{network}_two_community_manifest.json"

    with edge_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["source", "target"])
        writer.writerows(sorted(induced_edges))
    with community_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "community"])
        writer.writerows((remap[node_id], selected[node_id]) for node_id in original_nodes)

    community_sizes = {
        "0": sum(label == 0 for label in selected.values()),
        "1": sum(label == 1 for label in selected.values()),
    }
    manifest: dict[str, object] = {
        "format_version": 1,
        "network": network,
        "inputs": {
            "edge_list": {"filename": edge_list.name, "sha256": sha256_file(edge_list)},
            "community_mapping": {
                "filename": community_mapping.name,
                "sha256": sha256_file(community_mapping),
            },
        },
        "parameters": {
            "community_a": community_a,
            "community_b": community_b,
            "community_a_output_label": 0,
            "community_b_output_label": 1,
            "network": network,
            "output_dir": ".",
        },
        "statistics": {
            "nodes": len(original_nodes),
            "edges": len(induced_edges),
            "community_sizes": community_sizes,
        },
        "outputs": {
            "edges": {"filename": edge_output.name, "sha256": sha256_file(edge_output)},
            "communities": {
                "filename": community_output.name,
                "sha256": sha256_file(community_output),
            },
        },
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--community-mapping", type=Path, required=True)
    parser.add_argument("--community-a", type=int, required=True)
    parser.add_argument("--community-b", type=int, required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_two_community(
        args.edge_list,
        args.community_mapping,
        args.community_a,
        args.community_b,
        args.network,
        args.output_dir,
    )
    statistics = manifest["statistics"]
    print(
        f"Prepared {args.network}: {statistics['nodes']} nodes, "
        f"{statistics['edges']} undirected edges; outputs in {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
