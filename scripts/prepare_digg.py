#!/usr/bin/env python3
"""Prepare leakage-safe Digg 2009 cascades and baseline network data.

This stage cleans and sorts the raw files, builds the pre-cascade friendship
network, detects Leiden communities, and reports cascade eligibility.  It does
not construct exposure rows or fit a statistical model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import random
import sqlite3
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "digg2009"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "digg"
VOTES_FILENAME = "digg_votes.csv"
FRIENDS_FILENAME = "digg_friends.csv"
DEFAULT_SEED = 20240821
ELIGIBILITY_THRESHOLD = 20


class PreparationError(RuntimeError):
    """Raised when input data violate the expected Digg schema."""


@dataclass
class VoteImportStats:
    input_rows: int
    unique_rows: int
    duplicates_removed: int
    earliest_vote_date: int


@dataclass
class FriendImportStats:
    input_rows: int
    rows_after_self_edge_removal: int
    unique_rows: int
    self_edges_removed: int
    exact_duplicates_removed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def utc_string(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def strict_int_row(row: list[str], columns: int, row_number: int, source: Path) -> list[int]:
    if len(row) != columns:
        raise PreparationError(
            f"{source}: row {row_number} has {len(row)} columns; expected {columns}"
        )
    try:
        return [int(value) for value in row]
    except ValueError as error:
        raise PreparationError(f"{source}: row {row_number} contains a non-integer") from error


def configure_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")


def import_votes(connection: sqlite3.Connection, path: Path) -> VoteImportStats:
    connection.execute(
        """
        CREATE TABLE votes (
            story_id INTEGER NOT NULL,
            voter_id INTEGER NOT NULL,
            vote_date INTEGER NOT NULL,
            PRIMARY KEY (story_id, voter_id)
        ) WITHOUT ROWID
        """
    )
    statement = """
        INSERT INTO votes (story_id, voter_id, vote_date) VALUES (?, ?, ?)
        ON CONFLICT (story_id, voter_id) DO UPDATE SET
            vote_date = MIN(votes.vote_date, excluded.vote_date)
    """
    input_rows = 0
    batch: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            vote_date, voter_id, story_id = strict_int_row(row, 3, row_number, path)
            if vote_date < 0 or voter_id <= 0 or story_id <= 0:
                raise PreparationError(f"{path}: invalid timestamp or ID at row {row_number}")
            batch.append((story_id, voter_id, vote_date))
            input_rows += 1
            if len(batch) >= 50_000:
                connection.executemany(statement, batch)
                batch.clear()
        if batch:
            connection.executemany(statement, batch)
    connection.commit()

    unique_rows, earliest_vote_date = connection.execute(
        "SELECT COUNT(*), MIN(vote_date) FROM votes"
    ).fetchone()
    if not unique_rows or earliest_vote_date is None:
        raise PreparationError(f"{path}: no valid vote rows")
    return VoteImportStats(
        input_rows=input_rows,
        unique_rows=int(unique_rows),
        duplicates_removed=input_rows - int(unique_rows),
        earliest_vote_date=int(earliest_vote_date),
    )


def import_friends(connection: sqlite3.Connection, path: Path) -> FriendImportStats:
    connection.execute(
        """
        CREATE TABLE friends (
            friend_date INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            friend_id INTEGER NOT NULL,
            mutual INTEGER NOT NULL,
            PRIMARY KEY (friend_date, user_id, friend_id, mutual)
        ) WITHOUT ROWID
        """
    )
    statement = "INSERT OR IGNORE INTO friends VALUES (?, ?, ?, ?)"
    input_rows = 0
    self_edges_removed = 0
    batch: list[tuple[int, int, int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            mutual, friend_date, user_id, friend_id = strict_int_row(row, 4, row_number, path)
            if mutual not in (0, 1) or friend_date < 0 or user_id <= 0 or friend_id <= 0:
                raise PreparationError(f"{path}: invalid flag, timestamp, or ID at row {row_number}")
            input_rows += 1
            if user_id == friend_id:
                self_edges_removed += 1
                continue
            batch.append((friend_date, user_id, friend_id, mutual))
            if len(batch) >= 50_000:
                connection.executemany(statement, batch)
                batch.clear()
        if batch:
            connection.executemany(statement, batch)
    connection.commit()

    unique_rows = int(connection.execute("SELECT COUNT(*) FROM friends").fetchone()[0])
    after_self_edges = input_rows - self_edges_removed
    return FriendImportStats(
        input_rows=input_rows,
        rows_after_self_edge_removal=after_self_edges,
        unique_rows=unique_rows,
        self_edges_removed=self_edges_removed,
        exact_duplicates_removed=after_self_edges - unique_rows,
    )


class DeterministicGzipTextWriter:
    """Text writer producing gzip files without a wall-clock mtime."""

    def __init__(self, path: Path):
        self.raw = path.open("wb")
        self.compressed = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)
        self.text = io.TextIOWrapper(self.compressed, encoding="utf-8", newline="")

    def __enter__(self) -> TextIO:
        return self.text

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.text.close()


def atomic_gzip_csv(
    destination: Path, header: list[str], rows: Iterator[tuple[int, ...]]
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with DeterministicGzipTextWriter(temporary) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_clean_votes(connection: sqlite3.Connection, path: Path) -> int:
    rows = connection.execute(
        "SELECT vote_date, voter_id, story_id FROM votes "
        "ORDER BY story_id, vote_date, voter_id"
    )
    atomic_gzip_csv(path, ["vote_date", "voter_id", "story_id"], iter(rows))

    inversions = int(
        connection.execute(
            """
            WITH ordered AS (
                SELECT story_id, vote_date,
                       LAG(vote_date) OVER (
                           PARTITION BY story_id ORDER BY vote_date, voter_id
                       ) AS previous_vote_date
                FROM votes
            )
            SELECT COUNT(*) FROM ordered
            WHERE previous_vote_date IS NOT NULL AND vote_date < previous_vote_date
            """
        ).fetchone()[0]
    )
    if inversions:
        raise PreparationError(f"clean vote order contains {inversions} timestamp inversions")
    return inversions


def export_clean_friends(connection: sqlite3.Connection, path: Path) -> None:
    rows = connection.execute(
        "SELECT friend_date, user_id, friend_id, mutual FROM friends "
        "ORDER BY user_id, friend_id, friend_date, mutual"
    )
    atomic_gzip_csv(
        path, ["friend_date", "user_id", "friend_id", "mutual"], iter(rows)
    )


def create_baseline_tables(
    connection: sqlite3.Connection, earliest_vote_date: int
) -> tuple[int, int, int, int]:
    # Raw user_id -> friend_id means user follows friend.  Information therefore
    # travels in the reverse direction, friend_id -> user_id.
    connection.execute(
        """
        CREATE TABLE baseline_influence_edges AS
        SELECT DISTINCT friend_id AS source_id, user_id AS target_id
        FROM friends
        WHERE friend_date > 0 AND friend_date <= ?
        """,
        (earliest_vote_date,),
    )
    connection.execute(
        "CREATE UNIQUE INDEX baseline_edge_key "
        "ON baseline_influence_edges (source_id, target_id)"
    )
    connection.execute(
        """
        CREATE TABLE baseline_nodes (
            node_id INTEGER PRIMARY KEY
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        "INSERT INTO baseline_nodes SELECT source_id FROM baseline_influence_edges "
        "UNION SELECT target_id FROM baseline_influence_edges"
    )
    connection.execute(
        """
        CREATE TABLE community_edges AS
        SELECT DISTINCT
            MIN(source_id, target_id) AS node_a,
            MAX(source_id, target_id) AS node_b
        FROM baseline_influence_edges
        WHERE source_id != target_id
        GROUP BY node_a, node_b
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX community_edge_key ON community_edges (node_a, node_b)"
    )
    connection.commit()
    baseline_nodes = int(connection.execute("SELECT COUNT(*) FROM baseline_nodes").fetchone()[0])
    influence_edges = int(
        connection.execute("SELECT COUNT(*) FROM baseline_influence_edges").fetchone()[0]
    )
    undirected_edges = int(connection.execute("SELECT COUNT(*) FROM community_edges").fetchone()[0])
    excluded_zero_dates = int(
        connection.execute("SELECT COUNT(*) FROM friends WHERE friend_date = 0").fetchone()[0]
    )
    return baseline_nodes, influence_edges, undirected_edges, excluded_zero_dates


def detect_communities(
    connection: sqlite3.Connection, path: Path, seed: int
) -> dict[str, object]:
    try:
        import igraph as ig
    except ImportError as error:
        raise PreparationError(
            "Leiden community detection requires igraph; install project requirements first"
        ) from error

    node_ids = [row[0] for row in connection.execute("SELECT node_id FROM baseline_nodes ORDER BY node_id")]
    index_by_node = {node_id: index for index, node_id in enumerate(node_ids)}
    edges = [
        (index_by_node[node_a], index_by_node[node_b])
        for node_a, node_b in connection.execute(
            "SELECT node_a, node_b FROM community_edges ORDER BY node_a, node_b"
        )
    ]
    graph = ig.Graph(n=len(node_ids), edges=edges, directed=False)
    ig.set_random_number_generator(random.Random(seed))
    partition = graph.community_leiden(
        objective_function="modularity", n_iterations=-1
    )
    membership = list(partition.membership)
    raw_groups: dict[int, list[int]] = {}
    for node_id, community in zip(node_ids, membership):
        raw_groups.setdefault(community, []).append(node_id)
    ordered_groups = sorted(raw_groups.values(), key=lambda group: (-len(group), min(group)))
    canonical = {
        node_id: community
        for community, group in enumerate(ordered_groups)
        for node_id in group
    }
    canonical_membership = [canonical[node_id] for node_id in node_ids]
    modularity = float(graph.modularity(canonical_membership))
    degrees = graph.degree()
    isolated_nodes = sum(degree == 0 for degree in degrees)

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["node_id", "community"])
            writer.writerows((node_id, canonical[node_id]) for node_id in node_ids)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

    sizes = [len(group) for group in ordered_groups]
    return {
        "algorithm": "Leiden",
        "objective": "modularity",
        "seed": seed,
        "igraph_version": ig.__version__,
        "community_count": len(sizes),
        "community_sizes": sizes,
        "largest_community_fraction": max(sizes) / len(node_ids) if node_ids else 0.0,
        "modularity": modularity,
        "isolated_nodes": isolated_nodes,
    }


def export_cascade_eligibility(
    connection: sqlite3.Connection, path: Path
) -> tuple[int, int]:
    query = connection.execute(
        """
        SELECT
            v.story_id,
            COUNT(*) AS cleaned_size,
            SUM(CASE WHEN n.node_id IS NOT NULL THEN 1 ELSE 0 END) AS covered,
            MAX(v.vote_date) - MIN(v.vote_date) AS duration,
            COUNT(DISTINCT v.vote_date) AS unique_timestamps
        FROM votes AS v
        LEFT JOIN baseline_nodes AS n ON n.node_id = v.voter_id
        GROUP BY v.story_id
        ORDER BY v.story_id
        """
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    cascade_count = 0
    eligible_count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "story_id",
                    "cleaned_size",
                    "network_covered_adopters",
                    "network_coverage_rate",
                    "duration",
                    "unique_timestamps",
                    "at_least_20_network_covered_adopters",
                ]
            )
            for story_id, cleaned_size, covered, duration, unique_timestamps in query:
                eligible = int(covered) >= ELIGIBILITY_THRESHOLD
                writer.writerow(
                    [
                        story_id,
                        cleaned_size,
                        covered,
                        f"{covered / cleaned_size:.8f}",
                        duration,
                        unique_timestamps,
                        int(eligible),
                    ]
                )
                cascade_count += 1
                eligible_count += int(eligible)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return cascade_count, eligible_count


def write_report(
    path: Path,
    vote_stats: VoteImportStats,
    friend_stats: FriendImportStats,
    timestamp_inversions: int,
    baseline_nodes: int,
    influence_edges: int,
    undirected_edges: int,
    excluded_zero_dates: int,
    community_stats: dict[str, object],
    cascade_count: int,
    eligible_count: int,
) -> None:
    community_sizes = community_stats["community_sizes"]
    assert isinstance(community_sizes, list)
    lines = [
        "# Digg 2009 data preparation report",
        "",
        "## Vote cleaning",
        "",
        f"- Raw vote rows: **{vote_stats.input_rows:,}**",
        f"- Cleaned vote rows: **{vote_stats.unique_rows:,}**",
        f"- Duplicate `(story_id, voter_id)` rows removed: **{vote_stats.duplicates_removed:,}**",
        "- Duplicate rule: keep the earliest `vote_date`.",
        "- Sort key: `story_id, vote_date, voter_id`.",
        f"- Timestamp inversions after sorting: **{timestamp_inversions}**.",
        "- Original `voter_id` and `story_id` values are preserved.",
        f"- Earliest vote date: `{vote_stats.earliest_vote_date}` "
        f"(`{utc_string(vote_stats.earliest_vote_date)}`).",
        "",
        "## Friendship cleaning and leakage-safe baseline",
        "",
        f"- Raw friendship rows: **{friend_stats.input_rows:,}**",
        f"- Self-edges removed: **{friend_stats.self_edges_removed:,}**",
        f"- Exact duplicate rows removed: **{friend_stats.exact_duplicates_removed:,}**",
        f"- Cleaned friendship rows: **{friend_stats.unique_rows:,}**",
        "- Raw direction: `user_id -> friend_id` means that `user_id` follows `friend_id`.",
        "- Exposure/information direction: **`friend_id -> user_id`**.",
        f"- Baseline cutoff: `0 < friend_date <= {vote_stats.earliest_vote_date}`.",
        f"- Zero-date cleaned relationships excluded from baseline: **{excluded_zero_dates:,}**",
        f"- Baseline directed influence nodes: **{baseline_nodes:,}**",
        f"- Baseline directed influence edges: **{influence_edges:,}**",
        "",
        "## Leiden communities",
        "",
        "The baseline influence network was converted to an undirected simple graph only for "
        "community detection. Exposure construction must continue to use the directed influence "
        "network.",
        "",
        f"- Undirected simple-graph edges: **{undirected_edges:,}**",
        f"- Random seed: **{community_stats['seed']}**",
        f"- igraph version: `{community_stats['igraph_version']}`",
        f"- Objective: `{community_stats['objective']}`",
        f"- Community count: **{community_stats['community_count']:,}**",
        f"- Largest-community fraction: **{community_stats['largest_community_fraction']:.4%}**",
        f"- Modularity: **{community_stats['modularity']:.8f}**",
        f"- Isolated nodes: **{community_stats['isolated_nodes']:,}**",
        "",
        "| community | nodes | share |",
        "|---:|---:|---:|",
    ]
    for community, size in enumerate(community_sizes):
        lines.append(f"| {community} | {size:,} | {size / baseline_nodes:.4%} |")
    lines.extend(
        [
            "",
            "Community labels are canonicalized by descending community size and then by the "
            "smallest original node ID. Multiple communities are retained: future `m_in` means "
            "exposure from the adopter's own community and `m_out` means exposure from all other "
            "communities.",
            "",
            "## Cascade eligibility",
            "",
            f"- Total cleaned cascades: **{cascade_count:,}**",
            f"- Cascades with at least {ELIGIBILITY_THRESHOLD} baseline-network-covered adopters: "
            f"**{eligible_count:,}**",
            f"- Recommended cascade count for the first real-parameter estimation: "
            f"**{eligible_count:,}**.",
            "",
            "> This preparation stage intentionally does **not** construct an exposure table and "
            "does **not** fit logistic regression or any model parameters.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    votes_path = raw_dir / VOTES_FILENAME
    friends_path = raw_dir / FRIENDS_FILENAME
    missing = [str(path) for path in (votes_path, friends_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required Digg input file(s) not found: " + ", ".join(missing))
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="digg-preparation-") as temporary_dir:
        database_path = Path(temporary_dir) / "preparation.sqlite3"
        with sqlite3.connect(database_path) as connection:
            configure_database(connection)
            print("Importing and deduplicating votes...")
            vote_stats = import_votes(connection, votes_path)
            print("Importing and cleaning friendships...")
            friend_stats = import_friends(connection, friends_path)
            print("Building leakage-safe baseline network...")
            baseline = create_baseline_tables(connection, vote_stats.earliest_vote_date)
            baseline_nodes, influence_edges, undirected_edges, excluded_zero_dates = baseline
            print("Exporting cleaned data...")
            timestamp_inversions = export_clean_votes(
                connection, processed_dir / "digg_votes_clean.csv.gz"
            )
            export_clean_friends(connection, processed_dir / "digg_friends_clean.csv.gz")
            print("Running Leiden community detection...")
            community_stats = detect_communities(
                connection, processed_dir / "digg_communities.csv", args.seed
            )
            print("Computing cascade eligibility...")
            cascade_count, eligible_count = export_cascade_eligibility(
                connection, output_dir / "cascade_eligibility.csv"
            )

    write_report(
        output_dir / "data_preparation_report.md",
        vote_stats,
        friend_stats,
        timestamp_inversions,
        baseline_nodes,
        influence_edges,
        undirected_edges,
        excluded_zero_dates,
        community_stats,
        cascade_count,
        eligible_count,
    )
    print(f"Prepared data: {processed_dir}")
    print(f"Report: {output_dir / 'data_preparation_report.md'}")


if __name__ == "__main__":
    try:
        main()
    except (PreparationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
