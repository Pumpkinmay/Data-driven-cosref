#!/usr/bin/env python3
"""Audit the raw Digg 2009 vote cascades and friendship network.

The expected, headerless SNAP-derived layouts are:

* digg_votes.csv:   vote_timestamp, voter_id, story_id
* digg_friends.csv: mutual, friendship_timestamp, user_id, friend_id

Only the Python standard library is required.  Files are read as streams, which
also handles the CR-only line endings used by the distributed votes file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "digg2009"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "digg"
VOTES_FILENAME = "digg_votes.csv"
FRIENDS_FILENAME = "digg_friends.csv"
MAX_ISSUE_EXAMPLES = 10


class AuditError(RuntimeError):
    """Raised when the raw data cannot be audited reliably."""


@dataclass
class CascadeAccumulator:
    story_id: int
    votes: int = 0
    duplicate_votes: int = 0
    timestamp_inversions: int = 0
    start_timestamp: int | None = None
    end_timestamp: int | None = None
    previous_timestamp: int | None = None

    def add(self, timestamp: int, duplicate: bool) -> None:
        self.votes += 1
        self.duplicate_votes += int(duplicate)
        if self.start_timestamp is None or timestamp < self.start_timestamp:
            self.start_timestamp = timestamp
        if self.end_timestamp is None or timestamp > self.end_timestamp:
            self.end_timestamp = timestamp
        if self.previous_timestamp is not None and timestamp < self.previous_timestamp:
            self.timestamp_inversions += 1
        self.previous_timestamp = timestamp

    def finish(self, unique_voters: int) -> dict[str, int | str]:
        assert self.start_timestamp is not None and self.end_timestamp is not None
        return {
            "cascade_id": self.story_id,
            "votes": self.votes,
            "unique_voters": unique_voters,
            "duplicate_votes": self.duplicate_votes,
            "start_timestamp": self.start_timestamp,
            "start_utc": utc_string(self.start_timestamp),
            "end_timestamp": self.end_timestamp,
            "end_utc": utc_string(self.end_timestamp),
            "duration_seconds": self.end_timestamp - self.start_timestamp,
            "timestamp_inversions": self.timestamp_inversions,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing {VOTES_FILENAME} and {FRIENDS_FILENAME}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for audit_summary.json, cascade_summary.csv, and audit_report.md",
    )
    return parser.parse_args()


def utc_string(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    except (OverflowError, OSError, ValueError):
        return None


def percentile(values: Iterable[int], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[int]) -> dict[str, float | int | None]:
    return {
        "min": min(values) if values else None,
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
    }


def record_issue(examples: list[dict[str, object]], row_number: int, reason: str) -> None:
    if len(examples) < MAX_ISSUE_EXAMPLES:
        examples.append({"row": row_number, "reason": reason})


def audit_votes(path: Path) -> tuple[dict[str, object], list[dict[str, int | str]], set[int]]:
    total_rows = 0
    valid_rows = 0
    malformed_rows = 0
    invalid_value_rows = 0
    issue_examples: list[dict[str, object]] = []
    voter_ids: set[int] = set()
    closed_stories: set[int] = set()
    cascade_rows: list[dict[str, int | str]] = []
    current: CascadeAccumulator | None = None
    current_voters: set[int] = set()
    minimum_timestamp: int | None = None
    maximum_timestamp: int | None = None

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            total_rows += 1
            if len(row) != 3:
                malformed_rows += 1
                record_issue(issue_examples, row_number, f"expected 3 columns, found {len(row)}")
                continue
            try:
                timestamp, voter_id, story_id = (int(value) for value in row)
            except ValueError:
                malformed_rows += 1
                record_issue(issue_examples, row_number, "non-integer value")
                continue
            if timestamp < 0 or voter_id <= 0 or story_id <= 0:
                invalid_value_rows += 1
                record_issue(issue_examples, row_number, "negative timestamp or non-positive ID")
                continue

            if current is None or story_id != current.story_id:
                if current is not None:
                    cascade_rows.append(current.finish(len(current_voters)))
                    closed_stories.add(current.story_id)
                if story_id in closed_stories:
                    raise AuditError(
                        f"Votes are not grouped by story: story {story_id} reopens at row "
                        f"{row_number}. Exact per-cascade duplicate auditing would be unsafe."
                    )
                current = CascadeAccumulator(story_id=story_id)
                current_voters = set()

            duplicate = voter_id in current_voters
            current_voters.add(voter_id)
            voter_ids.add(voter_id)
            current.add(timestamp, duplicate)
            valid_rows += 1
            minimum_timestamp = timestamp if minimum_timestamp is None else min(minimum_timestamp, timestamp)
            maximum_timestamp = timestamp if maximum_timestamp is None else max(maximum_timestamp, timestamp)

    if current is not None:
        cascade_rows.append(current.finish(len(current_voters)))

    sizes = [int(row["votes"]) for row in cascade_rows]
    unique_sizes = [int(row["unique_voters"]) for row in cascade_rows]
    durations = [int(row["duration_seconds"]) for row in cascade_rows]
    duplicate_votes = sum(int(row["duplicate_votes"]) for row in cascade_rows)
    inversions = sum(int(row["timestamp_inversions"]) for row in cascade_rows)
    summary: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "schema": ["vote_timestamp", "voter_id", "story_id"],
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "malformed_rows": malformed_rows,
        "invalid_value_rows": invalid_value_rows,
        "issue_examples": issue_examples,
        "unique_voters": len(voter_ids),
        "unique_cascades": len(cascade_rows),
        "duplicate_voter_story_rows": duplicate_votes,
        "timestamp_inversions_within_file_order": inversions,
        "timestamp_min": minimum_timestamp,
        "timestamp_min_utc": utc_string(minimum_timestamp),
        "timestamp_max": maximum_timestamp,
        "timestamp_max_utc": utc_string(maximum_timestamp),
        "cascade_vote_count": distribution(sizes),
        "cascade_unique_voter_count": distribution(unique_sizes),
        "cascade_duration_seconds": distribution(durations),
    }
    return summary, cascade_rows, voter_ids


def audit_friends(path: Path) -> tuple[dict[str, object], set[int]]:
    total_rows = 0
    valid_rows = 0
    malformed_rows = 0
    invalid_value_rows = 0
    self_edges = 0
    zero_timestamps = 0
    flag_counts: Counter[int] = Counter()
    issue_examples: list[dict[str, object]] = []
    node_ids: set[int] = set()
    minimum_positive_timestamp: int | None = None
    maximum_timestamp: int | None = None

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            total_rows += 1
            if len(row) != 4:
                malformed_rows += 1
                record_issue(issue_examples, row_number, f"expected 4 columns, found {len(row)}")
                continue
            try:
                mutual, timestamp, user_id, friend_id = (int(value) for value in row)
            except ValueError:
                malformed_rows += 1
                record_issue(issue_examples, row_number, "non-integer value")
                continue
            if mutual not in (0, 1) or timestamp < 0 or user_id <= 0 or friend_id <= 0:
                invalid_value_rows += 1
                record_issue(issue_examples, row_number, "invalid mutual flag, timestamp, or ID")
                continue

            valid_rows += 1
            flag_counts[mutual] += 1
            zero_timestamps += int(timestamp == 0)
            self_edges += int(user_id == friend_id)
            node_ids.add(user_id)
            node_ids.add(friend_id)
            if timestamp > 0:
                minimum_positive_timestamp = (
                    timestamp
                    if minimum_positive_timestamp is None
                    else min(minimum_positive_timestamp, timestamp)
                )
            maximum_timestamp = timestamp if maximum_timestamp is None else max(maximum_timestamp, timestamp)

    summary: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "schema": ["mutual", "friendship_timestamp", "user_id", "friend_id"],
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "malformed_rows": malformed_rows,
        "invalid_value_rows": invalid_value_rows,
        "issue_examples": issue_examples,
        "unique_nodes": len(node_ids),
        "mutual_flag_counts": {str(flag): flag_counts[flag] for flag in (0, 1)},
        "self_edges": self_edges,
        "zero_timestamps": zero_timestamps,
        "positive_timestamp_min": minimum_positive_timestamp,
        "positive_timestamp_min_utc": utc_string(minimum_positive_timestamp),
        "timestamp_max": maximum_timestamp,
        "timestamp_max_utc": utc_string(maximum_timestamp),
    }
    return summary, node_ids


def write_cascade_csv(path: Path, rows: list[dict[str, int | str]]) -> None:
    fields = [
        "cascade_id",
        "votes",
        "unique_voters",
        "duplicate_votes",
        "start_timestamp",
        "start_utc",
        "end_timestamp",
        "end_utc",
        "duration_seconds",
        "timestamp_inversions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, object]) -> None:
    votes = summary["votes"]
    friends = summary["friends"]
    overlap = summary["network_vote_overlap"]
    assert isinstance(votes, dict) and isinstance(friends, dict) and isinstance(overlap, dict)
    lines = [
        "# Digg 2009 data audit",
        "",
        f"Generated at `{summary['generated_at_utc']}`.",
        "",
        "## Inputs",
        "",
        f"- Votes file: `{votes['path']}` ({votes['size_bytes']:,} bytes)",
        f"- Friends file: `{friends['path']}` ({friends['size_bytes']:,} bytes)",
        "",
        "## Main counts",
        "",
        f"- Valid votes: **{votes['valid_rows']:,}**",
        f"- Cascades (stories): **{votes['unique_cascades']:,}**",
        f"- Unique voters: **{votes['unique_voters']:,}**",
        f"- Valid friendship rows: **{friends['valid_rows']:,}**",
        f"- Unique friendship-network nodes: **{friends['unique_nodes']:,}**",
        f"- Voters represented in friendship network: **{overlap['voters_in_friend_network']:,}** "
        f"({overlap['voter_coverage_fraction']:.2%})",
        "",
        "## Integrity checks",
        "",
        f"- Malformed vote rows: {votes['malformed_rows']:,}",
        f"- Invalid-value vote rows: {votes['invalid_value_rows']:,}",
        f"- Duplicate voter-story rows: {votes['duplicate_voter_story_rows']:,}",
        f"- Within-cascade timestamp inversions: {votes['timestamp_inversions_within_file_order']:,}",
        f"- Malformed friendship rows: {friends['malformed_rows']:,}",
        f"- Invalid-value friendship rows: {friends['invalid_value_rows']:,}",
        f"- Friendship self-edges: {friends['self_edges']:,}",
        f"- Friendship rows with unknown (zero) timestamp: {friends['zero_timestamps']:,}",
        "",
        "Full metrics are in `audit_summary.json`; per-cascade metrics are in "
        "`cascade_summary.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    votes_path = data_dir / VOTES_FILENAME
    friends_path = data_dir / FRIENDS_FILENAME
    missing = [str(path) for path in (votes_path, friends_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required Digg input file(s) not found: " + ", ".join(missing))

    print(f"Auditing votes:   {votes_path}")
    votes_summary, cascade_rows, voter_ids = audit_votes(votes_path)
    print(f"Auditing friends: {friends_path}")
    friends_summary, friend_nodes = audit_friends(friends_path)

    voters_in_network = len(voter_ids & friend_nodes)
    network_nodes_who_voted = voters_in_network
    overlap = {
        "voters_in_friend_network": voters_in_network,
        "voters_missing_from_friend_network": len(voter_ids) - voters_in_network,
        "voter_coverage_fraction": voters_in_network / len(voter_ids) if voter_ids else 0.0,
        "friend_network_nodes_who_voted": network_nodes_who_voted,
        "friend_network_nodes_without_votes": len(friend_nodes) - network_nodes_who_voted,
        "friend_network_node_vote_fraction": (
            network_nodes_who_voted / len(friend_nodes) if friend_nodes else 0.0
        ),
    }
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "votes": votes_summary,
        "friends": friends_summary,
        "network_vote_overlap": overlap,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cascade_path = output_dir / "cascade_summary.csv"
    json_path = output_dir / "audit_summary.json"
    report_path = output_dir / "audit_report.md"
    write_cascade_csv(cascade_path, cascade_rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {cascade_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    try:
        main()
    except (AuditError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
