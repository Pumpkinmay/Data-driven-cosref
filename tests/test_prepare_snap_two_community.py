import csv
import json

from scripts.prepare_snap_two_community import prepare_two_community


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_filter_remap_deduplicate_and_deterministic_output(tmp_path):
    edges = tmp_path / "raw_edges.txt"
    communities = tmp_path / "raw_communities.csv"
    edges.write_text(
        "source target\n"
        "20 10\n"
        "10 20\n"
        "20 20\n"
        "20 30\n"
        "30 40\n"
        "40 30\n"
        "40 50\n",
        encoding="utf-8",
    )
    communities.write_text(
        "node_id,community_id\n"
        "40,9\n"
        "10,7\n"
        "50,12\n"
        "30,9\n"
        "20,7\n"
        "20,7\n",
        encoding="utf-8",
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = prepare_two_community(edges, communities, 7, 9, "toy", first)
    prepare_two_community(edges, communities, 7, 9, "toy", second)

    assert read_csv(first / "community_toy_remapped.csv") == [
        ["id", "community"],
        ["1", "0"],
        ["2", "0"],
        ["3", "1"],
        ["4", "1"],
    ]
    assert read_csv(first / "edges_toy_remapped.csv") == [
        ["source", "target"],
        ["1", "2"],
        ["2", "3"],
        ["3", "4"],
    ]
    assert manifest["statistics"] == {
        "nodes": 4,
        "edges": 3,
        "community_sizes": {"0": 2, "1": 2},
    }
    recorded = json.loads((first / "toy_two_community_manifest.json").read_text())
    assert recorded["parameters"]["community_a"] == 7
    assert recorded["parameters"]["community_b"] == 9
    for filename in (
        "edges_toy_remapped.csv",
        "community_toy_remapped.csv",
        "toy_two_community_manifest.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
