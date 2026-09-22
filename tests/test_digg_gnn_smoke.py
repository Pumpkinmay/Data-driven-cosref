import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch", reason="GraphSAGE smoke test requires PyTorch")
pytest.importorskip("torch_geometric", reason="GraphSAGE smoke test requires PyTorch Geometric")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_digg_gnn import TemporalGraph, TemporalGraphSAGE  # noqa: E402


def test_graphsage_forward_on_tiny_graph():
    torch.manual_seed(42)
    model = TemporalGraphSAGE(
        num_communities=2,
        numeric_features=4,
        context_features=2,
        community_dim=3,
        hidden=8,
        dropout=0.0,
    )
    numeric_x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.5, 1.0],
            [0.5, 2.0, 1.0, 0.0],
            [1.5, 1.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    community = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([2, 3], dtype=torch.long)
    context = torch.tensor([[0.0, 0.5], [1.0, 1.5]], dtype=torch.float32)

    logits = model(numeric_x, community, edge_index, targets, context)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_temporal_sampler_excludes_current_and_future_edges():
    graph = TemporalGraph(
        node_ids=[10, 20, 30, 40],
        node_to_index={10: 0, 20: 1, 30: 2, 40: 3},
        community_index=torch.tensor([0, 0, 1, 1]).numpy(),
        incoming_sources=[[], [0, 2, 3], [], []],
        incoming_dates=[[], [90, 100, 110], [], []],
        unknown_date_edges_excluded=0,
    )

    nodes, edge_index, targets = graph.sampled_subgraph(
        target_indices=[1],
        timestamp=100,
        fanouts=(10, 10),
        rng=__import__("random").Random(42),
    )

    local_edges = {
        (nodes[int(source)], nodes[int(target)])
        for source, target in edge_index.t().tolist()
    }
    assert local_edges == {(0, 1)}
    assert nodes[int(targets[0])] == 1
