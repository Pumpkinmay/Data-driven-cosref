import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recommendation_core import (  # noqa: E402
    DeepFM,
    _rank_metrics,
    generate_candidates,
    prepare_data,
    verify_candidates,
)


def tiny_data(tmp_path: Path):
    rows = []
    # Six interactions per user guarantee train/validation/test rows.  The
    # deliberately tied first timestamps test the story-id tie break.
    for user in range(1, 6):
        for offset in range(6):
            item = 1 + ((offset + user) % 6)
            rows.append((100 + (0 if offset < 2 else offset), user, item))
    # Add 150 globally available train stories so 99 temporal negatives exist.
    for user in range(100, 135):
        for offset in range(5):
            rows.append((1 + offset, user, 1000 + user * 5 + offset))
    votes = pd.DataFrame(rows, columns=["vote_date", "voter_id", "story_id"])
    vote_path = tmp_path / "votes.csv.gz"
    votes.to_csv(vote_path, index=False, compression="gzip")
    communities = pd.DataFrame({"node_id": votes.voter_id.unique(), "community": 1})
    community_path = tmp_path / "communities.csv"
    communities.to_csv(community_path, index=False)
    return prepare_data(vote_path, community_path)


def test_leave_last_two_out_and_temporal_candidates(tmp_path):
    data = tiny_data(tmp_path)
    assert len(data.validation) == len(data.user_to_index)
    assert len(data.test) == len(data.user_to_index)
    target_events = data.validation[data.validation.user_id < 10]
    candidates = generate_candidates(data, target_events, "validation", seed=42)
    verify_candidates(data, candidates)
    repeated = generate_candidates(data, target_events, "validation", seed=42)
    assert candidates.checksum == repeated.checksum
    assert np.array_equal(candidates.candidate_items, repeated.candidate_items)
    for timestamp, row in zip(candidates.timestamps, candidates.candidate_items):
        assert all(data.item_first_time[int(item)] < timestamp for item in row[1:])


def test_ranking_metrics_known_ranks():
    scores = np.array([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]])
    items = np.array([[1, 2, 3], [4, 5, 6]])
    metrics, per_user = _rank_metrics(scores, items)
    assert metrics["Recall@10"] == 1.0
    assert metrics["MRR"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert per_user.loc[0, "AUC"] == 1.0
    assert per_user.loc[1, "AUC"] == 0.0


def test_deepfm_contains_linear_fm_and_dnn_and_runs_forward():
    model = DeepFM(cardinalities=[4, 7, 3], numeric_count=4, embedding_dim=4, hidden=[8, 4], dropout=0.0)
    categorical = torch.tensor([[0, 1, 1], [2, 5, 2]], dtype=torch.long)
    numeric = torch.zeros((2, 4), dtype=torch.float32)
    output = model(categorical, numeric)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
