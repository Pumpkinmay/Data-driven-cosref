#!/usr/bin/env python3
"""Leakage-aware utilities and models for the Digg recommendation experiment.

This module is deliberately independent of the diffusion pipeline.  It treats a
vote as implicit feedback and an unobserved user--story pair as a sampled
non-interaction, never as an impression or an explicit negative.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PreparedData:
    interactions: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    user_to_index: dict[int, int]
    item_to_index: dict[int, int]
    index_to_item: np.ndarray
    item_first_time: dict[int, int]
    train_item_times: dict[int, np.ndarray]
    user_item_time: dict[tuple[int, int], int]
    user_times: dict[int, np.ndarray]
    community_index: np.ndarray
    community_to_index: dict[int, int]
    minimum_time: int


@dataclass
class CandidateSet:
    split: str
    user_ids: np.ndarray
    timestamps: np.ndarray
    positive_items: np.ndarray
    candidate_items: np.ndarray
    checksum: str
    source_events: int
    excluded_insufficient_pool: int

    @property
    def size(self) -> int:
        return len(self.user_ids)


@dataclass(frozen=True)
class NumericScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


def prepare_data(votes_path: Path, communities_path: Path, min_interactions: int = 5) -> PreparedData:
    votes = pd.read_csv(votes_path, usecols=["vote_date", "voter_id", "story_id"])
    if votes.empty:
        raise ValueError("Vote table is empty")
    if votes.duplicated(["voter_id", "story_id"]).any():
        raise ValueError("Expected cleaned votes with unique voter_id/story_id pairs")
    votes = votes.rename(columns={"vote_date": "timestamp", "voter_id": "user_id", "story_id": "item_id"})
    votes["_input_order"] = np.arange(len(votes), dtype=np.int64)
    # Catalog availability is a global property: a story can have appeared via
    # a user who is later excluded by the >=5-interaction evaluation filter.
    all_item_first_time = votes.groupby("item_id", sort=False)["timestamp"].min().astype(int).to_dict()
    all_minimum_time = int(votes["timestamp"].min())
    counts = votes.groupby("user_id", sort=False).size()
    eligible = counts.index[counts >= min_interactions]
    votes = votes[votes["user_id"].isin(eligible)].copy()
    # story_id is the explicit deterministic tie-breaker; input order is final fallback.
    votes.sort_values(["user_id", "timestamp", "item_id", "_input_order"], kind="mergesort", inplace=True)
    votes["_position"] = votes.groupby("user_id", sort=False).cumcount()
    votes["_count"] = votes.groupby("user_id", sort=False)["item_id"].transform("size")
    train = votes[votes["_position"] < votes["_count"] - 2].copy()
    validation = votes[votes["_position"] == votes["_count"] - 2].copy()
    test = votes[votes["_position"] == votes["_count"] - 1].copy()

    train_users = np.sort(train["user_id"].unique())
    train_items = np.sort(train["item_id"].unique())
    user_to_index = {int(value): idx for idx, value in enumerate(train_users)}
    item_to_index = {int(value): idx for idx, value in enumerate(train_items)}

    communities = pd.read_csv(communities_path, usecols=["node_id", "community"])
    community_by_user = dict(zip(communities["node_id"].astype(int), communities["community"].astype(int)))
    observed_communities = sorted({community_by_user[u] for u in user_to_index if u in community_by_user})
    # Zero is an explicit unknown category; known labels start at one.
    community_to_index = {value: idx + 1 for idx, value in enumerate(observed_communities)}
    community_index = np.zeros(len(user_to_index), dtype=np.int64)
    for user_id, user_index in user_to_index.items():
        label = community_by_user.get(user_id)
        if label is not None:
            community_index[user_index] = community_to_index[label]

    train_item_times = {
        int(item): np.sort(group["timestamp"].to_numpy(np.int64))
        for item, group in train.groupby("item_id", sort=False)
    }
    user_times = {
        int(user): np.sort(group["timestamp"].to_numpy(np.int64))
        for user, group in votes.groupby("user_id", sort=False)
    }
    user_item_time = {
        (int(user), int(item)): int(timestamp)
        for user, item, timestamp in votes[["user_id", "item_id", "timestamp"]].itertuples(index=False, name=None)
    }
    return PreparedData(
        interactions=votes,
        train=train,
        validation=validation,
        test=test,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_item=train_items.astype(np.int64),
        item_first_time={int(k): int(v) for k, v in all_item_first_time.items()},
        train_item_times=train_item_times,
        user_item_time=user_item_time,
        user_times=user_times,
        community_index=community_index,
        community_to_index=community_to_index,
        minimum_time=all_minimum_time,
    )


def warm_events(frame: pd.DataFrame, data: PreparedData) -> pd.DataFrame:
    return frame[frame["item_id"].isin(data.item_to_index)].copy()


def _event_seed(seed: int, split: str, user_id: int, timestamp: int, item_id: int) -> int:
    raw = f"{seed}|{split}|{user_id}|{timestamp}|{item_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _prior_user_items(data: PreparedData, user_id: int, timestamp: int) -> set[int]:
    group = data.interactions[data.interactions["user_id"] == user_id]
    return set(group.loc[group["timestamp"] < timestamp, "item_id"].astype(int))


def generate_candidates(
    data: PreparedData,
    events: pd.DataFrame,
    split: str,
    negatives: int = 99,
    seed: int = SEED,
) -> CandidateSet:
    events = warm_events(events, data).sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    ordered_items = np.array(
        sorted(data.item_to_index, key=lambda item: (data.item_first_time[item], item)), dtype=np.int64
    )
    ordered_first = np.array([data.item_first_time[int(item)] for item in ordered_items], dtype=np.int64)
    rows: list[np.ndarray] = []
    users: list[int] = []
    times: list[int] = []
    positives: list[int] = []
    digest = hashlib.sha256()
    # Cache each user's interactions once; strict timestamp comparisons preserve tie safety.
    histories = {
        int(user): group[["timestamp", "item_id"]].to_numpy(np.int64)
        for user, group in data.interactions.groupby("user_id", sort=False)
    }
    for user_id, timestamp, positive in events[["user_id", "timestamp", "item_id"]].itertuples(index=False, name=None):
        user_id, timestamp, positive = int(user_id), int(timestamp), int(positive)
        cutoff = int(np.searchsorted(ordered_first, timestamp, side="left"))
        appeared = ordered_items[:cutoff]
        history = histories[user_id]
        prior = set(history[history[:, 0] < timestamp, 1].tolist())
        pool = np.fromiter(
            (int(item) for item in appeared if int(item) not in prior and int(item) != positive),
            dtype=np.int64,
        )
        if len(pool) < negatives:
            # A 100-item sampled ranking task is undefined for these very early
            # targets.  Do not use replacement or future stories to pad it.
            continue
        rng = np.random.default_rng(_event_seed(seed, split, user_id, timestamp, positive))
        sampled = np.sort(rng.choice(pool, size=negatives, replace=False))
        candidates = np.concatenate((np.array([positive], dtype=np.int64), sampled))
        # Positive stays at column zero; ranking uses deterministic item-id tie breaks.
        rows.append(candidates)
        users.append(user_id)
        times.append(timestamp)
        positives.append(positive)
        digest.update(np.asarray([user_id, timestamp, positive], dtype="<i8").tobytes())
        digest.update(candidates.astype("<i8", copy=False).tobytes())
    if not rows:
        raise ValueError(f"No warm-start {split} events available")
    return CandidateSet(
        split=split,
        user_ids=np.asarray(users, dtype=np.int64),
        timestamps=np.asarray(times, dtype=np.int64),
        positive_items=np.asarray(positives, dtype=np.int64),
        candidate_items=np.stack(rows),
        checksum=digest.hexdigest(),
        source_events=len(events),
        excluded_insufficient_pool=len(events) - len(rows),
    )


def verify_candidates(data: PreparedData, candidates: CandidateSet, negatives: int = 99) -> None:
    if candidates.candidate_items.shape != (candidates.size, negatives + 1):
        raise AssertionError("Candidate matrix shape is incorrect")
    for user, timestamp, positive, items in zip(
        candidates.user_ids, candidates.timestamps, candidates.positive_items, candidates.candidate_items
    ):
        if int(items[0]) != int(positive) or len(set(items.tolist())) != negatives + 1:
            raise AssertionError("Candidate row does not contain one positive plus unique negatives")
        for item in items[1:]:
            if data.item_first_time[int(item)] >= int(timestamp):
                raise AssertionError("Candidate contains a future/unseen story")
            prior_time = data.user_item_time.get((int(user), int(item)))
            if prior_time is not None and prior_time < int(timestamp):
                raise AssertionError("Candidate contains a prior user interaction")


def popularity_before(data: PreparedData, item_id: int, timestamp: int) -> int:
    return int(np.searchsorted(data.train_item_times.get(int(item_id), np.empty(0, dtype=np.int64)), timestamp, side="left"))


def user_activity_before(data: PreparedData, user_id: int, timestamp: int) -> int:
    return int(np.searchsorted(data.user_times[int(user_id)], timestamp, side="left"))


def numeric_features(data: PreparedData, user_id: int, item_ids: np.ndarray, timestamp: int) -> np.ndarray:
    activity = math.log1p(user_activity_before(data, user_id, timestamp))
    popularity = np.array([math.log1p(popularity_before(data, int(item), timestamp)) for item in item_ids])
    hour = float((timestamp // 3600) % 24)
    relative_day = float(timestamp - data.minimum_time) / 86400.0
    return np.column_stack(
        (
            np.full(len(item_ids), activity),
            popularity,
            np.full(len(item_ids), hour),
            np.full(len(item_ids), relative_day),
        )
    ).astype(np.float32)


def fit_numeric_scaler(data: PreparedData, max_rows: int | None = None) -> NumericScaler:
    frame = data.train if max_rows is None else data.train.iloc[:max_rows]
    # `rank(method="min") - 1` is the number of rows with a strictly earlier
    # timestamp, so equal-time interactions never enter either history feature.
    activity = frame.groupby("user_id")["timestamp"].rank(method="min").to_numpy() - 1.0
    popularity = frame.groupby("item_id")["timestamp"].rank(method="min").to_numpy() - 1.0
    timestamps = frame["timestamp"].to_numpy(np.int64)
    values = np.column_stack(
        (
            np.log1p(activity),
            np.log1p(popularity),
            (timestamps // 3600) % 24,
            (timestamps - data.minimum_time) / 86400.0,
        )
    )
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return NumericScaler(mean=mean.astype(np.float32), scale=scale.astype(np.float32))


def temporal_negative(
    data: PreparedData, user_id: int, timestamp: int, rng: np.random.Generator
) -> int:
    available = [
        item for item in data.index_to_item
        if data.item_first_time[int(item)] < timestamp
        and not (
            (prior := data.user_item_time.get((user_id, int(item)))) is not None and prior < timestamp
        )
    ]
    if not available:
        raise ValueError(f"No temporal negative for user={user_id}, time={timestamp}")
    return int(available[int(rng.integers(len(available)))])


def precompute_train_negatives(data: PreparedData, seed: int = SEED, limit: int | None = None) -> np.ndarray:
    frame = data.train if limit is None else data.train.iloc[:limit]
    rng = np.random.default_rng(seed)
    output = np.empty(len(frame), dtype=np.int64)
    ordered_items = np.array(
        sorted(data.item_to_index, key=lambda item: (data.item_first_time[item], item)), dtype=np.int64
    )
    first = np.array([data.item_first_time[int(item)] for item in ordered_items], dtype=np.int64)
    for row, (user, timestamp) in enumerate(frame[["user_id", "timestamp"]].itertuples(index=False, name=None)):
        cutoff = int(np.searchsorted(first, int(timestamp), side="left"))
        if cutoff == 0:
            # The earliest story has no valid negative; mark and omit during training.
            output[row] = -1
            continue
        while True:
            item = int(ordered_items[int(rng.integers(cutoff))])
            prior = data.user_item_time.get((int(user), item))
            if prior is None or prior >= int(timestamp):
                output[row] = item
                break
    return output


def _rank_metrics(scores: np.ndarray, candidate_items: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    for row_scores, row_items in zip(scores, candidate_items):
        positive_score = float(row_scores[0])
        # Descending score, then ascending story ID gives deterministic ties.
        order = np.lexsort((row_items, -row_scores))
        rank = int(np.where(order == 0)[0][0]) + 1
        negatives = row_scores[1:]
        auc = float(
            np.mean((positive_score > negatives).astype(float) + 0.5 * (positive_score == negatives))
        )
        rows.append(
            {
                "Recall@10": float(rank <= 10),
                "Recall@20": float(rank <= 20),
                "NDCG@10": (1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0,
                "NDCG@20": (1.0 / math.log2(rank + 1)) if rank <= 20 else 0.0,
                "HitRate@10": float(rank <= 10),
                "MRR": 1.0 / rank,
                "AUC": auc,
            }
        )
    per_user = pd.DataFrame(rows)
    return {column: float(per_user[column].mean()) for column in per_user}, per_user


def bootstrap_intervals(per_user: pd.DataFrame, seed: int = SEED, repetitions: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = per_user.to_numpy(np.float64)
    means = np.empty((repetitions, values.shape[1]), dtype=np.float64)
    for repetition in range(repetitions):
        indices = rng.integers(0, len(values), size=len(values))
        means[repetition] = values[indices].mean(axis=0)
    return pd.DataFrame(
        {
            "metric": per_user.columns,
            "mean": values.mean(axis=0),
            "ci_lower": np.quantile(means, 0.025, axis=0),
            "ci_upper": np.quantile(means, 0.975, axis=0),
            "bootstrap_repetitions": repetitions,
        }
    )


class BPRMF(nn.Module):
    def __init__(self, users: int, items: int, dimension: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(users, dimension)
        self.item_embedding = nn.Embedding(items, dimension)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_embedding(users) * self.item_embedding(items)).sum(dim=-1)


class DeepFM(nn.Module):
    """Linear + second-order FM + DNN for categorical and continuous fields."""

    def __init__(
        self,
        cardinalities: Sequence[int],
        numeric_count: int,
        embedding_dim: int,
        hidden: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, embedding_dim) for size in cardinalities])
        self.linear_embeddings = nn.ModuleList([nn.Embedding(size, 1) for size in cardinalities])
        self.numeric_linear = nn.Linear(numeric_count, 1, bias=False) if numeric_count else None
        input_dim = len(cardinalities) * embedding_dim + numeric_count
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.dnn = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embedded = torch.stack(
            [embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)], dim=1
        )
        linear = self.bias + sum(
            embedding(categorical[:, index]) for index, embedding in enumerate(self.linear_embeddings)
        )
        if self.numeric_linear is not None:
            linear = linear + self.numeric_linear(numeric)
        summed = embedded.sum(dim=1)
        fm = 0.5 * (summed.square() - embedded.square().sum(dim=1)).sum(dim=1, keepdim=True)
        deep_input = torch.cat((embedded.flatten(start_dim=1), numeric), dim=1)
        return (linear + fm + self.dnn(deep_input)).squeeze(1)


def score_popularity(data: PreparedData, candidates: CandidateSet) -> np.ndarray:
    result = np.empty(candidates.candidate_items.shape, dtype=np.float64)
    for row, (timestamp, items) in enumerate(zip(candidates.timestamps, candidates.candidate_items)):
        result[row] = [popularity_before(data, int(item), int(timestamp)) for item in items]
    return result


@torch.no_grad()
def score_bpr(model: BPRMF, data: PreparedData, candidates: CandidateSet, device: torch.device) -> np.ndarray:
    model.eval()
    shape = candidates.candidate_items.shape
    user_indices = np.repeat(
        np.array([data.user_to_index[int(user)] for user in candidates.user_ids], dtype=np.int64), shape[1]
    )
    item_indices = np.array(
        [data.item_to_index[int(item)] for item in candidates.candidate_items.ravel()], dtype=np.int64
    )
    flat = np.empty(len(item_indices), dtype=np.float64)
    for start in range(0, len(flat), 65536):
        stop = min(start + 65536, len(flat))
        flat[start:stop] = model.score(
            torch.tensor(user_indices[start:stop], dtype=torch.long, device=device),
            torch.tensor(item_indices[start:stop], dtype=torch.long, device=device),
        ).cpu().numpy()
    return flat.reshape(shape)


def categorical_features(
    data: PreparedData, user_id: int, item_ids: np.ndarray, include_community: bool
) -> np.ndarray:
    user_index = data.user_to_index[int(user_id)]
    columns = [
        np.full(len(item_ids), user_index, dtype=np.int64),
        np.array([data.item_to_index[int(item)] for item in item_ids], dtype=np.int64),
    ]
    if include_community:
        columns.append(np.full(len(item_ids), data.community_index[user_index], dtype=np.int64))
    return np.column_stack(columns)


@torch.no_grad()
def score_deepfm(
    model: DeepFM,
    data: PreparedData,
    candidates: CandidateSet,
    scaler: NumericScaler,
    variant: str,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    include_context = variant != "DeepFM_ID"
    include_community = variant == "DeepFM_full"
    output = np.empty(candidates.candidate_items.shape, dtype=np.float64)
    for start in range(0, candidates.size, 512):
        stop = min(start + 512, candidates.size)
        cat_parts, num_parts = [], []
        for user, timestamp, items in zip(
            candidates.user_ids[start:stop],
            candidates.timestamps[start:stop],
            candidates.candidate_items[start:stop],
        ):
            cat_parts.append(categorical_features(data, int(user), items, include_community))
            if include_context:
                num_parts.append(scaler.transform(numeric_features(data, int(user), items, int(timestamp))))
        cat = np.concatenate(cat_parts)
        num = np.concatenate(num_parts) if include_context else np.empty((len(cat), 0), dtype=np.float32)
        logits = model(
            torch.tensor(cat, dtype=torch.long, device=device),
            torch.tensor(num, dtype=torch.float32, device=device),
        )
        output[start:stop] = logits.cpu().numpy().reshape(stop - start, -1)
    return output


def evaluate_scores(scores: np.ndarray, candidates: CandidateSet) -> tuple[dict[str, float], pd.DataFrame]:
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction score")
    return _rank_metrics(scores, candidates.candidate_items)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
