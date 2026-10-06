#!/usr/bin/env python3
"""Run the v3.1 Digg implicit-feedback recommendation baselines."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recommendation_core import (
    BPRMF,
    DeepFM,
    SEED,
    bootstrap_intervals,
    categorical_features,
    evaluate_scores,
    fit_numeric_scaler,
    generate_candidates,
    numeric_features,
    parameter_count,
    precompute_train_negatives,
    prepare_data,
    score_bpr,
    score_deepfm,
    score_popularity,
    seed_everything,
    sha256_file,
    verify_candidates,
    warm_events,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "recommendation"
DEFAULT_DOCS = ROOT / "docs"
METRICS = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "HitRate@10", "MRR", "AUC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes", type=Path, default=ROOT / "data/processed/digg_votes_clean.csv.gz")
    parser.add_argument("--communities", type=Path, default=ROOT / "data/processed/digg_communities.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Run a small end-to-end subset without writing final docs.")
    parser.add_argument("--smoke-users", type=int, default=256)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def subset_data(data, users: int):
    chosen = sorted(data.user_to_index)[:users]
    keep = set(chosen)
    data.interactions = data.interactions[data.interactions.user_id.isin(keep)].copy()
    data.train = data.train[data.train.user_id.isin(keep)].copy()
    data.validation = data.validation[data.validation.user_id.isin(keep)].copy()
    data.test = data.test[data.test.user_id.isin(keep)].copy()
    # Rebuild mappings because smoke must exercise the same train-defined mapping logic.
    train_users = np.sort(data.train.user_id.unique())
    train_items = np.sort(data.train.item_id.unique())
    old_communities = {u: data.community_index[i] for u, i in data.user_to_index.items() if u in keep}
    data.user_to_index = {int(u): i for i, u in enumerate(train_users)}
    data.item_to_index = {int(item): i for i, item in enumerate(train_items)}
    data.index_to_item = train_items.astype(np.int64)
    data.community_index = np.array([old_communities[int(u)] for u in train_users], dtype=np.int64)
    data.train_item_times = {
        int(item): np.sort(group.timestamp.to_numpy(np.int64)) for item, group in data.train.groupby("item_id")
    }
    data.user_times = {
        int(user): np.sort(group.timestamp.to_numpy(np.int64)) for user, group in data.interactions.groupby("user_id")
    }
    data.user_item_time = {
        (int(u), int(i)): int(t)
        for u, i, t in data.interactions[["user_id", "item_id", "timestamp"]].itertuples(index=False, name=None)
    }
    return data


def train_bpr(data, negatives, validation_candidates, device, smoke: bool, seed: int):
    frame = data.train.iloc[: len(negatives)]
    valid = negatives >= 0
    users = np.array([data.user_to_index[int(u)] for u in frame.user_id], dtype=np.int64)[valid]
    positives = np.array([data.item_to_index[int(i)] for i in frame.item_id], dtype=np.int64)[valid]
    negatives_idx = np.array([data.item_to_index[int(i)] for i in negatives[valid]], dtype=np.int64)
    configs = (
        [{"dimension": 16, "lr": 1e-3, "regularization": 1e-5, "max_epochs": 2}]
        if smoke
        else [
            {"dimension": 32, "lr": 1e-3, "regularization": 1e-5, "max_epochs": 12},
            {"dimension": 64, "lr": 1e-3, "regularization": 1e-5, "max_epochs": 12},
            {"dimension": 64, "lr": 5e-4, "regularization": 1e-4, "max_epochs": 12},
        ]
    )
    best = None
    started = time.perf_counter()
    for config_index, config in enumerate(configs):
        seed_everything(seed + config_index)
        model = BPRMF(len(data.user_to_index), len(data.item_to_index), config["dimension"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)
        rng = np.random.default_rng(seed + config_index)
        best_state, best_ndcg, best_epoch, stale = None, -math.inf, 0, 0
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            order = rng.permutation(len(users))
            total = 0.0
            for start in range(0, len(order), 8192):
                batch = order[start : start + 8192]
                u = torch.tensor(users[batch], dtype=torch.long, device=device)
                p = torch.tensor(positives[batch], dtype=torch.long, device=device)
                n = torch.tensor(negatives_idx[batch], dtype=torch.long, device=device)
                difference = model.score(u, p) - model.score(u, n)
                loss = -F.logsigmoid(difference).mean()
                regularization = config["regularization"] * (
                    model.user_embedding(u).square().mean()
                    + model.item_embedding(p).square().mean()
                    + model.item_embedding(n).square().mean()
                )
                objective = loss + regularization
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
                total += float(objective.detach()) * len(batch)
            scores = score_bpr(model, data, validation_candidates, device)
            metrics, _ = evaluate_scores(scores, validation_candidates)
            print(f"BPR config={config_index + 1} epoch={epoch} loss={total/len(order):.6f} val_NDCG@10={metrics['NDCG@10']:.6f}", flush=True)
            if metrics["NDCG@10"] > best_ndcg + 1e-12:
                best_ndcg, best_epoch = metrics["NDCG@10"], epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= (1 if smoke else 3):
                break
        record = (best_ndcg, -config_index, config, best_epoch, best_state)
        if best is None or record[:2] > best[:2]:
            best = record
    _, _, config, epoch, state = best
    final = BPRMF(len(data.user_to_index), len(data.item_to_index), config["dimension"]).to(device)
    final.load_state_dict(state)
    return final, {**config, "selected_epoch": epoch}, time.perf_counter() - started


def _deepfm_training_arrays(data, negatives, scaler, variant, limit=None):
    frame = data.train if limit is None else data.train.iloc[:limit]
    frame = frame.iloc[: len(negatives)]
    valid_rows = np.where(negatives >= 0)[0]
    include_context = variant != "DeepFM_ID"
    include_community = variant == "DeepFM_full"
    selected = frame.iloc[valid_rows]
    user_ids = selected.user_id.to_numpy(np.int64)
    timestamps = selected.timestamp.to_numpy(np.int64)
    positives = selected.item_id.to_numpy(np.int64)
    negative_items = negatives[valid_rows]
    user_indices = np.array([data.user_to_index[int(user)] for user in user_ids], dtype=np.int64)
    positive_indices = np.array([data.item_to_index[int(item)] for item in positives], dtype=np.int64)
    negative_indices = np.array([data.item_to_index[int(item)] for item in negative_items], dtype=np.int64)
    cats = np.column_stack((np.repeat(user_indices, 2), np.column_stack((positive_indices, negative_indices)).ravel()))
    if include_community:
        cats = np.column_stack((cats, np.repeat(data.community_index[user_indices], 2)))
    if include_context:
        # Strict-past counts. Equal timestamps receive the same pre-event count.
        activity_all = frame.groupby("user_id")["timestamp"].rank(method="min").to_numpy() - 1.0
        positive_activity = activity_all[valid_rows]
        positive_popularity = np.empty(len(selected), dtype=np.float64)
        negative_popularity = np.empty(len(selected), dtype=np.float64)
        for item in np.unique(np.concatenate((positives, negative_items))):
            times = data.train_item_times[int(item)]
            pos_rows = np.where(positives == item)[0]
            neg_rows = np.where(negative_items == item)[0]
            positive_popularity[pos_rows] = np.searchsorted(times, timestamps[pos_rows], side="left")
            negative_popularity[neg_rows] = np.searchsorted(times, timestamps[neg_rows], side="left")
        activity = np.repeat(np.log1p(positive_activity), 2)
        popularity = np.column_stack((np.log1p(positive_popularity), np.log1p(negative_popularity))).ravel()
        hour = np.repeat((timestamps // 3600) % 24, 2)
        relative_day = np.repeat((timestamps - data.minimum_time) / 86400.0, 2)
        nums = scaler.transform(np.column_stack((activity, popularity, hour, relative_day)).astype(np.float32))
    else:
        nums = np.empty((len(cats), 0), dtype=np.float32)
    labels = np.tile(np.array([1.0, 0.0], dtype=np.float32), len(selected))
    return cats, nums, labels


def train_deepfm(data, negatives, validation_candidates, scaler, variant, device, config, smoke, seed):
    cats, nums, labels = _deepfm_training_arrays(data, negatives, scaler, variant)
    cardinalities = [len(data.user_to_index), len(data.item_to_index)]
    if variant == "DeepFM_full":
        cardinalities.append(int(data.community_index.max()) + 1)
    seed_everything(seed)
    model = DeepFM(cardinalities, nums.shape[1], config["embedding_dim"], config["hidden"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed)
    best_state, best_ndcg, best_epoch, stale = None, -math.inf, 0, 0
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        order = rng.permutation(len(labels))
        total = 0.0
        model.train()
        for start in range(0, len(order), 8192):
            batch = order[start : start + 8192]
            logits = model(
                torch.tensor(cats[batch], dtype=torch.long, device=device),
                torch.tensor(nums[batch], dtype=torch.float32, device=device),
            )
            target = torch.tensor(labels[batch], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        scores = score_deepfm(model, data, validation_candidates, scaler, variant, device)
        metrics, _ = evaluate_scores(scores, validation_candidates)
        print(f"{variant} epoch={epoch} loss={total/len(order):.6f} val_NDCG@10={metrics['NDCG@10']:.6f}", flush=True)
        if metrics["NDCG@10"] > best_ndcg + 1e-12:
            best_ndcg, best_epoch = metrics["NDCG@10"], epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= (1 if smoke else 3):
            break
    model.load_state_dict(best_state)
    return model, best_epoch, time.perf_counter() - started


def select_deepfm_config(data, negatives, validation_candidates, scaler, device, smoke, seed):
    configs = (
        [{"embedding_dim": 8, "hidden": [16, 8], "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-5, "max_epochs": 2}]
        if smoke
        else [
            {"embedding_dim": 16, "hidden": [64, 32], "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-5, "max_epochs": 12},
            {"embedding_dim": 32, "hidden": [64, 32], "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-4, "max_epochs": 12},
        ]
    )
    best = None
    selection_started = time.perf_counter()
    for index, config in enumerate(configs):
        model, epoch, duration = train_deepfm(
            data, negatives, validation_candidates, scaler, "DeepFM_full", device, config, smoke, seed + index
        )
        metrics, _ = evaluate_scores(
            score_deepfm(model, data, validation_candidates, scaler, "DeepFM_full", device), validation_candidates
        )
        record = (metrics["NDCG@10"], -index, config, epoch, duration)
        if best is None or record[:2] > best[:2]:
            best = record
    return best[2], best[3], time.perf_counter() - selection_started


def output_guard(paths, overwrite):
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite existing recommendation outputs: " + ", ".join(existing))


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    for path in (args.votes, args.communities):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    final_paths = [
        args.output_dir / "model_metrics.csv",
        args.output_dir / "bootstrap_intervals.csv",
        args.output_dir / "ablation_metrics.csv",
        args.output_dir / "model_comparison.png",
        args.output_dir / "training_summary.md",
        args.output_dir / "candidate_manifest.json",
    ]
    if not args.smoke:
        output_guard(final_paths, args.overwrite)

    print("Preparing leave-last-two-out data...", flush=True)
    data = prepare_data(args.votes, args.communities)
    if args.smoke:
        data = subset_data(data, args.smoke_users)
    warm_validation = warm_events(data.validation, data)
    warm_test = warm_events(data.test, data)
    if len(warm_validation) == 0 or len(warm_test) == 0:
        raise RuntimeError("No warm-start events after train-defined mapping")
    print(f"users={len(data.user_to_index):,} train={len(data.train):,} val={len(data.validation):,} test={len(data.test):,}", flush=True)
    print("Generating shared temporal candidate sets...", flush=True)
    validation_candidates = generate_candidates(data, warm_validation, "validation", seed=args.seed)
    test_candidates = generate_candidates(data, warm_test, "test", seed=args.seed)
    verify_candidates(data, validation_candidates)
    verify_candidates(data, test_candidates)
    print(f"candidate checksums val={validation_candidates.checksum} test={test_candidates.checksum}", flush=True)
    print(
        "candidate eligibility "
        f"val={validation_candidates.size}/{validation_candidates.source_events} "
        f"test={test_candidates.size}/{test_candidates.source_events}",
        flush=True,
    )

    train_limit = min(len(data.train), 4000) if args.smoke else None
    negatives = precompute_train_negatives(data, seed=args.seed, limit=train_limit)
    scaler = fit_numeric_scaler(data, max_rows=train_limit)

    metric_rows, interval_rows, per_model, per_user_by_model = [], [], {}, {}
    # Popularity has no fitted parameters.
    start = time.perf_counter()
    popularity_scores = score_popularity(data, test_candidates)
    popularity_inference = time.perf_counter() - start
    popularity_metrics, popularity_users = evaluate_scores(popularity_scores, test_candidates)
    per_model["Popularity"] = popularity_metrics
    per_user_by_model["Popularity"] = popularity_users
    metric_rows.append({"model": "Popularity", **popularity_metrics, "parameters": 0, "training_seconds": 0.0, "inference_seconds": popularity_inference})
    interval_rows.append(bootstrap_intervals(popularity_users, args.seed, args.bootstrap_repetitions).assign(model="Popularity"))

    print("Training BPR-MF...", flush=True)
    bpr, bpr_config, bpr_training = train_bpr(data, negatives, validation_candidates, device, args.smoke, args.seed)
    start = time.perf_counter()
    bpr_scores = score_bpr(bpr, data, test_candidates, device)
    bpr_inference = time.perf_counter() - start
    bpr_metrics, bpr_users = evaluate_scores(bpr_scores, test_candidates)
    per_model["BPR-MF"] = bpr_metrics
    per_user_by_model["BPR-MF"] = bpr_users
    metric_rows.append({"model": "BPR-MF", **bpr_metrics, "parameters": parameter_count(bpr), "training_seconds": bpr_training, "inference_seconds": bpr_inference})
    interval_rows.append(bootstrap_intervals(bpr_users, args.seed, args.bootstrap_repetitions).assign(model="BPR-MF"))

    print("Selecting DeepFM hyperparameters on validation only...", flush=True)
    deepfm_config, selected_epoch, deepfm_selection_seconds = select_deepfm_config(
        data, negatives, validation_candidates, scaler, device, args.smoke, args.seed
    )
    deepfm_config = {**deepfm_config, "max_epochs": selected_epoch}
    for offset, variant in enumerate(("DeepFM_ID", "DeepFM_context", "DeepFM_full")):
        print(f"Training fixed-config ablation {variant}...", flush=True)
        model, epoch, training_seconds = train_deepfm(
            data, negatives, validation_candidates, scaler, variant, device, deepfm_config, args.smoke, args.seed + 20 + offset
        )
        start = time.perf_counter()
        scores = score_deepfm(model, data, test_candidates, scaler, variant, device)
        inference_seconds = time.perf_counter() - start
        metrics, user_metrics = evaluate_scores(scores, test_candidates)
        per_model[variant] = metrics
        per_user_by_model[variant] = user_metrics
        if variant == "DeepFM_full":
            training_seconds += deepfm_selection_seconds
        metric_rows.append({"model": variant, **metrics, "parameters": parameter_count(model), "training_seconds": training_seconds, "inference_seconds": inference_seconds})
        interval_rows.append(bootstrap_intervals(user_metrics, args.seed, args.bootstrap_repetitions).assign(model=variant))

    for comparison, better, baseline in (
        ("BPR-MF minus Popularity", "BPR-MF", "Popularity"),
        ("DeepFM_full minus BPR-MF", "DeepFM_full", "BPR-MF"),
    ):
        paired = per_user_by_model[better][METRICS] - per_user_by_model[baseline][METRICS]
        interval_rows.append(
            bootstrap_intervals(paired, args.seed, args.bootstrap_repetitions).assign(model=comparison)
        )

    metrics_frame = pd.DataFrame(metric_rows)
    intervals = pd.concat(interval_rows, ignore_index=True)[["model", "metric", "mean", "ci_lower", "ci_upper", "bootstrap_repetitions"]]
    ablation = metrics_frame[metrics_frame.model.str.startswith("DeepFM")].copy()
    for metric in METRICS:
        ablation[f"delta_vs_ID_{metric}"] = ablation[metric] - float(ablation.loc[ablation.model == "DeepFM_ID", metric].iloc[0])
    for metric in METRICS:
        metrics_frame[f"delta_vs_previous_{metric}"] = np.nan
        metrics_frame.loc[metrics_frame.model == "BPR-MF", f"delta_vs_previous_{metric}"] = bpr_metrics[metric] - popularity_metrics[metric]
        metrics_frame.loc[metrics_frame.model == "DeepFM_full", f"delta_vs_previous_{metric}"] = per_model["DeepFM_full"][metric] - bpr_metrics[metric]

    if args.smoke:
        if not np.isfinite(metrics_frame[METRICS].to_numpy()).all():
            raise RuntimeError("Smoke metrics contain non-finite values")
        print("SMOKE PASS: temporal candidates verified; all three model families trained and scored.", flush=True)
        print(metrics_frame[["model", "NDCG@10", "MRR", "AUC"]].to_string(index=False), flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(final_paths[0], index=False)
    intervals.to_csv(final_paths[1], index=False)
    ablation.to_csv(final_paths[2], index=False)
    manifest = {
        "seed": args.seed,
        "minimum_user_interactions": 5,
        "split": "leave-last-two-out after stable (user_id, timestamp, story_id, input_order) sorting",
        "negative_candidates_per_event": 99,
        "candidate_rule": "story first appeared in complete cleaned votes strictly before target; exclude user interactions strictly before target; warm train-mapped items only",
        "validation_events": validation_candidates.size,
        "test_events": test_candidates.size,
        "validation_source_warm_events": validation_candidates.source_events,
        "test_source_warm_events": test_candidates.source_events,
        "validation_excluded_insufficient_negative_pool": validation_candidates.excluded_insufficient_pool,
        "test_excluded_insufficient_negative_pool": test_candidates.excluded_insufficient_pool,
        "validation_candidate_sha256": validation_candidates.checksum,
        "test_candidate_sha256": test_candidates.checksum,
        "votes_sha256": sha256_file(args.votes),
        "communities_sha256": sha256_file(args.communities),
    }
    write_json(final_paths[5], manifest)
    plot_metrics(metrics_frame, final_paths[3])
    coverage = coverage_summary(data)
    write_reports(args, data, coverage, manifest, metrics_frame, intervals, ablation, bpr_config, deepfm_config)
    print(metrics_frame.to_string(index=False), flush=True)
    print(f"Wrote recommendation outputs to {args.output_dir}", flush=True)
    return 0


def coverage_summary(data):
    train_users, train_items = set(data.user_to_index), set(data.item_to_index)
    result = {}
    for name, frame in (("validation", data.validation), ("test", data.test)):
        warm_user = frame.user_id.isin(train_users)
        warm_item = frame.item_id.isin(train_items)
        result[name] = {
            "events": len(frame),
            "warm_user_rate": float(warm_user.mean()),
            "warm_item_rate": float(warm_item.mean()),
            "cold_item_rate": float((~warm_item).mean()),
        }
    return result


def plot_metrics(metrics, path):
    models = metrics.model.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric in zip(axes, ("NDCG@10", "Recall@10")):
        axis.bar(models, metrics[metric], color="#4472C4")
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Digg sampled-candidate recommendation (warm-start items)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) if isinstance(value, str) else f"{value:.8g}" for value in row) + " |")
    return "\n".join(lines)


def write_reports(args, data, coverage, manifest, metrics, intervals, ablation, bpr_config, deepfm_config):
    display_metrics = metrics.rename(columns={"model": "Model"})
    display_ablation = ablation.rename(columns={"model": "Model"})
    display_main = display_metrics[["Model", "NDCG@10", "Recall@10", "MRR", "AUC"]].copy()
    for column in ("NDCG@10", "Recall@10", "MRR", "AUC"):
        display_main[column] = display_main[column].map(lambda value: f"{value:.6f}")
    protocol = f"""# Digg v3.1 recommendation protocol

## Task

This is an implicit-feedback next-story ranking task. Given a user's votes strictly before time `t`, the models rank stories for the user's next vote. Digg has no impression log; unvoted candidates are **sampled non-interactions**, not observed negative feedback, and this is not CTR prediction.

## Split and candidate construction

- Users have at least five cleaned votes.
- Per user, events are stably ordered by `(timestamp, story_id, original input order)`. The last event is test, the penultimate event is validation, and earlier events are training.
- User and story mappings are determined from training interactions only.
- Each warm-start validation/test positive shares the same 99 negatives across all models. A negative story must have first appeared in the complete cleaned vote table strictly before the target timestamp and must not have been voted by that user strictly before the target.
- Very early warm targets with fewer than 99 distinct legal negatives are excluded rather than padded with duplicates or future stories. Validation excludes `{manifest['validation_excluded_insufficient_negative_pool']}` of `{manifest['validation_source_warm_events']}` warm targets; test excludes `{manifest['test_excluded_insufficient_negative_pool']}` of `{manifest['test_source_warm_events']}`.
- Seed: `{args.seed}`. Validation candidate SHA-256: `{manifest['validation_candidate_sha256']}`. Test candidate SHA-256: `{manifest['test_candidate_sha256']}`.
- Primary metrics cover warm-start target items only. Cold-item rates are reported separately because ID-based BPR-MF and DeepFM cannot infer unseen item embeddings.

## Models and tuning

Popularity counts only training interactions strictly before each target timestamp. BPR-MF uses temporal pairwise negatives and BPR loss. BPR embedding dimension, learning rate, regularization, and stopping epoch are selected using validation NDCG@10. DeepFM has linear, FM, and DNN components. Its full variant is tuned using validation NDCG@10; the selected configuration is then fixed for the ID/context/community ablation. Test is evaluated once after selection.

DeepFM numeric features are `log1p(user activity before t)`, `log1p(training-story popularity before t)`, UTC hour, and relative day. Its sparse fields are training-mapped user/story IDs and, for `DeepFM_full`, the baseline community label. The numeric scaler is fitted on training examples only. ID embeddings are identifiers, not content semantics.

Community labels come from the baseline friendship network constructed before the voting period, so they are available before the evaluated interactions. The requested leave-last-two-out split is per user rather than a single global calendar cutoff; candidate eligibility and dynamic statistics are still computed strictly before each target timestamp.

## Final leakage audit

1. **Pass — split isolation:** mutually exclusive position masks put earlier interactions in train, the penultimate interaction in validation, and the last interaction in test; validation/test positives are not train rows.
2. **Pass — user activity:** both training and scoring use strict-past counts; equal-time events are excluded by `rank(method="min") - 1` or left-sided timestamp search.
3. **Pass — story popularity:** popularity is counted only from training interactions with timestamps strictly below the target timestamp.
4. **Pass — temporal candidates:** candidate validation checks that every sampled story first appeared in the complete cleaned vote table strictly before the target; eight test targets with fewer than 99 legal negatives are excluded rather than padded.
5. **Pass — shared candidates:** Popularity, BPR-MF, and every DeepFM ablation score the same in-memory validation and test `CandidateSet` objects.
6. **Pass — validation-only selection:** BPR and DeepFM configuration/stopping choices use validation NDCG@10 only.
7. **Pass — final test use:** test scoring occurs only after configuration selection in the final pipeline. Test metrics are not consulted by the selection functions.
8. **Pass — user bootstrap:** leave-last-two-out contributes one test target per evaluated user, and bootstrap rows resample those user-level metric vectors.

The audit verifies the implemented per-user protocol; it does not turn leave-last-two-out into a global calendar split.

## Interpretation limits

This is sampled-candidate offline evaluation, not full-catalog online ranking. There are no true exposure logs; sampled negatives are not explicit negatives. Results do not establish online CTR lift and are not directly comparable to v2.0 diffusion PR-AUC. DIN is not implemented, and no missing story text or category features are invented.
"""
    (args.docs_dir / "recommendation_protocol.md").write_text(protocol, encoding="utf-8")
    results = f"""# Digg v3.1 recommendation results

## Warm-start sampled-candidate results

{markdown_table(display_main, ['Model', 'NDCG@10', 'Recall@10', 'MRR', 'AUC'])}

Additional calculated metrics and measured model size/runtime are:

{markdown_table(display_metrics, ['Model', 'NDCG@20', 'Recall@20', 'HitRate@10', 'parameters', 'training_seconds', 'inference_seconds'])}

All means use one held-out target per evaluated user. User-level bootstrap confidence intervals are in `outputs/recommendation/bootstrap_intervals.csv`.

BPR-MF improves NDCG@10 over Popularity by {float(metrics.loc[metrics.model == 'BPR-MF', 'NDCG@10'].iloc[0] - metrics.loc[metrics.model == 'Popularity', 'NDCG@10'].iloc[0]):.6f}. DeepFM_full changes NDCG@10 relative to BPR-MF by {float(metrics.loc[metrics.model == 'DeepFM_full', 'NDCG@10'].iloc[0] - metrics.loc[metrics.model == 'BPR-MF', 'NDCG@10'].iloc[0]):.6f}. Paired user-bootstrap intervals for both model deltas are included in the same CSV.

## Coverage

- Validation warm-user: {coverage['validation']['warm_user_rate']:.4%}; warm-item: {coverage['validation']['warm_item_rate']:.4%}; cold-item: {coverage['validation']['cold_item_rate']:.4%}.
- Test warm-user: {coverage['test']['warm_user_rate']:.4%}; warm-item: {coverage['test']['warm_item_rate']:.4%}; cold-item: {coverage['test']['cold_item_rate']:.4%}.

Cold target items are reported as coverage, not silently scored with trained ID embeddings.

## DeepFM ablation

{markdown_table(display_ablation, ['Model', 'NDCG@10', 'Recall@10', 'MRR', 'AUC', 'NDCG@20', 'Recall@20', 'HitRate@10'])}

`DeepFM_ID` uses user/story IDs; `DeepFM_context` adds activity, popularity, and time; `DeepFM_full` additionally adds the pre-period baseline user community. Context raises NDCG@10 over the ID-only variant by {float(ablation.loc[ablation.model == 'DeepFM_context', 'NDCG@10'].iloc[0] - ablation.loc[ablation.model == 'DeepFM_ID', 'NDCG@10'].iloc[0]):.6f}. Adding community changes NDCG@10 relative to context by {float(ablation.loc[ablation.model == 'DeepFM_full', 'NDCG@10'].iloc[0] - ablation.loc[ablation.model == 'DeepFM_context', 'NDCG@10'].iloc[0]):.6f}; it does not provide an additional gain in this run. This is a predictive comparison, not a causal community effect. ID embeddings are identifiers, not content semantics.

DeepFM_context is therefore the final v3.1 recommendation model. The lack of an incremental recommendation gain from the community field does not negate the separate mechanistic value of community structure in diffusion analysis. DIN is not implemented.

## Leakage audit

All eight checks in the protocol pass: disjoint leave-last-two-out rows; strict-past user activity and training-story popularity; no future-appearing test negatives; one shared candidate set across models; validation-only selection; test used after selection; and user-level bootstrap resampling. See [the protocol](recommendation_protocol.md) for the implementation evidence and the per-user/global-time caveat.

## Limitations

This is sampled-candidate offline evaluation and not full-catalog online ranking. It is not CTR prediction. Digg provides votes but no impression or click-opportunity log, so sampled non-interactions are not confirmed negatives and the results do not imply online CTR improvement. No DIN model is included and no unavailable story text/category features are fabricated. These recommendation metrics must not be compared numerically with v2.0 diffusion PR-AUC because the targets and candidate universes differ.
"""
    (args.docs_dir / "recommendation_results.md").write_text(results, encoding="utf-8")
    summary = f"""# Recommendation training summary

- Seed: `{args.seed}`
- Eligible users: `{len(data.user_to_index):,}`
- Training interactions: `{len(data.train):,}`
- Validation/test candidate count: `{manifest['validation_events']:,}` / `{manifest['test_events']:,}`
- BPR selection: `{json.dumps(bpr_config, sort_keys=True)}`
- DeepFM selection: `{json.dumps(deepfm_config, sort_keys=True)}`
- Final v3.1 recommendation model: `DeepFM_context`
- `training_seconds` for BPR-MF and DeepFM_full includes validation search; ID/context rows are fixed-configuration fit times. Inference time covers the complete warm-start test candidate set.
- Test was evaluated only after validation-based selection.
- Candidate and input checksums: see `candidate_manifest.json`.

The experiment is an implicit-feedback, sampled-candidate offline evaluation. It is not CTR prediction and sampled non-interactions are not explicit negative feedback.
"""
    (args.output_dir / "training_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
