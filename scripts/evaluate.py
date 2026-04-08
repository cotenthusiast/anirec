#!/usr/bin/env python3
"""CLI wrapper for evaluating a model.

Usage
-----
    python scripts/evaluate.py --model popularity_unfiltered
    python scripts/evaluate.py --model popularity_filtered --k 20
    python scripts/evaluate.py --model svd --load models/svd
"""

from __future__ import annotations

import argparse
import os

from anirec.config import load_config
from anirec.data.loader import load_truth
from anirec.eval.metrics import ndcg_at_k, recall_at_k
from anirec.models.popularity import PopularityFiltered, PopularityUnfiltered, PopularityBayesianFiltered, PopularityBayesianUnfiltered
from anirec.models.svd import SVD


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a recommender model.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config")
    ap.add_argument("--k", type=int, default=None, help="Top-k cut-off (overrides config)")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--load", type=str, default=None, help="Load model from path instead of fitting")
    args = ap.parse_args()

    os.makedirs(p["models_dir"], exist_ok=True)

    cfg = load_config(args.config)
    p = cfg["paths"]
    k = args.k or cfg["eval"]["default_k"]
    split_path = p["test_parquet"] if args.split == "test" else p["val_parquet"]

    m = cfg["models"]["bayesian_m"]
    svd_cfg = cfg["models"]["svd"]
    models = {
        "popularity_unfiltered": PopularityUnfiltered(),
        "popularity_filtered":   PopularityFiltered(),
        "bayesian_unfiltered":   PopularityBayesianUnfiltered(m),
        "bayesian_filtered":     PopularityBayesianFiltered(m),
        "svd": SVD(
            k=svd_cfg["k"],
            lr=svd_cfg["lr"],
            lambda_=svd_cfg["lambda_"],
            num_epochs=svd_cfg["num_epochs"],
        ),
    }

    if args.model not in models:
        ap.error(f"Unknown model. Choose from: {list(models.keys())}")

    model = models[args.model]

    if args.load:
        print(f"[load] {args.model} from {args.load}")
        model.load(args.load)
    else:
        print(f"[fit] {args.model} on {p['train_parquet']}")
        model.fit(p["train_parquet"])

    truth = load_truth(split_path)
    first_user = next(iter(truth))
    print(f"[debug] sample truth: user {first_user} -> items {truth[first_user]}")
    user_ids = list(truth.keys())
    print(f"[eval] {len(user_ids)} users, k={k}, split={args.split}")

    if hasattr(model, '_user_map'):
        overlap = len(set(model._user_map.keys()) & set(user_ids))
        print(f"Overlap: {overlap} / {len(user_ids)} test users found in training")
        model.save(p["svd_model_path"])

    recs = model.recommend(user_ids, k)
    r = recall_at_k(recs, truth, k)
    n = ndcg_at_k(recs, truth, k)
    print(f"Recall@{k}: {r:.4f}")
    print(f"NDCG@{k}:   {n:.4f}")


if __name__ == "__main__":
    main()