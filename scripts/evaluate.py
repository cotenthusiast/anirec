#!/usr/bin/env python3
"""CLI wrapper for evaluating a model.

Usage
-----
    python scripts/evaluate.py --model popularity_unfiltered
    python scripts/evaluate.py --model popularity_filtered --k 20
"""

from __future__ import annotations

import argparse

from anirec.config import load_config
from anirec.data.loader import load_truth
from anirec.eval.metrics import ndcg_at_k, recall_at_k
from anirec.models.popularity import PopularityFiltered, PopularityUnfiltered, PopularityBayesianFiltered, PopularityBayesianUnfiltered

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a recommender model.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config")
    ap.add_argument("--k", type=int, default=None, help="Top-k cut-off (overrides config)")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["paths"]
    k = args.k or cfg["eval"]["default_k"]
    split_path = p["test_parquet"] if args.split == "test" else p["val_parquet"]

    m = cfg["models"]["bayesian_m"]
    models = {
        "popularity_unfiltered": PopularityUnfiltered(),
        "popularity_filtered":   PopularityFiltered(),
        "bayesian_unfiltered":   PopularityBayesianUnfiltered(m=m),
        "bayesian_filtered":     PopularityBayesianFiltered(m=m),
    }

    if args.model not in models:
        ap.error(f"Unknown model. Choose from: {list(models.keys())}")

    # fit
    model = models[args.model]
    print(f"[fit] {args.model} on {p['train_parquet']}")
    model.fit(p["train_parquet"])

    # load ground truth
    truth = load_truth(split_path)
    user_ids = list(truth.keys())
    print(f"[eval] {len(user_ids)} users, k={k}, split={args.split}")

    # recommend
    recs = model.recommend(user_ids, k)

    # metrics
    r = recall_at_k(recs, truth, k)
    n = ndcg_at_k(recs, truth, k)
    print(f"Recall@{k}: {r:.4f}")
    print(f"NDCG@{k}:   {n:.4f}")


if __name__ == "__main__":
    main()
