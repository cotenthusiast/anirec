#!/usr/bin/env python3
"""CLI wrapper for train / val / test splitting.

Usage
-----
    python scripts/split.py
    python scripts/split.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse

from anirec.config import load_config
from anirec.data.split import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Split ratings into train/val/test.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["paths"]
    s = cfg["split"]

    run(
        ratings_path=p["ratings_parquet"],
        out_dir=p["processed_dir"],
        positive_threshold=s["positive_threshold"],
        min_positive_per_user=s["min_positive_per_user"],
        seed=str(s["seed"]),
        threads=s["threads"],
    )


if __name__ == "__main__":
    main()
