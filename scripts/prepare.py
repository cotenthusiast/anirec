#!/usr/bin/env python3
"""CLI wrapper for data preparation (raw CSV → Parquet).

Usage
-----
    python scripts/prepare.py
    python scripts/prepare.py --config configs/default.yaml
    python scripts/prepare.py --ratings-csv data/raw/my_ratings.csv
"""

from __future__ import annotations

import argparse

from anirec.config import load_config
from anirec.data.prepare import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert raw CSVs to cleaned Parquet.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config")
    ap.add_argument("--ratings-csv", type=str, default=None, help="Explicit ratings CSV path")
    ap.add_argument("--drop-nonpositive", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["paths"]
    prep = cfg["prepare"]

    run(
        raw_dir=p["raw_dir"],
        out_dir=p["processed_dir"],
        ratings_csv=args.ratings_csv,
        drop_nonpositive=args.drop_nonpositive or prep["drop_nonpositive"],
        sample_n=prep["sample_n"],
        threads=prep["threads"],
    )


if __name__ == "__main__":
    main()
