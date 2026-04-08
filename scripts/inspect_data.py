#!/usr/bin/env python3
"""Quick EDA on the processed ratings data.

Usage
-----
    python scripts/inspect_data.py
    python scripts/inspect_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse

import duckdb

from anirec.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect processed ratings data.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ratings_path = cfg["paths"]["ratings_parquet"]
    pos_thresh = cfg["split"]["positive_threshold"]
    threads = cfg["inspect"]["threads"]
    min_ratings = cfg["inspect"]["min_ratings_display"]

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads};")
    con.execute("PRAGMA enable_progress_bar=true;")

    total_positive = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{ratings_path}')
        WHERE rating >= {pos_thresh};
    """).fetchone()[0]
    print(f"number of anime × user combinations rated >= {pos_thresh}: {total_positive}")

    distinct = con.execute(f"""
        SELECT COUNT(DISTINCT user_id), COUNT(DISTINCT item_id)
        FROM read_parquet('{ratings_path}')
        WHERE rating >= {pos_thresh};
    """).fetchone()
    print(f"distinct users and animes: {distinct}")

    count_3plus = con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT user_id
            FROM read_parquet('{ratings_path}')
            WHERE rating >= {pos_thresh}
            GROUP BY user_id
            HAVING COUNT(*) >= {min_ratings}
        ) AS t
    """).fetchone()[0]
    print(f"users with at least {min_ratings} positive animes rated: {count_3plus}")

    con.close()


if __name__ == "__main__":
    main()
