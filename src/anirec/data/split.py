"""Train / validation / test splitting.

Uses a deterministic hash-based leave-one-out strategy:
- Each eligible user (≥ ``min_positive_per_user`` positively-rated items)
  contributes exactly **one** item to the test set and **one** to the
  validation set.  The remainder goes to training.
- The split is reproducible for a given ``seed``.

All heavy lifting is done inside DuckDB so the full 50 M-row dataset
never needs to fit in memory.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def run(
    ratings_path: str = "data/processed/ratings.parquet",
    out_dir: str = "data/processed",
    positive_threshold: float = 8.0,
    min_positive_per_user: int = 3,
    seed: str = "42",
    threads: int = 4,
) -> None:
    """Create train / val / test Parquet files from a ratings file.

    Parameters
    ----------
    ratings_path : str
        Path to the cleaned ratings Parquet.
    out_dir : str
        Directory for the output splits.
    positive_threshold : float
        Minimum rating to count as a positive interaction.
    min_positive_per_user : int
        Users with fewer positives than this are excluded.
    seed : str
        Seed string hashed into the deterministic ordering.
    threads : int
        DuckDB thread count.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute("PRAGMA enable_progress_bar=true")

    # ── eligible users ───────────────────────────────────────────────
    con.execute(f"""
        CREATE OR REPLACE VIEW eligible_users AS
        SELECT user_id
        FROM read_parquet('{ratings_path}')
        WHERE rating >= {positive_threshold}
        GROUP BY user_id
        HAVING COUNT(*) >= {min_positive_per_user}
    """)

    # ── eligible positive interactions ───────────────────────────────
    con.execute(f"""
        CREATE OR REPLACE VIEW eligible_pos AS
        SELECT r.user_id, r.item_id, r.rating
        FROM read_parquet('{ratings_path}') AS r
        JOIN eligible_users u ON r.user_id = u.user_id
        WHERE r.rating >= {positive_threshold}
    """)

    # ── deterministic per-user ranking via md5 hash ──────────────────
    con.execute(f"""
        CREATE OR REPLACE VIEW ranked_pos AS
        SELECT
            ep.user_id,
            ep.item_id,
            ep.rating,
            ROW_NUMBER() OVER (
                PARTITION BY ep.user_id
                ORDER BY md5(
                    CAST(ep.user_id AS VARCHAR) || ':' ||
                    CAST(ep.item_id AS VARCHAR) || ':' ||
                    '{seed}'
                )
            ) AS rn
        FROM eligible_pos ep
    """)

    # ── split views ──────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE VIEW split_test AS
        SELECT user_id, item_id, rating FROM ranked_pos WHERE rn = 1
    """)
    con.execute("""
        CREATE OR REPLACE VIEW split_val AS
        SELECT user_id, item_id, rating FROM ranked_pos WHERE rn = 2
    """)
    con.execute("""
        CREATE OR REPLACE VIEW split_train AS
        SELECT user_id, item_id, rating FROM ranked_pos WHERE rn >= 3
    """)

    # ── sanity checks ────────────────────────────────────────────────
    eligible_n = con.execute("SELECT COUNT(*) FROM eligible_users").fetchone()[0]
    test_n = con.execute("SELECT COUNT(*) FROM split_test").fetchone()[0]
    val_n = con.execute("SELECT COUNT(*) FROM split_val").fetchone()[0]
    train_n = con.execute("SELECT COUNT(*) FROM split_train").fetchone()[0]

    print(f"eligible_users: {eligible_n}")
    print(f"split_test rows: {test_n}")
    print(f"split_val rows: {val_n}")
    print(f"split_train rows: {train_n}")

    test_minmax = con.execute("""
        SELECT MIN(c), MAX(c)
        FROM (SELECT user_id, COUNT(*) c FROM split_test GROUP BY user_id) t
    """).fetchone()
    val_minmax = con.execute("""
        SELECT MIN(c), MAX(c)
        FROM (SELECT user_id, COUNT(*) c FROM split_val GROUP BY user_id) t
    """).fetchone()

    print(f"test per-user count (min, max): {test_minmax}")
    print(f"val  per-user count (min, max): {val_minmax}")

    # disjointness checks
    tv = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT user_id, item_id FROM split_test
            INTERSECT
            SELECT user_id, item_id FROM split_val
        ) t
    """).fetchone()[0]
    tt = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT user_id, item_id FROM split_test
            INTERSECT
            SELECT user_id, item_id FROM split_train
        ) t
    """).fetchone()[0]
    vt = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT user_id, item_id FROM split_val
            INTERSECT
            SELECT user_id, item_id FROM split_train
        ) t
    """).fetchone()[0]

    print(f"intersection test∩val:   {tv}")
    print(f"intersection test∩train: {tt}")
    print(f"intersection val∩train:  {vt}")

    # ── write parquet ────────────────────────────────────────────────
    for name, view in [("train", "split_train"), ("val", "split_val"), ("test", "split_test")]:
        dest = out_dir / f"{name}.parquet"
        con.execute(f"""
            COPY (SELECT * FROM {view})
            TO '{dest.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    print(f"wrote: {out_dir}/train.parquet, val.parquet, test.parquet")
    con.close()
