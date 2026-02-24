# scripts/prepare_data.py

# scripts/prepare_data.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import duckdb


RATINGS_ALIASES = {
    "user_id": {"user_id", "userid", "user", "uid", "profile", "profile_id"},
    "item_id": {"anime_id", "item_id", "mal_id", "anime", "item", "id"},
    "rating": {"rating", "score", "stars", "value"},
}

ITEMS_ALIASES = {
    "item_id": {"anime_id", "item_id", "mal_id", "id"},
    "title": {"name", "title", "anime_title"},
    "genres": {"genre", "genres"},
}


@dataclass
class CsvCandidate:
    path: Path
    header: list[str]
    size_bytes: int


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [h.strip().strip('"').strip("'").lower() for h in header]


def _score_header(header: list[str], aliases: dict[str, set[str]]) -> tuple[int, dict[str, str]]:
    colset = set(header)
    mapping: dict[str, str] = {}
    score = 0
    for canon, cands in aliases.items():
        found = None
        for c in cands:
            if c in colset:
                found = c
                break
        if found is not None:
            mapping[canon] = found
            score += 1
    return score, mapping


def _find_best_csv(raw_dir: Path, aliases: dict[str, set[str]]) -> tuple[Path, dict[str, str]]:
    csvs = sorted(raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv files found under {raw_dir}")

    best: tuple[int, int, Path, dict[str, str]] | None = None
    # best = (score, size_bytes, path, mapping)
    for p in csvs:
        try:
            header = _read_header(p)
        except Exception:
            continue
        score, mapping = _score_header(header, aliases)
        if score == 0:
            continue
        size = p.stat().st_size
        key = (score, size, p, mapping)
        if best is None or key[:2] > best[:2]:
            best = key

    if best is None:
        raise RuntimeError(f"Could not identify a suitable CSV in {raw_dir}")

    score, size, path, mapping = best
    if score < 3 and aliases is RATINGS_ALIASES:
        raise RuntimeError(
            f"Found a candidate ratings file ({path.name}) but missing required columns. "
            f"Header mapping detected: {mapping}"
        )
    return path, mapping


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, default="data/raw")
    ap.add_argument("--out-dir", type=str, default="data/processed")
    ap.add_argument("--ratings-csv", type=str, default=None, help="Optional explicit path to ratings CSV")
    ap.add_argument("--drop-nonpositive", action="store_true", help="Drop rating <= 0 (common for unrated/-1/0)")
    ap.add_argument("--sample-n", type=int, default=200_000, help="Rows to write to ratings_sample.parquet")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # Find ratings CSV + its columns
    if args.ratings_csv:
        ratings_path = Path(args.ratings_csv)
        if not ratings_path.exists():
            raise FileNotFoundError(ratings_path)
        header = _read_header(ratings_path)
        score, ratings_map = _score_header(header, RATINGS_ALIASES)
        if score < 3:
            raise RuntimeError(f"{ratings_path.name} does not look like ratings CSV. Detected: {ratings_map}")
    else:
        ratings_path, ratings_map = _find_best_csv(raw_dir, RATINGS_ALIASES)

    user_col = ratings_map["user_id"]
    item_col = ratings_map["item_id"]
    rating_col = ratings_map["rating"]

    print(f"[ratings] using: {ratings_path}")
    print(f"[ratings] columns: user={user_col} item={item_col} rating={rating_col}")

    # Optionally find items/anime metadata CSV
    items_path = None
    items_map = None
    try:
        items_path, items_map = _find_best_csv(raw_dir, ITEMS_ALIASES)
        if items_map.get("item_id") is None or items_map.get("title") is None:
            items_path = None
            items_map = None
    except Exception:
        items_path = None
        items_map = None

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(args.threads)};")
    con.execute("PRAGMA enable_progress_bar=true;")

    # Clean ratings out-of-core
    where_extra = ""
    if args.drop_nonpositive:
        where_extra = "AND r > 0"

    con.execute("DROP TABLE IF EXISTS cleaned_ratings;")
    con.execute(
        f"""
        CREATE TABLE cleaned_ratings AS
        SELECT
            TRY_CAST(u AS BIGINT)  AS user_id,
            TRY_CAST(i AS BIGINT)  AS item_id,
            TRY_CAST(r AS DOUBLE)  AS rating
        FROM (
            SELECT
                {user_col}   AS u,
                {item_col}   AS i,
                {rating_col} AS r
            FROM read_csv_auto('{ratings_path.as_posix()}', header=true)
        )
        WHERE TRY_CAST(u AS BIGINT) IS NOT NULL
          AND TRY_CAST(i AS BIGINT) IS NOT NULL
          AND TRY_CAST(r AS DOUBLE) IS NOT NULL
          {where_extra}
        """
    )

    # Stats
    stats = con.execute(
        """
        SELECT
            COUNT(*)                          AS rows,
            COUNT(DISTINCT user_id)           AS users,
            COUNT(DISTINCT item_id)           AS items,
            MIN(rating)                       AS rating_min,
            MAX(rating)                       AS rating_max,
            AVG(rating)                       AS rating_mean
        FROM cleaned_ratings
        """
    ).fetchone()
    print(f"[ratings] rows={stats[0]:,} users={stats[1]:,} items={stats[2]:,} "
          f"min={stats[3]} max={stats[4]} mean={float(stats[5]):.4f}")

    # Write Parquet
    ratings_out = out_dir / "ratings.parquet"
    con.execute(
        f"""
        COPY cleaned_ratings
        TO '{ratings_out.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    print(f"[write] {ratings_out}")

    # Write sample Parquet for tests/quick iteration
    sample_out = out_dir / "ratings_sample.parquet"
    con.execute(
        f"""
        COPY (
            SELECT * FROM cleaned_ratings
            USING SAMPLE {int(args.sample_n)} ROWS
        )
        TO '{sample_out.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    print(f"[write] {sample_out}")

    # Optional: items/anime metadata
    if items_path and items_map:
        item_id_col = items_map["item_id"]
        title_col = items_map["title"]
        genres_col = items_map.get("genres")

        print(f"[items] using: {items_path}")
        print(f"[items] columns: item_id={item_id_col} title={title_col}" + (f" genres={genres_col}" if genres_col else ""))

        con.execute("DROP TABLE IF EXISTS cleaned_items;")
        if genres_col:
            con.execute(
                f"""
                CREATE TABLE cleaned_items AS
                SELECT
                    TRY_CAST({item_id_col} AS BIGINT) AS item_id,
                    CAST({title_col} AS VARCHAR)      AS title,
                    CAST({genres_col} AS VARCHAR)     AS genres
                FROM read_csv_auto('{items_path.as_posix()}', header=true)
                WHERE TRY_CAST({item_id_col} AS BIGINT) IS NOT NULL
                  AND {title_col} IS NOT NULL
                """
            )
        else:
            con.execute(
                f"""
                CREATE TABLE cleaned_items AS
                SELECT
                    TRY_CAST({item_id_col} AS BIGINT) AS item_id,
                    CAST({title_col} AS VARCHAR)      AS title
                FROM read_csv_auto('{items_path.as_posix()}', header=true)
                WHERE TRY_CAST({item_id_col} AS BIGINT) IS NOT NULL
                  AND {title_col} IS NOT NULL
                """
            )

        items_out = out_dir / "items.parquet"
        con.execute(
            f"""
            COPY cleaned_items
            TO '{items_out.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )
        print(f"[write] {items_out}")

    con.close()


if __name__ == "__main__":
    main()