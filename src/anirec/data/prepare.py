"""Raw CSV → cleaned Parquet conversion.

Auto-detects column names across common anime dataset formats
(MAL exports, Kaggle dumps, etc.) and writes standardised Parquet
files using DuckDB for out-of-core processing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb


# ── column alias maps ────────────────────────────────────────────────

RATINGS_ALIASES: dict[str, set[str]] = {
    "user_id": {"user_id", "userid", "user", "uid", "profile", "profile_id"},
    "item_id": {"anime_id", "item_id", "mal_id", "anime", "item", "id", "anime_uid"},
    "rating": {"rating", "score", "stars", "value", "user_score", "mean"},
}

ITEMS_ALIASES: dict[str, set[str]] = {
    "item_id": {"anime_id", "item_id", "mal_id", "id"},
    "title": {"name", "title", "anime_title"},
    "genres": {"genre", "genres"},
}


# ── header detection helpers ─────────────────────────────────────────

def _read_header(path: Path) -> list[str]:
    """Return the lowercased, stripped header row of a CSV file."""
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [h.strip().strip('"').strip("'").lower() for h in header]


def _score_header(
    header: list[str],
    aliases: dict[str, set[str]],
) -> tuple[int, dict[str, str]]:
    """Score how well *header* matches the canonical *aliases*.

    Returns
    -------
    tuple[int, dict[str, str]]
        (number of matched canonical columns, {canonical_name: actual_col}).
    """
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


def _find_best_csv(
    raw_dir: Path,
    aliases: dict[str, set[str]],
) -> tuple[Path, dict[str, str]]:
    """Walk *raw_dir* for ``.csv`` files and return the best match.

    "Best" = most canonical columns matched, tie-broken by file size
    (larger files are preferred for ratings data).

    Raises
    ------
    FileNotFoundError
        No ``.csv`` files under *raw_dir*.
    RuntimeError
        No CSV scored above zero, or a ratings candidate is missing
        required columns.
    """
    csvs = sorted(raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv files found under {raw_dir}")

    best: tuple[int, int, Path, dict[str, str]] | None = None
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
            f"Found a candidate ratings file ({path.name}) but missing "
            f"required columns.  Header mapping detected: {mapping}"
        )
    return path, mapping


# ── main entry point ─────────────────────────────────────────────────

def run(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    ratings_csv: str | None = None,
    sample_n: int = 200_000,
    threads: int = 4,
) -> None:
    """Convert raw CSVs into cleaned, standardised Parquet files.

    Parameters
    ----------
    raw_dir : str
        Directory containing raw ``.csv`` files.
    out_dir : str
        Destination for Parquet outputs.
    ratings_csv : str | None
        Explicit path to a ratings CSV.  Auto-detected when *None*.
    sample_n : int
        Number of rows to write to ``ratings_sample.parquet``.
    threads : int
        DuckDB thread count.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── ratings ──────────────────────────────────────────────────────
    if ratings_csv:
        ratings_path = Path(ratings_csv)
        if not ratings_path.exists():
            raise FileNotFoundError(ratings_path)
        header = _read_header(ratings_path)
        score, ratings_map = _score_header(header, RATINGS_ALIASES)
        if score < 3:
            raise RuntimeError(
                f"{ratings_path.name} does not look like ratings CSV. "
                f"Detected: {ratings_map}"
            )
    else:
        ratings_path, ratings_map = _find_best_csv(raw_dir, RATINGS_ALIASES)

    user_col = ratings_map["user_id"]
    item_col = ratings_map["item_id"]
    rating_col = ratings_map["rating"]

    print(f"[ratings] using: {ratings_path}")
    print(f"[ratings] columns: user={user_col} item={item_col} rating={rating_col}")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)};")
    con.execute("PRAGMA enable_progress_bar=true;")

    con.execute("DROP TABLE IF EXISTS cleaned_ratings;")
    con.execute(
        f"""
        CREATE TABLE cleaned_ratings AS
        SELECT
            DENSE_RANK() OVER (ORDER BY u)  AS user_id,
            TRY_CAST(i AS BIGINT)           AS item_id,
            TRY_CAST(r AS DOUBLE)           AS rating
        FROM (
            SELECT
                CAST({user_col} AS VARCHAR)  AS u,
                {item_col}                   AS i,
                {rating_col}                 AS r
            FROM read_csv_auto('{ratings_path.as_posix()}', header=true)
        )
        WHERE u IS NOT NULL
          AND TRY_CAST(i AS BIGINT) IS NOT NULL
          AND TRY_CAST(r AS DOUBLE) IS NOT NULL
          AND r > 0
        """
    )

    stats = con.execute(
        """
        SELECT
            COUNT(*)                AS rows,
            COUNT(DISTINCT user_id) AS users,
            COUNT(DISTINCT item_id) AS items,
            MIN(rating)             AS rating_min,
            MAX(rating)             AS rating_max,
            AVG(rating)             AS rating_mean
        FROM cleaned_ratings
        """
    ).fetchone()
    print(
        f"[ratings] rows={stats[0]:,} users={stats[1]:,} items={stats[2]:,} "
        f"min={stats[3]} max={stats[4]} mean={float(stats[5]):.4f}"
    )

    ratings_out = out_dir / "ratings.parquet"
    con.execute(
        f"""
        COPY cleaned_ratings
        TO '{ratings_out.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    print(f"[write] {ratings_out}")

    sample_out = out_dir / "ratings_sample.parquet"
    con.execute(
        f"""
        COPY (
            SELECT * FROM cleaned_ratings
            USING SAMPLE {int(sample_n)} ROWS
        )
        TO '{sample_out.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    print(f"[write] {sample_out}")

    # ── items metadata (optional) ────────────────────────────────────
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

    if items_path and items_map:
        item_id_col = items_map["item_id"]
        title_col = items_map["title"]
        genres_col = items_map.get("genres")

        print(f"[items] using: {items_path}")
        print(
            f"[items] columns: item_id={item_id_col} title={title_col}"
            + (f" genres={genres_col}" if genres_col else "")
        )

        con.execute("DROP TABLE IF EXISTS cleaned_items;")
        genre_select = f", CAST({genres_col} AS VARCHAR) AS genres" if genres_col else ""
        con.execute(
            f"""
            CREATE TABLE cleaned_items AS
            SELECT
                TRY_CAST({item_id_col} AS BIGINT) AS item_id,
                CAST({title_col} AS VARCHAR)      AS title
                {genre_select}
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
