"""Popularity-based baseline recommenders.

Two variants:
- **Unfiltered**: recommend the globally most-popular items to every user.
- **Filtered**: exclude items the user has already rated.
"""

from __future__ import annotations

import duckdb

from anirec.models.base import Recommender


class PopularityUnfiltered(Recommender):
    """Recommend the global top-*k* items regardless of user history."""

    def __init__(self) -> None:
        self._top_items: list[int] = []

    def fit(self, train_path: str, **kwargs) -> None:
        """Compute global item popularity from the training split.

        Parameters
        ----------
        train_path : str
            Path to the training Parquet file.
        """
        con = duckdb.connect()
        rows = con.execute(f"""
            SELECT item_id, COUNT(*) AS cnt
            FROM read_parquet('{train_path}')
            GROUP BY item_id
            ORDER BY cnt DESC
        """).fetchall()
        con.close()
        self._top_items = [int(r[0]) for r in rows]

    def recommend(self, user_ids: list[int], k: int) -> dict[int, list[int]]:
        """Return the same global top-*k* for every user.

        Parameters
        ----------
        user_ids : list[int]
            Users to generate recommendations for.
        k : int
            Number of items to recommend.

        Returns
        -------
        dict[int, list[int]]
        """
        top_k = self._top_items[:k]
        return {uid: top_k for uid in user_ids}


class PopularityFiltered(Recommender):
    """Recommend the most-popular items the user has *not* already rated."""

    def __init__(self) -> None:
        self._top_items: list[int] = []
        self._user_seen: dict[int, set[int]] = {}

    def fit(self, train_path: str, **kwargs) -> None:
        """Compute global popularity and per-user history.

        Parameters
        ----------
        train_path : str
            Path to the training Parquet file.
        """
        con = duckdb.connect()

        rows = con.execute(f"""
            SELECT item_id, COUNT(*) AS cnt
            FROM read_parquet('{train_path}')
            GROUP BY item_id
            ORDER BY cnt DESC
        """).fetchall()
        self._top_items = [int(r[0]) for r in rows]

        history = con.execute(f"""
            SELECT user_id, item_id
            FROM read_parquet('{train_path}')
        """).fetchall()
        con.close()

        self._user_seen = {}
        for uid, iid in history:
            self._user_seen.setdefault(int(uid), set()).add(int(iid))

    def recommend(self, user_ids: list[int], k: int) -> dict[int, list[int]]:
        """Return top-*k* popular items the user hasn't seen.

        Parameters
        ----------
        user_ids : list[int]
            Users to generate recommendations for.
        k : int
            Number of items to recommend.

        Returns
        -------
        dict[int, list[int]]
        """
        recs = {}
        for uid in user_ids:
            seen = self._user_seen.get(uid, set())
            user_recs = []
            for iid in self._top_items:
                if iid not in seen:
                    user_recs.append(iid)
                if len(user_recs) == k:
                    break
            recs[uid] = user_recs
        return recs
