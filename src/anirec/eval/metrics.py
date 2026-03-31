"""Ranking metrics for top-*k* recommendation evaluation.

All functions follow the same convention:
- ``recs``:  ``{user_id: [item_id, …]}`` — predicted ranked lists.
- ``truth``: ``{user_id: item_id}``       — ground-truth item per user
  (leave-one-out setting).
"""

from __future__ import annotations

import numpy as np


def recall_at_k(recs: dict[int, list[int]], truth: dict[int, int], k: int) -> float:
    """Compute Recall@k (fraction of users whose ground-truth item appears in the top-*k*).

    Parameters
    ----------
    recs : dict[int, list[int]]
        Recommendations per user.
    truth : dict[int, int]
        Ground-truth item per user.
    k : int
        Cut-off length.

    Returns
    -------
    float
        Recall@k averaged over all users in *truth*.
    """
    hits = 0
    for user in truth:
        recommendations = recs[user][:k]
        actual = truth[user]
        if actual in recommendations:
            hits += 1
    return hits / len(truth)


def ndcg_at_k(recs: dict[int, list[int]], truth: dict[int, int], k: int) -> float:
    """Compute NDCG@k in a leave-one-out setting.

    With a single relevant item per user the ideal DCG is always 1,
    so NDCG reduces to ``1 / log2(rank + 1)`` when the item is found
    and 0 otherwise.

    Parameters
    ----------
    recs : dict[int, list[int]]
        Recommendations per user.
    truth : dict[int, int]
        Ground-truth item per user.
    k : int
        Cut-off length.

    Returns
    -------
    float
        NDCG@k averaged over all users in *truth*.
    """
    scores = []
    for user in truth:
        recommendations = recs[user][:k]
        actual = truth[user]
        if actual in recommendations:
            rank = recommendations.index(actual) + 1
            scores.append(1.0 / np.log2(rank + 1))
        else:
            scores.append(0.0)
    return float(np.mean(scores))
