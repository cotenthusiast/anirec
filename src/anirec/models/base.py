"""Abstract base class that every recommender must implement.

Adding a new model means:
1. Create a file in ``src/anirec/models/``
2. Subclass ``Recommender``
3. Implement ``fit`` and ``recommend``
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Recommender(ABC):
    """Interface shared by all recommendation models."""

    @abstractmethod
    def fit(self, train_path: str, **kwargs) -> None:
        """Train or precompute on the training split.

        Parameters
        ----------
        train_path : str
            Path to the training Parquet file.
        """

    @abstractmethod
    def recommend(self, user_ids: list[int], k: int) -> dict[int, list[int]]:
        """Return top-*k* item recommendations per user.

        Parameters
        ----------
        user_ids : list[int]
            Users to generate recommendations for.
        k : int
            Number of items to recommend.

        Returns
        -------
        dict[int, list[int]]
            ``{user_id: [item_id, …]}`` with at most *k* items each.
        """
