from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..utils.io import ensure_dir


@dataclass
class MemoryCase:
    case_id: str
    embedding: list[float]
    label: int  # 0 ADL, 1 Fall
    evidence_text: str
    features: dict[str, float]
    activity: str = ""

    def summary(self) -> str:
        tag = "FALL" if self.label == 1 else "ADL"
        return (
            f"Case {self.case_id}: {self.evidence_text.replace(chr(10), ' ')} → {tag}"
        )


class KNNMemory:
    """Growing case memory with cosine k-NN retrieval."""

    def __init__(self, k: int = 5, metric: str = "cosine"):
        self.k = k
        self.metric = metric
        self.cases: list[MemoryCase] = []
        self._index: NearestNeighbors | None = None
        self._emb: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.cases)

    def add(self, case: MemoryCase) -> None:
        self.cases.append(case)
        self._index = None

    def add_batch(self, cases: list[MemoryCase]) -> None:
        self.cases.extend(cases)
        self._index = None

    def _rebuild(self) -> None:
        if not self.cases:
            self._index = None
            self._emb = None
            return
        self._emb = np.asarray([c.embedding for c in self.cases], dtype=np.float32)
        n = len(self.cases)
        self._index = NearestNeighbors(
            n_neighbors=min(self.k, n),
            metric=self.metric,
            algorithm="auto",
        )
        self._index.fit(self._emb)

    def retrieve(self, embedding: np.ndarray, k: int | None = None) -> list[MemoryCase]:
        if not self.cases:
            return []
        if self._index is None:
            self._rebuild()
        assert self._index is not None
        kk = min(k or self.k, len(self.cases))
        q = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        _, idx = self._index.kneighbors(q, n_neighbors=kk)
        return [self.cases[i] for i in idx[0]]

    def contrastive_retrieve(
        self,
        embedding: np.ndarray,
        *,
        k_pos: int = 3,
        k_neg: int = 3,
        pool_k: int | None = None,
        hard_neg_activities: tuple[str, ...] = ("D18", "D19"),
    ) -> tuple[list[MemoryCase], list[MemoryCase]]:
        """Retrieve similar falls (positives) and similar ADLs (hard negatives).

        Hard negatives prefer near-fall ADL codes (D18/D19) when available,
        otherwise the nearest ADLs in the pool. Used by the Critic so it sees
        counter-evidence instead of only confirming fall neighbours.
        """
        need = max(k_pos + k_neg, self.k) * 4
        pool = self.retrieve(embedding, k=pool_k or min(max(need, 20), len(self.cases)))
        positives = [c for c in pool if int(c.label) == 1][:k_pos]
        hard = [
            c
            for c in pool
            if int(c.label) == 0 and str(c.activity) in hard_neg_activities
        ]
        other_adl = [
            c
            for c in pool
            if int(c.label) == 0 and str(c.activity) not in hard_neg_activities
        ]
        negatives = (hard + other_adl)[:k_neg]
        # Backfill if the local pool is class-imbalanced.
        if len(positives) < k_pos or len(negatives) < k_neg:
            if self._index is None:
                self._rebuild()
            if self._index is not None and self._emb is not None:
                q = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
                kk = min(len(self.cases), max(50, need))
                dists, idxs = self._index.kneighbors(q, n_neighbors=kk)
                order = list(idxs[0])
                if len(positives) < k_pos:
                    seen = {c.case_id for c in positives}
                    for i in order:
                        c = self.cases[int(i)]
                        if int(c.label) == 1 and c.case_id not in seen:
                            positives.append(c)
                            seen.add(c.case_id)
                        if len(positives) >= k_pos:
                            break
                if len(negatives) < k_neg:
                    seen = {c.case_id for c in negatives}
                    # Prefer hard activities first, then any ADL.
                    for prefer_hard in (True, False):
                        for i in order:
                            c = self.cases[int(i)]
                            if int(c.label) != 0 or c.case_id in seen:
                                continue
                            is_hard = str(c.activity) in hard_neg_activities
                            if prefer_hard and not is_hard:
                                continue
                            if (not prefer_hard) and is_hard:
                                continue
                            negatives.append(c)
                            seen.add(c.case_id)
                            if len(negatives) >= k_neg:
                                break
                        if len(negatives) >= k_neg:
                            break
        return positives, negatives

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        payload = [asdict(c) for c in self.cases]
        with path.open("w") as f:
            json.dump(payload, f)

    def load(self, path: str | Path) -> None:
        with Path(path).open() as f:
            payload = json.load(f)
        self.cases = [MemoryCase(**item) for item in payload]
        self._index = None

    def grow(self, case: MemoryCase) -> None:
        """Continual evidence accumulation (paper claim #5)."""
        self.add(case)
