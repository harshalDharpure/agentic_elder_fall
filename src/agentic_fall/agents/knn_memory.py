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
