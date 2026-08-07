from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def subject_folds(
    subjects: Sequence[str],
    n_folds: int = 5,
    seed: int = 42,
    strata: Sequence[str] | None = None,
) -> list[tuple[list[str], list[str]]]:
    """Subject-independent K-fold splits. Returns list of (train_ids, test_ids).

    If ``strata`` is provided (same length as unique subjects mapping), folds are
    stratified so each group (e.g. fall-capable vs ADL-only) is distributed.
    """
    uniq = sorted(set(subjects))
    rng = np.random.default_rng(seed)

    if strata is None:
        order = np.array(uniq)
        rng.shuffle(order)
        folds: list[list[str]] = [[] for _ in range(n_folds)]
        for i, sid in enumerate(order):
            folds[i % n_folds].append(str(sid))
    else:
        # map subject -> stratum from parallel arrays over windows is awkward;
        # expect strata as dict-like via zip of unique subject labels passed in.
        # Here strata aligns with ``uniq`` order if provided as list matching uniq.
        if len(strata) != len(uniq):
            # build from subject list + strata list of same length as subjects
            from collections import defaultdict

            subj_stratum: dict[str, str] = {}
            for s, st in zip(subjects, strata):
                subj_stratum[str(s)] = str(st)
            groups: dict[str, list[str]] = defaultdict(list)
            for s in uniq:
                groups[subj_stratum.get(str(s), "na")].append(str(s))
        else:
            from collections import defaultdict

            groups = defaultdict(list)
            for s, st in zip(uniq, strata):
                groups[str(st)].append(str(s))

        folds = [[] for _ in range(n_folds)]
        for _, members in groups.items():
            arr = np.array(members)
            rng.shuffle(arr)
            for i, sid in enumerate(arr):
                folds[i % n_folds].append(str(sid))

    splits = []
    for i in range(n_folds):
        test = folds[i]
        train = [s for j, fold in enumerate(folds) if j != i for s in fold]
        splits.append((train, test))
    return splits


def indices_by_subject(subject_ids: Sequence[str]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = defaultdict(list)
    for i, sid in enumerate(subject_ids):
        mapping[str(sid)].append(i)
    return dict(mapping)
