"""Turn pair probabilities into per-entity match lists, and score them.

Decision rule:
  1. keep pairs with p >= threshold;
  2. one-to-many constraint observed in training ground truth: every S2/S3
     record belongs to at most one S1 entity, so a candidate claimed by several
     S1 entities is given only to the one with the highest probability.
"""
import numpy as np
import pandas as pd


def decide(P: pd.DataFrame, s1: pd.DataFrame, c: pd.DataFrame, th: float) -> dict:
    """P needs columns s1_idx, c_idx, p. Returns {s1_id: [cand_id, ...]}."""
    Q = P.loc[P.p >= th, ["s1_idx", "c_idx", "p"]]
    Q = Q.sort_values("p", ascending=False).drop_duplicates("c_idx")
    ids1, idsc = s1.id.to_numpy(), c.id.to_numpy()
    out = {}
    for a, b in zip(Q.s1_idx.to_numpy(), Q.c_idx.to_numpy()):
        out.setdefault(ids1[a], []).append(idsc[b])
    return out


def f05(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict, ids, truth: dict) -> float:
    """Macro-average F0.5 over the given S1 ids (singletons included)."""
    return float(np.mean([f05(set(pred.get(i, ())), truth.get(i, set())) for i in ids]))
