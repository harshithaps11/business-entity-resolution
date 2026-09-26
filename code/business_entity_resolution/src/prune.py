"""Stage 2b: learned candidate pruning.

Retrieval keeps a generous top-K per Source 1 entity. A small LightGBM that sees
only the retrieval score and rank/competition features (no string work, so it
costs ~nothing per pair) then removes candidates that are clearly not matches.
Its threshold is chosen on the validation split so that a target share of the
true matches found by retrieval survives. The surviving set is the final
candidate set: it is what the matching model scores and what is written to
candidate_pairs.tsv.
"""
import lightgbm as lgb
import numpy as np

PRUNE_FEATURES = ["bscore", "rk", "bscore_rel", "n_cands", "rrk", "r_margin", "r_ncomp",
                  "gap_next", "gap_top", "src"]


def prune_frame(P, c_src):
    """Add the cheap pruning features to a competition_features() frame."""
    P = P.assign(src=c_src[P.c_idx.to_numpy()].astype(np.float32))
    s = P.bscore.to_numpy(np.float32)
    top = s / np.maximum(P.bscore_rel.to_numpy(np.float32), 1e-6)
    P["gap_top"] = (top - s).astype(np.float32)
    # gap to the next-ranked candidate of the same S1 (sorted by s1, rank)
    order = np.lexsort((P.rk.to_numpy(), P.s1_idx.to_numpy()))
    ss, si = s[order], P.s1_idx.to_numpy()[order]
    nxt = np.r_[ss[1:], 0.0]
    nxt[np.r_[si[1:] != si[:-1], True]] = 0.0
    g = np.empty(len(s), np.float32); g[order] = ss - nxt
    P["gap_next"] = g
    return P


def fit_pruner(X, y, Xva, yva):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.08, num_leaves=63,
                           min_child_samples=200, subsample=0.5, subsample_freq=1,
                           n_jobs=-1, verbose=-1)
    m.fit(X[PRUNE_FEATURES], y, eval_set=[(Xva[PRUNE_FEATURES], yva)],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def choose_threshold(p, y, n_s1, target_recall):
    """Largest threshold keeping >= target_recall of retrieved positives."""
    pos = np.sort(p[y == 1])
    th = float(pos[int(np.floor((1 - target_recall) * len(pos)))])
    keep = p >= th
    return th, dict(threshold=th, kept_recall=float(y[keep].sum() / y.sum()),
                    avg_cands=float(keep.sum() / n_s1), kept_pairs=int(keep.sum()))
