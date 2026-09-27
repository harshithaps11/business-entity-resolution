"""Stage 3b: two-stage matcher + expected-F0.5 decision, trained on cached features.

Stage 1: LightGBM on pair features. Out-of-fold (2-fold, by S1 entity)
         predictions on TRAIN feed stage 2 without leakage.
Stage 2: LightGBM on pair features + context features computed from stage-1
         probabilities of the other candidates of the same S1 entity and of the
         other S1 entities competing for the same candidate.
Decision: one-owner assignment, then per-entity choice of the match set that
          maximises expected F0.5 (compared with a global threshold on VALID).
"""
import glob
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from decide import f05
from train import load, truth_sets

ID_COLS = ["s1_idx", "c_idx", "y"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=127, num_threads=0, verbose=-1)


def read(work, name, cols=None):
    return pd.concat([pd.read_parquet(p, columns=cols) for p in sorted(glob.glob(f"{work}/feat_{name}/part_*.parquet"))],
                     ignore_index=True)


def fit(X, y, Xv=None, yv=None, rounds=3000):
    dtr = lgb.Dataset(X, y, free_raw_data=True, params={"max_bin": PARAMS["max_bin"]})
    if Xv is None:
        return lgb.train(PARAMS, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def context(D, p):
    """Features of a candidate relative to its S1's other candidates and to the
    other S1 entities that compete for it (all from stage-1 probabilities)."""
    D = D[["s1_idx", "c_idx"]].assign(p1=p.astype(np.float32))
    g = D.groupby("s1_idx").p1
    out = pd.DataFrame(index=D.index)
    out["p1"] = D.p1
    out["p1_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    mx = g.transform("max")
    out["p1_gap_best"] = (mx - D.p1).astype(np.float32)
    out["p1_sum_s1"] = g.transform("sum").astype(np.float32)
    out["p1_n05_s1"] = D.p1.gt(0.5).groupby(D.s1_idx).transform("sum").astype(np.float32)
    out["p1_n_s1"] = g.transform("size").astype(np.float32)
    h = D.groupby("c_idx").p1
    cmax = h.transform("max")
    rk = h.rank(ascending=False, method="first")
    second = D.p1.where(rk > 1).groupby(D.c_idx).transform("max").fillna(0.0)
    other = np.where(rk == 1, second, cmax)            # best competing S1's prob
    out["p1_other_s1"] = other.astype(np.float32)
    out["p1_margin_other"] = (D.p1 - other).astype(np.float32)
    out["p1_rank_c"] = rk.astype(np.float32)
    return out


def assign(D, p):
    """One-owner rule: a candidate is kept only for the S1 with the highest p."""
    best = pd.Series(p).groupby(D.c_idx.to_numpy()).transform("max").to_numpy()
    return np.where(p >= best, p, 0.0)


def decide_expected(D, p, miss_rate, floor):
    """Per S1: choose top-k maximising expected F0.5 (independence approx.)."""
    order = np.lexsort((-p, D.s1_idx.to_numpy()))
    s = D.s1_idx.to_numpy()[order]; q = p[order]
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    keep = np.zeros(len(p), bool)
    for a, b in zip(start, np.r_[start[1:], len(s)]):
        qq = q[a:b]
        qq = qq[qq >= floor]
        if not len(qq):
            continue
        et = qq.sum() + miss_rate
        ctp = np.cumsum(qq)
        k = np.arange(1, len(qq) + 1)
        ef = 1.25 * ctp / (0.25 * et + k)
        f0 = np.prod(1 - qq) * np.exp(-miss_rate)
        j = int(np.argmax(ef))
        if ef[j] > f0:
            keep[order[a:a + j + 1]] = True
    return keep


def macro(D, keep, s1_ids_eval, ids1, idsc, truth):
    pred = {}
    for a, b in zip(D.s1_idx.to_numpy()[keep], D.c_idx.to_numpy()[keep]):
        pred.setdefault(ids1[a], set()).add(idsc[b])
    return float(np.mean([f05(pred.get(i, set()), truth.get(i, set())) for i in s1_ids_eval]))


def main(work):
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    s1, c = load(work, "train")
    ids1, idsc = s1.id.to_numpy(), c.id.to_numpy()
    va_s1 = np.sort(np.random.default_rng(42).permutation(len(s1))[150_000:210_000])
    truth = truth_sets(pd.read_parquet(f"{work}/train_gt.parquet"), ids1[va_s1])
    del s1, c

    T, V = read(work, "train"), read(work, "valid")
    feats = [x for x in T.columns if x not in ID_COLS]
    log(f"train {T.shape} valid {V.shape} features {len(feats)}")
    # one float32 matrix per split; keep only the id columns as a frame
    X, y = T[feats].to_numpy(np.float32), T.y.to_numpy(); T = T[["s1_idx", "c_idx"]].copy()
    Xv, yv = V[feats].to_numpy(np.float32), V.y.to_numpy(); V = V[["s1_idx", "c_idx"]].copy()
    rest_parts = sorted(glob.glob(f"{work}/feat_rest/part_*.parquet"))
    R = pd.concat([pd.read_parquet(p, columns=["s1_idx", "c_idx"]) for p in rest_parts], ignore_index=True)

    def predict_rest(model, extra=None):
        out, off = [], 0
        for p in rest_parts:
            Z = pd.read_parquet(p, columns=feats).to_numpy(np.float32)
            if extra is not None:
                Z = np.hstack([Z, extra[off:off + len(Z)]])
            out.append(model.predict(Z).astype(np.float32)); off += len(Z)
        return np.concatenate(out)

    # ---- stage 1: out-of-fold on TRAIN, full model for VALID and REST
    fold = (T.s1_idx.to_numpy().astype(np.int64) * 2654435761 % 2**32) % 2
    oof = np.zeros(len(T), np.float32)
    for f in (0, 1):
        m = fit(X[fold != f], y[fold != f], Xv, yv)
        oof[fold == f] = m.predict(X[fold == f])
        log(f"stage1 fold {f}: {m.best_iteration} rounds")
    m1 = fit(X, y, Xv, yv)
    p1v = m1.predict(Xv).astype(np.float32)
    p1r = predict_rest(m1)
    log(f"stage1 full: {m1.best_iteration} rounds")

    # ---- stage 2: context over the FULL candidate graph (every S1 predicted)
    A = pd.concat([T, V, R], ignore_index=True)
    nT, nV = len(T), len(V)
    C = context(A, np.concatenate([oof, p1v, p1r])).to_numpy(np.float32)
    ctx_cols = list(context(A.iloc[:1], np.zeros(1, np.float32)).columns)
    feats2 = feats + ctx_cols
    X = np.hstack([X, C[:nT]]); Xv = np.hstack([Xv, C[nT:nT + nV]])
    m2 = fit(X, y, Xv, yv)
    p2t = m2.predict(X).astype(np.float32)
    p2v = m2.predict(Xv).astype(np.float32)
    p2r = predict_rest(m2, C[nT + nV:])
    log(f"stage2: {m2.best_iteration} rounds")
    del X, C
    sl = slice(nT, nT + nV)   # VALID rows inside A
    full = {"stage1": np.concatenate([oof, p1v, p1r]), "stage2": np.concatenate([p2t, p2v, p2r])}

    # ---- decisions on VALID
    res = {}
    for name in ("stage1", "stage2"):
        pa = assign(A, full[name])[sl]   # one-owner rule against ALL competing S1s
        for th in np.arange(0.40, 0.91, 0.025):
            res[(name, "threshold", round(th, 3))] = macro(V, pa >= th, ids1[va_s1], ids1, idsc, truth)
        for miss in (0.0, 0.1, 0.2, 0.3):
            for floor in (0.05, 0.2, 0.35):
                res[(name, "expected", miss, floor)] = macro(V, decide_expected(V, pa, miss, floor),
                                                            ids1[va_s1], ids1, idsc, truth)
    ranked = sorted(res.items(), key=lambda kv: -kv[1])
    for k, v in ranked[:8]:
        log(k, round(v, 5))
    for name in ("stage1", "stage2"):
        bt = max((v, k) for k, v in res.items() if k[0] == name and k[1] == "threshold")
        log(f"best {name} global-threshold: {bt[1]} -> {bt[0]:.5f}")
    best_key, best_val = ranked[0]
    log("BEST", best_key, round(best_val, 5))

    m1.save_model(f"{work}/stage1.txt"); m2.save_model(f"{work}/stage2.txt")
    json.dump({"feats1": feats, "feats2": feats2, "decision": list(best_key), "valid_macro_f05": best_val,
               "stage1_rounds": m1.best_iteration, "stage2_rounds": m2.best_iteration},
              open(f"{work}/v2_config.json", "w"), indent=2)
    imp = pd.Series(m2.feature_importance("gain"), index=feats2)
    (imp / imp.sum()).sort_values(ascending=False).to_csv(f"{work}/v2_feature_importance.csv")


if __name__ == "__main__":
    main(sys.argv[1])
