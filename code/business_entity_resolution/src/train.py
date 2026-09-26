"""Stage 3: candidate pruner + matching model, selected on a held-out split.

S1 entities are split into disjoint TRAIN / VALID groups (entity-level split,
so nothing leaks between them). On VALID we report:
  * retrieval recall ceiling (share of true pairs present after top-K retrieval),
  * pruning trade-off (candidates per S1 vs. share of true pairs kept),
  * pair-level AUC / AP / accuracy and end-to-end macro F0.5 per matching model.
The best matcher (by VALID macro F0.5), its threshold and the pruner are saved.
"""
import json
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as FE
import prune as PR
from decide import decide, macro_f05

N_TRAIN_S1, N_VALID_S1 = 150_000, 60_000
PRUNE_TARGET = 0.99          # share of retrieved true pairs the pruner must keep
PRUNE_SWEEP = [0.95, 0.97, 0.98, 0.99, 0.995, 0.998]


def load(work, split):
    s1 = pd.read_parquet(f"{work}/{split}_n1.parquet")
    c = pd.concat([pd.read_parquet(f"{work}/{split}_n{i}.parquet") for i in (2, 3)], ignore_index=True)
    return s1, c


def cand_src(work, split):
    """Source (2 or 3) of every candidate row, in the S2-then-S3 order used by c_idx."""
    return np.concatenate([pd.read_parquet(f"{work}/{split}_n{i}.parquet", columns=["src"]).src.to_numpy()
                           for i in (2, 3)])


def gt_keys(gt, s1, c):
    """Ground-truth pairs as int64 keys s1_idx * 2^32 + c_idx."""
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex[ex.m.str.len() > 0]
    i = pd.Index(s1.id).get_indexer(ex.source1_entity_id)
    j = pd.Index(c.id).get_indexer(ex.m)
    return np.unique(i.astype(np.int64) << 32 | j.astype(np.int64))


def labels(P, keys):
    k = P.s1_idx.to_numpy().astype(np.int64) << 32 | P.c_idx.to_numpy().astype(np.int64)
    return np.isin(k, keys).astype(np.int8)


def truth_sets(gt, ids):
    g = gt[gt.source1_entity_id.isin(ids)]
    return {r: set(filter(None, m.split(","))) for r, m in zip(g.source1_entity_id, g.matched_entity_ids)}


def tune_threshold(Pva, p, s1, c, va_ids, truth, grid=np.arange(0.30, 0.96, 0.025)):
    res = [(float(th), macro_f05(decide(Pva.assign(p=p), s1, c, th), va_ids, truth)) for th in grid]
    return max(res, key=lambda x: x[1])


def main(work):
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    # graph-level features on the full pair set first (needs every S1), then keep
    # only the sampled entities -- text tables are loaded afterwards to cap memory
    n_s1 = len(pd.read_parquet(f"{work}/train_n1.parquet", columns=["src"]))
    c_src = cand_src(work, "train")
    P = FE.competition_features(pd.read_parquet(f"{work}/train_pairs.parquet"))
    P = PR.prune_frame(P, c_src)
    log(f"retrieved pairs {len(P)} = {len(P) / n_s1:.1f} per S1")

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_s1)
    tr_idx, va_idx = perm[:N_TRAIN_S1], perm[N_TRAIN_S1:N_TRAIN_S1 + N_VALID_S1]
    Ptr = P[P.s1_idx.isin(tr_idx)].reset_index(drop=True)
    Pva = P[P.s1_idx.isin(va_idx)].reset_index(drop=True)
    del P
    s1, c = load(work, "train")
    gt = pd.read_parquet(f"{work}/train_gt.parquet")
    keys = gt_keys(gt, s1, c)
    ytr_all, yva_all = labels(Ptr, keys), labels(Pva, keys)
    va_ids = s1.id.to_numpy()[va_idx]
    truth = truth_sets(gt, va_ids)
    n_true = sum(len(v) for v in truth.values())
    report = {"retrieval": dict(pairs_per_s1=len(Pva) / len(va_idx),
                                recall_ceiling=float(yva_all.sum() / n_true))}
    log("retrieval", report["retrieval"])

    # ---------------------------------------------------------------- pruner
    pruner = PR.fit_pruner(Ptr, ytr_all, Pva, yva_all)
    q_tr = pruner.predict_proba(Ptr[PR.PRUNE_FEATURES])[:, 1]
    q_va = pruner.predict_proba(Pva[PR.PRUNE_FEATURES])[:, 1]
    sweep = {}
    for tgt in PRUNE_SWEEP:
        _, st = PR.choose_threshold(q_va, yva_all, len(va_idx), tgt)
        st["recall_ceiling"] = st["kept_recall"] * report["retrieval"]["recall_ceiling"]
        sweep[tgt] = st
        log(f"prune target {tgt}: {st}")
    th_prune = sweep[PRUNE_TARGET]["threshold"]
    loose = sweep[max(PRUNE_SWEEP)]["threshold"]

    # train matcher on the pruned distribution; featurise VALID at the loosest
    # setting so the pruning sweep can be scored end-to-end
    ktr = q_tr >= th_prune
    Ptr, ytr = Ptr[ktr].reset_index(drop=True), ytr_all[ktr]
    kva = q_va >= loose
    Pva, yva, q_va = Pva[kva].reset_index(drop=True), yva_all[kva], q_va[kva]
    idf = FE.build_idf([s1, c])
    Xtr = FE.pair_features(Ptr, s1, c, idf)
    log(f"train feats {Xtr.shape} pos={ytr.mean():.3f}")
    Xva = FE.pair_features(Pva, s1, c, idf)
    log(f"valid feats {Xva.shape}")
    Xva["prune_p"] = q_va; Xtr["prune_p"] = q_tr[ktr]
    feats = FE.FEATURES + ["prune_p"]

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                                             early_stopping=True, max_iter=40,
                                                             batch_size=4096, random_state=0)),
        "lightgbm": lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=127,
                                       min_child_samples=50, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1),
    }
    in_main = q_va >= th_prune
    Pm, Xm, ym = Pva[in_main], Xva[in_main], yva[in_main]
    sub = rng.choice(len(Xtr), size=min(len(Xtr), 3_000_000), replace=False)  # for slow models
    report["models"], probs = {}, {}
    for name, m in models.items():
        t = time.time()
        if name == "lightgbm":
            m.fit(Xtr[feats], ytr, eval_set=[(Xm[feats], ym)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        else:
            m.fit(Xtr[feats].iloc[sub], ytr[sub])
        p = m.predict_proba(Xva[feats])[:, 1]
        probs[name] = p
        pm = p[in_main]
        th, f = tune_threshold(Pm, pm, s1, c, va_ids, truth)
        report["models"][name] = dict(auc=roc_auc_score(ym, pm), ap=average_precision_score(ym, pm),
                                      pair_acc=accuracy_score(ym, pm > th), threshold=th,
                                      macro_f05=f, fit_s=time.time() - t)
        log(name, {k: round(v, 4) for k, v in report["models"][name].items()})

    best = max(report["models"], key=lambda k: report["models"][k]["macro_f05"])
    # end-to-end effect of the pruning level with the chosen matcher
    for tgt, st in sweep.items():
        k = q_va >= st["threshold"]
        st["macro_f05"] = tune_threshold(Pva[k], probs[best][k], s1, c, va_ids, truth)[1]
        log(f"prune {tgt}: avg_cands={st['avg_cands']:.2f} F0.5={st['macro_f05']:.4f}")
    report["prune_sweep"] = sweep
    report["best"] = best
    log("BEST", best, report["models"][best])

    joblib.dump(pruner, f"{work}/pruner.joblib")
    joblib.dump(models[best], f"{work}/model.joblib")
    json.dump(report, open(f"{work}/model_report.json", "w"), indent=2, default=float)
    json.dump({"model": best, "threshold": report["models"][best]["threshold"],
               "prune_threshold": th_prune, "prune_target": PRUNE_TARGET, "features": feats},
              open(f"{work}/model_config.json", "w"), indent=2)
    if "lightgbm" in models:
        imp = pd.Series(models["lightgbm"].booster_.feature_importance("gain"), index=feats)
        (imp / imp.sum()).sort_values(ascending=False).to_csv(f"{work}/feature_importance.csv")


if __name__ == "__main__":
    main(sys.argv[1])
