"""Stage 4 (v2): two-stage scoring of the cached test features -> submission files.

candidate_pairs.tsv = every pair in <work>/test_pruned.parquet (the set scored);
matching_results.tsv = pairs selected by stage 2 + one-owner rule + the decision
rule chosen on validation by experiment.py.
"""
import glob
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiment import assign, context, decide_expected
from predict import write_lists


def main(work, out_dir):
    t0 = time.time()
    cfg = json.load(open(f"{work}/v2_config.json"))
    m1, m2 = lgb.Booster(model_file=f"{work}/stage1.txt"), lgb.Booster(model_file=f"{work}/stage2.txt")
    parts = sorted(glob.glob(f"{work}/feat_test/part_*.parquet"))
    ids, p1 = [], []
    for p in parts:
        D = pd.read_parquet(p)
        ids.append(D[["s1_idx", "c_idx"]]); p1.append(m1.predict(D[cfg["feats1"]].to_numpy(np.float32)))
    A = pd.concat(ids, ignore_index=True); p1 = np.concatenate(p1).astype(np.float32)
    C = context(A, p1).to_numpy(np.float32)
    p2, off = [], 0
    for p in parts:
        Z = pd.read_parquet(p, columns=cfg["feats1"]).to_numpy(np.float32)
        p2.append(m2.predict(np.hstack([Z, C[off:off + len(Z)]]))); off += len(Z)
    p2 = np.concatenate(p2).astype(np.float32)
    print(f"scored {len(A)} pairs ({time.time() - t0:.0f}s)", flush=True)

    name, kind, *arg = cfg["decision"]
    pa = assign(A, p2 if name == "stage2" else p1)
    keep = pa >= arg[0] if kind == "threshold" else decide_expected(A, pa, arg[0], arg[1])

    s1 = pd.read_parquet(f"{work}/test_n1.parquet", columns=["id"]).id.to_numpy()
    c = np.concatenate([pd.read_parquet(f"{work}/test_n{i}.parquet", columns=["id"]).id.to_numpy() for i in (2, 3)])
    P = pd.read_parquet(f"{work}/test_pruned.parquet", columns=["s1_idx", "c_idx"])
    cand, match = {}, {}
    for a, b in zip(P.s1_idx.to_numpy(), P.c_idx.to_numpy()):
        cand.setdefault(s1[a], []).append(c[b])
    for a, b in zip(A.s1_idx.to_numpy()[keep], A.c_idx.to_numpy()[keep]):
        match.setdefault(s1[a], []).append(c[b])
    write_lists(f"{out_dir}/candidate_pairs.tsv", "candidate_entity_ids", s1, cand)
    write_lists(f"{out_dir}/matching_results.tsv", "matched_entity_ids", s1, match)
    n = sum(map(len, match.values()))
    print(f"done: {len(s1)} S1, {len(P)} candidates ({len(P) / len(s1):.2f}/S1), {n} matches, "
          f"{1 - len(match) / len(s1):.3f} singletons, decision={cfg['decision']} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
