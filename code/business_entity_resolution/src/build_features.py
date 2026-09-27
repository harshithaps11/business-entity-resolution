"""Stage 3a: featurise the final (pruned) candidate pairs and cache them on disk.

  train -> <work>/feat_train/part_*.parquet   (N_TRAIN_S1 entities, with labels)
           <work>/feat_valid/part_*.parquet   (the fixed validation entities)
           <work>/feat_rest/part_*.parquet    (all other train entities: only predicted,
                                               so competition features see every S1)
  test  -> <work>/feat_test/part_*.parquet    (every test entity)

Entity-level split (seed 42): validation = perm[150k:210k] (same as train.py),
training = the first N_TRAIN_S1 of the remaining entities.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import features as FE
from train import gt_keys, labels, load

N_TRAIN_S1 = 1_000_000
CHUNK_S1 = 120_000


def split_ids(n_s1):
    perm = np.random.default_rng(42).permutation(n_s1)
    va = perm[150_000:210_000]
    rest = np.concatenate([perm[:150_000], perm[210_000:]])
    return np.sort(rest[:N_TRAIN_S1]), np.sort(va), np.sort(rest[N_TRAIN_S1:])


def featurise(P, s1, c, idf, out_dir, keys=None):
    os.makedirs(out_dir, exist_ok=True)
    u = np.unique(P.s1_idx.to_numpy())
    for n, x in enumerate(range(0, len(u), CHUNK_S1)):
        path = f"{out_dir}/part_{n:03d}.parquet"
        if os.path.exists(path):
            continue
        Q = P[P.s1_idx.isin(u[x:x + CHUNK_S1])].reset_index(drop=True)
        X = FE.pair_features(Q, s1, c, idf)
        X["prune_p"] = Q.prune_p.to_numpy()
        X["gap_next"] = Q.gap_next.to_numpy(); X["gap_top"] = Q.gap_top.to_numpy()
        X["s1_idx"] = Q.s1_idx.to_numpy(); X["c_idx"] = Q.c_idx.to_numpy()
        if keys is not None:
            X["y"] = labels(Q, keys)
        X.to_parquet(path, index=False)
        print(f"  {path}: {len(X)} pairs", flush=True)


def main(work, split):
    t0 = time.time()
    s1, c = load(work, split)
    P = pd.read_parquet(f"{work}/{split}_pruned.parquet")
    idf = FE.build_idf([s1, c])
    print(f"loaded {len(P)} pairs ({time.time() - t0:.0f}s)", flush=True)
    if split == "train":
        keys = gt_keys(pd.read_parquet(f"{work}/train_gt.parquet"), s1, c)
        tr, va, rest = split_ids(len(s1))
        featurise(P[P.s1_idx.isin(va)], s1, c, idf, f"{work}/feat_valid", keys)
        featurise(P[P.s1_idx.isin(tr)], s1, c, idf, f"{work}/feat_train", keys)
        featurise(P[P.s1_idx.isin(rest)], s1, c, idf, f"{work}/feat_rest", keys)
    else:
        featurise(P, s1, c, idf, f"{work}/feat_test")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
