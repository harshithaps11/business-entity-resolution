"""Stage 4: prune + score test candidates and write the two submission files.

retrieval pairs -> learned pruner -> FINAL candidate set (= candidate_pairs.tsv)
-> full matching model -> decision rule -> matching_results.tsv.
Pairs are featurised in chunks of Source 1 entities so memory stays bounded.
"""
import gc
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

import features as FE
import prune as PR
from decide import decide
from train import cand_src


def write_lists(path, col, s1_ids, mapping):
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for i in s1_ids:
            f.write(f"{i}\t{','.join(mapping.get(i, ()))}\n")


def main(work, out_dir, chunk_s1=150_000):
    t0 = time.time()
    model = joblib.load(f"{work}/model.joblib")
    cfg = json.load(open(f"{work}/model_config.json"))
    pruner = joblib.load(f"{work}/pruner.joblib")
    P = FE.competition_features(pd.read_parquet(f"{work}/test_pairs.parquet"))
    P = PR.prune_frame(P, cand_src(work, "test"))
    n_ret = len(P)
    q = pruner.predict_proba(P[PR.PRUNE_FEATURES])[:, 1]
    keep = q >= cfg["prune_threshold"]
    P = P[keep].assign(prune_p=q[keep]).sort_values("s1_idx", kind="stable").reset_index(drop=True)
    del q, keep; gc.collect()
    s1 = pd.read_parquet(f"{work}/test_n1.parquet")
    c = pd.concat([pd.read_parquet(f"{work}/test_n{i}.parquet") for i in (2, 3)], ignore_index=True)
    print(f"retrieved {n_ret} ({n_ret / len(s1):.1f}/S1) -> final candidates {len(P)} "
          f"({len(P) / len(s1):.2f}/S1)", flush=True)
    idf = FE.build_idf([s1, c])
    print(f"loaded {len(P)} pairs ({time.time() - t0:.0f}s)", flush=True)

    p = np.empty(len(P), dtype=np.float32)
    cuts = np.searchsorted(P.s1_idx.to_numpy(), np.arange(0, len(s1) + chunk_s1, chunk_s1))
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a == b:
            continue
        X = FE.pair_features(P.iloc[a:b], s1, c, idf)
        X["prune_p"] = P.prune_p.to_numpy()[a:b]
        p[a:b] = model.predict_proba(X[cfg["features"]])[:, 1]
        print(f"  scored {b}/{len(P)} ({time.time() - t0:.0f}s)", flush=True)

    ids1, idsc = s1.id.to_numpy(), c.id.to_numpy()
    cand = {}
    for a, b in zip(P.s1_idx.to_numpy(), P.c_idx.to_numpy()):
        cand.setdefault(ids1[a], []).append(idsc[b])
    matches = decide(P[["s1_idx", "c_idx"]].assign(p=p), s1, c, cfg["threshold"])
    write_lists(f"{out_dir}/candidate_pairs.tsv", "candidate_entity_ids", ids1, cand)
    write_lists(f"{out_dir}/matching_results.tsv", "matched_entity_ids", ids1, matches)
    n_match = sum(len(v) for v in matches.values())
    print(f"done: {len(ids1)} S1, {n_match} matches, "
          f"{(len(ids1) - len(matches)) / len(ids1):.3f} predicted singletons ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
