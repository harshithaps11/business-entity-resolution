"""Stage 2c: apply the trained pruner to a split, one country block at a time.

The retrieval/competition features (ranks, maxima, second maxima, counts per
S1 and per candidate) are window functions over up to ~66M pairs per block.
They are computed with DuckDB under a hard memory cap (spilling to disk), so
this step runs in ~3 GB regardless of block size. Definitions match
features.competition_features + prune.prune_frame (ties broken arbitrarily).
Output: <work>/<split>_pruned.parquet.
"""
import glob
import json
import os
import sys
import time

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import prune as PR

KEEP = ["s1_idx", "c_idx", "bscore", "rk", "bscore_rel", "n_cands", "rrk", "r_margin", "r_ncomp",
        "gap_next", "gap_top", "prune_p"]

SQL = """
COPY (
  WITH a AS (
    SELECT s1_idx, c_idx, bscore,
      row_number() OVER (PARTITION BY s1_idx ORDER BY bscore DESC) AS rk,
      max(bscore)  OVER (PARTITION BY s1_idx) AS top,
      count(*)     OVER (PARTITION BY s1_idx) AS n_cands,
      row_number() OVER (PARTITION BY c_idx ORDER BY bscore DESC) AS rrk,
      max(bscore)  OVER (PARTITION BY c_idx) AS rtop,
      count(*)     OVER (PARTITION BY c_idx) AS r_ncomp
    FROM read_parquet('{src}')
  ), b AS (
    SELECT *,
      coalesce(lead(bscore) OVER (PARTITION BY s1_idx ORDER BY rk), 0) AS nxt,
      coalesce(max(CASE WHEN rrk = 2 THEN bscore END) OVER (PARTITION BY c_idx), 0) AS rsec
    FROM a
  )
  SELECT s1_idx, c_idx,
    bscore::FLOAT AS bscore, rk::FLOAT AS rk,
    (bscore / greatest(top, 1e-6))::FLOAT AS bscore_rel, n_cands::FLOAT AS n_cands,
    rrk::FLOAT AS rrk,
    (CASE WHEN rrk = 1 THEN bscore - rsec ELSE bscore - rtop END)::FLOAT AS r_margin,
    r_ncomp::FLOAT AS r_ncomp,
    (bscore - nxt)::FLOAT AS gap_next, (top - bscore)::FLOAT AS gap_top,
    (CASE WHEN c_idx < {n_s2} THEN 2 ELSE 3 END)::FLOAT AS src
  FROM b
) TO '{dst}' (FORMAT parquet)
"""


def block(path, n_s2, pruner, th, tmp):
    feat = f"{tmp}/feat.parquet"
    con = duckdb.connect()
    con.execute(f"SET memory_limit='2500MB'; SET threads=8; SET temp_directory='{tmp}';"
                "SET preserve_insertion_order=false")
    con.execute(SQL.format(src=path, dst=feat, n_s2=n_s2))
    con.close()
    parts, n = [], 0
    for batch in pq.ParquetFile(feat).iter_batches(batch_size=2_000_000):
        D = batch.to_pandas()
        n += len(D)
        q = pruner.predict_proba(D[PR.PRUNE_FEATURES])[:, 1].astype(np.float32)
        k = q >= th
        parts.append(D.loc[k, KEEP[:-1]].assign(prune_p=q[k]))
    os.remove(feat)
    return n, pd.concat(parts, ignore_index=True)


def main(work, split):
    t0 = time.time()
    cfg = json.load(open(f"{work}/model_config.json"))
    pruner = joblib.load(f"{work}/pruner.joblib")
    n_s2 = pq.ParquetFile(f"{work}/{split}_n2.parquet").metadata.num_rows
    tmp = f"{work}/_prune_tmp"; os.makedirs(tmp, exist_ok=True)
    parts = []
    for path in sorted(glob.glob(f"{work}/{split}_pairs_*.parquet")):
        part = path.replace("_pairs_", "_pruned_")
        if not os.path.exists(part):
            n, out = block(path, n_s2, pruner, cfg["prune_threshold"], tmp)
            out.to_parquet(part, index=False)
            print(f"  {path}: {n} -> {len(out)} ({time.time() - t0:.0f}s)", flush=True)
            del out
        parts.append(pd.read_parquet(part))
    out = pd.concat(parts, ignore_index=True).sort_values(["s1_idx", "rk"]).reset_index(drop=True)
    out.to_parquet(f"{work}/{split}_pruned.parquet", index=False)
    print(f"{split}: {len(out)} pruned pairs ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
