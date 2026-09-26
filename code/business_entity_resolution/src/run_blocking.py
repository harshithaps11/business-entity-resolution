"""Stage 2: candidate generation for one split -> <work>/<split>_pairs.parquet.

Candidate table = Source 2 rows followed by Source 3 rows (positional c_idx).
Each country block is checkpointed to <work>/<split>_pairs_<country>.parquet so
an interrupted run resumes where it stopped.
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd

from blocking import block

COLS = ["id", "country", "core", "awords", "anums"]

if __name__ == "__main__":
    work, split = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    t = time.time()
    s1 = pd.read_parquet(f"{work}/{split}_n1.parquet", columns=COLS)
    c = pd.concat([pd.read_parquet(f"{work}/{split}_n{i}.parquet", columns=COLS) for i in (2, 3)],
                  ignore_index=True)
    parts = []
    for country in pd.unique(s1.country):
        path = f"{work}/{split}_pairs_{re.sub(r'[^A-Za-z0-9]+', '_', country)}.parquet"
        if not os.path.exists(path):
            ia = np.flatnonzero(s1.country.values == country)
            P = block(s1.iloc[ia], c, k=k)          # c_idx is already global
            P["s1_idx"] = ia[P.s1_idx.to_numpy()]   # map back to global S1 index
            P.to_parquet(path, index=False)
        parts.append(pd.read_parquet(path))
    P = pd.concat(parts, ignore_index=True)
    P.to_parquet(f"{work}/{split}_pairs.parquet", index=False)
    print(f"{split}: {len(P)} pairs for {len(s1)} S1 in {time.time() - t:.0f}s", flush=True)
