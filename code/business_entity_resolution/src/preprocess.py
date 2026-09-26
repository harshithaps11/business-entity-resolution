"""Normalise every source table in parallel and write compact parquet files."""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import normalize as N


def _init(tpath):
    N.load_translit(tpath)


def _chunk(df):
    out = {k: [] for k in ("toks", "core", "legal", "alt", "f_native", "f_domain", "f_dba",
                           "awords", "anums", "state")}
    for name, addr in zip(df.business_name.values, df.business_address.values):
        toks, core, legal, alt, fl = N.normalize_name(name)
        w, nums, st = N.normalize_address(addr)
        out["toks"].append(" ".join(toks)); out["core"].append(" ".join(core))
        out["legal"].append(" ".join(legal)); out["alt"].append(" ".join(alt))
        out["f_native"].append(fl[0]); out["f_domain"].append(fl[1]); out["f_dba"].append(fl[2])
        out["awords"].append(" ".join(w)); out["anums"].append(" ".join(nums)); out["state"].append(st)
    r = pd.DataFrame(out, index=df.index)
    for c in ("f_native", "f_domain", "f_dba"):
        r[c] = r[c].astype(np.int8)
    return r


def preprocess(df, tpath, procs=None):
    procs = procs or max(1, (os.cpu_count() or 2) - 2)
    sub = df[["business_name", "business_address"]]
    bounds = np.linspace(0, len(sub), procs * 8 + 1).astype(int)
    chunks = [sub.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    with Pool(procs, initializer=_init, initargs=(tpath,)) as pool:
        parts = pool.map(_chunk, chunks)
    r = pd.concat(parts)
    r.insert(0, "id", df.entity_id.values)
    r.insert(1, "country", df.country.values)
    r["src"] = df.entity_id.str[1].astype(np.int8).values
    return r


if __name__ == "__main__":
    work = sys.argv[1]
    for split in ("train", "test"):
        for s in (1, 2, 3):
            df = pd.read_parquet(f"{work}/{split}_s{s}.parquet")
            preprocess(df, f"{work}/translit.json").to_parquet(f"{work}/{split}_n{s}.parquet", index=False)
            print(split, s, len(df), flush=True)
