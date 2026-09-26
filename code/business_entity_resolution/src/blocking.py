"""Candidate generation (blocking).

Each record becomes a sparse TF-IDF vector over field-prefixed tokens:
    n:<name core token>   name words (after normalisation / transliteration)
    c:<compact name 4-gram> character 4-grams of the space-free core name
                           (catches "kempgloba" vs "kemp globa", typos)
    a:<address word>      street / locality words
    d:<number>            house / plot / unit numbers
Blocks are the distinct country labels found in the data (open set). Within a
block S1 x Candidates^T is computed as a sparse product in parallel row chunks
and the top-K candidates per S1 record are kept. Very frequent tokens are left
out of the product (they add fan-out, not discrimination).
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp

FIELD_W = {"n": 1.0, "c": 0.35, "a": 0.55, "d": 0.7,
           "N": 1.2, "K": 1.2, "A": 0.6, "D": 0.8}


NBITS = 23
MASK = (1 << NBITS) - 1
_FIELD_CODE = {"n": 0, "c": 1, "a": 2, "d": 3, "N": 4, "K": 5, "A": 6, "D": 7}


def _hash_rows(df: pd.DataFrame, use_ngrams: bool):
    """Hash field-prefixed tokens straight to column ids (no vocabulary kept).

    Returns (cols int32, indptr int64, field int8 per entry)."""
    cols, fld, ptr = [], [], [0]
    for core, aw, an in zip(df.core.values, df.awords.values, df.anums.values):
        toks = {("n", t) for t in core.split()}
        if use_ngrams:
            comp = core.replace(" ", "")
            toks |= {("c", comp[i:i + 4]) for i in range(max(1, len(comp) - 3))}
        ct, at, nt = core.split(), aw.split(), an.split()
        # compound keys: rare even when their parts are common
        toks |= {("N", " ".join(sorted(p))) for p in zip(ct, ct[1:])}      # name bigram
        toks.add(("K", "".join(ct)))                                         # compact name
        toks |= {("a", t) for t in at}
        toks |= {("A", " ".join(sorted(p))) for p in zip(at, at[1:])}      # address bigram
        toks |= {("d", t) for t in nt}
        toks |= {("D", n + "|" + w) for n in nt[:3] for w in at if len(w) > 2}  # number x word
        hs = {}
        for f, t in toks:
            hs[hash(f + ":" + t) & MASK] = _FIELD_CODE[f]
        cols.extend(hs.keys()); fld.extend(hs.values()); ptr.append(len(cols))
    return (np.asarray(cols, dtype=np.int32), np.asarray(ptr, dtype=np.int64),
            np.asarray(fld, dtype=np.int8))


def _vectorize(dfa, dfb, max_df, use_ngrams):
    """Shared hashed TF-IDF space; rows L2-normalised; tokens with df>max_df dropped."""
    parts = []
    for df in (dfa, dfb):
        for x in range(0, len(df), 200_000):
            parts.append(_hash_rows(df.iloc[x:x + 200_000], use_ngrams))
    cols = np.concatenate([p[0] for p in parts]); fld = np.concatenate([p[2] for p in parts])
    ptr, off = [np.zeros(1, np.int64)], 0
    for p in parts:
        ptr.append(p[1][1:] + off); off += p[1][-1]
    ptr = np.concatenate(ptr); del parts
    n = len(ptr) - 1
    df = np.bincount(cols, minlength=MASK + 1)
    idf = (np.log((n + 1) / (df + 1)) + 1.0).astype(np.float32)
    fw = np.array([FIELD_W[k] for k in _FIELD_CODE], dtype=np.float32)
    w = idf[cols] * fw[fld]
    w[df[cols] > max_df] = 0.0
    del fld
    M = sp.csr_matrix((w, cols, ptr), shape=(n, MASK + 1)); del w, cols
    M.eliminate_zeros()
    norm = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()) + 1e-9
    M = (sp.diags((1.0 / norm).astype(np.float32)) @ M).tocsr()
    na = len(dfa)
    return M[:na], M[na:]


_G = {}


def _topk_chunk(bounds):
    a, b = bounds
    A, BT, k = _G["A"], _G["BT"], _G["k"]
    R = (A[a:b] @ BT).tocsr()
    out_r, out_c, out_s = [], [], []
    for i in range(R.shape[0]):
        s, e = R.indptr[i], R.indptr[i + 1]
        if s == e:
            continue
        d, c = R.data[s:e], R.indices[s:e]
        if e - s > k:
            sel = np.argpartition(-d, k)[:k]
            d, c = d[sel], c[sel]
        out_r.append(np.full(len(c), a + i, dtype=np.int32))
        out_c.append(c.astype(np.int32))
        out_s.append(d.astype(np.float32))
    if not out_r:
        return (np.empty(0, np.int32),) * 2 + (np.empty(0, np.float32),)
    return np.concatenate(out_r), np.concatenate(out_c), np.concatenate(out_s)


def block(s1: pd.DataFrame, cand: pd.DataFrame, k: int = 40, max_df: int = 20000,
          use_ngrams: bool = False, procs: int | None = None, chunk: int = 64) -> pd.DataFrame:
    """Return DataFrame(s1_idx, c_idx, bscore) with positional indices into s1/cand."""
    procs = procs or max(1, min(10, (os.cpu_count() or 2) - 4))
    parts = []
    for country in pd.unique(s1.country):
        ia = np.flatnonzero(s1.country.values == country)
        ib = np.flatnonzero(cand.country.values == country)
        if len(ia) == 0 or len(ib) == 0:
            continue
        A, B = _vectorize(s1.iloc[ia], cand.iloc[ib], max_df, use_ngrams)
        _G.update(A=A, BT=B.T.tocsr(), k=k); del B
        bnds = [(x, min(x + chunk, len(ia))) for x in range(0, len(ia), chunk)]
        with Pool(procs) as pool:  # fork: workers share A/BT copy-on-write
            res = pool.map(_topk_chunk, bnds)
        r = np.concatenate([x[0] for x in res]); c = np.concatenate([x[1] for x in res])
        s = np.concatenate([x[2] for x in res])
        parts.append(pd.DataFrame({"s1_idx": ia[r], "c_idx": ib[c], "bscore": s}))
        _G.clear()
        print(f"  block[{country}] s1={len(ia)} cand={len(ib)} pairs={len(r)}", flush=True)
    return pd.concat(parts, ignore_index=True)
