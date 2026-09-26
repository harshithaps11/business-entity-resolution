"""Pairwise feature extraction for (Source 1, candidate) pairs.

Inputs are the normalised tables from preprocess.py plus the candidate pairs
from blocking.py. Features are country-agnostic (no country one-hot), so an
unseen country such as France is scored with the same model.
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

_G = {}

FEATURES = [
    # name
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "n_comp_ratio", "n_comp_partial",
    "n_jacc", "n_idf_ovl", "n_idf_miss", "n_idf_extra", "n_extra_cnt", "n_miss_cnt",
    "n_extra_maxidf", "n_miss_maxidf", "n_first_eq", "n_len_a", "n_len_b", "n_full_tset",
    "n_alt_ratio", "legal_both", "legal_eq", "legal_conflict",
    "b_native", "b_domain", "b_dba", "a_native",
    # address
    "a_jacc", "a_idf_ovl", "a_idf_miss", "a_idf_extra", "a_tset", "a_partial", "a_empty_b",
    "num_jacc", "num_common", "num_a", "num_b", "num_extra_b", "num_first_in_b",
    "num_near", "num_conflict", "state_eq", "state_missing",
    # blocking / competition
    "bscore", "rk", "bscore_rel", "n_cands", "rrk", "r_margin", "r_ncomp", "src",
]


IDF_BITS = 22
IDF_MASK = (1 << IDF_BITS) - 1


def build_idf(tables):
    """Hashed IDF arrays for name-core and address tokens across all sources.

    Stored as numpy arrays (not dicts) so forked workers share them cheaply."""
    dn = np.zeros(IDF_MASK + 1, dtype=np.int32)
    da = np.zeros(IDF_MASK + 1, dtype=np.int32)
    n = 0
    for t in tables:
        for col, arr in (("core", dn), ("awords", da)):
            h = [hash(x) & IDF_MASK for v in t[col].values for x in set(v.split())]
            np.add.at(arr, np.asarray(h, dtype=np.int64), 1)
        n += len(t)
    to_idf = lambda d: (np.log((n + 1) / (d + 1.0)) + 1.0).astype(np.float32)
    return to_idf(dn), to_idf(da)


def _num_near(na, nb):
    """House-number near-miss: same length, one digit different (5655 vs 5665)."""
    best = 0.0
    for x in na:
        for y in nb:
            if x == y:
                return 1.0
            if len(x) == len(y) and len(x) >= 2 and Levenshtein.distance(x, y) == 1:
                best = max(best, 0.5)
    return best


def _pair_feats(a, b, dn, da):
    core_a, core_b = a["core"], b["core"]
    ta, tb = set(core_a.split()), set(core_b.split())
    inter, only_a, only_b = ta & tb, ta - tb, tb - ta
    ia = {t: float(dn[hash(t) & IDF_MASK]) for t in ta | tb}
    s_int = sum(ia[t] for t in inter)
    s_a = sum(ia[t] for t in only_a)
    s_b = sum(ia[t] for t in only_b)
    comp_a, comp_b = core_a.replace(" ", ""), core_b.replace(" ", "")
    la, lb = set(a["legal"].split()), set(b["legal"].split())
    alt_b = b["alt"]

    wa, wb = set(a["awords"].split()), set(b["awords"].split())
    ainter = wa & wb
    ai = {t: float(da[hash(t) & IDF_MASK]) for t in wa | wb}
    as_int = sum(ai[t] for t in ainter)
    as_a = sum(ai[t] for t in wa - wb)
    as_b = sum(ai[t] for t in wb - wa)
    na_l, nb_l = a["anums"].split(), b["anums"].split()
    na, nb = set(na_l), set(nb_l)
    ninter = na & nb
    sa, sb = a["state"], b["state"]

    return [
        fuzz.ratio(core_a, core_b), fuzz.token_set_ratio(core_a, core_b),
        fuzz.token_sort_ratio(core_a, core_b), fuzz.partial_ratio(core_a, core_b),
        JaroWinkler.similarity(core_a, core_b), fuzz.ratio(comp_a, comp_b),
        fuzz.partial_ratio(comp_a, comp_b) if comp_a and comp_b else 0.0,
        len(inter) / max(1, len(ta | tb)),
        s_int / max(1e-6, s_int + s_a + s_b), s_a, s_b, len(only_b), len(only_a),
        max((ia[t] for t in only_b), default=0.0), max((ia[t] for t in only_a), default=0.0),
        float(core_a.split(" ", 1)[0] == core_b.split(" ", 1)[0]) if core_a and core_b else 0.0,
        len(ta), len(tb), fuzz.token_set_ratio(a["toks"], b["toks"]),
        fuzz.token_set_ratio(core_a, alt_b) if alt_b else -1.0,
        float(bool(la) and bool(lb)), float(la == lb), float(bool(la) and bool(lb) and not (la & lb)),
        b["f_native"], b["f_domain"], b["f_dba"], a["f_native"],
        len(ainter) / max(1, len(wa | wb)), as_int / max(1e-6, as_int + as_a + as_b), as_a, as_b,
        fuzz.token_set_ratio(a["awords"], b["awords"]),
        fuzz.partial_ratio(a["awords"], b["awords"]) if wa and wb else 0.0,
        float(not wb and not nb),
        len(ninter) / max(1, len(na | nb)) if (na or nb) else -1.0, len(ninter), len(na), len(nb),
        len(nb - na), float(bool(na_l) and na_l[0] in nb) if na_l else -1.0,
        _num_near(na, nb) if na and nb else -1.0,
        float(bool(na) and bool(nb) and not ninter),
        float(sa == sb) if sa and sb else -1.0, float(not sa or not sb),
    ]


_COLS = ("toks", "core", "legal", "alt", "f_native", "f_domain", "f_dba", "awords", "anums", "state")


_SEP = "\x1f"
_INT = {"f_native", "f_domain", "f_dba"}


def _pack(df):
    """One string per row (fields joined by \\x1f): ~10x fewer Python objects."""
    parts = [df[c].astype(str) for c in _COLS]
    out = parts[0]
    for p in parts[1:]:
        out = out + _SEP + p
    return out.to_numpy(dtype=object)


def _unpack(s):
    d = dict(zip(_COLS, s.split(_SEP)))
    for c in _INT:
        d[c] = int(d[c])
    return d


def _work(bounds):
    s, e = bounds
    P, A, C, dn, da = _G["P"], _G["A"], _G["C"], _G["dn"], _G["da"]
    ai, ci = P[0][s:e], P[1][s:e]
    out = np.empty((e - s, len(FEATURES) - 8), dtype=np.float32)
    for k in range(e - s):
        i, j = ai[k], ci[k]
        a, b = _unpack(A[i]), _unpack(C[j])
        out[k] = _pair_feats(a, b, dn, da)
    return out


def _group_stats(key, score):
    """Per-row rank (1 = best) within its key group, group max, 2nd max, group size."""
    order = np.lexsort((-score, key))
    ks = key[order]
    start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    size = np.diff(np.r_[start, len(ks)])
    pos = np.arange(len(ks)) - np.repeat(start, size)
    rank = np.empty(len(ks), np.float32); rank[order] = pos + 1
    top = score[order][start]
    sec = np.where(size > 1, score[order][np.minimum(start + 1, len(ks) - 1)], 0.0)
    gtop = np.empty(len(ks), np.float32); gtop[order] = np.repeat(top, size)
    gsec = np.empty(len(ks), np.float32); gsec[order] = np.repeat(sec, size)
    gsz = np.empty(len(ks), np.float32); gsz[order] = np.repeat(size, size)
    return rank, gtop, gsec, gsz


def competition_features(P: pd.DataFrame) -> pd.DataFrame:
    """Rank/margin features from the full candidate graph (needs all S1 rows).

    rk / bscore_rel / n_cands: position of the candidate among this S1's candidates.
    rrk / r_margin / r_ncomp: position of this S1 among all S1s that retrieved the
    same candidate -- exploits "each S2/S3 record belongs to at most one S1"."""
    s = P.bscore.to_numpy(np.float32)
    rk, top, _, n = _group_stats(P.s1_idx.to_numpy(), s)
    P = P.assign(rk=rk, bscore_rel=s / np.maximum(top, 1e-6), n_cands=n)
    rrk, rtop, rsec, rn = _group_stats(P.c_idx.to_numpy(), s)
    P["rrk"] = rrk
    P["r_margin"] = np.where(rrk == 1, s - rsec, s - rtop).astype(np.float32)
    P["r_ncomp"] = rn
    return P


def pair_features(P, s1n, cn, idf, procs=None, chunk=20000):
    """P: competition_features() output restricted to the rows to featurise."""
    procs = procs or max(1, (os.cpu_count() or 2) - 2)
    dn, da = idf
    ua, uc = np.unique(P.s1_idx.to_numpy()), np.unique(P.c_idx.to_numpy())
    # only pack rows referenced by these pairs; re-index locally
    li = np.searchsorted(ua, P.s1_idx.to_numpy()).astype(np.int32)
    lj = np.searchsorted(uc, P.c_idx.to_numpy()).astype(np.int32)
    _G.update(P=(li, lj), A=_pack(s1n.iloc[ua]), C=_pack(cn.iloc[uc]), dn=dn, da=da)
    bnds = [(x, min(x + chunk, len(P))) for x in range(0, len(P), chunk)]
    with Pool(procs) as pool:
        X = np.concatenate(pool.map(_work, bnds))
    _G.clear()
    F = pd.DataFrame(X, columns=FEATURES[:-8], index=P.index)
    for c in ("bscore", "rk", "bscore_rel", "n_cands", "rrk", "r_margin", "r_ncomp"):
        F[c] = P[c].values.astype(np.float32)
    F["src"] = cn.src.values[P.c_idx.values].astype(np.float32)
    return F[FEATURES]
