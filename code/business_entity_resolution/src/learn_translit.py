"""Learn native-script -> Latin token maps from training ground-truth pairs.

Names: when a matched S2/S3 name is in a non-Latin script and has the same
number of tokens as its S1 name, tokens are aligned positionally.
Addresses: each non-Latin address segment (typically the state) is aligned with
the state code parsed from the matched S1 address.
Only training data is used.
"""
import collections
import json
import re
import sys

import pandas as pd

from normalize import NON_ASCII, NONALNUM_RE, LEGAL_CANON, STATE_RE, STATE_CANON, translit

NATIVE_SEG = re.compile(r"[^\x00-\x7f][^,]*")


def main(work, out_path):
    gt = pd.read_parquet(f"{work}/train_gt.parquet")
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex[ex.m.str.len() > 0]
    s1 = pd.read_parquet(f"{work}/train_s1.parquet").set_index("entity_id")
    cand = pd.concat([pd.read_parquet(f"{work}/train_s{k}.parquet") for k in (2, 3)]).set_index("entity_id")
    ex = ex.join(s1[["business_name", "business_address"]], on="source1_entity_id")
    ex = ex.join(cand[["business_name", "business_address"]], on="m", rsuffix="_c")

    name_counts = collections.defaultdict(collections.Counter)
    nat = ex[ex.business_name_c.str.contains(NON_ASCII)]
    for a, b in zip(nat.business_name, nat.business_name_c):
        if not NON_ASCII.search(b) or NON_ASCII.search(a):
            continue
        ta = [LEGAL_CANON.get(t, t) for t in NONALNUM_RE.split(a.lower()) if t]
        tb = [t for t in NONALNUM_RE.split(translit(b)) if t]
        if len(ta) == len(tb):
            for x, y in zip(tb, ta):
                name_counts[x][y] += 1

    addr_counts = collections.defaultdict(collections.Counter)
    nat = ex[ex.business_address_c.str.contains(NON_ASCII, na=False)]
    for a, b in zip(nat.business_address, nat.business_address_c):
        st = STATE_RE.findall(a.lower())
        if not st:
            continue
        code = STATE_CANON[st[-1]]
        for seg in NATIVE_SEG.findall(b):
            key = " ".join(t for t in NONALNUM_RE.split(translit(seg)) if t)
            if key:
                addr_counts[key][code] += 1

    def select(counts, min_n, min_share):
        out = {}
        for k, c in counts.items():
            (v, n), tot = c.most_common(1)[0], sum(c.values())
            if n >= min_n and n / tot >= min_share and v != k:
                out[k] = v
        return out

    name_map = select(name_counts, 3, 0.6)
    addr_map = select(addr_counts, 5, 0.8)
    json.dump({"name": name_map, "addr": addr_map}, open(out_path, "w"), ensure_ascii=False)
    print(f"name map {len(name_map)} tokens, addr map {len(addr_map)} segments")
    for k in list(name_map)[:15]:
        print("  ", k, "->", name_map[k])
    for k in list(addr_map)[:10]:
        print("  ", k, "->", addr_map[k])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
