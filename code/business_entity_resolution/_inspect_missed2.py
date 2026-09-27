"""Inspect the true matches STILL missed after combined (weak+strong) blocking.

These ~8.5% decide whether more blocking keys can help or we need a different
method. For a sample of S1 entities, we take the existing weak candidates + a
fresh strong-key pass, find true matches not recovered, and PRINT the actual
name/address text of S1 vs the missed match so we can see the failure mode.

Memory-safe: strong-key index is hashed int64, strong-keys only (same as v2).

Run:  python _inspect_missed2.py --sample 20000 --show 40
"""
from __future__ import annotations
import argparse, array, sys, time, hashlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C
import normalize as N

DELIM = "\t"

def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little", signed=True)

def strong_keys(name_raw, addr_raw):
    nm = N.normalize_name(name_raw); ad = N.normalize_address(addr_raw)
    ks = []
    if nm.core: ks.append(_h("NC:" + nm.core))
    if nm.tokens: ks.append(_h("NS:" + "_".join(sorted(nm.tokens))))
    if nm.phonetic: ks.append(_h("NP:" + "".join(nm.phonetic.split())))
    if ad.present:
        if ad.pin: ks.append(_h("Z:" + ad.pin))
        if ad.full and len(ad.full) >= 10: ks.append(_h("AF:" + ad.full))
    return ks

def stream(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            if not line.strip(): continue
            p = line.rstrip("\n").split(DELIM)
            yield p[0], (p[1] if len(p)>1 else ""), (p[2] if len(p)>2 else ""), (p[3] if len(p)>3 else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()

    gt = {}
    with open(C.TRAIN_GROUND_TRUTH, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            gt[s1] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()

    # sample S1 with gt
    sample = set()
    for eid,*_ in stream(C.TRAIN_SOURCE1):
        if eid in gt and gt[eid]:
            sample.add(eid)
            if len(sample) >= args.sample: break

    # existing weak candidates for sample
    existing = {}
    with open(C.WORK_DIR / "candidates_train.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1,_,rest = line.rstrip("\n").partition(DELIM)
            if s1 in sample:
                existing[s1] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()
    print(f"  sample={len(sample):,} loaded existing ({time.time()-t0:.0f}s)", flush=True)

    # strong index over S2/S3, but only keep posting text for needed matches
    ids = []; index = defaultdict(lambda: array.array("i"))
    n=0
    for path in (C.TRAIN_SOURCE2, C.TRAIN_SOURCE3):
        for eid,name,addr,cc in stream(path):
            idx=len(ids); ids.append(eid)
            for k in strong_keys(name,addr): index[k].append(idx)
            n+=1
            if n%2_000_000==0: print(f"  idx {n:,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  strong index built {n:,} ({time.time()-t0:.0f}s)", flush=True)

    # strong candidates per sample S1
    strong_cand = {}
    for eid,name,addr,cc in stream(C.TRAIN_SOURCE1):
        if eid in sample:
            hits=set()
            for k in strong_keys(name,addr):
                pl=index.get(k)
                if pl: hits.update(pl)
            strong_cand[eid]=set(ids[i] for i in hits)

    # find missed = truth - (existing | strong)
    missed_pairs = []   # (s1, missed_id)
    for s1 in sample:
        combined = existing.get(s1,set()) | strong_cand.get(s1,set())
        for mid in (gt[s1] - combined):
            missed_pairs.append((s1, mid))
    missed_ids = set(m for _,m in missed_pairs)
    print(f"  missed pairs: {len(missed_pairs):,}  distinct missed S2/S3: {len(missed_ids):,}", flush=True)

    # fetch text for the s1 side and missed side to display
    want_s1 = set(s for s,_ in missed_pairs)
    s1txt={}; mtxt={}
    for eid,name,addr,cc in stream(C.TRAIN_SOURCE1):
        if eid in want_s1: s1txt[eid]=(name,addr,cc)
    for path in (C.TRAIN_SOURCE2, C.TRAIN_SOURCE3):
        for eid,name,addr,cc in stream(path):
            if eid in missed_ids: mtxt[eid]=(name,addr,cc)

    print(f"\n===== SAMPLE OF MISSED TRUE MATCHES (first {args.show}) =====")
    shown=0
    for s1,mid in missed_pairs:
        if s1 not in s1txt or mid not in mtxt: continue
        n1,a1,c1 = s1txt[s1]; n2,a2,c2 = mtxt[mid]
        print(f"\n[{s1} ({c1})]  name={n1!r}")
        print(f"     addr={a1!r}")
        print(f"  MISSED {mid} ({c2}) name={n2!r}")
        print(f"     addr={a2!r}")
        shown+=1
        if shown>=args.show: break
    print(f"\n[done {time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
