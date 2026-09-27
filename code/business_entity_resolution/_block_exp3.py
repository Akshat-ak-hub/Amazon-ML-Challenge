"""Blocking recall EXPERIMENT v3 — target the ACTUAL failure modes seen in
_missed_out.txt. Memory-safe (hashed strong keys only).

Observed failure modes (all recoverable):
  1. Address number is the anchor even when name is destroyed. But numbers are
     noisy: leading zeros ('0321' vs '321'), off-by-one digits, reordering.
     -> KEY: normalized house-number (strip leading zeros) + first alpha street
        token + locality token. Robust to reordering.
  2. Domain names: 'moorebitwise.com' == 'Moore Bitwise Inc'. -> strip TLD/.com,
     the concatenated alpha string matches the concatenated name core.
  3. Native-script names but matching address -> address keys catch them.
  4. Partial names + shared address -> address keys catch them.

We test how much recall the NEW address/domain keys add on top of existing+strong.

Run:  python _block_exp3.py --sample 20000
"""
from __future__ import annotations
import argparse, array, sys, time, hashlib, re
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C
import normalize as N

DELIM = "\t"

def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little", signed=True)

_NUM = re.compile(r"\d+")

def new_keys(name_raw, addr_raw):
    """Address-anchored + domain keys — LEAN: at most ~3 keys per record to
    avoid index explosion (v3a hit 33M keys and swapped)."""
    nm = N.normalize_name(name_raw)
    ad = N.normalize_address(addr_raw)
    ks = []

    # ---- DOMAIN normalization: concatenated alpha of core name (1 key) ----
    concat = "".join(nm.tokens)                       # 'moorebitwise'
    if len(concat) >= 6:
        ks.append(_h("DC:" + concat))                 # 'moorebitwise.com' -> same concat

    # ---- ADDRESS: ONE strong number-anchored key (reordering robust) ----
    if ad.present and ad.tokens:
        # the FIRST meaningful house number, leading zeros stripped
        first_num = ""
        for t in ad.tokens:
            for mnum in _NUM.findall(t):
                z = mnum.lstrip("0") or "0"
                if len(z) >= 2:
                    first_num = z
                    break
            if first_num:
                break
        # the longest alpha token = most distinctive locality/street word
        alphas = sorted((t for t in ad.tokens if t.isalpha() and len(t) >= 4),
                        key=len, reverse=True)
        if first_num and alphas:
            # number + single most-distinctive token (1 key, order independent)
            ks.append(_h("AZ:" + first_num + "_" + alphas[0]))
        # concatenated alpha of address (catches reordered/abbrev addresses) (1 key)
        addr_concat = "".join(t for t in ad.tokens if t.isalpha())
        if len(addr_concat) >= 12:
            ks.append(_h("AC:" + addr_concat[:24]))
    return ks


def strong_keys(name_raw, addr_raw):
    """existing strong keys from v2 (for the 'existing+strong' baseline)."""
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
            yield p[0], (p[1] if len(p)>1 else ""), (p[2] if len(p)>2 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()
    t0 = time.time()

    gt = {}
    with open(C.TRAIN_GROUND_TRUTH, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1,_,rest = line.rstrip("\n").partition(DELIM)
            gt[s1] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()

    sample=set()
    for eid,*_ in stream(C.TRAIN_SOURCE1):
        if eid in gt and gt[eid]:
            sample.add(eid)
            if len(sample)>=args.sample: break

    existing={}
    with open(C.WORK_DIR / "candidates_train.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1,_,rest=line.rstrip("\n").partition(DELIM)
            if s1 in sample:
                existing[s1]=set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()
    print(f"  sample={len(sample):,} ({time.time()-t0:.0f}s)", flush=True)

    # build combined index: strong keys + new keys, hashed
    ids=[]; index=defaultdict(lambda: array.array("i"))
    n=0
    MAXF=25000
    for path in (C.TRAIN_SOURCE2, C.TRAIN_SOURCE3):
        for eid,name,addr in stream(path):
            idx=len(ids); ids.append(eid)
            for k in strong_keys(name,addr): index[k].append(idx)
            for k in new_keys(name,addr): index[k].append(idx)
            n+=1
            if n%2_000_000==0: print(f"  idx {n:,} keys={len(index):,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  index built {n:,} keys={len(index):,} ({time.time()-t0:.0f}s)", flush=True)
    dropped=0
    for k in list(index.keys()):
        if len(index[k])>MAXF: del index[k]; dropped+=1
    print(f"  dropped {dropped:,} over-common ({time.time()-t0:.0f}s)", flush=True)

    # candidates from strong+new keys for the sample
    newcand={}
    for eid,name,addr in stream(C.TRAIN_SOURCE1):
        if eid in sample:
            hits=set()
            for k in strong_keys(name,addr)+new_keys(name,addr):
                pl=index.get(k)
                if pl: hits.update(pl)
            newcand[eid]=set(ids[i] for i in hits)
    print(f"  scored new candidates ({time.time()-t0:.0f}s)", flush=True)

    def recall_of(get):
        tt=rec=0; ps=0.0; ent=0; ctot=0
        for s1 in sample:
            truth=gt.get(s1,set()); cand=get(s1); ctot+=len(cand); ent+=1
            if not truth: ps+=1.0; continue
            got=truth&cand; tt+=len(truth); rec+=len(got); ps+=len(got)/len(truth)
        return (rec/tt if tt else 0),(ps/ent if ent else 0),(ctot/ent if ent else 0)

    e=recall_of(lambda s: existing.get(s,set()))
    c=recall_of(lambda s: existing.get(s,set())|newcand.get(s,set()))
    print("\n===== RECALL v3 (existing + strong + NEW addr/domain keys) =====")
    print(f"sample: {len(sample):,}")
    print(f"EXISTING only        : MACRO {e[1]*100:.2f}%  cand/ent {e[2]:.1f}")
    print(f"EXISTING+strong+NEW  : MACRO {c[1]*100:.2f}%  cand/ent {c[2]:.1f}")
    print(f"=> ceiling now {c[1]*100:.2f}%  (v2 combined was ~91.5%)")
    print(f"[done {time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
