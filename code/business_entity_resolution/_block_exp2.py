"""Blocking recall EXPERIMENT v2 — MEMORY-SAFE (fits ~8-11 GB).

Why v2: v1 held two giant string-keyed dict indexes (weak ~13M + strong ~20M
entries) over all 10.3M S2/S3 records -> ~11 GB, which thrashed/swapped on this
machine. v2 fixes the ROOT cause, not a tweak:

  * Build ONLY the strong-key index (exact name, sorted-name signature, full
    phonetic, exact PIN, exact address). Strong keys are where NEW recall comes
    from; the existing 87% blocking already exploits weak fuzzy keys.
  * Hash every key to an int64 (8 bytes) instead of storing the string (~40-80 B).
    Posting lists are array('i'). This cuts index memory ~3-4x.
  * Load the existing candidates_train.tsv (weak-key result, ~87% recall) and
    measure how much the strong keys ADD on top -> the achievable ceiling.

This directly answers: "if we add strong keys to blocking, what recall do we get?"

Run:  python _block_exp2.py --sample 20000
"""
from __future__ import annotations
import argparse, array, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C
import normalize as N

DELIM = "\t"
TOP_K = 40
MAX_FANOUT_STRONG = 20000     # strong keys are precise; allow large postings


def _h(s: str) -> int:
    # stable 64-bit hash (Python's hash is salted per-run; use a fixed one)
    import hashlib
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


def strong_keys(name_raw, addr_raw):
    nm = N.normalize_name(name_raw)
    ad = N.normalize_address(addr_raw)
    ks = []
    core = nm.core
    toks = nm.tokens
    if core:
        ks.append(_h("NC:" + core))
    if toks:
        ks.append(_h("NS:" + "_".join(sorted(toks))))
    if nm.phonetic:
        ks.append(_h("NP:" + "".join(nm.phonetic.split())))
    if ad.present:
        if ad.pin:
            ks.append(_h("Z:" + ad.pin))
        if ad.full and len(ad.full) >= 10:
            ks.append(_h("AF:" + ad.full))
    return ks


def stream(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            p = line.rstrip("\n").split(DELIM)
            yield p[0], (p[1] if len(p)>1 else ""), (p[2] if len(p)>2 else "")


def load_existing_candidates(path, keep):
    cand = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            if s1 in keep:
                cand[s1] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()
    t0 = time.time()

    # ground truth first (small) + pick sample of S1 that HAVE gt
    gt = {}
    with open(C.TRAIN_GROUND_TRUTH, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            gt[s1] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()
    print(f"  loaded gt: {len(gt):,} S1 ({time.time()-t0:.0f}s)", flush=True)

    # build STRONG-key index over S2/S3 with hashed keys (memory-safe)
    ids = []
    index = defaultdict(lambda: array.array("i"))
    n = 0
    for path in (C.TRAIN_SOURCE2, C.TRAIN_SOURCE3):
        for eid, name, addr in stream(path):
            idx = len(ids); ids.append(eid)
            for k in strong_keys(name, addr):
                index[k].append(idx)
            n += 1
            if n % 1_000_000 == 0:
                print(f"  strong-index {n:,} recs, keys={len(index):,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  strong-index built: {n:,} recs, keys={len(index):,} ({time.time()-t0:.0f}s)", flush=True)

    dropped = 0
    for k in list(index.keys()):
        if len(index[k]) > MAX_FANOUT_STRONG:
            del index[k]; dropped += 1
    print(f"  dropped {dropped:,} over-common strong keys ({time.time()-t0:.0f}s)", flush=True)

    # existing weak-key candidates (the current ~87% blocking output), for sample
    sample_s1 = set()
    for eid, *_ in stream(C.TRAIN_SOURCE1):
        if eid in gt:
            sample_s1.add(eid)
            if len(sample_s1) >= args.sample:
                break
    existing = load_existing_candidates(C.WORK_DIR / "candidates_train.tsv", sample_s1)
    print(f"  loaded existing candidates for {len(existing):,} sample S1 ({time.time()-t0:.0f}s)", flush=True)

    # measure recall: existing-only, strong-only, and COMBINED
    def recall_of(getcands):
        tt = rec = 0; ps = 0.0; ent = 0; ctot = 0
        for s1 in sample_s1:
            truth = gt.get(s1, set())
            cand = getcands(s1)
            ctot += len(cand)
            ent += 1
            if not truth:
                ps += 1.0; continue
            got = truth & cand
            tt += len(truth); rec += len(got)
            ps += len(got)/len(truth)
        return (rec/tt if tt else 0), (ps/ent if ent else 0), (ctot/ent if ent else 0)

    # precompute strong candidates per sample S1
    strong_cand = {}
    for eid, name, addr in stream(C.TRAIN_SOURCE1):
        if eid in sample_s1:
            hits = set()
            for k in strong_keys(name, addr):
                pl = index.get(k)
                if pl:
                    hits.update(pl)
            # cap
            strong_cand[eid] = set(ids[i] for i in list(hits)[:TOP_K*2])
    print(f"  scored strong candidates ({time.time()-t0:.0f}s)", flush=True)

    e_mi, e_ma, e_c = recall_of(lambda s: existing.get(s, set()))
    s_mi, s_ma, s_c = recall_of(lambda s: strong_cand.get(s, set()))
    c_mi, c_ma, c_c = recall_of(lambda s: existing.get(s, set()) | strong_cand.get(s, set()))

    print("\n===== RECALL COMPARISON (sample) =====")
    print(f"sample S1: {len(sample_s1):,}")
    print(f"EXISTING (weak) only : MICRO {e_mi*100:.2f}%  MACRO {e_ma*100:.2f}%  cand/ent {e_c:.1f}")
    print(f"STRONG keys only     : MICRO {s_mi*100:.2f}%  MACRO {s_ma*100:.2f}%  cand/ent {s_c:.1f}")
    print(f"COMBINED (weak+strong): MICRO {c_mi*100:.2f}%  MACRO {c_ma*100:.2f}%  cand/ent {c_c:.1f}")
    print(f"\n=> gain from adding strong keys: "
          f"MACRO {e_ma*100:.2f}% -> {c_ma*100:.2f}%  (+{(c_ma-e_ma)*100:.2f} pts)")
    print(f"[done {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
