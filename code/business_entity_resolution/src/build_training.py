"""Stage 5b — Build the labeled training matrix from blocking candidates.

Steps:
  1. Load candidates_train.tsv  (S1 -> [candidate ids])
  2. Split S1 entities into TRAIN / VALID by hashing the id (entity-level split so
     an entity's pairs never straddle the split -> no leakage).
  3. Stream all three sources once, keeping normalized (name, addr, country) only
     for the ids we actually need (S1 entities + their candidates). Memory-bounded.
  4. For each (S1, candidate) pair: compute features, set country flag, label 1 if
     the candidate is a true match (from ground truth) else 0.
  5. Write two npz-friendly arrays (X, y, groups) for train and valid to WORK_DIR.

Run:  python src/build_training.py            # full
      python src/build_training.py --limit 50000   # first N S1 entities (dev)
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import normalize as N  # noqa: E402
import features as F  # noqa: E402

DELIM = "\t"
VALID_FRACTION = 0.15


def _hash_split(eid: str) -> str:
    """Deterministic entity-level split by crc32 hash."""
    h = zlib.crc32(eid.encode()) & 0xFFFFFFFF
    return "valid" if (h % 100) < int(VALID_FRACTION * 100) else "train"


def load_candidates(path, limit=None):
    cand = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            cand[s1] = [x for x in rest.split(",") if x.strip()] if rest.strip() else []
            if limit and len(cand) >= limit:
                break
    return cand


def load_ground_truth(path, needed):
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            if s1 in needed:
                gt[s1] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()
    return gt


def load_records(paths, needed):
    """id -> (NormName, NormAddr, country_str) for ids in `needed`. Streaming."""
    out = {}
    t0 = time.time()
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split(DELIM)
                eid = p[0]
                if eid in needed:
                    name = p[1] if len(p) > 1 else ""
                    addr = p[2] if len(p) > 2 else ""
                    ctry = p[3] if len(p) > 3 else ""
                    out[eid] = (N.normalize_name(name), N.normalize_address(addr), ctry)
        print(f"  loaded records after {path.name}: {len(out):,} ({time.time()-t0:.0f}s)", flush=True)
    return out


def build(limit=None):
    C.ensure_dirs()
    t0 = time.time()
    print("[build_training] loading candidates ...", flush=True)
    cand = load_candidates(C.WORK_DIR / "candidates_train.tsv", limit=limit)
    print(f"  {len(cand):,} S1 entities", flush=True)

    # which ids we need text for
    needed = set(cand)
    for lst in cand.values():
        needed.update(lst)
    print(f"  need text for {len(needed):,} records", flush=True)

    gt = load_ground_truth(C.TRAIN_GROUND_TRUTH, set(cand))
    print(f"  loaded ground truth for {len(gt):,} S1 entities", flush=True)

    print("[build_training] loading record text (streaming) ...", flush=True)
    recs = load_records([C.TRAIN_SOURCE1, C.TRAIN_SOURCE2, C.TRAIN_SOURCE3], needed)

    print("[build_training] computing features ...", flush=True)
    rows = {"train": [], "valid": []}
    labels = {"train": [], "valid": []}
    groups = {"train": [], "valid": []}   # per-S1 group sizes (for ranking/metrics)
    pair_ids = {"train": [], "valid": []}  # (s1, cand) for later assembly

    n = 0
    for s1, cands in cand.items():
        if s1 not in recs or not cands:
            continue
        split = _hash_split(s1)
        nm1, ad1, cc1 = recs[s1]
        truth = gt.get(s1, set())
        grp = 0
        for cid in cands:
            if cid not in recs:
                continue
            nm2, ad2, cc2 = recs[cid]
            fv = F.pair_features(nm1, ad1, nm2, ad2)
            fv[-1] = F.country_flag(cc1, cc2)   # fill country flag
            rows[split].append(fv)
            labels[split].append(1 if cid in truth else 0)
            pair_ids[split].append((s1, cid))
            grp += 1
        if grp:
            groups[split].append(grp)
        n += 1
        if n % 100_000 == 0:
            print(f"  processed {n:,} S1 ({time.time()-t0:.0f}s)", flush=True)

    for split in ("train", "valid"):
        X = np.asarray(rows[split], dtype=np.float32)
        y = np.asarray(labels[split], dtype=np.int8)
        g = np.asarray(groups[split], dtype=np.int32)
        np.savez(C.WORK_DIR / f"trainmat_{split}.npz", X=X, y=y, groups=g)
        with open(C.WORK_DIR / f"pairids_{split}.tsv", "w", encoding="utf-8") as f:
            for s1, cid in pair_ids[split]:
                f.write(f"{s1}\t{cid}\n")
        pos = int(y.sum())
        print(f"  {split}: X={X.shape} positives={pos:,} ({pos/max(len(y),1)*100:.2f}%)", flush=True)

    print(f"[build_training] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(limit=args.limit)
