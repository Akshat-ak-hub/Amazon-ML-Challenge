"""Realistic local scorer — the honest F0.5 that mirrors the leaderboard.

The existing model.py tune_threshold only scores over rows that BLOCKING produced.
That silently ignores every true match blocking never proposed (the 12.5% recall
gap), so its 0.928 is optimistic. The leaderboard scores EVERY S1 entity, counting
a missed true match as a false negative.

This scorer reproduces the leaderboard exactly:
  * evaluate over ALL S1 entities on the held-out VALID split
  * predictions = model score >= threshold, restricted to blocking candidates
    (a missed true match simply never gets predicted -> counts as FN, as it should)
  * macro-average F0.5 per entity, singletons scored 1.0 when predicted empty

It rebuilds the SAME validation split model.py uses (crc32 hash on S1 id), scores
each entity's candidates with the current model, and sweeps thresholds to report
the realistic best F0.5. Also reports the recall ceiling implied by blocking.

Reuses: work/candidates_train.tsv, work/match_model.txt, ground truth. Builds a
small record store only for the needed valid-split records (memory-safe).

Run:  python realistic_scorer.py            # uses current model + candidates
      python realistic_scorer.py --limit N  # first N S1 for a fast check
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C          # noqa: E402
import normalize as N       # noqa: E402
import features as F        # noqa: E402

DELIM = "\t"
VALID_FRACTION = 0.15
BETA = 0.5
B2 = BETA * BETA


def is_valid(eid: str) -> bool:
    h = zlib.crc32(eid.encode()) & 0xFFFFFFFF
    return (h % 100) < int(VALID_FRACTION * 100)


def load_candidates(path, valid_only=True, limit=None):
    cand = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            if valid_only and not is_valid(s1):
                continue
            cand[s1] = [x for x in rest.split(",") if x.strip()] if rest.strip() else []
            if limit and len(cand) >= limit:
                break
    return cand


def load_gt(path, keep):
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            if s1 in keep:
                gt[s1] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()
    return gt


def _pack(name, addr, ctry):
    nm = N.normalize_name(name); ad = N.normalize_address(addr)
    return json.dumps([nm.full, nm.core, nm.phonetic, nm.tokens,
                       ad.present, ad.full, ad.tokens, ad.pin,
                       (ctry or "").strip().lower()], ensure_ascii=False)


def _unpack(js):
    a = json.loads(js)
    nm = N.NormName(raw="", full=a[0], core=a[1], tokens=a[3], legal=[], phonetic=a[2])
    ad = N.NormAddr(raw="", full=a[5], tokens=a[6], landmark="", pin=a[7], present=a[4])
    return nm, ad, a[8]


def build_store(db_path, needed):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF"); conn.execute("PRAGMA synchronous=OFF")
    conn.execute("DROP TABLE IF EXISTS rec")
    conn.execute("CREATE TABLE rec (id TEXT PRIMARY KEY, js TEXT)")
    cur = conn.cursor(); buf = []
    t0 = time.time()
    for path in (C.TRAIN_SOURCE1, C.TRAIN_SOURCE2, C.TRAIN_SOURCE3):
        with open(path, encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split(DELIM)
                if p[0] in needed:
                    buf.append((p[0], _pack(p[1] if len(p) > 1 else "",
                                            p[2] if len(p) > 2 else "",
                                            p[3] if len(p) > 3 else "")))
                    if len(buf) >= 50_000:
                        cur.executemany("INSERT OR REPLACE INTO rec VALUES (?,?)", buf); buf.clear()
        print(f"  scanned {path.name} ({time.time()-t0:.0f}s)", flush=True)
    if buf:
        cur.executemany("INSERT OR REPLACE INTO rec VALUES (?,?)", buf)
    conn.commit()
    return conn


def fbeta_from_counts(tp, fp, fn):
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    denom = B2 * prec + rec
    return (1 + B2) * prec * rec / denom if denom else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--candidates", default=str(C.WORK_DIR / "candidates_train.tsv"))
    ap.add_argument("--model", default=str(C.WORK_DIR / "match_model.txt"))
    ap.add_argument("--exact-accept", action="store_true",
                    help="auto-accept exact normalized-name + same-country pairs")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[scorer] loading VALID-split candidates from {Path(args.candidates).name}", flush=True)
    cand = load_candidates(args.candidates, valid_only=True, limit=args.limit)
    print(f"  {len(cand):,} valid-split S1 entities", flush=True)

    gt = load_gt(C.TRAIN_GROUND_TRUTH, set(cand))
    needed = set(cand)
    for lst in cand.values():
        needed.update(lst)
    print(f"  need {len(needed):,} records; building store ...", flush=True)

    db = C.WORK_DIR / "recstore_score.sqlite"
    if db.exists():
        db.unlink()
    conn = build_store(db, needed)
    cur = conn.cursor()

    model = lgb.Booster(model_file=args.model)
    print(f"[scorer] scoring candidates with model ...", flush=True)

    # For each entity, collect (cid, score, is_true, exact_flag). Missed true
    # matches (not in candidates) are tracked separately as guaranteed FN.
    per_ent = {}          # s1 -> list of (score, is_true, exact)
    missed_fn = {}        # s1 -> count of true matches never proposed as candidate
    true_count = {}       # s1 -> total true matches
    n = 0
    CH = 5000
    items = list(cand.items())
    for start in range(0, len(items), CH):
        block = items[start:start + CH]
        need = set()
        for s1, cs in block:
            need.add(s1); need.update(cs)
        rows = {}
        need = list(need)
        for i in range(0, len(need), 30000):
            sub = need[i:i + 30000]
            q = ",".join("?" * len(sub))
            rows.update(cur.execute(f"SELECT id, js FROM rec WHERE id IN ({q})", sub).fetchall())
        for s1, cs in block:
            truth = gt.get(s1, set())
            true_count[s1] = len(truth)
            if s1 not in rows:
                per_ent[s1] = []
                missed_fn[s1] = len(truth)
                continue
            nm1, ad1, cc1 = _unpack(rows[s1])
            feats, meta = [], []
            cand_set = set()
            for cid in cs:
                js = rows.get(cid)
                if js is None:
                    continue
                nm2, ad2, cc2 = _unpack(js)
                fv = F.pair_features(nm1, ad1, nm2, ad2)
                fv[-1] = F.country_flag(cc1, cc2)
                feats.append(fv)
                exact = 1 if (nm1.core and nm1.core == nm2.core and cc1 == cc2) else 0
                meta.append((cid, 1 if cid in truth else 0, exact))
                cand_set.add(cid)
            scores = model.predict(np.asarray(feats, dtype=np.float32)) if feats else []
            per_ent[s1] = [(float(sc), m[1], m[2]) for sc, m in zip(scores, meta)]
            # true matches that never appeared as candidates:
            missed_fn[s1] = len(truth - cand_set)
        n += len(block)
        if (start // CH) % 5 == 0:
            print(f"  scored {n:,}/{len(items):,} ({time.time()-t0:.0f}s)", flush=True)
    conn.close(); db.unlink()

    # Sweep thresholds; compute REALISTIC macro-F0.5 including missed_fn.
    print("[scorer] sweeping thresholds (realistic, includes blocking misses) ...", flush=True)
    ents = list(per_ent.keys())
    best_t, best_f = 0.5, -1.0
    results = []
    for t in np.arange(0.30, 0.96, 0.02):
        total = 0.0
        for s1 in ents:
            tp = fp = 0
            for sc, is_true, exact in per_ent[s1]:
                pred = 1 if (sc >= t or (args.exact_accept and exact)) else 0
                if pred and is_true:
                    tp += 1
                elif pred and not is_true:
                    fp += 1
            fn = (true_count[s1] - tp) if true_count[s1] else 0
            # fn already includes missed candidates because tp<=candidate trues
            total += fbeta_from_counts(tp, fp, fn)
        f = total / len(ents) if ents else 0.0
        results.append((t, f))
        if f > best_f:
            best_f, best_t = f, float(t)

    # recall ceiling on this split
    tot_true = sum(true_count.values())
    tot_missed = sum(missed_fn.values())
    ceiling = 1 - tot_missed / tot_true if tot_true else 1.0

    print("\n===== REALISTIC LOCAL SCORE (valid split) =====")
    print(f"entities: {len(ents):,}   true links: {tot_true:,}   "
          f"blocking-missed: {tot_missed:,}  (recall ceiling {ceiling*100:.2f}%)")
    print(f"exact-accept layer: {'ON' if args.exact_accept else 'off'}")
    print(f"BEST threshold = {best_t:.2f}   realistic macro-F0.5 = {best_f:.4f}")
    print("\nthreshold : F0.5")
    for t, f in results:
        mark = "  <== best" if abs(t - best_t) < 1e-9 else ""
        print(f"  {t:.2f}   : {f:.4f}{mark}")
    print(f"\n[scorer done {time.time()-t0:.0f}s]")

    with open(C.WORK_DIR / "realistic_score.txt", "w", encoding="utf-8") as fo:
        fo.write(f"best_threshold\t{best_t}\nrealistic_f05\t{best_f}\n"
                 f"recall_ceiling\t{ceiling}\nexact_accept\t{args.exact_accept}\n")


if __name__ == "__main__":
    main()
