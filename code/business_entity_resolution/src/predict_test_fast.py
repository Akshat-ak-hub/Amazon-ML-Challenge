"""Stage 6 (fast v2) — Chunked BULK-fetch scoring.

Profiling showed SQLite lookups were 79% of runtime (1.7M separate queries).
Fix: process entities in CHUNKS. For each chunk of N entities, fetch ALL needed
records in ONE bulk query, hold that small slice in a dict, and score the chunk
from memory. This reduces ~1.7M queries to ~350, while peak RAM stays small
(only one chunk's records in memory at a time).

Profile (per 2000 entities): sqlite 79% / features 12% / predict 8%
-> eliminating query overhead should bring the full run to roughly 35-50 min.

Identical output to predict_test.py: same features, model, threshold, conflict
resolution. Reuses work/candidates_test.tsv and work/recstore_test.sqlite.

Run:  python src/predict_test_fast.py
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import normalize as N  # noqa: E402
import features as F  # noqa: E402

DELIM = "\t"
CHUNK_ENTITIES = 5_000       # entities per bulk fetch (~70k records/query)


def load_candidates(path):
    cand = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            cand[s1] = [x for x in rest.split(",") if x.strip()] if rest.strip() else []
    return cand


def _unpack(js):
    a = json.loads(js)
    nm = N.NormName(raw="", full=a[0], core=a[1], tokens=a[3], legal=[], phonetic=a[2])
    ad = N.NormAddr(raw="", full=a[5], tokens=a[6], landmark="", pin=a[7], present=a[4])
    return nm, ad, a[8]


def main():
    C.ensure_dirs()
    t0 = time.time()

    cand_path = C.WORK_DIR / "candidates_test.tsv"
    store_path = C.WORK_DIR / "recstore_test.sqlite"
    for p in (cand_path, store_path):
        if not p.exists():
            print(f"ERROR: {p} missing."); return

    print(f"[fast2] loading candidates from {cand_path.name}", flush=True)
    cand = load_candidates(cand_path)
    print(f"  {len(cand):,} test S1 entities", flush=True)

    conn = sqlite3.connect(str(store_path))
    conn.execute("PRAGMA cache_size=-524288")   # ~512 MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")
    cur = conn.cursor()

    model = lgb.Booster(model_file=str(C.WORK_DIR / "match_model.txt"))
    thr = float(open(C.WORK_DIR / "threshold.txt").readline().strip())
    print(f"[fast2] model loaded, threshold={thr}", flush=True)

    best_claim = {}
    scored_by_s1 = {}
    items = list(cand.items())
    total = len(items)
    print(f"[fast2] scoring in chunks of {CHUNK_ENTITIES:,} ...", flush=True)

    for start in range(0, total, CHUNK_ENTITIES):
        block = items[start:start + CHUNK_ENTITIES]
        # collect every id this chunk needs
        need = set()
        for s1, cands in block:
            need.add(s1)
            need.update(cands)
        need = list(need)
        # ONE bulk query for the whole chunk (split if > SQLite param limit)
        rows = {}
        STEP = 30_000                      # stay under SQLite variable limits
        for i in range(0, len(need), STEP):
            sub = need[i:i + STEP]
            q = ",".join("?" * len(sub))
            rows.update(cur.execute(f"SELECT id, js FROM rec WHERE id IN ({q})", sub).fetchall())

        # score this chunk from the in-memory slice
        for s1, cands in block:
            if not cands or s1 not in rows:
                scored_by_s1[s1] = []
                continue
            nm1, ad1, cc1 = _unpack(rows[s1])
            feats, valid = [], []
            for cid in cands:
                js = rows.get(cid)
                if js is None:
                    continue
                nm2, ad2, cc2 = _unpack(js)
                fv = F.pair_features(nm1, ad1, nm2, ad2)
                fv[-1] = F.country_flag(cc1, cc2)
                feats.append(fv); valid.append(cid)
            kept = []
            if feats:
                preds = model.predict(np.asarray(feats, dtype=np.float32))
                for cid, sc in zip(valid, preds):
                    if sc >= thr:
                        kept.append((cid, float(sc)))
                        if cid not in best_claim or sc > best_claim[cid][0]:
                            best_claim[cid] = (float(sc), s1)
            scored_by_s1[s1] = kept
        rows.clear()

        done = min(start + CHUNK_ENTITIES, total)
        if (start // CHUNK_ENTITIES) % 10 == 0 or done == total:
            el = time.time() - t0
            rate = done / el if el else 0
            eta = (total - done) / rate / 60 if rate else 0
            print(f"  scored {done:,}/{total:,} ({el:.0f}s, {rate:.0f} ent/s, ETA {eta:.0f} min)", flush=True)

    conn.close()

    print("[fast2] resolving conflicts + writing outputs ...", flush=True)
    out_match = C.OUTPUT_DIR / "matching_results.tsv"
    with open(out_match, "w", encoding="utf-8") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, kept in scored_by_s1.items():
            keep = [cid for cid, sc in kept if best_claim.get(cid, (0, None))[1] == s1]
            out.write(f"{s1}\t{','.join(keep)}\n")
    out_cand = C.OUTPUT_DIR / "candidate_pairs.tsv"
    shutil.copy(cand_path, out_cand)

    print(f"[fast2] DONE in {time.time()-t0:.0f}s", flush=True)
    print(f"  -> {out_match}")
    print(f"  -> {out_cand}")


if __name__ == "__main__":
    main()
