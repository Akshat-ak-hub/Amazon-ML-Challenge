"""Stage 6 — Test inference: produce matching_results.tsv + candidate_pairs.tsv.

Reuses the already-trained model (WORK_DIR/match_model.txt) and tuned threshold
(WORK_DIR/threshold.txt). Memory-safe (SQLite record store like build_training).

Steps:
  1. Blocking on TEST -> candidate_pairs.tsv  (via blocking.build_blocking('test'))
  2. Load test record text into a SQLite store.
  3. Stream S1 test entities; for each, fetch candidate text, compute features,
     score with the model. Keep candidates with score >= threshold.
  4. Greedy conflict resolution: each S2/S3 record assigned to its highest-scoring
     S1 entity. Write matching_results.tsv.

Run:  python src/predict_test.py
"""
from __future__ import annotations

import json
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
import blocking as B  # noqa: E402

DELIM = "\t"


def _pack(name, addr, ctry):
    nm = N.normalize_name(name)
    ad = N.normalize_address(addr)
    return json.dumps([nm.full, nm.core, nm.phonetic, nm.tokens,
                       ad.present, ad.full, ad.tokens, ad.pin,
                       (ctry or "").strip().lower()], ensure_ascii=False)


def _unpack(js):
    (full, core, phon, toks, present, afull, atoks, pin, ctry) = json.loads(js)
    nm = N.NormName(raw="", full=full, core=core, tokens=toks, legal=[], phonetic=phon)
    ad = N.NormAddr(raw="", full=afull, tokens=atoks, landmark="", pin=pin, present=present)
    return nm, ad, ctry


def build_test_store(db_path, needed):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF"); conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-131072")
    conn.execute("DROP TABLE IF EXISTS rec")
    conn.execute("CREATE TABLE rec (id TEXT PRIMARY KEY, js TEXT)")
    cur = conn.cursor()
    buf = []
    t0 = time.time()
    for path in (C.TEST_SOURCE1, C.TEST_SOURCE2, C.TEST_SOURCE3):
        with open(path, encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split(DELIM)
                eid = p[0]
                if eid in needed:
                    buf.append((eid, _pack(p[1] if len(p) > 1 else "",
                                           p[2] if len(p) > 2 else "",
                                           p[3] if len(p) > 3 else "")))
                    if len(buf) >= 50_000:
                        cur.executemany("INSERT OR REPLACE INTO rec VALUES (?,?)", buf); buf.clear()
        print(f"  scanned {path.name} ({time.time()-t0:.0f}s)", flush=True)
    if buf:
        cur.executemany("INSERT OR REPLACE INTO rec VALUES (?,?)", buf)
    conn.commit()
    return conn


def main():
    C.ensure_dirs()
    t0 = time.time()

    # 1. Test blocking -> candidate_pairs.tsv in WORK_DIR
    print("[predict] blocking on test ...", flush=True)
    cand_path = B.build_blocking("test")
    print(f"  candidates at {cand_path} ({time.time()-t0:.0f}s)", flush=True)

    # load candidates
    cand = {}
    with open(cand_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            cand[s1] = [x for x in rest.split(",") if x.strip()] if rest.strip() else []
    print(f"  {len(cand):,} test S1 entities", flush=True)

    needed = set(cand)
    for lst in cand.values():
        needed.update(lst)
    print(f"  need text for {len(needed):,} records", flush=True)

    # 2. record store
    db_path = C.WORK_DIR / "recstore_test.sqlite"
    if db_path.exists():
        db_path.unlink()
    print("[predict] building test record store ...", flush=True)
    conn = build_test_store(db_path, needed)
    cur = conn.cursor()

    # 3. load model + threshold
    model = lgb.Booster(model_file=str(C.WORK_DIR / "match_model.txt"))
    thr = float(open(C.WORK_DIR / "threshold.txt").readline().strip())
    print(f"[predict] model loaded, threshold={thr}", flush=True)

    # 4. score all candidate pairs, remember scores for conflict resolution
    print("[predict] scoring candidates ...", flush=True)
    best_claim = {}      # cand id -> (score, s1)
    scored_by_s1 = {}    # s1 -> list[(cid, score)]
    n = 0
    for s1, cands in cand.items():
        if not cands:
            scored_by_s1[s1] = []
            n += 1
            continue
        row = cur.execute("SELECT js FROM rec WHERE id=?", (s1,)).fetchone()
        if not row:
            scored_by_s1[s1] = []
            n += 1
            continue
        nm1, ad1, cc1 = _unpack(row[0])
        feats = []
        valid_cids = []
        for cid in cands:
            r = cur.execute("SELECT js FROM rec WHERE id=?", (cid,)).fetchone()
            if not r:
                continue
            nm2, ad2, cc2 = _unpack(r[0])
            fv = F.pair_features(nm1, ad1, nm2, ad2)
            fv[-1] = F.country_flag(cc1, cc2)
            feats.append(fv)
            valid_cids.append(cid)
        kept = []
        if feats:
            preds = model.predict(np.asarray(feats, dtype=np.float32))
            for cid, sc in zip(valid_cids, preds):
                if sc >= thr:
                    kept.append((cid, float(sc)))
                    if cid not in best_claim or sc > best_claim[cid][0]:
                        best_claim[cid] = (float(sc), s1)
        scored_by_s1[s1] = kept
        n += 1
        if n % 100_000 == 0:
            print(f"  scored {n:,} S1 ({time.time()-t0:.0f}s)", flush=True)

    conn.close()
    db_path.unlink()

    # 5. conflict resolution + write matching_results.tsv
    print("[predict] resolving conflicts + writing output ...", flush=True)
    out_match = C.OUTPUT_DIR / "matching_results.tsv"
    with open(out_match, "w", encoding="utf-8") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, kept in scored_by_s1.items():
            keep = [cid for cid, sc in kept if best_claim.get(cid, (0, None))[1] == s1]
            out.write(f"{s1}\t{','.join(keep)}\n")

    # copy candidate_pairs.tsv to OUTPUT_DIR
    out_cand = C.OUTPUT_DIR / "candidate_pairs.tsv"
    import shutil
    shutil.copy(cand_path, out_cand)

    print(f"[predict] DONE in {time.time()-t0:.0f}s", flush=True)
    print(f"  -> {out_match}")
    print(f"  -> {out_cand}")


if __name__ == "__main__":
    main()
