"""Stage 5b (v2) — Memory-safe chunked training-matrix builder.

Problem with v1: it held all ~11.2M records' normalized text in a Python dict
(~4-5 GB) and paged to disk on an 8 GB machine. v2 caps RAM at a low fixed level:

  1. Collect the set of record IDs we actually need (S1 entities + their candidates).
  2. Stream the 3 sources ONCE, writing normalized text for needed IDs into a
     SQLite key-value table on disk (id -> name_core, name_full, phonetic, tokens,
     addr fields, country). RAM holds only a small insert buffer.
  3. Stream S1 entities; for each, fetch its + candidates' rows from SQLite, compute
     features, and APPEND them to a growing memmapped array on disk (flush per
     chunk). Never accumulate the whole matrix in RAM.

Output identical to v1: trainmat_{train,valid}.npz + pairids_{train,valid}.tsv, so
the model/verify steps are unchanged.

Run:  python src/build_training.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
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
CHUNK = 50_000            # S1 entities per feature-flush chunk
NFEAT = len(F.FEATURE_NAMES)


def _hash_split(eid: str) -> str:
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


def load_ground_truth(path, needed_s1):
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            if s1 in needed_s1:
                gt[s1] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()
    return gt


def _connect(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-131072")  # ~128 MB
    return conn


def _pack(name, addr, ctry):
    """Normalize a record and pack the fields we need into a compact tuple/json."""
    nm = N.normalize_name(name)
    ad = N.normalize_address(addr)
    obj = [
        nm.full, nm.core, nm.phonetic, nm.tokens,
        ad.present, ad.full, ad.tokens, ad.pin,
        (ctry or "").strip().lower(),
    ]
    return json.dumps(obj, ensure_ascii=False)


def _unpack(js):
    (full, core, phon, toks, present, afull, atoks, pin, ctry) = json.loads(js)
    nm = N.NormName(raw="", full=full, core=core, tokens=toks,
                    legal=[], phonetic=phon)
    ad = N.NormAddr(raw="", full=afull, tokens=atoks, landmark="",
                    pin=pin, present=present)
    return nm, ad, ctry


def build_record_store(db_path, needed, logf):
    conn = _connect(db_path)
    conn.execute("DROP TABLE IF EXISTS rec")
    conn.execute("CREATE TABLE rec (id TEXT PRIMARY KEY, js TEXT)")
    cur = conn.cursor()
    t0 = time.time()
    n = 0
    buf = []
    for path in (C.TRAIN_SOURCE1, C.TRAIN_SOURCE2, C.TRAIN_SOURCE3):
        with open(path, encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split(DELIM)
                eid = p[0]
                if eid in needed:
                    name = p[1] if len(p) > 1 else ""
                    addr = p[2] if len(p) > 2 else ""
                    ctry = p[3] if len(p) > 3 else ""
                    buf.append((eid, _pack(name, addr, ctry)))
                    if len(buf) >= 50_000:
                        cur.executemany("INSERT OR REPLACE INTO rec VALUES (?,?)", buf)
                        buf.clear()
                n += 1
        print(f"  scanned {path.name} (store now filling) {time.time()-t0:.0f}s", flush=True)
    if buf:
        cur.executemany("INSERT OR REPLACE INTO rec VALUES (?,?)", buf)
    conn.commit()
    (count,) = conn.execute("SELECT COUNT(*) FROM rec").fetchone()
    print(f"  record store built: {count:,} rows in {time.time()-t0:.0f}s", flush=True)
    return conn


class ChunkWriter:
    """Appends feature rows to a growing .npy on disk, flushing per chunk."""
    def __init__(self, split):
        self.split = split
        self.X_parts = []      # list of temp .npy paths
        self.y = []
        self.groups = []
        self.pair_f = open(C.WORK_DIR / f"pairids_{split}.tsv", "w", encoding="utf-8")
        self._buf_X = []
        self._buf_y = []
        self._part = 0

    def add(self, fv, label, s1, cid):
        self._buf_X.append(fv)
        self._buf_y.append(label)
        self.pair_f.write(f"{s1}\t{cid}\n")

    def add_group(self, size):
        if size:
            self.groups.append(size)

    def flush(self):
        if not self._buf_X:
            return
        arr = np.asarray(self._buf_X, dtype=np.float32)
        path = C.WORK_DIR / f"_part_{self.split}_{self._part}.npy"
        np.save(path, arr)
        self.X_parts.append(path)
        self.y.extend(self._buf_y)
        self._buf_X.clear()
        self._buf_y.clear()
        self._part += 1

    def finalize(self):
        self.flush()
        self.pair_f.close()
        if self.X_parts:
            X = np.concatenate([np.load(p) for p in self.X_parts], axis=0)
        else:
            X = np.zeros((0, NFEAT), dtype=np.float32)
        y = np.asarray(self.y, dtype=np.int8)
        g = np.asarray(self.groups, dtype=np.int32)
        np.savez(C.WORK_DIR / f"trainmat_{self.split}.npz", X=X, y=y, groups=g)
        for p in self.X_parts:
            p.unlink()
        pos = int(y.sum())
        print(f"  {self.split}: X={X.shape} positives={pos:,} "
              f"({pos/max(len(y),1)*100:.2f}%)", flush=True)


def build(limit=None):
    C.ensure_dirs()
    t0 = time.time()
    logf = None
    print("[build v2] loading candidates ...", flush=True)
    cand = load_candidates(C.WORK_DIR / "candidates_train.tsv", limit=limit)
    print(f"  {len(cand):,} S1 entities", flush=True)

    needed = set(cand)
    for lst in cand.values():
        needed.update(lst)
    print(f"  need text for {len(needed):,} records", flush=True)

    gt = load_ground_truth(C.TRAIN_GROUND_TRUTH, set(cand))
    print(f"  ground truth for {len(gt):,} S1 entities", flush=True)

    db_path = C.WORK_DIR / "recstore.sqlite"
    if db_path.exists():
        db_path.unlink()
    print("[build v2] building on-disk record store ...", flush=True)
    conn = build_record_store(db_path, needed, logf)
    cur = conn.cursor()

    print("[build v2] computing features (chunked) ...", flush=True)
    writers = {"train": ChunkWriter("train"), "valid": ChunkWriter("valid")}
    n = 0
    cache = {}   # tiny per-chunk cache of unpacked rows
    for s1, cands in cand.items():
        if not cands:
            n += 1
            continue
        row = cur.execute("SELECT js FROM rec WHERE id=?", (s1,)).fetchone()
        if not row:
            n += 1
            continue
        nm1, ad1, cc1 = _unpack(row[0])
        truth = gt.get(s1, set())
        split = _hash_split(s1)
        w = writers[split]
        grp = 0
        for cid in cands:
            r = cur.execute("SELECT js FROM rec WHERE id=?", (cid,)).fetchone()
            if not r:
                continue
            nm2, ad2, cc2 = _unpack(r[0])
            fv = F.pair_features(nm1, ad1, nm2, ad2)
            fv[-1] = F.country_flag(cc1, cc2)
            w.add(fv, 1 if cid in truth else 0, s1, cid)
            grp += 1
        w.add_group(grp)
        n += 1
        if n % CHUNK == 0:
            writers["train"].flush()
            writers["valid"].flush()
            print(f"  processed {n:,} S1 ({time.time()-t0:.0f}s)", flush=True)

    for w in writers.values():
        w.finalize()
    conn.close()
    db_path.unlink()
    print(f"[build v2] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(limit=args.limit)
