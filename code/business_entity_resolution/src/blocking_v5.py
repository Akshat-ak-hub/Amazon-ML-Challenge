"""Stage 4 (v5) — DISK-based SCORED-RANKER blocking (production).

Combines the wins from the experiments:
  * DISK (SQLite) inverted index -> no RAM wall on 10M+ records (RAM stays flat).
  * Rich keys targeting the observed misses: exact name, sorted-name sig, phonetic
    sig, domain-concat name, exact PIN, exact address, number+token, concat-address.
  * WEIGHTED scoring + top-K cap -> high recall WITHOUT the candidate flood
    (v5 K=50: ~92% recall @ ~43 cand/ent vs v4 union 94% @ 198 cand/ent).
  * UNION with the existing weak-key candidates (candidates_{split}.tsv) so we keep
    the fuzzy-name recall the old blocking already had.

Output: work/candidates2_{split}.tsv  (source1_entity_id\tcandidate_entity_ids)
  — same format downstream expects. Also measures recall ceiling on train.

Run:
  python src/blocking_v5.py --split train
  python src/blocking_v5.py --split test
"""
from __future__ import annotations
import argparse, sys, time, hashlib, re, sqlite3
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import normalize as N

DELIM = "\t"
_NUM = re.compile(r"\d+")
TOP_K = 50
MAX_FANOUT = 25000

WEIGHT = {
    "NC": 10.0, "AF": 10.0, "DC": 8.0, "NS": 7.0,
    "Z": 6.0, "NP": 5.0, "AC": 4.0, "AZ": 3.0,
}


def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
                          "little", signed=True)


def keyed(name_raw, addr_raw):
    """(key_hash, weight) list for one record."""
    nm = N.normalize_name(name_raw); ad = N.normalize_address(addr_raw)
    out = []
    if nm.core: out.append((_h("NC:" + nm.core), WEIGHT["NC"]))
    if nm.tokens: out.append((_h("NS:" + "_".join(sorted(nm.tokens))), WEIGHT["NS"]))
    if nm.phonetic: out.append((_h("NP:" + "".join(nm.phonetic.split())), WEIGHT["NP"]))
    concat = "".join(nm.tokens)
    if len(concat) >= 6: out.append((_h("DC:" + concat), WEIGHT["DC"]))
    if ad.present and ad.tokens:
        if ad.pin: out.append((_h("Z:" + ad.pin), WEIGHT["Z"]))
        if ad.full and len(ad.full) >= 10: out.append((_h("AF:" + ad.full), WEIGHT["AF"]))
        # Precise address key: full concatenated alpha of the address (leading
        # zeros of numbers stripped, digits kept). Only records with essentially
        # the SAME address collide -> small posting lists (fast) while still
        # catching 'same address, garbled name' matches. Replaces the broad
        # AZ (number+token) key that had huge posting lists and was slow.
        nums = []
        for t in ad.tokens:
            for mm in _NUM.findall(t):
                z = mm.lstrip("0") or "0"
                if len(z) >= 2:
                    nums.append(z)
        alpha_concat = "".join(t for t in ad.tokens if t.isalpha())
        if len(alpha_concat) >= 10:
            sig = alpha_concat[:28] + "|" + (nums[0] if nums else "")
            out.append((_h("AC:" + sig), WEIGHT["AC"]))
    return out


def stream(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            if not line.strip(): continue
            p = line.rstrip("\n").split(DELIM)
            yield p[0], (p[1] if len(p)>1 else ""), (p[2] if len(p)>2 else "")


def load_existing(path):
    """existing weak-key candidates to UNION with (may not exist for a split)."""
    cand = {}
    if not Path(path).exists():
        return cand
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1,_,rest = line.rstrip("\n").partition(DELIM)
            cand[s1] = [x for x in rest.split(",") if x.strip()] if rest.strip() else []
    return cand


def build(split):
    C.ensure_dirs()
    t0 = time.time()
    if split == "train":
        s1p, s23 = C.TRAIN_SOURCE1, (C.TRAIN_SOURCE2, C.TRAIN_SOURCE3)
        existing_path = C.WORK_DIR / "candidates_train.tsv"
    else:
        s1p, s23 = C.TEST_SOURCE1, (C.TEST_SOURCE2, C.TEST_SOURCE3)
        existing_path = C.WORK_DIR / "candidates_test.tsv"

    logf = open(C.WORK_DIR / f"blocking_v5_{split}.log", "w", encoding="utf-8")
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); logf.write(line+"\n"); logf.flush()

    log(f"[blocking_v5:{split}] START")

    db = C.WORK_DIR / f"blockidx_v5_{split}.sqlite"
    if db.exists(): db.unlink()
    conn = sqlite3.connect(str(db))
    for pragma in ("journal_mode=OFF","synchronous=OFF","cache_size=-262144","temp_store=FILE"):
        conn.execute("PRAGMA " + pragma)
    conn.execute("CREATE TABLE posting (key INTEGER, idx INTEGER, w REAL)")
    cur = conn.cursor()

    ids = []; buf = []; n = 0
    for path in s23:
        for eid, name, addr in stream(path):
            idx = len(ids); ids.append(eid)
            for k, w in keyed(name, addr):
                buf.append((k, idx, w))
            if len(buf) >= 500_000:
                cur.executemany("INSERT INTO posting VALUES (?,?,?)", buf); buf.clear()
            n += 1
            if n % 2_000_000 == 0:
                log(f"  streamed {n:,} recs ({time.time()-t0:.0f}s)")
    if buf: cur.executemany("INSERT INTO posting VALUES (?,?,?)", buf); buf.clear()
    conn.commit()
    log(f"  postings written for {n:,} recs; indexing ...")
    cur.execute("CREATE INDEX ix_key ON posting(key)")
    conn.commit()
    over = set(k for (k,) in cur.execute(
        "SELECT key FROM posting GROUP BY key HAVING COUNT(*) > ?", (MAX_FANOUT,)))
    log(f"  index built, {len(over):,} over-common keys ignored ({time.time()-t0:.0f}s)")

    existing = load_existing(existing_path)
    log(f"  loaded {len(existing):,} existing weak-candidate rows to union")

    # Chunked BULK scoring: per chunk of S1, collect all distinct keys, fetch ALL
    # their postings in ONE query (chunked under SQLite's variable limit), build a
    # key->postings map in memory, then score the chunk from memory. Turns ~15M
    # point queries into a few hundred bulk queries.
    out_path = C.WORK_DIR / f"candidates2_{split}.tsv"
    CHUNK = 20_000            # S1 entities per bulk fetch
    STEP = 20_000            # keys per IN(...) query (under SQLite param limit)
    m = 0
    out = open(out_path, "w", encoding="utf-8")
    out.write("source1_entity_id\tcandidate_entity_ids\n")

    chunk_rows = []          # (eid, [(key,w),...]) for the current chunk
    def flush_chunk(chunk_rows):
        if not chunk_rows:
            return 0
        # distinct keys needed by this chunk
        need = set()
        for _, kws in chunk_rows:
            for k, _w in kws:
                if k not in over:
                    need.add(k)
        # bulk fetch postings for those keys
        postings = defaultdict(list)   # key -> [(idx, w), ...]
        need = list(need)
        for i in range(0, len(need), STEP):
            sub = need[i:i+STEP]
            q = ",".join("?" * len(sub))
            for (kk, idx, pw) in cur.execute(
                    f"SELECT key, idx, w FROM posting WHERE key IN ({q})", sub):
                postings[kk].append((idx, pw))
        # score each entity from the in-memory postings
        written = 0
        for eid, kws in chunk_rows:
            scores = defaultdict(float)
            for k, _w in kws:
                pl = postings.get(k)
                if pl:
                    for idx, pw in pl:
                        scores[idx] += pw
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
            cand = [ids[i] for i, _ in ranked]
            merged = list(dict.fromkeys(cand + existing.get(eid, [])))
            out.write(f"{eid}\t{','.join(merged)}\n")
            written += 1
        return written

    for eid, name, addr in stream(s1p):
        chunk_rows.append((eid, keyed(name, addr)))
        if len(chunk_rows) >= CHUNK:
            m += flush_chunk(chunk_rows)
            chunk_rows = []
            log(f"  wrote {m:,} S1 candidates ({time.time()-t0:.0f}s)")
    m += flush_chunk(chunk_rows)
    out.close()
    conn.close(); db.unlink()
    log(f"  candidates -> {out_path}")
    log(f"[blocking_v5:{split}] DONE total {time.time()-t0:.0f}s")

    if split == "train":
        measure_recall(out_path, C.TRAIN_GROUND_TRUTH, log)
    logf.close()
    return out_path


def measure_recall(cand_path, gt_path, log):
    cand = {}
    with open(cand_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1,_,rest = line.rstrip("\n").partition(DELIM)
            cand[s1] = set(rest.split(",")) if rest.strip() else set()
    tot=rec=ent=0; ps=0.0; ctot=0
    with open(gt_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1,_,rest = line.rstrip("\n").partition(DELIM)
            if s1 not in cand: continue
            truth = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()
            ent += 1; ctot += len(cand[s1])
            if not truth: ps += 1.0; continue
            got = truth & cand[s1]
            tot += len(truth); rec += len(got); ps += len(got)/len(truth)
    log(f"  RECALL ceiling: MICRO {rec/tot*100:.2f}%  MACRO {ps/ent*100:.2f}%  "
        f"avg cand/ent {ctot/ent:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train","test"])
    args = ap.parse_args()
    build(args.split)
