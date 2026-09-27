"""Blocking recall EXPERIMENT v4 — DISK-BASED index (no RAM wall).

Root fix for the memory thrashing: instead of holding a 24-33M-entry inverted
index in RAM, we stream (key_hash, entity_idx) pairs to a SQLite table on disk,
build an index on key_hash, then look up candidates per S1 by JOIN. RAM stays
low and flat; disk (D: has plenty) absorbs the size.

Keys (full rich set — the ones that target the observed misses):
  strong: exact core-name, sorted-name sig, full phonetic sig, exact PIN, exact addr
  new:    domain-concat name, number+distinctive-token, concat-address prefix

Measures recall of (existing weak candidates) UNION (v4 keys) on a sample.

Run:  python _block_exp4.py --sample 20000
"""
from __future__ import annotations
import argparse, sys, time, hashlib, re, sqlite3, array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C
import normalize as N

DELIM = "\t"
_NUM = re.compile(r"\d+")


def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
                          "little", signed=True)


def all_keys(name_raw, addr_raw):
    nm = N.normalize_name(name_raw); ad = N.normalize_address(addr_raw)
    ks = []
    if nm.core: ks.append(_h("NC:" + nm.core))
    if nm.tokens: ks.append(_h("NS:" + "_".join(sorted(nm.tokens))))
    if nm.phonetic: ks.append(_h("NP:" + "".join(nm.phonetic.split())))
    concat = "".join(nm.tokens)
    if len(concat) >= 6: ks.append(_h("DC:" + concat))
    if ad.present and ad.tokens:
        if ad.pin: ks.append(_h("Z:" + ad.pin))
        if ad.full and len(ad.full) >= 10: ks.append(_h("AF:" + ad.full))
        first_num = ""
        for t in ad.tokens:
            for m in _NUM.findall(t):
                z = m.lstrip("0") or "0"
                if len(z) >= 2:
                    first_num = z; break
            if first_num: break
        alphas = sorted((t for t in ad.tokens if t.isalpha() and len(t) >= 4),
                        key=len, reverse=True)
        if first_num and alphas:
            ks.append(_h("AZ:" + first_num + "_" + alphas[0]))
        addr_concat = "".join(t for t in ad.tokens if t.isalpha())
        if len(addr_concat) >= 12:
            ks.append(_h("AC:" + addr_concat[:24]))
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
    ap.add_argument("--maxfan", type=int, default=25000)
    args = ap.parse_args()
    t0 = time.time()

    gt = {}
    with open(C.TRAIN_GROUND_TRUTH, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1,_,rest = line.rstrip("\n").partition(DELIM)
            gt[s1] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()

    sample = set()
    for eid,*_ in stream(C.TRAIN_SOURCE1):
        if eid in gt and gt[eid]:
            sample.add(eid)
            if len(sample) >= args.sample: break

    existing = {}
    with open(C.WORK_DIR / "candidates_train.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1,_,rest = line.rstrip("\n").partition(DELIM)
            if s1 in sample:
                existing[s1] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()
    print(f"  sample={len(sample):,} ({time.time()-t0:.0f}s)", flush=True)

    # ---- build DISK index: table posting(key INTEGER, idx INTEGER) ----
    db = C.WORK_DIR / "blockidx_exp.sqlite"
    if db.exists(): db.unlink()
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-262144")   # ~256MB page cache
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("CREATE TABLE posting (key INTEGER, idx INTEGER)")
    cur = conn.cursor()

    ids = []            # idx -> entity_id  (this list is the only big RAM item: ~10M strings)
    buf = []
    n = 0
    for path in (C.TRAIN_SOURCE2, C.TRAIN_SOURCE3):
        for eid, name, addr in stream(path):
            idx = len(ids); ids.append(eid)
            for k in all_keys(name, addr):
                buf.append((k, idx))
            if len(buf) >= 500_000:
                cur.executemany("INSERT INTO posting VALUES (?,?)", buf); buf.clear()
            n += 1
            if n % 2_000_000 == 0:
                print(f"  streamed {n:,} recs ({time.time()-t0:.0f}s)", flush=True)
    if buf:
        cur.executemany("INSERT INTO posting VALUES (?,?)", buf); buf.clear()
    conn.commit()
    print(f"  postings written for {n:,} recs ({time.time()-t0:.0f}s); indexing ...", flush=True)
    cur.execute("CREATE INDEX ix_key ON posting(key)")
    conn.commit()
    print(f"  index built ({time.time()-t0:.0f}s)", flush=True)

    # drop over-common keys (write their hashes to a set)
    over = set(k for (k,) in cur.execute(
        "SELECT key FROM posting GROUP BY key HAVING COUNT(*) > ?", (args.maxfan,)))
    print(f"  {len(over):,} over-common keys (> {args.maxfan}) will be ignored ({time.time()-t0:.0f}s)", flush=True)

    # ---- candidates for sample via disk lookups ----
    newcand = {}
    done = 0
    for eid, name, addr in stream(C.TRAIN_SOURCE1):
        if eid in sample:
            hits = set()
            for k in all_keys(name, addr):
                if k in over: continue
                for (i,) in cur.execute("SELECT idx FROM posting WHERE key=?", (k,)):
                    hits.add(i)
            newcand[eid] = set(ids[i] for i in hits)
            done += 1
            if done % 5000 == 0:
                print(f"  looked up {done:,}/{len(sample):,} ({time.time()-t0:.0f}s)", flush=True)
    conn.close(); db.unlink()

    def recall_of(get):
        tt=rec=0; ps=0.0; ent=0; ctot=0
        for s1 in sample:
            truth=gt.get(s1,set()); cand=get(s1); ctot+=len(cand); ent+=1
            if not truth: ps+=1.0; continue
            got=truth&cand; tt+=len(truth); rec+=len(got); ps+=len(got)/len(truth)
        return (rec/tt if tt else 0),(ps/ent if ent else 0),(ctot/ent if ent else 0)

    e = recall_of(lambda s: existing.get(s,set()))
    c = recall_of(lambda s: existing.get(s,set()) | newcand.get(s,set()))
    nk = recall_of(lambda s: newcand.get(s,set()))
    print("\n===== RECALL v4 (DISK index, full rich keys) =====")
    print(f"sample: {len(sample):,}")
    print(f"EXISTING only         : MACRO {e[1]*100:.2f}%  cand/ent {e[2]:.1f}")
    print(f"v4 keys only          : MACRO {nk[1]*100:.2f}%  cand/ent {nk[2]:.1f}")
    print(f"EXISTING + v4 keys    : MACRO {c[1]*100:.2f}%  cand/ent {c[2]:.1f}")
    print(f"=> ceiling now {c[1]*100:.2f}%  (v2 was 91.5%, baseline 87.9%)")
    print(f"[done {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
