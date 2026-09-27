"""Path B — SCORED candidate ranker (disk index). Keep high recall, kill the flood.

v4 got 93.99% recall but 197.6 cand/entity (union of all key matches). Most are
junk from broad address keys. Fix: score each candidate by WEIGHTED shared-key
evidence, rank, and cap at K. Strong precise keys (exact name, PIN, exact addr,
sorted-name, phonetic, domain-concat) weigh a lot; broad address keys weigh a
little. True matches share high-weight keys -> rank near the top -> survive the cap.

We measure recall at several K (30/50/80) AND candidates/entity, to find the
sweet spot: recall ~93% at a manageable K.

Reuses the disk index build from v4 but stores (key, idx) with an implicit weight
derived from the key PREFIX (looked up at query time). Memory-safe.

Run:  python _block_exp5.py --sample 20000
"""
from __future__ import annotations
import argparse, sys, time, hashlib, re, sqlite3
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C
import normalize as N

DELIM = "\t"
_NUM = re.compile(r"\d+")

# key weights by type: precise keys high, broad keys low
WEIGHT = {
    "NC": 10.0,   # exact core name
    "AF": 10.0,   # exact full address
    "NS": 7.0,    # sorted-token name signature
    "NP": 5.0,    # full phonetic signature
    "Z":  6.0,    # exact PIN/ZIP
    "DC": 8.0,    # domain-concat name
    "AZ": 3.0,    # number + distinctive token (broad-ish)
    "AC": 4.0,    # concat-address prefix
}


def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
                          "little", signed=True)


def keyed(name_raw, addr_raw):
    """Return list of (key_hash, weight)."""
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
        first_num = ""
        for t in ad.tokens:
            for m in _NUM.findall(t):
                z = m.lstrip("0") or "0"
                if len(z) >= 2: first_num = z; break
            if first_num: break
        alphas = sorted((t for t in ad.tokens if t.isalpha() and len(t) >= 4),
                        key=len, reverse=True)
        if first_num and alphas:
            out.append((_h("AZ:" + first_num + "_" + alphas[0]), WEIGHT["AZ"]))
        addr_concat = "".join(t for t in ad.tokens if t.isalpha())
        if len(addr_concat) >= 12:
            out.append((_h("AC:" + addr_concat[:24]), WEIGHT["AC"]))
    return out


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

    # disk index: posting(key, idx, w)  -- store weight too
    db = C.WORK_DIR / "blockidx_exp5.sqlite"
    if db.exists(): db.unlink()
    conn = sqlite3.connect(str(db))
    for pragma in ("journal_mode=OFF","synchronous=OFF","cache_size=-262144","temp_store=FILE"):
        conn.execute("PRAGMA " + pragma)
    conn.execute("CREATE TABLE posting (key INTEGER, idx INTEGER, w REAL)")
    cur = conn.cursor()
    ids = []; buf = []; n = 0
    for path in (C.TRAIN_SOURCE2, C.TRAIN_SOURCE3):
        for eid, name, addr in stream(path):
            idx = len(ids); ids.append(eid)
            for k, w in keyed(name, addr):
                buf.append((k, idx, w))
            if len(buf) >= 500_000:
                cur.executemany("INSERT INTO posting VALUES (?,?,?)", buf); buf.clear()
            n += 1
            if n % 2_000_000 == 0:
                print(f"  streamed {n:,} ({time.time()-t0:.0f}s)", flush=True)
    if buf: cur.executemany("INSERT INTO posting VALUES (?,?,?)", buf); buf.clear()
    conn.commit()
    print(f"  postings written ({time.time()-t0:.0f}s); indexing ...", flush=True)
    cur.execute("CREATE INDEX ix_key ON posting(key)")
    conn.commit()
    over = set(k for (k,) in cur.execute(
        "SELECT key FROM posting GROUP BY key HAVING COUNT(*) > ?", (args.maxfan,)))
    print(f"  index built, {len(over):,} over-common keys ignored ({time.time()-t0:.0f}s)", flush=True)

    # scored candidates for the sample
    Ks = [30, 50, 80]
    ranked_cache = {}
    done = 0
    for eid, name, addr in stream(C.TRAIN_SOURCE1):
        if eid in sample:
            scores = defaultdict(float)
            for k, w in keyed(name, addr):
                if k in over: continue
                for (i, pw) in cur.execute("SELECT idx, w FROM posting WHERE key=?", (k,)):
                    scores[i] += pw
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            ranked_cache[eid] = [ids[i] for i, _ in ranked]
            done += 1
            if done % 5000 == 0:
                print(f"  scored {done:,}/{len(sample):,} ({time.time()-t0:.0f}s)", flush=True)
    conn.close(); db.unlink()

    def recall_at(K):
        tt=rec=0; ps=0.0; ent=0; ctot=0
        for s1 in sample:
            truth = gt.get(s1, set())
            topk = set(ranked_cache.get(s1, [])[:K]) | existing.get(s1, set())
            ctot += len(topk); ent += 1
            if not truth: ps += 1.0; continue
            got = truth & topk; tt += len(truth); rec += len(got); ps += len(got)/len(truth)
        return (rec/tt if tt else 0),(ps/ent if ent else 0),(ctot/ent if ent else 0)

    print("\n===== PATH B: SCORED RANKER recall (existing + ranked top-K) =====")
    print(f"sample: {len(sample):,}")
    for K in Ks:
        mi, ma, c = recall_at(K)
        print(f"  K={K:3d} : MACRO {ma*100:.2f}%  MICRO {mi*100:.2f}%  cand/ent {c:.1f}")
    print(f"(v4 union was 93.99% @ 197.6 cand/ent; baseline 87.9% @ 18.4)")
    print(f"[done {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
