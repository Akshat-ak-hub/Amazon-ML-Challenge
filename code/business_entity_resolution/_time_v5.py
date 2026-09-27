"""Time the FIXED blocking_v5 scoring on a small S1 slice (fast go/no-go check).

Builds the disk index (unavoidable ~13 min) then times scoring the first
N S1 entities with the chunked bulk method. Reports entities/sec so we can
extrapolate the full 2.2M runtime BEFORE committing to it.

Run:  python _time_v5.py --limit 40000
"""
import argparse, sys, time, sqlite3
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C
import blocking_v5 as B

DELIM = "\t"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40000)
    args = ap.parse_args()
    t0 = time.time()

    db = C.WORK_DIR / "blockidx_time.sqlite"
    if db.exists(): db.unlink()
    conn = sqlite3.connect(str(db))
    for p in ("journal_mode=OFF","synchronous=OFF","cache_size=-262144","temp_store=FILE"):
        conn.execute("PRAGMA "+p)
    conn.execute("CREATE TABLE posting (key INTEGER, idx INTEGER, w REAL)")
    cur = conn.cursor()
    ids=[]; buf=[]; n=0
    for path in (C.TRAIN_SOURCE2, C.TRAIN_SOURCE3):
        for eid,name,addr in B.stream(path):
            idx=len(ids); ids.append(eid)
            for k,w in B.keyed(name,addr): buf.append((k,idx,w))
            if len(buf)>=500_000: cur.executemany("INSERT INTO posting VALUES (?,?,?)",buf); buf.clear()
            n+=1
            if n%2_000_000==0: print(f"  streamed {n:,} ({time.time()-t0:.0f}s)",flush=True)
    if buf: cur.executemany("INSERT INTO posting VALUES (?,?,?)",buf); buf.clear()
    conn.commit()
    print(f"  postings written ({time.time()-t0:.0f}s); indexing...",flush=True)
    cur.execute("CREATE INDEX ix_key ON posting(key)"); conn.commit()
    over=set(k for (k,) in cur.execute("SELECT key FROM posting GROUP BY key HAVING COUNT(*)>?",(B.MAX_FANOUT,)))
    print(f"  index built, {len(over):,} over-common ({time.time()-t0:.0f}s)",flush=True)

    # time scoring the first --limit S1
    ts = time.time()
    CHUNK=20000; STEP=20000; done=0
    chunk=[]
    def flush(chunk):
        if not chunk: return 0
        need=set()
        for _,kws in chunk:
            for k,_w in kws:
                if k not in over: need.add(k)
        postings=defaultdict(list); need=list(need)
        for i in range(0,len(need),STEP):
            sub=need[i:i+STEP]; q=",".join("?"*len(sub))
            for (kk,idx,pw) in cur.execute(f"SELECT key,idx,w FROM posting WHERE key IN ({q})",sub):
                postings[kk].append((idx,pw))
        w=0
        for eid,kws in chunk:
            sc=defaultdict(float)
            for k,_w in kws:
                pl=postings.get(k)
                if pl:
                    for idx,pw in pl: sc[idx]+=pw
            ranked=sorted(sc.items(),key=lambda kv:kv[1],reverse=True)[:B.TOP_K]
            w+=1
        return w
    for eid,name,addr in B.stream(C.TRAIN_SOURCE1):
        chunk.append((eid,B.keyed(name,addr)))
        if len(chunk)>=CHUNK:
            done+=flush(chunk); chunk=[]
            el=time.time()-ts
            print(f"  scored {done:,} in {el:.0f}s -> {done/el:.0f} ent/s; "
                  f"full 2.2M ETA {2206821/(done/el)/60:.0f} min",flush=True)
        if done>=args.limit: break
    conn.close(); db.unlink()
    print(f"[done {time.time()-t0:.0f}s]")

if __name__=="__main__":
    main()
