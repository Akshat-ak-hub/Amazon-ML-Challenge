"""Build recstore_test.sqlite for the CURRENT candidates_test.tsv (new v5 set).

predict_test_fast.py reuses recstore_test.sqlite; after we swap in the new v5
candidates, the store must be rebuilt so it contains every needed S1/S2/S3 record.
Reuses predict_test.build_test_store.
"""
import sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))
import config as C
import predict_test as PT

DELIM = "\t"


def main():
    t0 = time.time()
    cand_path = C.WORK_DIR / "candidates_test.tsv"
    cand = {}
    with open(cand_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            cand[s1] = [x for x in rest.split(",") if x.strip()] if rest.strip() else []
    needed = set(cand)
    for lst in cand.values():
        needed.update(lst)
    print(f"[store] {len(cand):,} S1, need {len(needed):,} records", flush=True)

    db_path = C.WORK_DIR / "recstore_test.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = PT.build_test_store(db_path, needed)
    (count,) = conn.execute("SELECT COUNT(*) FROM rec").fetchone()
    conn.close()
    print(f"[store] built recstore_test.sqlite: {count:,} rows ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
