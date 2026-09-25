"""Stage 1 — Exploratory Data Analysis (memory-safe, streaming).

Never loads a full 5M-row source into RAM. Streams line-by-line and accumulates
only small aggregates. Produces the numbers that drive every downstream decision:
row counts, missing rates, country distribution, and — most importantly — the
ground-truth match-count distribution (singleton rate, S2/S3 split).

Run:  python src/eda.py
"""
import sys
from collections import Counter
from pathlib import Path

# Allow running as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

DELIM = "\t"


def stream_lines(path):
    """Yield decoded lines (excluding header), tolerant of bad bytes."""
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f, None)  # skip header
        for line in f:
            if line.strip():
                yield line.rstrip("\n")


def analyze_source(path, label):
    """Stream one source file; return small aggregate stats."""
    n = 0
    empty_name = empty_addr = 0
    country = Counter()
    for line in stream_lines(path):
        parts = line.split(DELIM)
        # entity_id, business_name, business_address, country
        name = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""
        ctry = parts[3] if len(parts) > 3 else ""
        n += 1
        if not name.strip():
            empty_name += 1
        if not addr.strip():
            empty_addr += 1
        country[ctry.strip()] += 1
    print(f"\n=== {label} ===")
    print(f"  rows:              {n:,}")
    print(f"  empty name:        {empty_name:,} ({empty_name/n*100:.2f}%)")
    print(f"  empty address:     {empty_addr:,} ({empty_addr/n*100:.2f}%)")
    print(f"  country breakdown: {dict(country.most_common())}")
    return n


def analyze_ground_truth(path):
    """Stream the ground truth; report match-count distribution & singleton rate."""
    n = 0
    singletons = 0
    match_count_hist = Counter()   # number of matches -> how many S1 entities
    s2_matches = s3_matches = 0
    total_matches = 0
    for line in stream_lines(path):
        s1, _, rest = line.partition(DELIM)
        ids = [x for x in rest.split(",") if x.strip()] if rest.strip() else []
        n += 1
        k = len(ids)
        match_count_hist[k] += 1
        if k == 0:
            singletons += 1
        total_matches += k
        for i in ids:
            if i.startswith("S2-"):
                s2_matches += 1
            elif i.startswith("S3-"):
                s3_matches += 1

    print("\n=== GROUND TRUTH ===")
    print(f"  S1 entities:            {n:,}")
    print(f"  singletons (0 matches): {singletons:,} ({singletons/n*100:.2f}%)")
    print(f"  with >=1 match:         {n-singletons:,} ({(n-singletons)/n*100:.2f}%)")
    print(f"  total match links:      {total_matches:,}")
    print(f"    -> S2 links:          {s2_matches:,}")
    print(f"    -> S3 links:          {s3_matches:,}")
    if n - singletons:
        print(f"  avg matches (non-singleton): {total_matches/(n-singletons):.2f}")
    print("  match-count distribution (k -> #entities, top 15):")
    for k in sorted(match_count_hist)[:15]:
        cnt = match_count_hist[k]
        print(f"    {k:>3} matches: {cnt:>10,} ({cnt/n*100:5.2f}%)")
    mx = max(match_count_hist)
    print(f"  max matches for a single S1 entity: {mx}")


def main():
    print("Stage 1 EDA — streaming, memory-safe")
    print("Reading from:", C.DATASET_DIR)

    print("\n########## TRAIN ##########")
    analyze_source(C.TRAIN_SOURCE1, "train_source1")
    analyze_source(C.TRAIN_SOURCE2, "train_source2")
    analyze_source(C.TRAIN_SOURCE3, "train_source3")
    analyze_ground_truth(C.TRAIN_GROUND_TRUTH)

    print("\n########## TEST ##########")
    analyze_source(C.TEST_SOURCE1, "test_source1")
    analyze_source(C.TEST_SOURCE2, "test_source2")
    analyze_source(C.TEST_SOURCE3, "test_source3")


if __name__ == "__main__":
    import io
    log_path = C.WORK_DIR / "eda_report.txt"
    C.WORK_DIR.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    _real = sys.stdout
    sys.stdout = buf
    try:
        main()
    finally:
        sys.stdout = _real
    text = buf.getvalue()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
    # Also echo to console with safe encoding.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    print("\n[EDA report written to]", log_path)
