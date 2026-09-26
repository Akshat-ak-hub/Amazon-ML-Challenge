"""Stage 4 (v2) — Blocking via in-memory inverted index (fast, high recall).

Why v2: the SQL GROUP BY join was ~2h and, after dropping char n-grams to save
time, recall ceiling was only ~53%. v2 restores char n-grams (the strongest fuzzy
signal for typos + Devanagari transliteration) and replaces the slow SQL join with
a compact in-memory inverted index scored by shared-key counting.

Memory plan (fits ~8 GB):
  * S2/S3 entity IDs stored once in a list; referenced everywhere by int32 index.
  * Inverted index: dict[key] -> array('i', [entity_idx, ...]) (posting list).
  * Over-common keys (fanout > MAX_FANOUT) dropped: they cost memory and add noise.
  * Scoring one S1: walk its keys, tally Counter over posting lists, take top-K.

Keys per record:
  * char n-grams (n=3) over core name           -> typos / transliteration / fuzzy
  * phonetic Soundex per core token             -> spelling variants
  * sorted token bigrams                         -> word-order
  * long tokens (len>=6) and first token         -> distinctive anchors
  * address PIN/ZIP                              -> strong structured signal

Run:
  python src/blocking.py --split train                 # full run + recall ceiling
  python src/blocking.py --split train --sample 30000  # fast subset validation
  python src/blocking.py --split test                  # writes test candidates
"""
from __future__ import annotations

import argparse
import array
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import normalize as N  # noqa: E402

DELIM = "\t"
TOP_K = C.BLOCK_K
NGRAM_N = 3
MAX_FANOUT = 2000          # drop keys with more postings than this
MIN_SHARED = 2             # a candidate must share >= this many keys to qualify


def _log(logf, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logf:
        logf.write(line + "\n"); logf.flush()


def char_ngrams(text: str, n: int = NGRAM_N):
    t = text.replace(" ", "")
    if len(t) <= n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def record_keys(name_raw: str, addr_raw: str):
    """Set of blocking keys for one record (name AND address signals)."""
    nm = N.normalize_name(name_raw)
    ad = N.normalize_address(addr_raw)
    keys = set()

    # ---- NAME keys ----
    core = nm.core
    for g in char_ngrams(core, NGRAM_N):
        keys.add("n:" + g)
    for p in nm.phonetic.split():
        if p:
            keys.add("p:" + p)
    toks = nm.tokens
    for i in range(len(toks) - 1):
        keys.add("b:" + "_".join(sorted((toks[i], toks[i + 1]))))
    for t in toks:
        if len(t) >= 6:
            keys.add("t:" + t)
    if toks:
        keys.add("f:" + toks[0])

    # ---- ADDRESS keys (crucial: catches name-garbled / same-address matches) ----
    if ad.present:
        if ad.pin:
            keys.add("z:" + ad.pin)
        atoks = ad.tokens
        # distinctive address tokens (street names, numbers): length>=4, skip pure
        # generic words already standardized
        addr_core = "".join(t for t in atoks if t.isalpha())
        for g in char_ngrams(addr_core, 4):        # 4-grams over address letters
            keys.add("an:" + g)
        # house/building number + first alpha token combo (strong locator)
        nums = [t for t in atoks if t.isdigit()]
        alphas = [t for t in atoks if t.isalpha() and len(t) >= 4]
        for num in nums[:2]:
            for a in alphas[:3]:
                keys.add("ai:" + num + "_" + a)
        # rare-ish standalone address tokens
        for t in alphas:
            if len(t) >= 5:
                keys.add("at:" + t)
    return keys


def _stream_records(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(DELIM)
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            yield eid, name, addr


def build_index(paths, logf, log_every=1_000_000):
    """Build inverted index over S2/S3. Returns (ids, index).

    ids:   list of entity_id strings; position = int index.
    index: dict[key] -> array('i', posting list of int indices)
    """
    ids = []
    index = defaultdict(lambda: array.array("i"))
    n = 0
    t0 = time.time()
    for path in paths:
        for eid, name, addr in _stream_records(path):
            idx = len(ids)
            ids.append(eid)
            for k in record_keys(name, addr):
                index[k].append(idx)
            n += 1
            if n % log_every == 0:
                _log(logf, f"  index: {n:,} records, {len(index):,} keys ({time.time()-t0:.0f}s)")
    _log(logf, f"  index built: {n:,} records, {len(index):,} keys ({time.time()-t0:.0f}s)")

    # Drop over-common keys (memory + noise).
    dropped = 0
    for k in list(index.keys()):
        if len(index[k]) > MAX_FANOUT:
            del index[k]
            dropped += 1
    _log(logf, f"  dropped {dropped:,} over-common keys (>{MAX_FANOUT} postings); {len(index):,} remain")
    return ids, index


def candidates_for(keys, index, ids):
    """Score candidates by shared-key count; return list of (eid, score) top-K."""
    counter = Counter()
    for k in keys:
        postings = index.get(k)
        if postings:
            counter.update(postings)
    if not counter:
        return []
    # keep those with >= MIN_SHARED shared keys; fall back to best if none reach it
    items = [(i, c) for i, c in counter.items() if c >= MIN_SHARED]
    if not items:
        items = counter.most_common(TOP_K)
    items.sort(key=lambda kv: (-kv[1], ids[kv[0]]))
    return [(ids[i], c) for i, c in items[:TOP_K]]


def build_blocking(split: str, sample: int | None = None):
    C.ensure_dirs()
    if split == "train":
        s1p, s23 = C.TRAIN_SOURCE1, [C.TRAIN_SOURCE2, C.TRAIN_SOURCE3]
    else:
        s1p, s23 = C.TEST_SOURCE1, [C.TEST_SOURCE2, C.TEST_SOURCE3]

    suffix = f"_sample{sample}" if sample else ""
    logf = open(C.WORK_DIR / f"blocking_{split}{suffix}.log", "w", encoding="utf-8")
    t_all = time.time()
    _log(logf, f"[blocking v2:{split}] START (sample={sample})")

    ids, index = build_index(s23, logf)

    out_path = C.WORK_DIR / f"candidates_{split}{suffix}.tsv"
    _log(logf, f"  scoring S1 -> {out_path}")
    n = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid, name, addr in _stream_records(s1p):
            keys = record_keys(name, addr)
            cands = candidates_for(keys, index, ids)
            out.write(f"{eid}\t{','.join(c for c, _ in cands)}\n")
            n += 1
            if n % 100_000 == 0:
                _log(logf, f"  scored {n:,} S1 ({time.time()-t0:.0f}s)")
            if sample and n >= sample:
                break
    _log(logf, f"  scored {n:,} S1 in {time.time()-t0:.0f}s")
    _log(logf, f"[blocking v2:{split}] DONE total {time.time()-t_all:.0f}s")
    logf.close()
    return out_path


def measure_recall_ceiling(candidates_path, gt_path, tag=""):
    cand = {}
    with open(candidates_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            cand[s1] = set(rest.split(",")) if rest.strip() else set()

    total_links = recovered = entities = singletons = 0
    per_entity_recall_sum = 0.0
    with open(gt_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition(DELIM)
            if s1 not in cand:
                continue
            truth = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()
            entities += 1
            if not truth:
                singletons += 1
                per_entity_recall_sum += 1.0
                continue
            got = truth & cand[s1]
            total_links += len(truth)
            recovered += len(got)
            per_entity_recall_sum += len(got) / len(truth)

    micro = recovered / total_links if total_links else 0.0
    macro = per_entity_recall_sum / entities if entities else 0.0
    out = [
        f"===== RECALL CEILING {tag} =====",
        f"S1 entities evaluated: {entities:,} ({singletons:,} singletons)",
        f"true links: {total_links:,}  recovered: {recovered:,}",
        f"MICRO recall (links):      {micro*100:.2f}%",
        f"MACRO recall (per-entity): {macro*100:.2f}%",
    ]
    txt = "\n".join(out)
    print(txt, flush=True)
    with open(C.WORK_DIR / f"recall_ceiling{tag}.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return micro, macro


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--sample", type=int, default=None,
                    help="score only first N S1 entities (fast validation)")
    args = ap.parse_args()
    path = build_blocking(args.split, sample=args.sample)
    if args.split == "train":
        tag = f"_sample{args.sample}" if args.sample else ""
        measure_recall_ceiling(path, C.TRAIN_GROUND_TRUTH, tag=tag)
