"""
Amazon ML Challenge 2026 — Business Entity Resolution
SELF-CONTAINED KAGGLE PIPELINE (single file, runs end-to-end)

Designed for Kaggle's ~30 GB RAM: loads everything in-memory (fast), no SQLite
chunking needed. Produces matching_results.tsv + candidate_pairs.tsv for the TEST set.

FAIR PLAY: uses ONLY the provided challenge files. No external data, APIs, or lookups.

SETUP ON KAGGLE
---------------
1. Create a NEW notebook.
2. Add your challenge data as a PRIVATE Kaggle dataset (the 7 .tsv files). Note its path,
   e.g. /kaggle/input/amazon-ml-be/  (adjust DATA_DIR below to match).
3. Settings -> turn OFF internet (fair-play; not needed).
4. pip install rapidfuzz unidecode  (allowed: libraries, not data) -- or add via
   "Add-ons -> Install dependencies". If internet is off, add rapidfuzz+unidecode as a
   Kaggle "utility dataset" or enable internet ONLY for pip install then disable.
5. Paste this file into a cell (or upload as a utility script) and run.

OUTPUT
------
Writes /kaggle/working/matching_results.tsv and /kaggle/working/candidate_pairs.tsv
which you can download and submit.

The logic MIRRORS the local pipeline (same normalization, blocking keys, features,
LightGBM model, F0.5 threshold) so results are consistent with the repo.
"""
from __future__ import annotations

import argparse
import array
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

# ---- config (EDIT DATA_DIR for your Kaggle dataset path) --------------------
DATA_DIR = "/kaggle/input/datasetforamazon/student_resource/dataset"  # <-- your Kaggle dataset
OUT_DIR = "/kaggle/working"
DELIM = "\t"
NGRAM_NAME = 3
NGRAM_ADDR = 4
MAX_FANOUT = 2000
MIN_SHARED = 2
TOP_K = 20
VALID_FRACTION = 0.15
SEED = 42
BETA = 0.5

# ---- dependencies -----------------------------------------------------------
from unidecode import unidecode          # noqa: E402
from rapidfuzz import fuzz, distance     # noqa: E402
import lightgbm as lgb                   # noqa: E402


def path(split, src):
    # Files live under train/ and test/ subfolders (Kaggle student_resource layout).
    return f"{DATA_DIR}/{split}/{split}_source{src}.tsv"


GT_PATH = f"{DATA_DIR}/train/train_ground_truth.tsv"

# =============================================================================
# 1. NORMALIZATION  (mirrors src/normalize.py)
# =============================================================================
LEGAL_SUFFIXES = {
    "corp": "corp", "corporation": "corp", "inc": "inc", "incorporated": "inc",
    "llc": "llc", "llp": "llp", "ltd": "ltd", "limited": "ltd", "co": "co",
    "company": "co", "plc": "plc", "lp": "lp", "pvt": "pvt", "private": "pvt",
    "pl": "pvt", "sarl": "sarl", "sas": "sas", "sa": "sa", "sci": "sci",
    "eurl": "eurl", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "srl": "srl", "spa": "spa",
}
ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court",
    "sq": "square", "hwy": "highway", "pkwy": "parkway", "apt": "apartment",
    "ste": "suite", "fl": "floor", "bldg": "building", "no": "number",
    "ph": "phase", "opp": "opposite", "nr": "near", "kh": "khasra",
    "gali": "lane", "marg": "road", "rue": "rue", "r": "rue", "bd": "boulevard",
}
import re
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI = re.compile(r"\s+")
_PIN = re.compile(r"\b(\d{5,6})\b")
_SNDX = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
         **dict.fromkeys("dt", "3"), **dict.fromkeys("l", "4"),
         **dict.fromkeys("mn", "5"), **dict.fromkeys("r", "6")}


def soundex(w):
    w = re.sub(r"[^a-z]", "", unidecode(w).lower())
    if not w:
        return ""
    out = [w[0].upper()]
    prev = _SNDX.get(w[0], "")
    for c in w[1:]:
        code = _SNDX.get(c, "")
        if code and code != prev:
            out.append(code)
        if c not in "hw":
            prev = code
    return (out[0] + "".join(out[1:])[:3] + "000")[:4]


def clean(t):
    if not t:
        return ""
    t = unidecode(t).lower().replace("&", " and ")
    return _MULTI.sub(" ", _PUNCT.sub(" ", t)).strip()


def norm_name(raw):
    c = clean(raw or "")
    toks = c.split()
    # merge dotted acronym suffixes
    merged, i = [], 0
    while i < len(toks):
        if len(toks[i]) == 1 and toks[i].isalpha():
            j, run = i, []
            while j < len(toks) and len(toks[j]) == 1 and toks[j].isalpha():
                run.append(toks[j]); j += 1
            joined = "".join(run)
            if len(run) >= 2 and joined in LEGAL_SUFFIXES:
                merged.append(joined); i = j; continue
            merged.extend(run); i = j
        else:
            merged.append(toks[i]); i += 1
    toks = merged
    legal, core = [], []
    for t in toks:
        (legal if t in LEGAL_SUFFIXES else core).append(t)
    core_s = " ".join(core)
    phon = " ".join(soundex(t) for t in core if t)
    return {"full": c, "core": core_s, "tokens": core, "phon": phon}


def norm_addr(raw):
    if not raw or not raw.strip():
        return {"present": False, "full": "", "tokens": [], "pin": ""}
    pin = ""
    m = _PIN.search(unidecode(raw))
    if m:
        pin = m.group(1)
    c = clean(raw)
    toks = [ADDR_ABBREV.get(t, t) for t in c.split()]
    return {"present": True, "full": " ".join(toks), "tokens": toks, "pin": pin}


# =============================================================================
# 2. BLOCKING KEYS  (mirrors src/blocking.py v2)
# =============================================================================
def ngrams(s, n):
    t = s.replace(" ", "")
    if len(t) <= n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def record_keys(nm, ad):
    keys = set()
    for g in ngrams(nm["core"], NGRAM_NAME):
        keys.add("n:" + g)
    for p in nm["phon"].split():
        if p:
            keys.add("p:" + p)
    toks = nm["tokens"]
    for i in range(len(toks) - 1):
        keys.add("b:" + "_".join(sorted((toks[i], toks[i + 1]))))
    for t in toks:
        if len(t) >= 6:
            keys.add("t:" + t)
    if toks:
        keys.add("f:" + toks[0])
    if ad["present"]:
        if ad["pin"]:
            keys.add("z:" + ad["pin"])
        atoks = ad["tokens"]
        addr_core = "".join(t for t in atoks if t.isalpha())
        for g in ngrams(addr_core, NGRAM_ADDR):
            keys.add("an:" + g)
        nums = [t for t in atoks if t.isdigit()]
        alphas = [t for t in atoks if t.isalpha() and len(t) >= 4]
        for num in nums[:2]:
            for a in alphas[:3]:
                keys.add("ai:" + num + "_" + a)
        for t in alphas:
            if len(t) >= 5:
                keys.add("at:" + t)
    return keys


# =============================================================================
# 3. FEATURES  (mirrors src/features.py)
# =============================================================================
FEATURE_NAMES = [
    "name_tok_jaccard", "name_lev_ratio", "name_jaro_winkler", "name_token_sort",
    "name_token_set", "name_trigram_jaccard", "name_phonetic_overlap",
    "name_exact_core", "name_full_core_delta", "name_len_ratio",
    "name_tokcount_delta", "addr_present_both", "addr_tok_jaccard",
    "addr_token_sort", "addr_4gram_jaccard", "addr_pin_both", "addr_pin_exact",
    "addr_shared_nums", "country_flag",
]


def jac(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(nm1, ad1, cc1, nm2, ad2, cc2):
    c1, c2 = nm1["core"], nm2["core"]
    t1, t2 = set(nm1["tokens"]), set(nm2["tokens"])
    name_tok_jaccard = jac(t1, t2)
    name_lev = distance.Levenshtein.normalized_similarity(c1, c2) if (c1 or c2) else 1.0
    name_jw = distance.JaroWinkler.normalized_similarity(c1, c2) if (c1 or c2) else 1.0
    name_tsort = fuzz.token_sort_ratio(c1, c2) / 100.0
    name_tset = fuzz.token_set_ratio(c1, c2) / 100.0
    name_tri = jac(ngrams(c1, 3), ngrams(c2, 3))
    p1, p2 = set(nm1["phon"].split()), set(nm2["phon"].split())
    name_phon = jac(p1, p2)
    name_exact = 1.0 if (c1 and c1 == c2) else 0.0
    fdelta = abs(fuzz.ratio(nm1["full"], nm2["full"]) / 100.0 - fuzz.ratio(c1, c2) / 100.0)
    l1, l2 = len(c1), len(c2)
    name_lenr = min(l1, l2) / max(l1, l2) if max(l1, l2) else 1.0
    name_tcd = abs(len(nm1["tokens"]) - len(nm2["tokens"]))
    both = 1.0 if (ad1["present"] and ad2["present"]) else 0.0
    if both:
        at1, at2 = set(ad1["tokens"]), set(ad2["tokens"])
        a_jac = jac(at1, at2)
        a_ts = fuzz.token_sort_ratio(ad1["full"], ad2["full"]) / 100.0
        a_4g = jac(ngrams(ad1["full"], 4), ngrams(ad2["full"], 4))
        n1 = {t for t in ad1["tokens"] if t.isdigit()}
        n2 = {t for t in ad2["tokens"] if t.isdigit()}
        a_sn = float(len(n1 & n2))
    else:
        a_jac = a_ts = a_4g = a_sn = 0.0
    pin_both = 1.0 if (ad1["pin"] and ad2["pin"]) else 0.0
    pin_exact = 1.0 if (ad1["pin"] and ad1["pin"] == ad2["pin"]) else 0.0
    a, b = (cc1 or "").strip().lower(), (cc2 or "").strip().lower()
    cflag = 0.0 if (not a or not b) else (1.0 if a == b else -1.0)
    return [name_tok_jaccard, name_lev, name_jw, name_tsort, name_tset, name_tri,
            name_phon, name_exact, fdelta, name_lenr, float(name_tcd), both,
            a_jac, a_ts, a_4g, pin_both, pin_exact, a_sn, cflag]


# =============================================================================
# 4. IO helpers
# =============================================================================
def load_source(p):
    """Return list of (eid, nm, ad, country). Normalizes inline."""
    out = []
    with open(p, encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(DELIM)
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            ctry = parts[3] if len(parts) > 3 else ""
            out.append((eid, norm_name(name), norm_addr(addr), ctry))
    return out


def build_index(records):
    ids, index = [], defaultdict(lambda: array.array("i"))
    for eid, nm, ad, cc in records:
        idx = len(ids)
        ids.append(idx)  # placeholder; we store eid separately
        for k in record_keys(nm, ad):
            index[k].append(idx)
    for k in list(index.keys()):
        if len(index[k]) > MAX_FANOUT:
            del index[k]
    return index


def candidates_for(keys, index, eids):
    counter = Counter()
    for k in keys:
        post = index.get(k)
        if post:
            counter.update(post)
    if not counter:
        return []
    items = [(i, c) for i, c in counter.items() if c >= MIN_SHARED] or counter.most_common(TOP_K)
    items.sort(key=lambda kv: (-kv[1], eids[kv[0]]))
    return [eids[i] for i, _ in items[:TOP_K]]


# =============================================================================
# 5. PIPELINE
# =============================================================================
def run(split_for_output="test"):
    t0 = time.time()
    print("Loading TRAIN sources ...", flush=True)
    s1 = load_source(path("train", 1))
    s2 = load_source(path("train", 2))
    s3 = load_source(path("train", 3))
    print(f"  train s1={len(s1):,} s2={len(s2):,} s3={len(s3):,} ({time.time()-t0:.0f}s)", flush=True)

    # index S2+S3
    s23 = s2 + s3
    eids23 = [r[0] for r in s23]
    print("Building blocking index (train) ...", flush=True)
    index = build_index(s23)
    rec23 = {r[0]: (r[1], r[2], r[3]) for r in s23}
    print(f"  index keys={len(index):,} ({time.time()-t0:.0f}s)", flush=True)

    # ground truth
    gt = {}
    with open(GT_PATH, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, _, rest = line.rstrip("\n").partition(DELIM)
            gt[a] = set(x for x in rest.split(",") if x.strip()) if rest.strip() else set()

    # build labeled features from train candidates
    print("Generating train candidates + features ...", flush=True)
    X, y, s1ids = [], [], []
    for eid, nm, ad, cc in s1:
        cands = candidates_for(record_keys(nm, ad), index, eids23)
        truth = gt.get(eid, set())
        for cid in cands:
            nm2, ad2, cc2 = rec23[cid]
            X.append(pair_features(nm, ad, cc, nm2, ad2, cc2))
            y.append(1 if cid in truth else 0)
            s1ids.append(eid)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    s1ids = np.array(s1ids)
    print(f"  train pairs={len(y):,} positives={int(y.sum()):,} ({time.time()-t0:.0f}s)", flush=True)

    # entity-level split
    def is_valid(e):
        return (zlib.crc32(e.encode()) & 0xFFFFFFFF) % 100 < int(VALID_FRACTION * 100)
    vmask = np.array([is_valid(e) for e in s1ids])
    Xtr, ytr = X[~vmask], y[~vmask]
    Xva, yva, s1va = X[vmask], y[vmask], s1ids[vmask]

    # train
    print("Training LightGBM ...", flush=True)
    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    params = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                  num_leaves=63, min_child_samples=100, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, scale_pos_weight=pos_w,
                  seed=SEED, verbose=-1)
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

    # tune F0.5 threshold
    def fbeta(ids, yt, yp, beta=BETA):
        tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int); ent = set()
        for s, a, b in zip(ids, yt, yp):
            ent.add(s)
            if b and a: tp[s] += 1
            elif b and not a: fp[s] += 1
            elif a and not b: fn[s] += 1
        b2 = beta * beta; tot = 0.0
        for s in ent:
            t, f_, n_ = tp[s], fp[s], fn[s]
            if t == 0 and f_ == 0 and n_ == 0: tot += 1.0; continue
            if t == 0: continue
            pr = t / (t + f_); rc = t / (t + n_); d = b2 * pr + rc
            tot += (1 + b2) * pr * rc / d if d else 0.0
        return tot / len(ent) if ent else 0.0
    sc = model.predict(Xva, num_iteration=model.best_iteration)
    best_t, best_f = 0.5, -1
    for t in np.arange(0.1, 0.95, 0.01):
        f = fbeta(s1va, yva, (sc >= t).astype(np.int8))
        if f > best_f:
            best_f, best_t = f, float(t)
    print(f"  BEST threshold={best_t:.2f}  valid macro-F0.5={best_f:.4f}", flush=True)

    # --- FREE TRAINING MEMORY before loading TEST (prevents OOM) ---
    # Explicit `del` genuinely releases these; keep only `model` and `best_t`.
    print("Freeing training memory ...", flush=True)
    import gc
    del s1, s2, s3, s23, eids23, index, rec23
    del X, y, s1ids, Xtr, ytr, Xva, yva, s1va, vmask, dtr, dva, sc
    gc.collect()
    # ---------------------------------------------------------------

    # ------ TEST inference ------
    print("Loading TEST sources ...", flush=True)
    ts1 = load_source(path(split_for_output, 1))
    ts2 = load_source(path(split_for_output, 2))
    ts3 = load_source(path(split_for_output, 3))
    ts23 = ts2 + ts3
    teids23 = [r[0] for r in ts23]
    tindex = build_index(ts23)
    trec23 = {r[0]: (r[1], r[2], r[3]) for r in ts23}
    print(f"  test s1={len(ts1):,} s2/3={len(ts23):,} ({time.time()-t0:.0f}s)", flush=True)

    print("Scoring test + writing outputs ...", flush=True)
    cand_out = open(f"{OUT_DIR}/candidate_pairs.tsv", "w", encoding="utf-8")
    match_out = open(f"{OUT_DIR}/matching_results.tsv", "w", encoding="utf-8")
    cand_out.write("source1_entity_id\tcandidate_entity_ids\n")
    match_out.write("source1_entity_id\tmatched_entity_ids\n")
    # collect scored matches, then greedy conflict resolution
    best_claim = {}  # cand id -> (score, s1)
    pending = []     # (s1, [(cid, score)])
    for eid, nm, ad, cc in ts1:
        cands = candidates_for(record_keys(nm, ad), tindex, teids23)
        cand_out.write(f"{eid}\t{','.join(cands)}\n")
        scored = []
        if cands:
            feats = []
            for cid in cands:
                nm2, ad2, cc2 = trec23[cid]
                feats.append(pair_features(nm, ad, cc, nm2, ad2, cc2))
            preds = model.predict(np.asarray(feats, dtype=np.float32),
                                  num_iteration=model.best_iteration)
            for cid, sc_ in zip(cands, preds):
                if sc_ >= best_t:
                    scored.append((cid, float(sc_)))
                    if cid not in best_claim or sc_ > best_claim[cid][0]:
                        best_claim[cid] = (float(sc_), eid)
        pending.append((eid, scored))
    # resolve: keep a cand for s1 only if s1 is its highest scorer
    for eid, scored in pending:
        keep = [cid for cid, sc_ in scored if best_claim.get(cid, (0, None))[1] == eid]
        match_out.write(f"{eid}\t{','.join(keep)}\n")
    cand_out.close(); match_out.close()
    print(f"DONE in {time.time()-t0:.0f}s. Outputs in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    # Notebook-safe: parse_known_args ignores the kernel's injected "-f ...json" arg.
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    args, _unknown = ap.parse_known_args()
    run(args.split)
