# Business Entity Resolution — Pipeline (Amazon ML Challenge 2026)

Links records that refer to the same real-world business across three noisy sources
(Source 1 = deduplicated reference; Sources 2 & 3 = noisy fragments). Optimised for the
competition metric **macro-averaged F₀.₅** (precision weighted 2× over recall).

**Headline results (held-out validation, full training data):**
- Blocking recall ceiling: **86.8% micro / 87.5% macro** (US 90.6%, India 81.9%)
- Matching model validation **macro-F₀.₅ = 0.9279** at decision threshold **0.89**
- Final model: **LightGBM** (MIT-licensed, ~4 MB, far under the 8B-parameter limit)

---

## 1. Environment

```bash
python -m pip install -r requirements.txt
```
Python 3.10+ (developed on 3.14). No internet or external data is used at runtime
(fair-play compliant). See `requirements.txt` for pinned versions.

## 2. Data layout

The pipeline reads the provided challenge files. Paths are configured in
`src/config.py` — edit `DATA_ROOT` to point at your dataset root:

```
<DATA_ROOT>/dataset/train/train_source1.tsv
<DATA_ROOT>/dataset/train/train_source2.tsv
<DATA_ROOT>/dataset/train/train_source3.tsv
<DATA_ROOT>/dataset/train/train_ground_truth.tsv
<DATA_ROOT>/dataset/test/test_source1.tsv
<DATA_ROOT>/dataset/test/test_source2.tsv
<DATA_ROOT>/dataset/test/test_source3.tsv
```
Intermediates are written to `<DATA_ROOT>/work/`, final outputs to `<DATA_ROOT>/output/`.

## 3. End-to-end reproduction

Run from `code/business_entity_resolution/`:

```bash
# (optional) exploratory data analysis — row counts, singleton rate, country mix
python src/eda.py

# STEP 1 — Blocking on TRAIN + measure recall ceiling (writes work/candidates_train.tsv)
python src/blocking.py --split train

# STEP 2 — Build labelled pairwise feature matrix (memory-safe, chunked; SQLite store)
#          writes work/trainmat_{train,valid}.npz + pairids_*.tsv
python src/build_training.py

# STEP 3 — Train LightGBM + tune the F0.5 threshold on a held-out entity-level split
#          writes work/match_model.txt + work/threshold.txt
python src/model.py

# STEP 4 — Blocking + scoring on TEST -> final submission files
#          writes output/candidate_pairs.tsv and output/matching_results.tsv
python src/predict_test.py

# STEP 5 — Validate the outputs against the official rules before submitting
python ../../utils/validate_submission.py \
    --matching <DATA_ROOT>/output/matching_results.tsv \
    --candidate <DATA_ROOT>/output/candidate_pairs.tsv \
    --test-dir  <DATA_ROOT>/dataset/test
```

### Single-file alternative (high-RAM machines: Kaggle/Colab ≥ 25 GB)

`kaggle_pipeline.py` runs the whole thing in-memory (no chunking) in one process:
edit `DATA_DIR`/`OUT_DIR` at the top, then `python kaggle_pipeline.py`. It mirrors the
same normalization, blocking, features, model, and threshold as the modular pipeline.

## 4. Module overview (`src/`)

| File | Role |
| --- | --- |
| `config.py` | Paths + tunables (`BLOCK_K`, seeds, threshold). |
| `eda.py` | Streaming exploratory analysis (memory-safe). |
| `normalize.py` | Name/address/country normalization: transliteration (unidecode), legal-suffix splitting, address-abbreviation standardization, PIN/ZIP + landmark extraction, pure-Python Soundex. Unit-tested. |
| `blocking.py` | Candidate generation via an in-memory inverted index over **name AND address** keys (char n-grams, phonetic codes, token bigrams, address n-grams / number-combos, PIN). Measures the recall ceiling on train. |
| `features.py` | 19 pairwise similarity features (name: Jaccard/Levenshtein/Jaro-Winkler/token-sort/trigram/phonetic/…; address: token Jaccard/4-gram/PIN/shared-numbers; country same/diff/unknown flag). Unit-tested. |
| `build_training.py` | Memory-safe chunked builder: SQLite record store + incremental feature flush; entity-level train/valid split (no leakage). |
| `model.py` | LightGBM training + **vectorized** F₀.₅ threshold search. |
| `predict_test.py` | Test blocking + scoring + greedy conflict resolution → output TSVs. |

Tests: `tests/test_normalize.py` (14 cases), `tests/test_features.py` (7 cases).

```bash
python tests/test_normalize.py
python tests/test_features.py
```

## 5. Design notes

- **Blocking is the recall ceiling.** Adding address-based keys (not just name) lifted the
  ceiling from ~53% to ~87% — many true matches have garbled names but identical addresses.
- **Memory-safe by design.** Streaming TSV reads + a disk-backed SQLite record store keep
  peak RAM low, so the full ~12M-record dataset runs on an 8 GB machine.
- **Open-set country.** Country is never one-hot/hard-filtered — only a same/different/unknown
  flag — so the test-only country (France) flows through unchanged.
- **Precision-first threshold.** F₀.₅ weights precision 2×, so the tuned threshold (0.89) is
  deliberately conservative; a greedy conflict-resolution pass assigns each S2/S3 record to
  its highest-scoring S1 entity.

## 6. Fair play

Uses only the provided challenge files. No external databases, APIs, geocoding, or internet
data augmentation. Final model is MIT-licensed and well under 8B parameters.
