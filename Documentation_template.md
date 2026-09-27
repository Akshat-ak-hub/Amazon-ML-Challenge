# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We link business records across three noisy sources with a two-stage pipeline: an
**inverted-index blocking stage** that keys on **both name and address** signals, followed by a
**LightGBM pairwise classifier** whose decision threshold is tuned directly against the
competition metric. Our central finding — discovered by inspecting the matches our first
blocker *missed* — is that a large fraction of true matches have **badly corrupted names but
near-identical addresses**; adding address-derived blocking keys lifted the recall ceiling from
**52.9% to 86.8%** and is the single biggest contributor to our score. The final model achieves
**macro-F₀.₅ = 0.9279** on a held-out, entity-level validation split of 330,378 Source-1 entities.

---

## 2. Methodology

### 2.1 Problem Analysis

We ran a streaming EDA pass over all ~23M records (never loading a full source into RAM). Key
findings that shaped every later decision:

| Observation | Value | Design consequence |
| --- | --- | --- |
| Train Source 1 / 2 / 3 records | 2,206,821 / 5,034,616 / 5,285,603 | All-pairs (≈1.1×10¹³ comparisons) is impossible → blocking is mandatory |
| True match links (train) | 7,638,365 | Split almost evenly: 3,693,619 to S2, 3,944,746 to S3 — neither source can be ignored |
| Singleton rate | **5.58%** (123,247 entities) | Worth 5.6% of the score for free if predicted correctly; punishes over-merging |
| Matches per non-singleton entity | mean **3.67**, **max 11** | A candidate cap of K=20 cannot clip a true match |
| Empty `business_address` | ~3.3% in S2/S3, 0% in S1 | Address features need an explicit "present on both sides" flag, never imputation |
| France in test only | **259,452 entities = 15.0%** of test Source 1 | Country must be an open-set label; no country-specific branching anywhere |

**Noise patterns** were catalogued from real data rather than assumed. We wrote a diagnostic that
prints true-match pairs our blocker *failed* to recover, which exposed five distinct failure
modes:

1. **Name completely different, address identical** — e.g. `BS Projects` ↔ `Gildcalo`, both at
   *"Shop No.- 4, Ground Floor, 59/101 Kanhaiya Plaza, Dalmandi, Canal Road, Kanpur"*.
   **Name-only blocking can never recover these.**
2. **Severe intra-word typos / character scrambling** — `Silver`→`Silvre`,
   `Intelligence`→`Ínelllighcnce`, `Specialists`→`Stpianlsts`, `Hypnosis`→`HPYOSRIS`.
3. **Word-order transposition** — `Physical Therapy Associates Group` ↔
   `Group Physical Therapy Asseatiaes`.
4. **Domain-style trade names** — `Secure Marketing Technologies LLC` ↔ `technologiesmarketing.com`,
   `Hall, Rushing & Hall LLC` ↔ `hallrushinghall.com` (no shared tokens at all).
5. **Devanagari transliteration** — `Pioneer Tech Private Limited` ↔ `पायोनियर टेक प्राइवेट लिमिटेड`.
   Transliteration is *lossy*: unidecode yields `paayoniyr ttek praaivett limittedd`, which shares
   almost no tokens with the Romanised original, so token-level matching fails and only
   sub-word / phonetic signals bridge the gap.

Additional noise: junk prefixes (`-- Holloway Peak Inc`, `<< Team Ecole`), accent variants
(`Léarning`), legal-suffix inconsistency, `&` vs `and`, dotted acronyms (`S.A.S`), and
municipal/landmark address formats (`KH NO. -570/13`, `Near SBI ATM`).

### 2.2 Solution Strategy

**Approach Type:** Blocking + GBDT pairwise classifier (with greedy assignment post-processing)

**Core Innovation:** **Address-derived blocking keys.** Most entity-resolution blockers key on
the business name. Our error analysis proved that is insufficient here: when a vendor corrupts the
name beyond recognition, the *address* is the only surviving signal. Adding character-4-grams over
address text, house-number × street-token composites, and distinctive address tokens to the
blocking key set raised the recall ceiling **52.9% → 86.8%** (micro) in a single change. The
classifier independently confirmed this, ranking `addr_tok_jaccard` as its joint-top feature.

Every stage is engineered to be **memory-safe**: TSVs are streamed, record text lives in a
disk-backed SQLite store, and the feature matrix is written to disk in chunks. This lets the full
~12M-record problem run end-to-end on an 8 GB laptop.

---

## 3. Candidate Generation (Blocking)

We build an **in-memory inverted index** (`key → posting list of int32 record indices`) over all
Source-2 and Source-3 records, then score each Source-1 entity's candidates by **shared-key count**
and keep the top-K. Deliberately, several cheap independent signals are unioned rather than
relying on one.

**Blocking keys used** (per record, all derived from normalized text):

| Prefix | Key | Targets noise pattern |
| --- | --- | --- |
| `n:` | character **3-grams** of the core name | typos, scrambling, transliteration drift, domain-style names |
| `p:` | **Soundex** code per core-name token (pure-Python) | phonetic spelling and transliteration variants |
| `b:` | **sorted token bigrams** | word-order transposition |
| `t:` | distinctive single tokens (length ≥ 6) | strong anchors for rare words |
| `f:` | first core token | short-name coverage |
| `z:` | **PIN / ZIP** code (5–6 digit) | strong structured locator |
| `an:` | character **4-grams** over address letters | *name-garbled / address-intact matches* |
| `ai:` | house-number × street-token composites | precise address locator |
| `at:` | distinctive address tokens (length ≥ 5) | street/locality anchors |

**Parameters:** `MAX_FANOUT = 2000` (keys with more postings are dropped as non-discriminative and
memory-costly), `MIN_SHARED = 2` (a candidate must share ≥2 keys, with a graceful fallback to the
best-scoring candidates when none reach 2), `TOP_K = 20`.

**Candidate pairs generated (test set):** **32,898,155** across 1,732,544 entities
(mean **18.99**, median 20, max 20; only 7 entities received zero candidates).

**Reduction ratio:** a naive all-pairs comparison on the test split would be
1,732,544 × 9,969,589 ≈ **1.73 × 10¹³** pairs. We score **3.29 × 10⁷** — a
**≈525,000× reduction**, i.e. we examine 0.00019% of the search space.

**How we ensured true matches were not lost.** Blocking recall is a hard ceiling on the whole
pipeline, so we measured it *before* any modelling and re-measured after every change:

| Blocking configuration | Micro recall (links) | Macro recall (per entity) |
| --- | --- | --- |
| v1 — name-only keys (phonetic + bigrams + tokens + PIN) | 52.94% | 55.61% |
| v2 — v1 + name 3-grams | 54.87% | 57.44% |
| **v3 — v2 + address keys (final)** | **86.81%** | **87.54%** |

The v3 ceiling was verified on the **full** training set (2,206,821 entities; 6,630,693 of
7,638,365 true links recovered) and matched a 30,000-entity sample (87.13% / 87.87%), confirming
the measurement is stable. Per-country recall: **US 90.61%**, **India 81.92%** (transliteration
remains the harder case), and **zero** entities were left with an empty candidate list — down from
124,731 in v1.

Because true matches peak at 11 per entity, `K = 20` provides headroom without clipping, while
keeping the candidate set small enough to be precision-friendly for the downstream model.

---

## 4. Matching Model

Each (Source 1, candidate) pair is reduced to a **19-dimensional numeric feature vector** — no raw
text reaches the model.

**Features used:**

- **Name features (11):** token Jaccard; normalized Levenshtein similarity; Jaro–Winkler;
  `token_sort_ratio` (word-order invariant); `token_set_ratio`; character-trigram Jaccard;
  Soundex-code overlap; exact core-name match flag; full-vs-core similarity delta (isolates the
  effect of legal suffixes); length ratio; token-count delta.
- **Address features (7):** present-on-both-sides flag; token Jaccard; `token_sort_ratio`;
  character-4-gram Jaccard; PIN-available-on-both flag; PIN exact-match flag; count of shared
  numeric tokens (house/building numbers).
- **Other (1):** country as a three-way flag — `same (+1)` / `different (−1)` / `unknown (0)`.
  This is the *only* country-derived feature; it is never one-hot encoded and never used to filter,
  which is precisely what lets the unseen France records flow through the identical pipeline.

Normalization (shared by every source, so nothing is source-specific): unidecode transliteration to
ASCII, lowercasing, punctuation stripping, `&`↔`and`, legal-suffix extraction into a separate field
(with dotted-acronym reassembly so `S.A.S` → `sas`), address-abbreviation standardization
(`Rd`→`Road`, `R.`→`Rue`, …), PIN/ZIP extraction, landmark-phrase isolation, and a pure-Python
Soundex. Normalization is covered by 14 unit tests built from the real noisy strings above.

**Model type:** **LightGBM** binary classifier (MIT-licensed; the saved model is ~4 MB — orders of
magnitude below the 8B-parameter cap). Chosen over an end-to-end neural approach because, with a
compact set of hand-engineered similarity features, GBDTs are more data-efficient, far cheaper to
train on CPU, easier to calibrate for a precision-sensitive metric, and trivially satisfy the
licence/size constraints.

Configuration: `learning_rate 0.05`, `num_leaves 63`, `min_child_samples 100`,
`feature_fraction 0.9`, `bagging_fraction 0.8` (freq 1), `scale_pos_weight` set to the negative/positive
ratio, 600 boosting rounds, seed 42. Validation binary log-loss improved monotonically to
**0.0724** (early stopping never triggered).

**Training data:** labels are derived from `train_ground_truth.tsv` — a candidate is positive iff it
is a true match. Train **34,642,022** pairs / validation **6,099,384** pairs, **16.28%** positive
(16.26% in validation, confirming an unbiased split).

**Threshold selection method:** F₀.₅ optimization on the validation set. We scan thresholds in
0.01 steps and compute the **exact competition metric** — macro-averaged per-entity F₀.₅ with
singletons scored 1.0 when correctly predicted empty. The search is vectorized with
`numpy.bincount` over entity codes (a naive Python implementation needed ~30+ minutes on 6.1M rows;
the vectorized version is effectively instant and was verified to return bit-identical results).
Selected threshold: **0.89** — far above the default 0.5, exactly as the 2× precision weighting
predicts.

**Assignment / conflict resolution.** Since Source 1 is deduplicated, one Source-2/3 record should
not be claimed by multiple Source-1 entities. After thresholding we apply a greedy pass assigning
each contested record to its **highest-scoring** Source-1 entity and dropping it elsewhere.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro): 0.9279** — held-out validation split of **330,378** Source-1 entities
  (6,099,384 scored pairs), at threshold 0.89. The split is by **entity** (CRC32 hash of the
  Source-1 id, 15% held out) so no entity's pairs ever straddle train and validation.

**Feature importance (by gain)** — the model's own view of what matters:

| Feature | Share of total gain |
| --- | --- |
| `name_trigram_jaccard` | 26.6% |
| `addr_tok_jaccard` | 26.3% |
| `name_phonetic_overlap` | 25.6% |
| `addr_4gram_jaccard` | 3.7% |
| `addr_token_sort` | 3.1% |
| `addr_shared_nums` | 2.8% |
| `name_token_sort` | 2.7% |
| remaining 12 features | 9.2% |

Three fuzzy signals — name trigrams, **address token overlap**, and phonetic overlap — carry ~78%
of the decision. This independently validates the blocking design: the address is as informative as
the name, and no brittle exact-match feature dominates. Notably `country_flag` contributes ~0%,
which is reassuring for France generalization — the model never learned to lean on country.

**Prediction distribution (test set):** 5,097,053 predicted matches over 1,732,544 entities
(mean **2.94**, median 3, max 12) with **129,944 (7.5%)** predicted singletons. The shape closely
tracks the training ground truth (mean 3.67 for non-singletons, max 11), and the singleton rate sits
slightly above the training rate of 5.58% — the expected signature of a deliberately conservative,
precision-weighted threshold.

**Common false positives (wrong merges).** The dominant risk is **co-located but distinct
businesses**. Because address token overlap is a top-weighted feature, two different companies that
genuinely share a building, plaza, or office complex (very common in the Indian data, e.g. multiple
firms in *"Kanhaiya Plaza"*) produce a strong address signal. The model must then rely on name
dissimilarity alone to separate them, and when one side's name is also noisy it can over-merge. Our
mitigations are the high threshold (0.89) and the greedy one-record-one-entity assignment pass. We
note this as a reasoned diagnosis from feature importance plus the co-location patterns observed in
the data, rather than from an exhaustive labelled FP audit.

**Common false negatives (missed matches).** These are dominated by **blocking misses**, which we
characterised directly (Section 2.1): ~13% of true links never enter the candidate set, so the
classifier cannot recover them. Concretely, the residual misses are concentrated in
(a) **Devanagari transliteration** — India micro-recall is 81.9% vs 90.6% for the US, because
transliteration destroys both token and trigram overlap; and (b) pairs where the name is corrupted
*and* the address is abbreviated or partially missing, leaving too few shared keys to clear
`MIN_SHARED`. A smaller second group is model-side: true pairs that are retrieved but score below
the deliberately conservative 0.89 threshold — an accepted trade, since F₀.₅ penalises a wrong merge
roughly twice as much as a miss.

---

## 6. Conclusion

Treating blocking as a measured, first-class problem — not plumbing — produced the decisive gain:
inspecting the matches our blocker missed revealed that corrupted names frequently sit on intact
addresses, and adding address-derived keys lifted the recall ceiling from 52.9% to 86.8%. On top of
that candidate set, a compact 19-feature LightGBM classifier with an F₀.₅-tuned threshold of 0.89
reaches **macro-F₀.₅ = 0.9279** on 330K held-out entities while compressing the search space
≈525,000×. The main lesson: in entity resolution the *upper bound you set in blocking* matters more
than the sophistication of the matcher, and the fastest route to finding that bound is to read the
examples you are getting wrong.

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/` — all source in `src/`, with
`README.md` (exact reproduction steps) and `requirements.txt` (pinned versions).

| Module | Role |
| --- | --- |
| `config.py` | All paths and tunables (`BLOCK_K`, seed, threshold) in one place. |
| `eda.py` | Streaming exploratory analysis (Section 2.1 numbers). |
| `normalize.py` | Shared name/address/country normalization + pure-Python Soundex. |
| `blocking.py` | Inverted-index candidate generation **and** the recall-ceiling measurement. |
| `features.py` | The 19 pairwise similarity features. |
| `build_training.py` | Memory-safe chunked labelled-matrix builder (SQLite record store, entity-level split). |
| `model.py` | LightGBM training + vectorized F₀.₅ threshold search. |
| `predict_test.py` / `predict_test_fast.py` | Test blocking + scoring + greedy conflict resolution → output TSVs. |
| `kaggle_pipeline.py` | Single-file in-memory variant of the identical pipeline for high-RAM (≥25 GB) environments. |
| `tests/` | 14 normalization tests + 7 feature tests, all passing. |

**Entry points to reproduce both output files** (from `code/business_entity_resolution/`):

```bash
python src/blocking.py --split train     # candidates + recall ceiling
python src/build_training.py             # labelled feature matrix
python src/model.py                      # train + tune threshold
python src/predict_test_fast.py          # -> output/matching_results.tsv, candidate_pairs.tsv
python ../../utils/validate_submission.py --matching ... --candidate ... --test-dir ...
```

### B. Additional Results

**Blocking ablation (full train set, 2,206,821 entities):**

| Configuration | Micro recall | Macro recall | Entities with 0 candidates |
| --- | --- | --- | --- |
| Name-only keys | 52.94% | 55.61% | 124,731 |
| + name 3-grams | 54.87% | 57.44% | — |
| **+ address keys (final)** | **86.81%** | **87.54%** | **0** |

**Per-country blocking recall (final configuration):** US **90.61%**, India **81.92%**.

**Candidate-set efficiency (test):** 32,898,155 candidates for 1,732,544 entities — mean 18.99 per
entity, versus 1.73×10¹³ naive pairs (**≈525,000× reduction**).

**Submission validation:** `utils/validate_submission.py` reports `PASS` on both files —
1,732,544 rows each, matching the required entity set exactly, with final matches a strict subset of
the candidate set.

**Engineering notes.** Peak RAM stays under ~5 GB for every stage, achieved via streaming TSV reads,
a disk-backed SQLite record store, chunked feature flushing, and int32 posting lists in the
inverted index; the full pipeline therefore runs on commodity hardware without a GPU.

---

**Fair play.** The pipeline uses only the provided challenge files. No external databases, APIs,
geocoding services, or internet data augmentation are used at any stage. The final model
(LightGBM) is MIT-licensed and far below the 8B-parameter limit.
