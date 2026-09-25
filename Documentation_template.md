# Business Entity Resolution — Methodology Document

## 1. Overview

The task is to link records for the same real-world business across three noisy,
independently-collected sources, treating Source 1 as the deduplicated reference set.
Because the scoring metric (F₀.₅, macro-averaged per Source 1 entity) weights precision
twice as heavily as recall, the entire pipeline is designed around a simple principle:
**it is better to leave an entity unmatched than to guess wrong.**

The pipeline has three stages:

1. **Normalization** — clean and decompose noisy name/address strings into comparable parts.
2. **Blocking / candidate generation** — cheaply find a small, high-recall candidate set per
   Source 1 entity (written out as `candidate_pairs.tsv`).
3. **Pairwise matching model + assignment** — score each (S1, candidate) pair, threshold
   for precision, and resolve conflicts (written out as `matching_results.tsv`).

Country is treated throughout as an **open-set string label**, never a fixed categorical
feature, since the test set introduces France, which never appears in training.

---

## 2. Data Normalization

Applied identically to Source 1, 2, and 3 records so nothing is source-specific.

**Business name:**
- Lowercase, strip punctuation, normalize `&` ↔ `and`.
- Recognize and separate legal-suffix tokens (Corp/Corporation, Pvt/Private, Ltd/Limited,
  LLC, Inc, and their regional equivalents) into a separate "legal form" field, and also
  keep a "core name" version with these tokens removed — the model sees both the full and
  stripped forms so legal-suffix inconsistency doesn't tank similarity scores.
- Generate a phonetic code (Soundex / NYSIIS) on the core name tokens to catch typos and
  transliteration variants that edit-distance measures might under-score.

**Business address:**
- Standardize common abbreviations (Rd/Road, St/Street, Ave/Avenue, etc.).
- Extract structured sub-fields where detectable: PIN/ZIP code, city, state — left null when
  absent rather than imputed.
- Detect and split out landmark-style fragments (e.g. "Near SBI ATM") into a separate field
  so they don't dilute core address token overlap.
- Keep a raw normalized full-address string for whole-string similarity measures as well.

**Country:**
- Treated as a raw string, never one-hot encoded and never used to hard-filter the pipeline.
  The only feature derived from it is a three-way match flag: `same` / `different` / `unknown`
  (when either side is missing). This is what lets the pipeline handle France at test time
  without retraining or code changes.

---

## 3. Candidate Generation / Blocking Strategy

Goal: maximize recall (the ceiling on the whole pipeline's score) while keeping the
candidate set small enough for a pairwise model to score efficiently. Several
independent, cheap blocking signals are unioned rather than relying on one:

1. **Name token / n-gram blocking** — TF-IDF vectors over character n-grams and word
   tokens of the normalized core name; candidates are records above a cosine-similarity
   threshold, retrieved efficiently via sparse top-N matching (avoiding a full
   all-pairs comparison).
2. **Phonetic blocking** — records sharing a Soundex/NYSIIS code on their core name are
   pulled in as candidates, catching typos and transliteration spellings that n-gram
   overlap can miss.
3. **Address token blocking** — shared PIN code, or shared city/state tokens when present,
   as a secondary, lower-precision signal.
4. **Country as a soft filter, not a hard one** — candidates are *down-weighted*, not
   dropped, when country differs, since address noise or mislabeling could otherwise cause
   a silent recall loss, and the pipeline must not assume a closed set of countries.

The union of all four blockers, deduplicated, forms the candidate set per Source 1 entity,
capped at a maximum candidate count (tuned against the validation recall ceiling — see
Section 6). This capped, final list is exactly what is written to `candidate_pairs.tsv` and
is the same set fed to the matching model at inference time.

**Recall ceiling check:** before any modeling work, the blocking stage is scored on its own
by measuring what fraction of true matches (from `train_ground_truth.tsv`) survive into the
candidate set. Since no matching model can recover a true match blocking never produced,
this number is treated as a hard upper bound on achievable recall and is re-checked whenever
blocking thresholds change.

---

## 4. Feature Engineering

Each (Source 1, candidate) pair is turned into a fixed feature vector rather than raw text
being passed to the model:

**Name features:** exact-match flag (post-normalization), Jaccard similarity of name
tokens, normalized Levenshtein distance, Jaro-Winkler similarity, TF-IDF cosine similarity,
core-name vs. full-name similarity (to isolate legal-suffix effects), phonetic code match,
substring/containment flag, token count and length deltas.

**Address features:** token Jaccard similarity on normalized addresses, edit distance,
PIN/ZIP exact-match flag (only meaningful when both present — a separate "PIN available on
both sides" flag avoids conflating "no PIN" with "different PIN"), city/state exact-match
flags, landmark-stripped address similarity.

**Country feature:** the single `same` / `different` / `unknown` match flag described above.

**Structural features:** blocking-stage similarity score(s) themselves are passed through as
features, letting the classifier learn how much to trust each blocking signal rather than
treating all candidates as equally likely.

No feature is derived from a fixed vocabulary of country- or source-specific values, which is
what allows the same feature pipeline to run unchanged on the unseen France records at test
time.

---

## 5. Matching Model

**Architecture:** a gradient-boosted decision tree classifier (e.g. LightGBM or XGBoost,
both MIT/Apache-2.0 licensed and orders of magnitude under the 8B-parameter cap) trained on
the engineered pairwise features, with a binary match / no-match label derived from
`train_ground_truth.tsv` (true matches within the candidate set are positives; every other
candidate pair is a negative).

This choice is deliberate over an end-to-end neural approach: with a modest number of
hand-engineered, interpretable similarity features, GBDTs are typically both more
data-efficient and easier to threshold/calibrate for a precision-sensitive metric than a
deep model would be, and they keep the license/parameter-count constraint trivially
satisfied.

*(Optional extension, if pursued: a small pretrained sentence-embedding model, MIT/Apache-2.0
licensed and under the size cap, can supply an additional semantic-similarity feature over
name/address strings. Only a general-purpose pretrained model is used for this — nothing
fine-tuned or looked up against any business registry or external database, in line with the
fair-play rules.)*

**Threshold selection:** rather than using a default 0.5 cutoff, the decision threshold is
tuned directly against F₀.₅ on the held-out validation split, since the metric's 2x weighting
of precision over recall consistently pushes the optimal threshold well above 0.5.

**Assignment / conflict resolution:** because Source 1 is the deduplicated reference, a
single Source 2 or Source 3 record should generally not be claimed by more than one Source 1
entity. After per-pair scoring, a light conflict-resolution pass assigns each contested
candidate to its highest-scoring Source 1 entity (dropping it from lower-scoring matches)
rather than allowing independent per-pair thresholding to create implausible many-to-one
links across different Source 1 entities.

**Singleton handling:** a Source 1 entity with no candidate scoring above threshold is
emitted with an empty match list. Given F₀.₅'s scoring rule (a correct empty prediction scores
1.0; any false match on a true singleton scores 0.0), the tuned threshold is intentionally
conservative.

---

## 6. Validation Strategy

A validation split is held out from the training data at the Source 1 entity level (to avoid
leaking a Source 1 entity's matches across train/validation). On this split:

- Blocking recall ceiling is measured first, independent of the model.
- The classifier is trained on the remaining training entities and scored on the held-out
  split using the same per-entity F₀.₅ / macro-average formula used on the leaderboard.
- Blocking cap size and classification threshold are both tuned against this validation
  F₀.₅, not against raw precision or recall alone.

---

## 7. Generalization to Unseen Countries

No stage of the pipeline is allowed to branch on a fixed list of country values:
normalization rules are generic string operations, blocking never hard-filters on country,
and the only country-derived feature is a same/different/unknown flag. This is what allows
France — absent from all training data — to flow through the same pipeline at test time
without special-casing.

---

## 8. Fair-Play Compliance

The pipeline uses only the provided training/test files. No external entity-resolution
APIs, business registries, or geocoding services are called at any stage, and no data outside
the provided dataset is used to inform normalization, blocking, or matching. The final model
is MIT/Apache-2.0 licensed and well under the 8B-parameter limit.

---

## 9. Limitations & Possible Extensions

- Blocking recall is the pipeline's hard ceiling; if validation shows a persistent recall
  gap, adding a blocking signal targeted at the specific failure mode (e.g. more aggressive
  transliteration handling) is preferable to trying to recover lost recall in the classifier.
- The conflict-resolution assignment step is greedy; a global optimal assignment (e.g. a
  min-cost matching formulation) could be explored if greedy conflicts prove to hurt
  precision materially.
- Country-specific address grammar (e.g. French postal formatting) cannot be learned from
  training data by construction; the pipeline relies on generic string similarity to
  degrade gracefully rather than modeling it explicitly.
