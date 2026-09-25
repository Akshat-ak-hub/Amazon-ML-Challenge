# Business Entity Resolution — Project Plan

## Objective

Build an ML pipeline that, for every Source 1 entity in the test set, correctly identifies
all matching records from Source 2 and Source 3, optimizing for the F₀.₅ macro-average
metric (precision weighted 2x over recall). Deliver both leaderboard predictions and a
complete, reproducible submission package.

## Status

No dataset files have been provided yet. This plan assumes they will be uploaded in the
structure the challenge describes (`dataset/train/*.tsv`, `dataset/test/*.tsv`,
`utils/validate_submission.py`). Phase 1 is currently blocking all downstream work.

---

## Project Structure

Target layout for the working project and the final submission zip. Paths under `dataset/`,
`utils/`, `output/`, and `code/` come directly from the challenge PDF (Phase 1 intake + Phase 8
packaging); the `src/` module breakdown is the one-module-per-phase decomposition of Phases 3–6.

```
AMAZON/                                          # student_resource working root
├── amazon_ml_challenge_problem_statement.pdf    # provided — challenge spec (binding)
├── Project_Plan.md                              # this file — 8-phase plan
├── Documentation_template.md                    # methodology write-up (update w/ real numbers)
├── Video_Notes.md                               # explainer-video summary (reference)
│
├── dataset/                                     # Phase 1 — NOT YET PROVIDED
│   ├── train/
│   │   ├── train_source1.tsv                    # entity_id, business_name, business_address, country
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv               # source1_entity_id, matched_entity_ids (labels)
│   └── test/
│       ├── test_source1.tsv                     # predict a row for every entity here
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── utils/
│   └── validate_submission.py                   # official stdlib-only format checker
│
├── output/                                      # Phase 6 outputs
│   ├── matching_results.tsv                     # final predictions (leaderboard-scored)
│   └── candidate_pairs.tsv                       # last-stage blocking set (audited, not scored)
│
└── code/
    └── business_entity_resolution/
        ├── README.md                            # exact end-to-end reproduction steps
        ├── requirements.txt                     # pinned deps (lightgbm/xgboost, pandas, …)
        ├── src/
        │   ├── config.py                        # paths, K cap, threshold, random seeds
        │   ├── data_io.py                       # TSV load/write, schema checks (Phase 1)
        │   ├── eda.py                            # exploratory analysis (Phase 2)
        │   ├── normalize.py                      # name/address/country cleaning (Phase 3)
        │   ├── blocking.py                       # TF-IDF + phonetic + address union (Phase 4)
        │   ├── features.py                       # pairwise feature vectors (Phase 5)
        │   ├── model.py                          # GBDT train + F₀.₅ threshold tuning (Phase 5)
        │   ├── assignment.py                     # conflict resolution / singletons (Phase 6)
        │   ├── validate.py                       # entity-level split + F₀.₅ scoring (Phase 6)
        │   └── run_pipeline.py                   # end-to-end orchestrator
        └── tests/
            ├── test_normalize.py                 # unit tests vs. noisy examples (Phase 3)
            ├── test_blocking.py                  # recall-ceiling check (Phase 4)
            └── test_features.py
```

**Data flow:**

```
raw TSVs → data_io → normalize → blocking → features → model → assignment → output TSVs
                                    │                     │
                              recall ceiling        threshold tuned on
                          (hard cap on recall)      validate.py entity-level F₀.₅ split
```

---

## Phase 1 — Data Intake

**Goal:** get the raw files in place and confirm they match the documented format.

- Obtain `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`,
  `train_ground_truth.tsv`, `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`.
- Obtain `utils/validate_submission.py` so local format-checking is available from day one.
- Confirm files parse correctly with `sep="\t"` and have the expected columns
  (`entity_id`, `business_name`, `business_address`, `country`).

**Deliverable:** dataset loaded and verified, no format surprises.
**Blocks:** everything below.

---

## Phase 2 — Exploratory Data Analysis

**Goal:** understand the data before writing any pipeline code.

- Row counts per source; ID prefix consistency (`S1-`, `S2-`, `S3-`).
- Missing-value rates for `business_address` and `country`, per source.
- Distribution of `country` values in train vs. test (confirm France only appears in test).
- Ground-truth match distribution: fraction of Source 1 entities with 0, 1, and 2+ matches
  (this sets expectations for the singleton rate F₀.₅ rewards).
- Sample and manually inspect ~20–30 true-match pairs from `train_ground_truth.tsv` to
  catalogue which noise patterns (abbreviations, typos, transliteration, landmark
  addresses, word-order swaps) actually appear, rather than assuming from the prompt alone.

**Deliverable:** short EDA notes/notebook informing normalization and blocking design.
**Depends on:** Phase 1.

---

## Phase 3 — Normalization Pipeline

**Goal:** a shared, source-agnostic cleaning module.

- Name normalization: lowercasing, punctuation handling, legal-suffix extraction
  (Corp/Pvt/Ltd/etc.) into a separate field, phonetic code generation.
- Address normalization: abbreviation standardization, PIN/city/state extraction where
  detectable, landmark-phrase isolation.
- Country handling: kept as a raw string; no fixed vocabulary, no one-hot encoding.
- Unit-test the normalization functions against the noisy examples catalogued in Phase 2.

**Deliverable:** `normalize.py` (or equivalent module) with test cases.
**Depends on:** Phase 2.

---

## Phase 4 — Blocking / Candidate Generation

**Goal:** a high-recall, size-bounded candidate set per Source 1 entity.

- Implement TF-IDF/n-gram blocking, phonetic blocking, and address-token blocking as
  independent signals; take their union.
- Cap candidates per Source 1 entity at a tunable `K`.
- Measure the **recall ceiling**: what fraction of true matches in
  `train_ground_truth.tsv` survive into the candidate set. This is the single most
  important number in the project — no later stage can recover a match blocking drops.
- Iterate on blocking thresholds/`K` until the recall ceiling is acceptably high without
  making the candidate set unmanageably large.
- Write `candidate_pairs.tsv` for the training split to enable feature engineering.

**Deliverable:** blocking module + measured recall ceiling on a held-out split.
**Depends on:** Phase 3.
**Exit criterion:** recall ceiling high enough to make Phase 5's job realistic (target
established after seeing actual data — cannot be fixed in advance).

---

## Phase 5 — Feature Engineering & Matching Model

**Goal:** a calibrated pairwise classifier tuned for F₀.₅.

- Build the pairwise feature vector (name similarity, address similarity, country
  match flag, structural features, blocking scores) per candidate pair.
- Split validation data by Source 1 entity (no entity split across train/validation).
- Train a GBDT classifier (LightGBM/XGBoost — MIT/Apache-2.0, well under 8B parameters).
- Tune the decision threshold directly against F₀.₅ on the validation split, not against
  raw accuracy or a default 0.5 cutoff.

**Deliverable:** trained model, chosen threshold, validation F₀.₅ score.
**Depends on:** Phase 4.

---

## Phase 6 — Post-Processing, Output Generation & Local Validation

**Goal:** turn model scores into the two required TSV files, and confirm both format and
quality before spending a leaderboard submission.

- Apply the tuned threshold; resolve cases where one S2/S3 record scores above threshold
  against more than one Source 1 entity (assign to the highest-scoring entity).
- Write `matching_results.tsv` and `candidate_pairs.tsv` for the test set.
- Run `utils/validate_submission.py` locally and fix any flagged issues.
- Compute F₀.₅ on the held-out validation split as a sanity check before uploading.

**Deliverable:** validated `matching_results.tsv` and `candidate_pairs.tsv` in `output/`.
**Depends on:** Phase 5.

---

## Phase 7 — Iteration

**Goal:** use leaderboard feedback (public split) and validation error analysis to improve
precision/recall trade-offs.

- Analyze false positives (name/address pairs the model over-trusted) and false negatives
  (true matches blocking missed or the model under-scored).
- Revisit blocking parameters or add a targeted feature/blocking signal if a specific
  failure mode (e.g. a particular transliteration pattern) is recurring.
- Re-tune threshold if precision/recall balance shifts after changes.

**Deliverable:** updated model/pipeline, improved validation F₀.₅.
**Depends on:** Phase 6; repeats as needed within the challenge timeline.

---

## Phase 8 — Final Packaging

**Goal:** assemble the submission zip.

- `output/matching_results.tsv`, `output/candidate_pairs.tsv` — final run outputs.
- `code/business_entity_resolution/src/` — all pipeline source code.
- `code/business_entity_resolution/README.md` — exact reproduction steps, data → blocking
  → matching → output.
- `code/business_entity_resolution/requirements.txt` — pinned dependencies.
- `Documentation_template.md` — methodology write-up, updated with real numbers from the
  final run (actual recall ceiling, chosen threshold, validation F₀.₅). A first draft of
  this document already exists and needs updating once real results are in.

**Deliverable:** `<team_name>_submission.zip` matching the required structure exactly.
**Depends on:** Phase 7 (or Phase 6 if iteration is skipped).

---

## Risks / Open Questions

- **Recall ceiling unknown until real data is seen** — blocking parameters in this plan are
  placeholders; they'll need real tuning once Phase 2/4 run on actual data.
- **France generalization** — no training signal exists for French address patterns;
  relying on generic string similarity is the only option available under the fair-play
  rules, and its effectiveness can only be estimated once test predictions are scored.
- **Conflict resolution approach (greedy vs. optimal assignment)** — greedy is simpler and
  is the default in this plan; revisit only if validation analysis shows it costing
  meaningful precision.
