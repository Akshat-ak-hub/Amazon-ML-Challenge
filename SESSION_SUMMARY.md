# Amazon ML Challenge 2026 — Session Summary & Handoff

**Date:** 2026-09-27
**Project:** Business Entity Resolution (match S1 records to S2/S3 across 3 noisy sources)
**Metric:** macro-averaged F0.5 (precision-weighted, per S1 entity, singletons count)

---

## 1. CURRENT STANDING

- **Leaderboard score: 0.857** (submitted 2026-09-27 11:54 IST, status Evaluated).
- **Top-500 cutoff: 0.984.** Leader pack clusters at 0.990–0.991 (25+ teams).
- **Time / attempts left (as of ~12:00 IST): ~11 hrs, 4 leaderboard attempts.**

## 2. THE ROOT-CAUSE DIAGNOSIS (verified with real numbers)

The gap from 0.857 → 0.984 is a **blocking recall problem**, not a model problem.

- Current blocking recall ceiling = **~87.5%** (measured: micro 86.81%, macro 87.54%
  on full train; 86.88% on a valid sample). ~12.5% of true matches are NEVER
  proposed as candidates, so the model can't recover them. This caps the score.
- Built a **realistic local scorer** (`realistic_scorer.py`) that mirrors the
  leaderboard (includes blocking misses, not just candidate pairs). It reported
  realistic F0.5 = **0.8790** at threshold 0.88 — matches the 0.857 leaderboard,
  confirming the scorer is trustworthy. The OLD 0.928 "validation" number was
  optimistic because it only scored over candidate pairs.
- **Threshold sweep (Attempt 1, no retrain): does NOT beat 0.857.** Current
  threshold 0.88 is already near-optimal; model already extracts almost everything
  blocking gives it. So we did NOT waste an attempt on it.

## 3. WHY TEAMS HIT 0.984+ — THE KEY INSIGHT (from inspecting missed matches)

Ran `_inspect_missed2.py` — printed 50 real true-matches that blocking misses.
The misses are NOT random hard cases; they follow fixable patterns:

1. **Name destroyed but ADDRESS nearly identical.** The match lives in the address
   (house number + locality), e.g.
   - `Konrad Heritage City` ↔ `DOVAVERA`, both `8935/893 Parkview Cir, Chisago City, MN`
   - `Shree It Pvt Ltd` ↔ `Fayeviocalo`, both `Behind 66Kv Electric Substation, Odhav Road`
2. **Address number noise:** leading zeros (`321` vs `0321`), digit typos
   (`818` vs `218`, `894` vs `8949`), `th`→`nd`.
3. **Domain-name variants:** `moorebitwise.com` ↔ `Moore Bitwise Inc`;
   `bellasadvancedsupply.com` ↔ `Bella's Advanced Supply`.
4. **Native-script names (Devanagari/Kannada/Telugu/Tamil/Malayalam)** but matching
   address — address keys catch these.
5. **Partial/truncated names + shared address:** `Great` ↔ `Great Royal Business Pvt Ltd`.

**=> The path to 0.984 is aggressive ADDRESS-ANCHORED blocking + number
normalization + domain normalization**, not a fancier model.

## 4. EXPERIMENTS RUN (recall ceiling, on 20k valid sample)

| Blocking approach | MACRO recall | cand/ent | Notes |
|---|---|---|---|
| Existing (weak keys) | 87.94% | 18.4 | current pipeline |
| + strong keys (exact name, sorted-name, phonetic, PIN, exact addr) | **91.47%** | 44.2 | v2, memory-safe, DONE |
| + address-number + domain keys | (not finished) | — | v3 — kept hitting memory wall |

**Strong keys added +3.53 pts** (87.9→91.5). Still short of 0.984-level (~97-98%
ceiling needed). v3 (address-number + domain keys) is the promising next step but
DID NOT COMPLETE — see blocker below.

## 5. THE BLOCKER: MEMORY (definitive finding after a full day)

Current machine has **11.8 GB RAM**. EVERY recall-improving approach hit a wall:

- **In-RAM inverted index** (original blocking.py, and the enriched version with
  strong keys): index grows to 30M+ keys over 10.3M records -> ~6 GB+ and SWAPS
  (thrashes) around 7-8M records with <1 GB free. Stalls.
- **Disk-based SQLite index** (blocking_v5.py): RAM stays flat (~2.5 GB, no wall!)
  BUT per-entity scoring against the 60M-row disk table is SLOW: ~35-60 entities/sec
  at full scale -> ~10 HOURS for train blocking alone. Not viable in the time left.

Conclusion: the recall gain toward 0.984 fundamentally needs either MORE RAM (fast
in-RAM path) or a vectorized/SQL-JOIN blocking rewrite (disk path). Neither fit the
11.8 GB machine within the remaining time.

### Experiments run (recall ceiling, 20k valid sample)
| Approach | MACRO recall | cand/ent | speed at full scale |
|---|---|---|---|
| baseline (existing weak keys) | 87.9% | 18.4 | fast (already done) |
| + strong keys only (v2) | 91.47% | 44.2 | in-RAM: OOM on this machine |
| + address/domain keys union (v4) | 93.99% | 197.6 | disk: fits RAM, candidate flood |
| + scored ranker K=50 (v5) | 91.90% | 43.0 | disk: fits RAM, ~10hr (too slow) |

## 5b. WHAT IS READY TO RUN ON THE 16 GB MACHINE

`src/blocking.py` has ALREADY been enriched (committed) with the strong keys:
  - NC: exact core name, NS: sorted-token signature (word-order),
    DC: domain-concat (moorebitwise.com<->Moore Bitwise), NP: full phonetic,
    AC: near-exact address signature (same-address/garbled-name),
    z: exact PIN (existing). Strong keys weighted 3x in candidates_for(), qualify
    a candidate alone. TOP_K raised 20 -> 40.
On 16 GB the in-RAM index should FIT (it needs ~7-9 GB) and run fast.

### Exact commands on the 16 GB machine (after clone + data + config.py DATA_ROOT):
```
cd code/business_entity_resolution
# 1. verify recall on a fast sample first:
python src/blocking.py --split train --sample 30000     # want MACRO recall > 90%
# 2. if good, run the full fast cycle (in-RAM blocking, not the disk v5):
#    NOTE: run_full_cycle.py currently calls the SLOW blocking_v5. On 16GB, instead
#    edit it to call the enriched src/blocking.py (build_blocking) for train+test,
#    OR run stages manually:
python src/blocking.py --split train        # full train candidates + recall ceiling
#   then swap work/candidates_train.tsv is already the output path used by build_training
python src/build_training.py                # rebuild training matrix
python src/model.py                         # retrain + tune threshold
python realistic_scorer.py                  # HONEST F0.5 gate (must beat 0.857)
python src/predict_test.py                  # test blocking + predict -> output/*.tsv
python ../../utils/validate_submission.py -m ../../output/matching_results.tsv \
    -c ../../output/candidate_pairs.tsv -t <test_dir>
```
GATE RULE: only upload output/matching_results.tsv if realistic_scorer F0.5 > 0.857.

## 5c. IF STAYING ON 11.8 GB: options
- Keep the 0.857 submission (validated, safe, already on leaderboard).
- OR vectorized blocking rewrite (SQL JOIN of S1-keys table x posting table) —
  real dev work, ~1-2 hrs, untested.


## 6. NEXT STEPS (in order) ON THE NEW MACHINE

1. Set up: clone repo, copy dataset, edit `src/config.py` DATA_ROOT to new path.
2. Re-run the LEAN v3 experiment (`_block_exp3.py`) — confirm address+domain keys
   push recall ceiling toward 96-98%. If yes, proceed; if not, iterate on keys.
3. Fold the winning keys into `src/blocking.py` (add strong + address-number +
   domain keys; raise TOP_K ~40; lower MIN_SHARED to 1 for strong keys).
4. Add a high-precision **exact-match auto-accept** layer (exact norm-name + same
   country, or exact norm-address) — free precision for F0.5.
5. Full cycle (detached, ~3 hrs): blocking train → build_training → retrain model
   → realistic_scorer → blocking test → predict_test_fast → validate.
6. **GATE:** only upload if `realistic_scorer.py` F0.5 beats 0.857. Never upload a
   guess. 4 attempts remain.

## 7. FILE MAP

Code repo: `C:\Users\Akashat\Downloads\AMAZON`  (GitHub:
https://github.com/Akshat-ak-hub/Amazon-ML-Challenge.git)
- `code/business_entity_resolution/src/`
  - `config.py`      — all paths (EDIT DATA_ROOT on new machine), tunables
  - `normalize.py`   — name/address normalization, Soundex, legal suffixes
  - `blocking.py`    — inverted-index blocking (v2). record_keys() is where new
                       keys go. TOP_K=BLOCK_K=20, MAX_FANOUT=2000, MIN_SHARED=2.
  - `features.py`    — pairwise similarity features for the model
  - `build_training.py` — chunked, memory-safe training-matrix builder
  - `model.py`       — LightGBM train + F0.5 threshold tune (NOTE: tunes on
                       candidate pairs only = optimistic; use realistic_scorer)
  - `predict_test.py` / `predict_test_fast.py` — inference on test
- `code/business_entity_resolution/realistic_scorer.py` — HONEST F0.5 scorer (NEW,
  local only, not on GitHub yet)
- Experiment scripts (NEW, local only, prefixed `_`): `_block_exp2.py` (strong keys,
  works), `_block_exp3.py` (addr+domain keys, lean version ready to run),
  `_inspect_missed2.py` (prints missed matches)

Data + artifacts (off-repo): `D:\CODE SNAP\ml_challenge`  (move these or re-extract
from `C:\Users\Akashat\Downloads\6ab10eb3b23ba_student_resource.zip`, 1.02 GB)
- `dataset/train/` train_source1/2/3.tsv + train_ground_truth.tsv
- `dataset/test/`  test_source1/2/3.tsv  (test has France = 15% of entities, unseen
  in train; keep country handling OPEN-SET, never hard-code US/India)
- `work/match_model.txt` (trained model), `candidates_train.tsv`,
  `candidates_test.tsv`, `threshold.txt` (0.89), `realistic_score.txt`
- `output/matching_results.tsv` (85.7 MB, the 0.857 submission),
  `candidate_pairs.tsv`

Data sizes: zip 1.02 GB → extracted 2.35 GB (7 TSVs). Submission = matching_results.tsv
(85.7 MB) only, to the portal. Validator: `utils/validate_submission.py` (both output
files already PASS, incl. --check-ids).

## 8. KEY RULES / CONSTRAINTS
- F0.5 weights precision 2x recall → false merges hurt more than misses.
- NO external data/APIs/lookups (disqualification). Provided data only.
- Final model must be MIT/Apache-2.0 and ≤8B params (LightGBM is fine).
- Every test S1 entity must appear exactly once in output; singletons = empty list.
- Keep country open-set (France appears only in test).
- Pipeline streams TSVs with stdlib (no pandas) for memory safety.

## 9. HYGIENE
- Sleep disabled on old machine via: `powercfg /change standby-timeout-ac 0` and
  `monitor-timeout-ac 0`. Do the same on the new machine for long runs.
- Long runs launched as detached PowerShell windows that Tee-Object to a log file,
  so progress can be read anytime and they survive chat closure (but NOT shutdown).
