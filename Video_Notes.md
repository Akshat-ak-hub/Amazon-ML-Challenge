# Problem-Statement Explainer Video — Summary & Transcript Notes

> **Source:** Embedded "Click Here" link on page 1 of
> `amazon_ml_challenge_problem_statement.pdf`.
> **Status:** ✅ Transcript provided by user and summarized below.

## Video Link

```
https://d8it4huxumps7.cloudfront.net/files/6ab509c5b7036_ml_challenge_2026_video.mp4
```

---

## TL;DR

An explainer walkthrough of the **Amazon ML Challenge 2026 — Business Entity Resolution**
problem. It confirms everything in the PDF and adds intuition, most usefully a concrete
**blocking → matching** mental model. No new binding rules beyond the PDF.

---

## 1. Problem & Where the Data Comes From

- Task: **business entity resolution** — decide which records across 3 independent, noisy
  sources describe the same real-world business.
- Real-world framing: a business signs up on **Amazon Business** (name + address captured),
  then extra info is pulled from other **data vendors**, each with its own formats and
  conventions. Crucially, **no shared identifier** links the sources.
- Only two fields are usable: **business name and address** (deliberately limited for the
  challenge).
- Core difficulty: the same business is written differently by each source —
  e.g. "Acme Robotics Incorporated" vs. an abbreviated address vs. a nearby-landmark reference.
- **Source 1** = clean, deduplicated reference list. **Sources 2 & 3** = noisy fragments to
  reconcile against it. For every Source 1 entity, find **all** its matches in Sources 2 & 3
  — which may be **many, exactly one, or none**.

## 2. Blocking → Matching Workflow (the key intuition)

- Comparing every S1 record against every S2/S3 record is **too expensive at scale**.
- **Blocking:** sort records into buckets using a **cheap key built from both name and
  address**, so likely-matches land in the same bucket.
- Records can group via a **similar name** OR a **shared address**.
- Blocking **favors recall**, so a bucket deliberately pulls in lookalikes:
  - a business with a *similar name* at a *different address*
  - a *different business* that happens to *share an address*
- The **matching model** then scores each candidate pair and **removes the false ones**,
  keeping only true matches.
- Blocking narrows a huge comparison space down to a manageable **candidate set**; the model
  does the fine-grained filtering.

## 3. Datasets & Label Format

- **Training set:** all 3 sources + ground-truth labels (which S2/S3 records each S1 business
  matches).
- **Test set:** same 3 sources, **no labels** — you predict for these.
- **Label format:** **one row per Source 1 entity**, its ID → comma-separated list of all
  matches; list is **empty when it matches nothing**. This mirrors exactly what you submit.
- **All files are tab-separated (`.tsv`)** — read with an explicit tab separator or columns
  won't parse.

## 4. Deliverables & Evaluation

- **`matching_results.tsv`** — one row per S1 entity with predicted matches; **the only file
  scored on the leaderboard**.
- **Final archive** — final matches + **`candidate_pairs.tsv`** (blocking output, *not*
  scored, used to **audit blocking quality**) + runnable pipeline + methodology doc. Top
  teams' packages reviewed before final rankings.
- **Run the validation script before submitting** so a formatting error doesn't waste a
  submission.
- **Metric: macro F₀.₅**, precision weighted **2× over recall**. Wrongly merging two
  different businesses is penalized ~2× more than missing a true match →
  **when in doubt, don't merge.**

## 5. Key Recommendations (from the video)

1. **Understand singletons** — an S1 entity with no S2/S3 match. Correct empty prediction =
   full **1.0**; any predicted match = **0**. Identifying no-match businesses is *as
   important* as finding matches.
2. **Invest in blocking first** — it sets the **ceiling on recall**; you can't match a record
   you never considered.
3. **Address (and name) patterns** — pay attention to **region-specific** patterns.
- **Firm rule:** pure ML challenge — **no external databases, APIs, or lookups**; use only
  the provided data.

---

## What the Video Adds Beyond the PDF

- The **Amazon Business signup → vendor enrichment** origin story for the data.
- An explicit, concrete **blocking bucket** example (the "Acme Robotics" orange block) and
  *why* buckets intentionally over-include lookalikes (recall-first, model filters after).
- The plain-language framing **"when in doubt, don't merge"** for the F₀.₅ precision bias.
- No new rules, thresholds, or file-format details — the PDF remains the binding spec.

## Impact on Our Plan

Nothing in the transcript changes the end-to-end plan — it reinforces it:
- Blocking-first, recall-ceiling focus → already Stage 3, flagged as the most important number.
- Conservative threshold ("don't merge") → already Stage 5 (tune threshold well above 0.5).
- Singleton care → already Stage 6 (empty list for no-candidate entities).
- Both name AND address blocking keys → already in the union-of-signals blocking design.
