"""Central configuration: all paths and tunable constants live here.

Design note: code lives on C: (in the git repo); the dataset and all heavy
intermediates live off-repo on D: (188 GB free). Only this file knows where the
data is, so the rest of the pipeline is path-agnostic and portable — anyone who
clones the repo just edits DATA_ROOT.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Data + scratch live off-repo (big, not version-controlled).
DATA_ROOT = Path(r"D:\CODE SNAP\ml_challenge")

DATASET_DIR = DATA_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Scratch space for intermediates: blocking DB, features, models, logs.
WORK_DIR = DATA_ROOT / "work"
# Final outputs (matching_results.tsv, candidate_pairs.tsv).
OUTPUT_DIR = DATA_ROOT / "output"

# Training source files
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Test source files
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
DELIM = "\t"

# ---------------------------------------------------------------------------
# Pipeline tunables (placeholders — tuned against validation in later phases)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
BLOCK_K = 20          # max candidates per S1 entity (capped; tuned vs recall ceiling)
MATCH_THRESHOLD = 0.5 # decision threshold (tuned vs F0.5 later; will rise above 0.5)


def ensure_dirs():
    """Create scratch/output dirs if missing (safe to call repeatedly)."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
