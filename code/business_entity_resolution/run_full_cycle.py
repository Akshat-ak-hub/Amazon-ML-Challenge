"""Full improvement cycle orchestrator (K=50 scored-ranker blocking).

Runs the whole pipeline end-to-end, gating the expensive test-prediction on the
REALISTIC local F0.5 (must beat the 0.857 baseline before we bother predicting).

Sequence:
  1. v5 blocking on TRAIN  -> work/candidates2_train.tsv (+ recall ceiling)
  2. swap candidates2_train.tsv -> candidates_train.tsv (backup original)
  3. build_training.py     -> new trainmat_{train,valid}.npz + pairids
  4. model.py              -> retrain LightGBM, tune threshold
  5. realistic_scorer.py   -> HONEST F0.5 on held-out valid  ***GATE***
        if F0.5 <= 0.857: STOP here (do not waste time predicting). Print verdict.
        else: continue.
  6. v5 blocking on TEST   -> work/candidates2_test.tsv
  7. swap candidates2_test.tsv -> candidates_test.tsv (backup original)
  8. build test record store + predict (predict_test_fast) -> output/*.tsv
  9. validate output

Each step logs to work/cycle.log and prints progress. Designed to run detached.

Run:  python run_full_cycle.py
"""
from __future__ import annotations
import subprocess, sys, time, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))
import config as C  # noqa

PY = sys.executable
BASELINE = 0.857
LOG = C.WORK_DIR / "cycle.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(desc, args, log_to=None):
    log(f">>> {desc}")
    log(f"    cmd: {' '.join(str(a) for a in args)}")
    t0 = time.time()
    if log_to:
        with open(log_to, "w", encoding="utf-8") as out:
            p = subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, cwd=str(HERE))
    else:
        p = subprocess.run(args, cwd=str(HERE))
    log(f"    done rc={p.returncode} ({time.time()-t0:.0f}s)")
    if p.returncode != 0:
        log(f"!!! STEP FAILED: {desc} (rc={p.returncode}). See {log_to}")
        raise SystemExit(1)


def swap(new_name, target_name):
    """Back up target, move new file into target's place."""
    new = C.WORK_DIR / new_name
    tgt = C.WORK_DIR / target_name
    bak = C.WORK_DIR / (target_name + ".orig")
    if tgt.exists() and not bak.exists():
        shutil.copy2(tgt, bak)
        log(f"    backed up {target_name} -> {target_name}.orig")
    shutil.copy2(new, tgt)
    log(f"    swapped {new_name} -> {target_name}")


def read_realistic_f05():
    p = C.WORK_DIR / "realistic_score.txt"
    if not p.exists():
        return None
    for line in open(p, encoding="utf-8"):
        if line.startswith("realistic_f05"):
            try:
                return float(line.split("\t")[1])
            except Exception:
                return None
    return None


def main():
    LOG.write_text("", encoding="utf-8")
    t0 = time.time()
    log("========== FULL CYCLE START (K=50 scored ranker) ==========")

    # 1. train blocking
    run("1/9 v5 blocking on TRAIN", [PY, str(SRC/"blocking_v5.py"), "--split", "train"])

    # 2. swap train candidates
    log("2/9 swap train candidates")
    swap("candidates2_train.tsv", "candidates_train.tsv")

    # 3. build training matrix
    run("3/9 build training matrix", [PY, str(SRC/"build_training.py")],
        log_to=C.WORK_DIR/"cycle_build.log")

    # 4. retrain
    run("4/9 retrain LightGBM + tune threshold", [PY, str(SRC/"model.py")],
        log_to=C.WORK_DIR/"cycle_model.log")

    # 5. realistic score GATE
    run("5/9 realistic scorer (GATE)", [PY, str(HERE/"realistic_scorer.py")],
        log_to=C.WORK_DIR/"cycle_score.log")
    f05 = read_realistic_f05()
    log(f"*** REALISTIC LOCAL F0.5 = {f05}  (baseline {BASELINE}) ***")
    if f05 is None:
        log("!!! could not read realistic F0.5; stopping for manual check")
        raise SystemExit(1)
    if f05 <= BASELINE:
        log(f"*** GATE NOT PASSED ({f05:.4f} <= {BASELINE}). Stopping before test "
            f"prediction. The new blocking did not improve the honest score. "
            f"Original candidates are backed up as *.orig. ***")
        log(f"========== CYCLE STOPPED AT GATE ({time.time()-t0:.0f}s) ==========")
        return
    log(f"*** GATE PASSED ({f05:.4f} > {BASELINE}). Proceeding to test prediction. ***")

    # 6. test blocking
    run("6/9 v5 blocking on TEST", [PY, str(SRC/"blocking_v5.py"), "--split", "test"])

    # 7. swap test candidates
    log("7/9 swap test candidates")
    swap("candidates2_test.tsv", "candidates_test.tsv")

    # 8. build test store + predict (predict_test builds its own store from
    #    candidates_test.tsv — but it re-runs OLD blocking; so we use the store-
    #    building path of predict_test via a thin call). Simpler: predict_test_fast
    #    needs recstore_test.sqlite matching new candidates -> build it first.
    run("8a/9 build test record store for new candidates",
        [PY, str(HERE/"build_test_store.py")], log_to=C.WORK_DIR/"cycle_store.log")
    run("8b/9 predict test (fast)", [PY, str(SRC/"predict_test_fast.py")],
        log_to=C.WORK_DIR/"cycle_predict.log")

    # 9. validate
    run("9/9 validate output",
        [PY, str(HERE.parent.parent/"utils"/"validate_submission.py"),
         "-m", str(C.OUTPUT_DIR/"matching_results.tsv"),
         "-c", str(C.OUTPUT_DIR/"candidate_pairs.tsv"),
         "-t", str(C.TEST_DIR)],
        log_to=C.WORK_DIR/"cycle_validate.log")

    log(f"========== CYCLE COMPLETE ({time.time()-t0:.0f}s) ==========")
    log(f"New output written. Realistic F0.5={f05:.4f} beat baseline {BASELINE}.")
    log(f"Review, then upload output/matching_results.tsv if satisfied.")


if __name__ == "__main__":
    main()
