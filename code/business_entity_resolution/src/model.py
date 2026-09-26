"""Stage 5c — Train LightGBM matching model + tune F0.5 threshold.

Trains a binary classifier on the pairwise features to predict match/no-match,
then selects the decision threshold that maximizes the competition metric
(macro-averaged F0.5 per S1 entity, singletons included) on the validation split.

Why LightGBM: MIT-licensed, tiny (well under 8B params), fast on CPU, strong on
tabular similarity features, easy to threshold for a precision-weighted metric.

Run:  python src/model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import features as F  # noqa: E402

BETA = 0.5


def load_split(split):
    d = np.load(C.WORK_DIR / f"trainmat_{split}.npz")
    return d["X"], d["y"], d["groups"]


def load_pair_s1(split):
    """Return array of S1 id per row (to group predictions by entity)."""
    s1s = []
    with open(C.WORK_DIR / f"pairids_{split}.tsv", encoding="utf-8") as f:
        for line in f:
            s1s.append(line.split("\t", 1)[0])
    return np.array(s1s)


def fbeta_macro(s1_ids, y_true, y_pred, beta=BETA):
    """Macro-average F_beta per S1 entity (kept for reference / tests)."""
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    ent = set()
    for s1, yt, yp in zip(s1_ids, y_true, y_pred):
        ent.add(s1)
        if yp and yt:
            tp[s1] += 1
        elif yp and not yt:
            fp[s1] += 1
        elif yt and not yp:
            fn[s1] += 1
    b2 = beta * beta
    total = 0.0
    for s1 in ent:
        t, f_, n_ = tp[s1], fp[s1], fn[s1]
        if t == 0 and f_ == 0 and n_ == 0:
            total += 1.0
            continue
        if t == 0:
            total += 0.0
            continue
        prec = t / (t + f_)
        rec = t / (t + n_)
        denom = (b2 * prec + rec)
        total += (1 + b2) * prec * rec / denom if denom else 0.0
    return total / len(ent) if ent else 0.0


def tune_threshold(s1_ids, y_true, scores, beta=BETA):
    """Vectorized F_beta threshold search.

    Groups rows by S1 entity once (integer codes), then for each candidate
    threshold computes per-entity TP/FP/FN with numpy bincount — no Python loops
    over millions of rows. Handles singletons (entities with no true positive and
    no prediction score above threshold -> perfect 1.0).
    """
    codes, _ = _factorize(s1_ids)
    n_ent = codes.max() + 1 if len(codes) else 0
    y_true = y_true.astype(np.int64)
    # per-entity count of true matches (fixed across thresholds)
    true_per_ent = np.bincount(codes, weights=y_true, minlength=n_ent)
    b2 = beta * beta
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.10, 0.95, 0.01):
        pred = (scores >= t).astype(np.int64)
        tp = np.bincount(codes, weights=(pred & y_true), minlength=n_ent)
        pp = np.bincount(codes, weights=pred, minlength=n_ent)   # predicted positives
        fp = pp - tp
        fn = true_per_ent - tp
        # per-entity F_beta
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
            rec = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
            denom = b2 * prec + rec
            fbeta = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
        # singletons: no true match AND no predicted -> perfect 1.0
        singleton_perfect = (true_per_ent == 0) & (pp == 0)
        fbeta = np.where(singleton_perfect, 1.0, fbeta)
        score = fbeta.mean() if n_ent else 0.0
        if score > best_f:
            best_f, best_t = score, float(t)
    return best_t, best_f


def _factorize(arr):
    """Map arbitrary ids to 0..k-1 integer codes (like pandas.factorize)."""
    uniq, codes = np.unique(arr, return_inverse=True)
    return codes.astype(np.int64), uniq


def main():
    print("[model] loading data ...", flush=True)
    Xtr, ytr, _ = load_split("train")
    Xva, yva, _ = load_split("valid")
    s1_va = load_pair_s1("valid")
    print(f"  train={Xtr.shape} valid={Xva.shape}", flush=True)

    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        scale_pos_weight=pos_w,
        seed=C.RANDOM_SEED,
        verbose=-1,
    )
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=F.FEATURE_NAMES)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)

    print("[model] training LightGBM ...", flush=True)
    model = lgb.train(
        params, dtr, num_boost_round=600, valid_sets=[dva],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    print("[model] tuning F0.5 threshold on validation ...", flush=True)
    scores = model.predict(Xva, num_iteration=model.best_iteration)
    best_t, best_f = tune_threshold(s1_va, yva, scores)
    print(f"  BEST threshold={best_t:.2f}  validation macro-F0.5={best_f:.4f}", flush=True)

    # feature importance
    imp = sorted(zip(F.FEATURE_NAMES, model.feature_importance("gain")),
                 key=lambda kv: -kv[1])
    print("[model] top features by gain:")
    for name, g in imp[:12]:
        print(f"   {name:24} {g:,.0f}")

    model.save_model(str(C.WORK_DIR / "match_model.txt"),
                     num_iteration=model.best_iteration)
    with open(C.WORK_DIR / "threshold.txt", "w") as f:
        f.write(f"{best_t}\n{best_f}\n")
    print(f"[model] saved model + threshold (F0.5={best_f:.4f})", flush=True)


if __name__ == "__main__":
    main()
