"""Trains and compares several classifiers to predict the best LLVM optimization variant from
static features (loopcost.ml.llvm_dataset), and saves the best one to models/.

Two cross-validation schemes are used, because they answer different questions:

- Stratified k-fold (shuffled, over rows): reports raw accuracy but LEAKS -- each kernel
  contributes 4 rows (one per njit_variant) that share every feature except fastmath/
  boundscheck, so a random split can put 3 near-duplicate rows of the same kernel in the
  training fold and the 4th in the test fold. This inflates accuracy relative to real use.
- Leave-one-kernel-out (grouped by kernel, one kernel held out per fold): every row for a
  given kernel is held out together, so each fold tests generalization to a kernel the model
  never saw at all -- the real use case (predicting a good LLVM variant for a NEW kernel).
  With only ~20 distinct kernels, this also gets far more folds (and a more stable estimate)
  than a 5-way grouped split would. This is the metric model selection uses.

Accuracy (exact label match) is reported alongside a second, arguably more meaningful metric:
%-of-peak GFLOP/s actually achieved (from the real sweep) had the model's prediction been
used -- a wrong label that still lands within a few % of the true best is a much smaller
mistake than a wrong label that picks a badly slow variant, and exact-match accuracy can't
tell the two apart.
"""

import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from loopcost.heuristic.llvm_choice import _LLVM_CHOICE_TABLE
from loopcost.ml.llvm_dataset import FEATURE_COLUMNS, SWEEP_CSV, load_dataset

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODEL_DIR / "llvm_choice_model.pkl"
REPORT_TXT = MODEL_DIR / "llvm_choice_ml_report.txt"

N_SPLITS = 5


class HeuristicTableClassifier(BaseEstimator, ClassifierMixin):
    """Wraps the existing hand-crafted style x boundscheck lookup table
    (heuristic/llvm_choice.py's _LLVM_CHOICE_TABLE) as an sklearn-compatible "classifier" with
    no learned parameters at all, so it can be cross-validated in the exact same harness as
    the real ML candidates -- the fairest possible comparison of "was training a model on this
    data worth it" against the 3-bucket rule that was hand-derived from the same sweep.
    """

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        preds = []
        for _, row in X.iterrows():
            if row["has_transcendental"]:
                style = "transcendental"
            elif row["float_ops"] == 0:
                style = "non_fp"
            else:
                style = "vectorizable"
            variant, _ = _LLVM_CHOICE_TABLE[(style, bool(row["boundscheck"]))]
            preds.append(variant)
        return np.array(preds)


CANDIDATES = {
    "heuristic_table": HeuristicTableClassifier(),
    "decision_tree_d3": DecisionTreeClassifier(max_depth=3, random_state=0),
    "decision_tree_d3_bal": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0),
    "decision_tree_d5": DecisionTreeClassifier(max_depth=5, random_state=0),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=0),
    "random_forest_bal": RandomForestClassifier(
        n_estimators=300, max_depth=5, class_weight="balanced", random_state=0
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=0),
    "hist_gb": HistGradientBoostingClassifier(random_state=0, max_depth=3, max_iter=100),
    "knn3": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=3))]),
    "knn5": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "logreg": Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, C=1.0))]
    ),
    "logreg_bal": Pipeline(
        [("scale", StandardScaler()),
         ("clf", LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"))]
    ),
    "svm_rbf": Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf", C=2.0))]),
    "gaussian_nb": GaussianNB(),
}


def _pct_of_peak(meta, y_true, y_pred, csv_path=None):
    """For each (kernel, njit_variant) row, what fraction of the true peak GFLOP/s (max over
    the 6 real, measured LLVM variants) the predicted variant actually achieved."""
    sweep_path = Path(csv_path) if csv_path else SWEEP_CSV
    with open(sweep_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gflops_per_sec"] = float(r["gflops_per_sec"])

    groups = {}
    for r in rows:
        groups.setdefault((r["kernel"], r["njit_variant"]), []).append(r)

    fracs = []
    for (_, meta_row), pred in zip(meta.iterrows(), y_pred):
        grp = groups[(meta_row["kernel"], meta_row["njit_variant"])]
        peak = max(r["gflops_per_sec"] for r in grp)
        by_llvm = {r["llvm_variant"]: r["gflops_per_sec"] for r in grp}
        got = by_llvm.get(pred, 0.0)
        fracs.append((got / peak) if peak > 0 else 1.0)
    return np.array(fracs)


def evaluate_candidates(X, y, meta):
    """Cross-validates every candidate in CANDIDATES both ways (see module docstring) and
    returns a results DataFrame sorted by the grouped (honest) %-of-peak metric, descending.
    """
    groups = meta["kernel"].values
    y_arr = y.values

    stratified = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    loko = LeaveOneGroupOut()

    results = []
    for name, model in CANDIDATES.items():
        strat_pred = cross_val_predict(model, X, y_arr, cv=stratified)
        strat_acc = (strat_pred == y_arr).mean()

        loko_pred = cross_val_predict(model, X, y_arr, cv=loko, groups=groups)
        loko_acc = (loko_pred == y_arr).mean()
        loko_pct_of_peak = _pct_of_peak(meta, y_arr, loko_pred).mean()

        results.append(
            {
                "model": name,
                "stratified_acc": strat_acc,
                "loko_acc": loko_acc,
                "loko_pct_of_peak": loko_pct_of_peak,
            }
        )

    df = pd.DataFrame(results).sort_values(
        ["loko_pct_of_peak", "loko_acc"], ascending=False
    ).reset_index(drop=True)
    return df


def _baseline_pct_of_peak(meta, always_variant="llvm_O3_loopvec", csv_path=None):
    """%-of-peak achieved by always predicting a single fixed variant (e.g. numba's own
    default LLVM pipeline) -- the naive baseline any model needs to beat."""
    y_pred = [always_variant] * len(meta)
    return _pct_of_peak(meta, None, y_pred, csv_path).mean()


def _majority_class_pct_of_peak(meta, y, csv_path=None):
    """%-of-peak achieved by always predicting the single most common label in the training
    data -- the naive ML baseline (no features used at all)."""
    majority = y.value_counts().idxmax()
    y_pred = [majority] * len(meta)
    return _pct_of_peak(meta, None, y_pred, csv_path).mean(), majority


def train_best_model(csv_path=None):
    """Loads the dataset, compares every candidate classifier via leave-one-kernel-out
    cross-validation, retrains the best actually-trained one on the full dataset, and saves it
    (+ a text report) to models/. Returns (comparison_df, best_model_name, fitted_pipeline).

    "heuristic_table" (the hand-derived style x boundscheck lookup from heuristic/llvm_choice.py)
    is included in the comparison as a reference point, not as a candidate to select -- the
    point of this module is to report honestly whether training beat it, not to silently save
    it as "the ML model" if it happens to win.
    """
    X, y, meta = load_dataset(csv_path)
    print(f"dataset: {len(X)} rows, {y.nunique()} distinct best-LLVM-variant labels")
    print(y.value_counts().to_string())

    baseline_default = _baseline_pct_of_peak(meta, "llvm_O3_loopvec", csv_path)
    majority_pct, majority_label = _majority_class_pct_of_peak(meta, y, csv_path)
    print(f"\nbaseline (always numba default llvm_O3_loopvec): {100 * baseline_default:.1f}% of peak")
    print(f"baseline (always majority class {majority_label}): {100 * majority_pct:.1f}% of peak")

    print(f"\ncomparing {len(CANDIDATES)} classifiers "
          f"(stratified {N_SPLITS}-fold acc -- leaky -- vs leave-one-kernel-out "
          f"acc/%-of-peak -- honest generalization to unseen kernels)...")
    df = evaluate_candidates(X, y, meta)
    print(df.to_string(index=False, formatters={
        "stratified_acc": "{:.3f}".format,
        "loko_acc": "{:.3f}".format,
        "loko_pct_of_peak": "{:.3f}".format,
    }))

    heuristic_row = df[df["model"] == "heuristic_table"].iloc[0]
    ml_df = df[df["model"] != "heuristic_table"].reset_index(drop=True)
    best_name = ml_df.iloc[0]["model"]
    best_model = CANDIDATES[best_name]
    best_model.fit(X, y)

    beat_heuristic = ml_df.iloc[0]["loko_pct_of_peak"] > heuristic_row["loko_pct_of_peak"]
    verdict = (
        f"beats the hand-crafted heuristic_table ({100 * ml_df.iloc[0]['loko_pct_of_peak']:.1f}% "
        f"vs {100 * heuristic_row['loko_pct_of_peak']:.1f}% of peak, held-out kernels)"
        if beat_heuristic else
        f"does NOT beat the hand-crafted heuristic_table ({100 * ml_df.iloc[0]['loko_pct_of_peak']:.1f}% "
        f"vs {100 * heuristic_row['loko_pct_of_peak']:.1f}% of peak, held-out kernels) -- "
        f"with only {meta['kernel'].nunique()} distinct kernels to learn from, no trained "
        f"classifier reliably out-generalized the 3-bucket rule it was compared against"
    )
    print(f"\nbest trained model: {best_name} -- {verdict}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)

    report_lines = [
        f"dataset: {len(X)} rows, {y.nunique()} labels, {meta['kernel'].nunique()} distinct kernels",
        f"baseline always-numba-default: {100 * baseline_default:.1f}% of peak",
        f"baseline always-majority-class ({majority_label}): {100 * majority_pct:.1f}% of peak",
        "",
        df.to_string(index=False),
        "",
        f"best trained model: {best_name}",
        f"  leave-one-kernel-out accuracy:   {ml_df.iloc[0]['loko_acc']:.3f}",
        f"  leave-one-kernel-out %-of-peak:  {ml_df.iloc[0]['loko_pct_of_peak']:.3f}",
        f"  {verdict}",
    ]
    REPORT_TXT.write_text("\n".join(report_lines) + "\n")

    print(f"saved to {MODEL_PATH}")
    print(f"report written to {REPORT_TXT}")
    return df, best_name, best_model


if __name__ == "__main__":
    train_best_model()
