"""Trains a machine-learning model to predict loop transformation benefit from extracted features."""

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from loopcost.ml.dataset import load_dataset

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MAX_DEPTH = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_all(path="data/training_data.csv"):
    """Trains one DecisionTreeClassifier per transform type on an 80/20 split.

    Saves each model to models/{transform_type}_model.pkl (joblib) and its decision rules
    to models/{transform_type}_rules.txt (sklearn.tree.export_text). Returns a dict keyed by
    transform type with n_train/n_test/accuracy/confusion_matrix, for reporting.
    """
    X_by_transform, y_by_transform = load_dataset(path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    report = {}
    for transform_type, X in X_by_transform.items():
        y = y_by_transform[transform_type]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )

        model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred, labels=[False, True])

        joblib.dump(model, MODELS_DIR / f"{transform_type}_model.pkl")
        rules = export_text(model, feature_names=list(X.columns))
        (MODELS_DIR / f"{transform_type}_rules.txt").write_text(rules)

        report[transform_type] = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": accuracy,
            "confusion_matrix": matrix,
        }

    return report


def _print_report(report):
    for transform_type, info in report.items():
        print(
            f"{transform_type}: n_train={info['n_train']} n_test={info['n_test']} "
            f"accuracy={info['accuracy']:.3f}"
        )
        print("  confusion matrix [rows=true(False,True), cols=pred(False,True)]:")
        for line in info["confusion_matrix"]:
            print(f"    {line}")


if __name__ == "__main__":
    _print_report(train_all())
