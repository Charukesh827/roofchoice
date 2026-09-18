"""Tests that training runs end-to-end on the real sweep CSV and predict_profitability explains itself."""
import pytest
from sklearn.model_selection import train_test_split

from loopcost.ml.dataset import TRANSFORM_TYPE_SOURCES, load_dataset
from loopcost.ml.predict import predict_profitability
from loopcost.ml.train import MAX_DEPTH, RANDOM_STATE, TEST_SIZE, train_all

TRANSFORM_TYPES = list(TRANSFORM_TYPE_SOURCES)


@pytest.fixture(scope="module")
def report():
    return train_all()


def test_train_all_produces_a_report_for_every_transform_type(report):
    assert set(report) == set(TRANSFORM_TYPES)
    for transform_type, info in report.items():
        assert info["n_train"] > 0
        assert info["n_test"] > 0
        assert 0.0 <= info["accuracy"] <= 1.0
        assert info["confusion_matrix"].shape == (2, 2)


def test_train_all_saves_model_and_rules_files(report):
    from loopcost.ml.train import MODELS_DIR

    for transform_type in TRANSFORM_TYPES:
        model_path = MODELS_DIR / f"{transform_type}_model.pkl"
        rules_path = MODELS_DIR / f"{transform_type}_rules.txt"
        assert model_path.exists(), f"missing {model_path}"
        assert rules_path.exists(), f"missing {rules_path}"
        assert rules_path.read_text().strip() != ""


def test_max_depth_is_capped_for_interpretability():
    assert 4 <= MAX_DEPTH <= 5


@pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
def test_predict_profitability_explains_known_test_split_rows(report, transform_type):
    X_by_transform, y_by_transform = load_dataset()
    X = X_by_transform[transform_type]
    y = y_by_transform[transform_type]

    _, X_test, _, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    for _, row in X_test.iterrows():
        features = {
            "oi": row["oi"],
            "bound": "compute-bound" if row["bound"] else "memory-bound",
            "trip_count": row["trip_count"],
            "working_set_bytes": row["working_set_bytes"],
            "has_loop_carried_dep": bool(row["has_loop_carried_dep"]),
        }

        profitable, confidence, explanation = predict_profitability(features, transform_type)

        assert isinstance(profitable, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(explanation, str) and explanation.strip() != ""
        assert "predict" in explanation  # every explanation ends in a "leaf: predict <bool>" step
