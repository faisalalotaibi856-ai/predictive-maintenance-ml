"""Tests use real UCI rows, with explicit mutations only to exercise validation."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import (
    DATA_PATH,
    EXPECTED_SHA256,
    FAILURE_FLAGS,
    FEATURES,
    NUMERIC_FEATURES,
    ROOT,
    TARGET,
    audit_dataset,
    load_dataset,
    prepare_xy,
    sha256_file,
    split_data,
    validate_features,
)
from src.evaluate import evaluate_predictions, threshold_analysis
from src.predict import predict_sample
from src.preprocessing import build_models, build_preprocessor


@pytest.fixture(scope="module")
def frame():
    return load_dataset()


def test_official_data_provenance_and_audit(frame):
    assert sha256_file(DATA_PATH) == EXPECTED_SHA256
    audit = audit_dataset(frame)
    assert (audit["rows"], audit["columns"]) == (10000, 14)
    assert audit["flag_or_target_mismatches"] == 27


def test_target_and_failure_flags_cannot_enter_features(frame):
    X, _ = prepare_xy(frame)
    changed = frame.copy()
    changed[FAILURE_FLAGS + [TARGET]] = 1 - changed[FAILURE_FLAGS + [TARGET]]
    changed["UDI"] += 100000
    changed["Product ID"] = "ignored"
    X_changed, _ = prepare_xy(changed)
    pd.testing.assert_frame_equal(X, X_changed)
    assert list(X.columns) == FEATURES


def test_splits_are_disjoint_exhaustive_and_stratified(frame):
    split = split_data(frame)
    groups = [set(index) for index in split.values()]
    assert (
        len(groups[0] & groups[1])
        == len(groups[0] & groups[2])
        == len(groups[1] & groups[2])
        == 0
    )
    assert set.union(*groups) == set(frame.index)
    assert [len(index) for index in split.values()] == [6400, 1600, 2000]
    for index in split.values():
        assert abs(frame.loc[index, TARGET].mean() - frame[TARGET].mean()) < 0.001


def test_imputation_is_fit_local_and_encoded_output_is_finite(frame):
    X, _ = prepare_xy(frame)
    training = X.iloc[:100].copy()
    training.loc[training.index[0], NUMERIC_FEATURES[0]] = np.nan
    preprocessor = build_preprocessor(scale_numeric=True)
    preprocessor.fit(training)
    medians = (
        preprocessor.named_transformers_["numeric"]
        .named_steps["impute"]
        .statistics_.copy()
    )
    np.testing.assert_allclose(medians, training[NUMERIC_FEATURES].median())
    held_out = X.iloc[100:105].copy()
    held_out[NUMERIC_FEATURES[0]] = 999999
    held_out.loc[held_out.index[0], NUMERIC_FEATURES[1]] = np.nan
    transformed = preprocessor.transform(held_out)
    assert transformed.shape == (5, 8)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(
        medians,
        preprocessor.named_transformers_["numeric"].named_steps["impute"].statistics_,
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing_column",
        "leakage_column",
        "bad_numeric",
        "infinity",
        "bad_type",
        "missing_value",
        "negative_wear",
    ],
)
def test_prediction_schema_rejects_invalid_inputs(frame, case):
    sample = frame[FEATURES].iloc[[0]].copy()
    if case == "missing_column":
        sample = sample.drop(columns=[FEATURES[0]])
    elif case == "leakage_column":
        sample[TARGET] = 1
    elif case == "bad_numeric":
        sample[NUMERIC_FEATURES[0]] = "hot"
    elif case == "infinity":
        sample[NUMERIC_FEATURES[0]] = np.inf
    elif case == "bad_type":
        sample["Type"] = "UNKNOWN"
    elif case == "missing_value":
        sample[NUMERIC_FEATURES[0]] = np.nan
    else:
        sample["Tool wear [min]"] = -1
    with pytest.raises(ValueError):
        validate_features(sample, allow_missing=False)


def test_pipeline_serialization_and_prediction_use_stored_threshold(frame, tmp_path):
    subset = frame.groupby(TARGET, group_keys=False).head(30)
    X, y = prepare_xy(subset)
    model = build_models()["Logistic Regression"].fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "pipeline": model,
            "threshold": 0.35,
            "features": FEATURES,
            "model_name": "test fixture",
        },
        path,
    )
    sample = X.iloc[0].to_dict()
    result = predict_sample(sample, path)
    expected = float(model.predict_proba(X.iloc[[0]])[0, 1])
    assert result["failure_probability"] == pytest.approx(expected)
    assert 0 <= result["failure_probability"] <= 1
    assert result["predicted_class"] == int(expected >= 0.35)


def test_saved_results_are_consistent_with_saved_predictions():
    path = ROOT / "reports/results.json"
    if not path.exists():
        pytest.skip("Run python -m src.train to verify generated artifacts.")
    results = json.loads(path.read_text())
    saved = pd.read_csv(ROOT / "reports/test_predictions.csv")
    metrics, predictions = evaluate_predictions(
        saved.y_true, saved.failure_probability, results["threshold"]
    )
    for name in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]:
        assert metrics[name] == pytest.approx(results["test_metrics"][name])
    assert metrics["confusion_matrix"] == results["test_metrics"]["confusion_matrix"]
    np.testing.assert_array_equal(predictions, saved.prediction)
    validation = pd.read_csv(ROOT / "reports/validation_predictions.csv")
    _, threshold = threshold_analysis(
        validation.y_true, validation.failure_probability.to_numpy()
    )
    assert threshold == results["threshold"]
