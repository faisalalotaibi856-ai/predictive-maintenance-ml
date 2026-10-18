"""Predict one machine record from a JSON file containing the six named inputs."""

import argparse
import json
from pathlib import Path
import warnings

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data import FEATURES, ROOT, validate_features


def predict_sample(sample, model_path=ROOT / "models/failure_model.joblib"):
    """Load a trusted local artifact and apply its stored preprocessing and threshold."""
    if not isinstance(sample, dict):
        raise ValueError("Sample must be a JSON object with six named features.")
    if not Path(model_path).exists():
        raise FileNotFoundError("Model missing. Run: python -m src.train")
    X = validate_features(pd.DataFrame([sample]), allow_missing=False)
    # Joblib can execute code when loading: use only an artifact you trust.
    bundle = joblib.load(model_path)
    if bundle["features"] != FEATURES:
        raise ValueError("The saved model uses a different feature schema.")
    outside = [
        name
        for name, (low, high) in bundle.get("training_ranges", {}).items()
        if not low <= X.iloc[0][name] <= high
    ]
    if outside:
        warnings.warn(
            "Outside the fitting-data range: " + ", ".join(outside),
            UserWarning,
            stacklevel=2,
        )
    with threadpool_limits(limits=2):
        probability = float(bundle["pipeline"].predict_proba(X)[0, 1])
    threshold = float(bundle["threshold"])
    return {
        "predicted_class": int(probability >= threshold),
        "failure_probability": probability,
        "threshold": threshold,
        "model": bundle["model_name"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-file", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/failure_model.joblib"
    )
    args = parser.parse_args()
    try:
        sample = json.loads(args.sample_file.read_text(encoding="utf-8"))
        print(json.dumps(predict_sample(sample, args.model), indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Prediction error: {exc}\n")


if __name__ == "__main__":
    main()
