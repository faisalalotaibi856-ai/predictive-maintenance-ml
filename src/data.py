"""Download the official dataset, validate its schema, and select safe inputs."""

import argparse
import hashlib
import io
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/ai4i2020.csv"
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
EXPECTED_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"
TARGET = "Machine failure"
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FAILURE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDENTIFIERS = ["UDI", "Product ID"]
SEED = 42


def sha256_file(path):
    """Hash a file so a run can be traced to exact input bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_dataset(path=DATA_PATH):
    """Reuse or download the unchanged official CSV; never generate replacement data."""
    path = Path(path)
    if path.exists():
        if sha256_file(path) != EXPECTED_SHA256:
            raise ValueError(
                "Existing CSV differs from the verified UCI file. See data/README.md."
            )
        return path
    try:
        with urlopen(DATA_URL, timeout=60) as response:
            archive_bytes = response.read(5_000_001)
        if len(archive_bytes) > 5_000_000:
            raise ValueError("Unexpectedly large dataset archive.")
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            content = archive.read("ai4i2020.csv")
    except (URLError, TimeoutError, BadZipFile, KeyError) as exc:
        raise RuntimeError(
            "Could not download UCI AI4I. Download the official ZIP manually, "
            "extract ai4i2020.csv into data/raw/, then retry. See data/README.md."
        ) from exc
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise ValueError(
            "UCI file has changed: review its provenance before updating the checksum."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return path


def validate_features(frame, *, allow_missing=True):
    """Validate six named inputs; return a numeric/categorical copy in stable order."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Supply a nonempty pandas DataFrame.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES))
    if missing or extra:
        raise ValueError(
            f"Input columns mismatch. Missing: {missing}; unexpected: {extra}"
        )
    result = frame.loc[:, FEATURES].copy()
    for column in NUMERIC_FEATURES:
        if result[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{column} must contain numbers, not booleans.")
        try:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain numeric values.") from exc
        known = result[column].dropna()
        if not np.isfinite(known).all():
            raise ValueError(f"{column} cannot contain infinity.")
        if (known < 0).any() or (column != "Tool wear [min]" and (known == 0).any()):
            raise ValueError(f"{column} has physically invalid nonpositive values.")
    invalid_types = (
        result["Type"].dropna().loc[lambda values: ~values.isin(["L", "M", "H"])]
    )
    if not invalid_types.empty:
        raise ValueError("Type must be L, M, or H.")
    result["Type"] = result["Type"].astype(object).where(result["Type"].notna(), np.nan)
    if not allow_missing and result.isna().any().any():
        raise ValueError(
            "Prediction samples must provide every feature without missing values."
        )
    return result


def load_dataset(path=DATA_PATH):
    """Read the verified CSV and fail clearly on schema or target problems."""
    path = download_dataset(path)
    frame = pd.read_csv(path)
    required = set(FEATURES + IDENTIFIERS + [TARGET] + FAILURE_FLAGS)
    if set(frame.columns) != required:
        raise ValueError(
            "Dataset schema does not match the official 14-column AI4I CSV."
        )
    validate_features(frame[FEATURES])
    for column in [TARGET] + FAILURE_FLAGS:
        if frame[column].isna().any() or not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be a complete binary label.")
    if frame["UDI"].duplicated().any() or frame["UDI"].isna().any():
        raise ValueError("UDI must uniquely identify each row for split auditing.")
    return frame


def prepare_xy(frame):
    """Use an explicit feature allowlist: IDs, target, and failure flags cannot enter X."""
    return validate_features(frame.loc[:, FEATURES]), frame[TARGET].astype(int).copy()


def split_data(frame):
    """Lock 20% test; reserve 20% of the other 80% for threshold validation."""
    development_idx, test_idx = train_test_split(
        frame.index,
        test_size=0.20,
        stratify=frame[TARGET],
        random_state=SEED,
    )
    fit_idx, validation_idx = train_test_split(
        development_idx,
        test_size=0.20,
        stratify=frame.loc[development_idx, TARGET],
        random_state=SEED + 1,
    )
    return {"fit": fit_idx, "validation": validation_idx, "test": test_idx}


def audit_dataset(frame):
    """Check quality and flag/target inconsistencies without changing supplied labels."""
    flags_or = frame[FAILURE_FLAGS].max(axis=1)
    mismatches = flags_or.ne(frame[TARGET])
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "full_duplicate_rows": int(frame.duplicated().sum()),
        "feature_duplicate_rows": int(frame[FEATURES].duplicated().sum()),
        "target_counts": {
            str(k): int(v) for k, v in frame[TARGET].value_counts().items()
        },
        "failure_rate": float(frame[TARGET].mean()),
        "failure_flag_counts": frame[FAILURE_FLAGS].sum().astype(int).to_dict(),
        "flag_or_target_mismatches": int(mismatches.sum()),
        "target_1_without_flag": int(((frame[TARGET] == 1) & (flags_or == 0)).sum()),
        "target_0_with_flag": int(((frame[TARGET] == 0) & (flags_or == 1)).sum()),
        "mismatching_udi": frame.loc[mismatches, "UDI"].astype(int).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    frame = load_dataset(args.path)
    print(f"Verified UCI AI4I: {len(frame):,} rows; {frame[TARGET].sum()} failures.")
    print(f"SHA256: {sha256_file(args.path)}")


if __name__ == "__main__":
    main()
