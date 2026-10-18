"""Run the fixed experiment: compare, tune, choose threshold, then test once."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    cross_validate,
)
from threadpoolctl import threadpool_limits

from src.data import (
    DATA_PATH,
    FEATURES,
    NUMERIC_FEATURES,
    ROOT,
    SEED,
    TARGET,
    audit_dataset,
    load_dataset,
    prepare_xy,
    sha256_file,
    split_data,
)
from src.eda import create_eda
from src.evaluate import (
    evaluate_predictions,
    plot_importance,
    plot_test_evaluation,
    plot_thresholds,
    threshold_analysis,
)
from src.preprocessing import build_models, tuning_grid


def write_json(value, path):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def compare_models(models, X, y, cv):
    """Compare all pipelines on the same training-only folds."""
    scoring = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    records, fold_records = [], []
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
        )
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        records.append(row)
        for fold in range(cv.n_splits):
            fold_records.append(
                {
                    "model": name,
                    "fold": fold + 1,
                    **{
                        metric: float(scores[f"test_{metric}"][fold])
                        for metric in scoring
                    },
                }
            )
        print(f"  {name}: CV AP = {row['average_precision_mean']:.4f}", flush=True)
    return pd.DataFrame(records).sort_values(
        "average_precision_mean", ascending=False
    ), pd.DataFrame(fold_records)


def _run_experiment(data_path, output_root):
    started = time.perf_counter()
    output_root = Path(output_root)
    reports, models_dir = output_root / "reports", output_root / "models"
    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures = reports / "figures"
    frame = load_dataset(data_path)
    X, y = prepare_xy(frame)
    partitions = split_data(frame)
    fit_idx, val_idx, test_idx = (
        partitions[name] for name in ["fit", "validation", "test"]
    )
    X_fit, y_fit = X.loc[fit_idx], y.loc[fit_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    audit = audit_dataset(frame)
    write_json(audit, reports / "data_quality.json")
    assignment = pd.concat(
        [
            pd.DataFrame({"UDI": frame.loc[index, "UDI"], "partition": name})
            for name, index in partitions.items()
        ]
    ).sort_values("UDI")
    assignment.to_csv(reports / "split_assignments.csv", index=False)
    print("Creating EDA from fitting rows only ...", flush=True)
    create_eda(frame.loc[fit_idx], reports)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = build_models()
    print("Comparing models using five stratified folds ...", flush=True)
    comparison, fold_scores = compare_models(candidates, X_fit, y_fit, cv)
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    fold_scores.to_csv(reports / "cv_fold_scores.csv", index=False)
    selected_name = comparison.loc[comparison.model != "Dummy"].iloc[0]["model"]
    grid = tuning_grid(selected_name)
    print(
        f"Tuning {selected_name}: {len(list(ParameterGrid(grid)))} configurations ...",
        flush=True,
    )
    search = GridSearchCV(
        candidates[selected_name],
        grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_fit, y_fit)
    final_pipeline = search.best_estimator_
    pd.DataFrame(search.cv_results_).to_csv(reports / "tuning_results.csv", index=False)

    print(
        "Choosing threshold and explaining the fitted model on validation rows ...",
        flush=True,
    )
    val_probability = final_pipeline.predict_proba(X_val)[:, 1]
    threshold_table, threshold = threshold_analysis(y_val, val_probability)
    threshold_table.to_csv(reports / "threshold_analysis.csv", index=False)
    plot_thresholds(threshold_table, threshold, figures / "threshold_analysis.png")
    pd.DataFrame(
        {
            "UDI": frame.loc[val_idx, "UDI"].to_numpy(),
            "y_true": y_val.to_numpy(),
            "failure_probability": val_probability,
        }
    ).to_csv(reports / "validation_predictions.csv", index=False)
    importance = permutation_importance(
        final_pipeline,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=10,
        random_state=SEED,
        n_jobs=1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_table.to_csv(reports / "feature_importance.csv", index=False)
    plot_importance(importance_table, figures / "feature_importance.png")

    # Freeze pipeline and threshold before accessing test features for prediction.
    # Do not refit on validation rows: that could shift the selected score threshold.
    bundle = {
        "pipeline": final_pipeline,
        "threshold": threshold,
        "features": FEATURES,
        "model_name": selected_name,
        "dataset_sha256": sha256_file(data_path),
        "training_ranges": {
            name: [float(X_fit[name].min()), float(X_fit[name].max())]
            for name in NUMERIC_FEATURES
        },
        "sklearn_version": importlib.metadata.version("scikit-learn"),
    }
    model_path = models_dir / "failure_model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    print("Running the single final held-out test evaluation ...", flush=True)
    test_probability = final_pipeline.predict_proba(X.loc[test_idx])[:, 1]
    metrics, test_prediction = evaluate_predictions(
        y.loc[test_idx], test_probability, threshold
    )
    pd.DataFrame(
        {
            "UDI": frame.loc[test_idx, "UDI"].to_numpy(),
            "y_true": y.loc[test_idx].to_numpy(),
            "failure_probability": test_probability,
            "prediction": test_prediction,
        }
    ).to_csv(reports / "test_predictions.csv", index=False)
    plot_test_evaluation(y.loc[test_idx], test_probability, test_prediction, figures)
    # Overall accuracy is reported separately; it has no per-class support row.
    pd.DataFrame(metrics["classification_report"]).T.drop(index="accuracy").to_csv(
        reports / "classification_report.csv"
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "pandas",
            "matplotlib",
            "scikit-learn",
            "scipy",
            "joblib",
            "threadpoolctl",
        ]
    }
    source_hashes = {
        path.name: sha256_file(path)
        for path in sorted((ROOT / "src").glob("*.py"))
        if path.name != "reporting.py"
    }
    results = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256_file(data_path),
        "model_sha256": sha256_file(model_path),
        "python_version": platform.python_version(),
        "package_versions": versions,
        "source_sha256": source_hashes,
        "random_state": SEED,
        "split": {
            name: {"rows": len(index), "failures": int(y.loc[index].sum())}
            for name, index in partitions.items()
        },
        "features": FEATURES,
        "selected_model": selected_name,
        "selection_metric": "average_precision",
        "cv_folds": cv.n_splits,
        "tuning_grid": grid,
        "tuning_configurations": len(list(ParameterGrid(grid))),
        "best_parameters": search.best_params_,
        "best_cv_average_precision": float(search.best_score_),
        "best_cv_average_precision_std": float(
            search.cv_results_["std_test_score"][search.best_index_]
        ),
        "threshold": threshold,
        "threshold_rule": "Maximum validation F1 over 0.05..0.95 in 0.05 steps; highest threshold wins ties.",
        "validation_at_selected_threshold": threshold_table.loc[
            threshold_table.threshold == threshold
        ]
        .iloc[0]
        .to_dict(),
        "test_metrics": metrics,
        "training_seconds": round(time.perf_counter() - started, 2),
    }
    write_json(results, reports / "results.json")
    from src.reporting import refresh_reports

    refresh_reports(output_root)
    print(f"Selected: {selected_name}; threshold={threshold:.2f}")
    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]:
        print(f"  Test {metric}: {metrics[metric]:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    return results


def run_experiment(data_path=DATA_PATH, output_root=ROOT):
    """Use two numerical-library threads to keep laptop resource use predictable."""
    with threadpool_limits(limits=2):
        return _run_experiment(Path(data_path), Path(output_root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT, help="Root for models/ and reports/"
    )
    args = parser.parse_args()
    run_experiment(args.data, args.output_dir)


if __name__ == "__main__":
    main()
