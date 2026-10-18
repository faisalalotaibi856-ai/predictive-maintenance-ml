"""Generate result tables from measured artifacts; keep documentation synchronized."""

import json
from pathlib import Path

import pandas as pd

from src.data import ROOT

START = "<!-- RUN_RESULTS_START -->"
END = "<!-- RUN_RESULTS_END -->"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def load_results(root=ROOT):
    """Read the last measured run; notebooks display it without reevaluating test data."""
    return json.loads((Path(root) / "reports/results.json").read_text(encoding="utf-8"))


def refresh_reports(root=ROOT):
    root = Path(root)
    results = load_results(root)
    reports = root / "reports"
    comparison = pd.read_csv(reports / "model_comparison.csv")
    importance = pd.read_csv(reports / "feature_importance.csv")
    audit = json.loads((reports / "data_quality.json").read_text(encoding="utf-8"))
    metrics = results["test_metrics"]
    rows = []
    for _, row in comparison.iterrows():
        rows.append(
            [
                row.model,
                f"{row.average_precision_mean:.4f} ± {row.average_precision_std:.4f}",
                f"{row.roc_auc_mean:.4f}",
                f"{row.precision_mean:.4f}",
                f"{row.recall_mean:.4f}",
                f"{row.f1_mean:.4f}",
                f"{row.accuracy_mean:.4f}",
            ]
        )
    cv_table = markdown_table(
        [
            "Model",
            "CV AP (mean ± fold SD)",
            "CV ROC-AUC",
            "Precision",
            "Recall",
            "F1",
            "Accuracy",
        ],
        rows,
    )
    metric_table = markdown_table(
        ["Held-out test metric", "Value"],
        [
            [name, f"{metrics[key]:.4f}"]
            for name, key in [
                ("Accuracy", "accuracy"),
                ("Failure precision", "precision"),
                ("Failure recall", "recall"),
                ("Failure F1", "f1"),
                ("ROC-AUC", "roc_auc"),
                ("Average Precision (AP)", "average_precision"),
            ]
        ],
    )
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    parameters = ", ".join(
        f"`{key.replace('classifier__', '')}={value}`"
        for key, value in results["best_parameters"].items()
    )
    top = ", ".join(f"**{name}**" for name in importance.feature.head(3))
    splits = "; ".join(
        f"{name}: {info['rows']:,} rows / {info['failures']} failures"
        for name, info in results["split"].items()
    )
    summary = f"""Measured run: `{results['completed_at_utc']}`. Seed: `{results['random_state']}`.

**Selected model: {results['selected_model']}**. Decision threshold: **{results['threshold']:.2f}**, selected by validation F1 before test evaluation.

Partition sizes — {splits}.

Training-only, five-fold comparison (classification metrics use each model's default decision rule):

{cv_table}

The winner was tuned with {results['tuning_configurations']} configurations. Best parameters: {parameters}.
Best tuning CV AP: **{results['best_cv_average_precision']:.4f}**. This is a selection score, not an unbiased estimate; model and parameter selection reused these folds.

Final test results, calculated once after freezing the fitted pipeline and threshold:

{metric_table}

Confusion matrix `[[TN, FP], [FN, TP]]`: **[[{tn}, {fp}], [{fn}, {tp}]]**.
The model detected **{tp} of {tp + fn}** test failures, missed **{fn}**, and raised **{fp}** false alarms.

Largest validation permutation-importance values: {top}. Values are AP decreases when a feature is shuffled; they are associations, not causal effects.

AP is scikit-learn's non-interpolated Average Precision, not trapezoidal PR-AUC. CV and final test figures describe different partitions and decision rules.
"""
    for filename in ["README.md", "INTERVIEW_GUIDE.md"]:
        path = root / filename
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if START in text and END in text:
                before = text.split(START, 1)[0]
                after = text.split(END, 1)[1]
                path.write_text(
                    before + START + "\n\n" + summary + "\n" + END + after,
                    encoding="utf-8",
                )
    importance_table = markdown_table(
        ["Feature", "Mean AP decrease", "Shuffle SD"],
        [
            [row.feature, f"{row.importance_mean:.4f}", f"{row.importance_std:.4f}"]
            for _, row in importance.iterrows()
        ],
    )
    detail = f"""# Measured experiment results

{summary}
## Validation-only interpretation

{importance_table}

Ten repeats per feature. The standard deviations measure shuffle variability, not confidence intervals for deployment performance. Correlated features can share or mask importance.

## Data audit

The unchanged CSV has {audit['rows']:,} rows, {audit['columns']} columns, {sum(audit['missing_by_column'].values())} missing cells, {audit['full_duplicate_rows']} complete duplicate rows, and {audit['feature_duplicate_rows']} duplicate six-feature combinations.
The target has {audit['target_counts']['1']} failures ({audit['failure_rate']:.2%}).
There are {audit['flag_or_target_mismatches']} disagreements between the OR of the five failure flags and the supplied target: {audit['target_1_without_flag']} target-positive rows without a flag and {audit['target_0_with_flag']} target-negative rows with a flag.
The original target is preserved; none of these flags is a feature.

## Reproduction record

- Dataset SHA-256: `{results['dataset_sha256']}`
- Python: `{results['python_version']}`
- Training duration in the recorded environment: {results['training_seconds']} seconds (not a laptop guarantee).
- Exact observed package versions and source/model hashes: `results.json`.
- All per-row test scores: `test_predictions.csv`.
- All fold scores: `cv_fold_scores.csv`.
- Every tuning configuration: `tuning_results.csv`.
- Threshold grid and validation confusion counts: `threshold_analysis.csv`.
- Classification report: `classification_report.csv`.

The stored model was fit on the fitting partition only. Validation data selected the threshold and supported interpretation; it was not used for refitting. Test metrics estimate random-holdout performance on this synthetic benchmark, not forward-time factory performance.
"""
    (reports / "RESULTS.md").write_text(detail, encoding="utf-8")
    return summary


if __name__ == "__main__":
    refresh_reports()
