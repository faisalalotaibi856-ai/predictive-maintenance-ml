"""Threshold selection on validation data and one final test evaluation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def save_figure(figure, path):
    """Save a readable static plot and release its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def threshold_analysis(y_true, probabilities):
    """Predeclared 0.05-step grid; maximize validation F1, break ties upward."""
    records = []
    for threshold in np.round(np.arange(0.05, 1.0, 0.05), 2):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        records.append(
            {
                "threshold": float(threshold),
                "precision": float(
                    precision_score(y_true, predictions, zero_division=0)
                ),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    table = pd.DataFrame(records)
    chosen = table.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    return table, float(chosen["threshold"])


def plot_thresholds(table, threshold, path):
    figure, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    for metric, color in [
        ("precision", "#174a70"),
        ("recall", "#14887b"),
        ("f1", "#d07b22"),
    ]:
        ax.plot(
            table.threshold,
            table[metric],
            label=metric.capitalize(),
            color=color,
            marker="o",
            ms=3,
        )
    ax.axvline(
        threshold, color="#66717e", linestyle="--", label=f"Selected: {threshold:.2f}"
    )
    ax.set(
        title="Threshold trade-off (validation set only)",
        xlabel="Failure-score threshold",
        ylabel="Score",
        ylim=(0, 1.03),
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    save_figure(figure, path)


def evaluate_predictions(y_true, probabilities, threshold):
    """Compute all metrics from one saved vector of positive-class scores."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["No failure", "Failure"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, predictions


def plot_test_evaluation(y_true, probabilities, predictions, figures):
    figure, ax = plt.subplots(figsize=(5.4, 4.4), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=["No failure", "Failure"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Final held-out test confusion matrix")
    save_figure(figure, figures / "confusion_matrix.png")
    figure, ax = plt.subplots(figsize=(6.0, 4.6), layout="constrained")
    RocCurveDisplay.from_predictions(
        y_true, probabilities, ax=ax, name="Selected model", color="#174a70"
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="No-skill reference")
    ax.set_title("Final held-out test ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    save_figure(figure, figures / "roc_curve.png")
    figure, ax = plt.subplots(figsize=(6.0, 4.6), layout="constrained")
    PrecisionRecallDisplay.from_predictions(
        y_true, probabilities, ax=ax, name="Selected model", color="#14887b"
    )
    ax.axhline(
        np.mean(y_true), color="#999999", linestyle="--", label="Test prevalence"
    )
    ax.set_title("Final held-out test precision–recall curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)
    save_figure(figure, figures / "precision_recall_curve.png")


def plot_importance(table, path):
    ordered = table.sort_values("importance_mean")
    figure, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    ax.barh(
        ordered.feature,
        ordered.importance_mean,
        xerr=ordered.importance_std,
        color="#14887b",
        capsize=3,
    )
    ax.axvline(0, color="#66717e", linewidth=0.8)
    ax.set(
        title="Permutation importance (validation set)",
        xlabel="Decrease in Average Precision (mean ± shuffle SD)",
        ylabel="Input feature",
    )
    ax.grid(axis="x", alpha=0.2)
    save_figure(figure, path)
