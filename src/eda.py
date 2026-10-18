"""Compact exploratory analysis restricted to the model-fitting partition."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import FEATURES, NUMERIC_FEATURES, TARGET
from src.evaluate import save_figure


def create_eda(frame, reports):
    """Write descriptive tables and four useful figures, never using test rows."""
    figures = reports / "figures"
    frame[FEATURES].describe(include="all").to_csv(
        reports / "fit_descriptive_statistics.csv"
    )
    grouped = frame.groupby(TARGET)[NUMERIC_FEATURES].mean()
    grouped.to_csv(reports / "fit_means_by_target.csv")
    rates = frame.groupby("Type")[TARGET].agg(["count", "sum", "mean"])
    rates.columns = ["rows", "failures", "failure_rate"]
    rates.to_csv(reports / "fit_failure_rate_by_type.csv")
    correlations = frame[NUMERIC_FEATURES].corr()
    correlations.to_csv(reports / "fit_correlations.csv")

    counts = frame[TARGET].value_counts().reindex([0, 1], fill_value=0)
    figure, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    bars = ax.bar(["No failure", "Failure"], counts, color=["#174a70", "#d07b22"])
    ax.bar_label(
        bars, labels=[f"{n:,} ({n / len(frame):.2%})" for n in counts], padding=4
    )
    ax.set(
        title="Class imbalance (fitting partition)",
        xlabel="Machine failure label",
        ylabel="Number of records",
        ylim=(0, counts.max() * 1.15),
    )
    save_figure(figure, figures / "target_distribution.png")

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for column, ax in zip(NUMERIC_FEATURES, axes.flat):
        ax.hist(
            frame[column], bins=32, color="#174a70", edgecolor="white", linewidth=0.3
        )
        ax.set(xlabel=column, ylabel="Records")
    rates["failure_rate"].plot.bar(ax=axes.flat[-1], color="#d07b22", rot=0)
    axes.flat[-1].set(xlabel="Product type", ylabel="Failure fraction")
    figure.suptitle("Input distributions and failure rate by type (fitting partition)")
    save_figure(figure, figures / "feature_distributions.png")

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for column, ax in zip(NUMERIC_FEATURES, axes.flat):
        ax.boxplot(
            [frame.loc[frame[TARGET] == k, column] for k in [0, 1]],
            tick_labels=["No failure", "Failure"],
            showfliers=False,
        )
        ax.set(xlabel="Machine failure label", ylabel=column)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.05,
        0.6,
        "Boxes show median and middle 50%.\nWhiskers use 1.5 × IQR.\nOutliers hidden for readability.\nAssociation does not imply causation.",
        fontsize=11,
    )
    figure.suptitle("Sensor values by target (fitting partition)")
    save_figure(figure, figures / "features_by_target.png")

    short = ["Air temperature", "Process temperature", "Speed", "Torque", "Tool wear"]
    figure, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    grid = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(short)), short, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(short)), short)
    for row in range(len(short)):
        for col in range(len(short)):
            value = correlations.iloc[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.65 else "black",
            )
    ax.set(
        title="Pearson correlation (fitting partition)",
        xlabel="Numeric input",
        ylabel="Numeric input",
    )
    figure.colorbar(grid, ax=ax, label="Pearson r", shrink=0.8)
    save_figure(figure, figures / "correlation_matrix.png")
