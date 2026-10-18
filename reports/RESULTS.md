# Measured experiment results

Measured run: `2026-10-05T09:26:10.112041+00:00`. Seed: `42`.

**Selected model: HistGradientBoosting**. Decision threshold: **0.10**, selected by validation F1 before test evaluation.

Partition sizes — fit: 6,400 rows / 217 failures; validation: 1,600 rows / 54 failures; test: 2,000 rows / 68 failures.

Training-only, five-fold comparison (classification metrics use each model's default decision rule):

| Model | CV AP (mean ± fold SD) | CV ROC-AUC | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| HistGradientBoosting | 0.7373 ± 0.0752 | 0.9642 | 0.6393 | 0.7647 | 0.6943 | 0.9770 |
| Random Forest | 0.7268 ± 0.0968 | 0.9615 | 0.8551 | 0.3865 | 0.5309 | 0.9770 |
| Logistic Regression | 0.4594 ± 0.0361 | 0.9040 | 0.1438 | 0.8016 | 0.2437 | 0.8314 |
| Dummy | 0.0339 ± 0.0004 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.9661 |

The winner was tuned with 8 configurations. Best parameters: `class_weight=None`, `l2_regularization=0.0`, `max_leaf_nodes=31`.
Best tuning CV AP: **0.7752**. This is a selection score, not an unbiased estimate; model and parameter selection reused these folds.

Final test results, calculated once after freezing the fitted pipeline and threshold:

| Held-out test metric | Value |
| --- | --- |
| Accuracy | 0.9835 |
| Failure precision | 0.7465 |
| Failure recall | 0.7794 |
| Failure F1 | 0.7626 |
| ROC-AUC | 0.9664 |
| Average Precision (AP) | 0.8430 |

Confusion matrix `[[TN, FP], [FN, TP]]`: **[[1914, 18], [15, 53]]**.
The model detected **53 of 68** test failures, missed **15**, and raised **18** false alarms.

Largest validation permutation-importance values: **Torque [Nm]**, **Air temperature [K]**, **Rotational speed [rpm]**. Values are AP decreases when a feature is shuffled; they are associations, not causal effects.

AP is scikit-learn's non-interpolated Average Precision, not trapezoidal PR-AUC. CV and final test figures describe different partitions and decision rules.

## Validation-only interpretation

| Feature | Mean AP decrease | Shuffle SD |
| --- | --- | --- |
| Torque [Nm] | 0.4929 | 0.0223 |
| Air temperature [K] | 0.4623 | 0.0235 |
| Rotational speed [rpm] | 0.3239 | 0.0365 |
| Process temperature [K] | 0.2963 | 0.0349 |
| Tool wear [min] | 0.2913 | 0.0334 |
| Type | 0.0594 | 0.0211 |

Ten repeats per feature. The standard deviations measure shuffle variability, not confidence intervals for deployment performance. Correlated features can share or mask importance.

## Data audit

The unchanged CSV has 10,000 rows, 14 columns, 0 missing cells, 0 complete duplicate rows, and 0 duplicate six-feature combinations.
The target has 339 failures (3.39%).
There are 27 disagreements between the OR of the five failure flags and the supplied target: 9 target-positive rows without a flag and 18 target-negative rows with a flag.
The original target is preserved; none of these flags is a feature.

## Reproduction record

- Dataset SHA-256: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`
- Python: `3.12.14`
- Training duration in the recorded environment: 15.53 seconds (not a laptop guarantee).
- Exact observed package versions and source/model hashes: `results.json`.
- All per-row test scores: `test_predictions.csv`.
- All fold scores: `cv_fold_scores.csv`.
- Every tuning configuration: `tuning_results.csv`.
- Threshold grid and validation confusion counts: `threshold_analysis.csv`.
- Classification report: `classification_report.csv`.

The stored model was fit on the fitting partition only. Validation data selected the threshold and supported interpretation; it was not used for refitting. Test metrics estimate random-holdout performance on this synthetic benchmark, not forward-time factory performance.
