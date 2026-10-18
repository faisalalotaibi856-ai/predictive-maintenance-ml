# Interview Guide

Study the code, run it locally, and explain the choices in your own words. Describe this as an academic portfolio project. Be honest about any tools or assistance you used and about which parts you personally changed, tested, and understood.

## 1. Thirty-second project explanation

“This is a machine-failure classification project using the UCI AI4I benchmark. I used six operational inputs, excluded outcome-related columns to prevent leakage, and compared logistic regression, random forest, and histogram gradient boosting. The workflow selects a model through stratified cross-validation, chooses a decision threshold on separate validation data, and evaluates once on a held-out test set. I also included a prediction command, tests, and reproducible reports.”

## 2. Two-minute project explanation

“The aim was to recognize observations associated with machine failure. The official dataset contains 10,000 synthetic industrial-style observations, with failure in about 3.4% of rows. That imbalance makes accuracy insufficient: an always-no-failure model looks accurate but detects nothing.

“I first checked the schema, missing data, duplicates, and target. A critical decision was removing the five failure-mode flags. Those are outcome labels, so using them would leak the answer. I also excluded row and product identifiers. The original CSV has some flag/target inconsistencies, which I documented without silently changing labels.

“I split the data into fitting, threshold-validation, and test partitions. Within fitting data, a Pipeline keeps imputation, encoding, and optional scaling inside cross-validation. I compared three classifiers against a dummy reference and selected by Average Precision. Then I tuned the strongest family with a small eight-configuration search.

“I chose a probability threshold by validation F1, froze the fitted model and threshold, and evaluated the test data once. The final report gives precision, recall, F1, AP, ROC-AUC, and the actual missed-failure and false-alarm counts. Permutation importance helps explain the fitted model. The limitations are important: synthetic data, no defined forecast horizon, possible dependence between observations, and no real maintenance costs or deployment validation.”

Add the current selected model and measured figures from answer 25 when practicing.

## 3. What problem does the model solve?

It assigns a failure score and binary prediction to an operating observation. It demonstrates how a maintenance-screening model could be developed and evaluated. It does not establish an advance warning such as “this machine will fail tomorrow.” That would require timestamps and a future-horizon target.

## 4. Why is this classification?

The target has two discrete values: 0 for no recorded failure and 1 for recorded failure. We are not predicting a continuous quantity such as remaining tool life.

## 5. What are the input features?

Product `Type` plus air temperature in kelvin, process temperature in kelvin, rotational speed in rpm, torque in Nm, and tool wear in minutes. There are six raw inputs and eight columns after one-hot encoding three product categories. No future or failure-mode labels are inputs.

## 6. What is the target?

The original `Machine failure` column. We keep its supplied labels, including rows that disagree with the failure flags. Changing those labels would define a different experiment and would require justification.

## 7. What is class imbalance?

One class is much less frequent than the other. Here the official CSV contains 339 failures and 9,661 non-failures. Balanced class weights make minority-class mistakes count more in training; they do not create new observations or guarantee better results. The tuning grid tests whether weighting actually helps the selected family.

## 8. Why is accuracy insufficient?

An always-no-failure rule would be correct on 96.61% of all CSV rows while having zero failure recall. The dummy baseline makes this visible. Its actual cross-validation scores appear in answer 25. Precision, recall, F1, and AP tell us about the rare class that matters.

## 9. What are precision and recall?

- **Precision = TP / (TP + FP):** how many raised failure alerts are correct?
- **Recall = TP / (TP + FN):** how many actual failures were detected?

A false positive is an unnecessary alert; a false negative is a missed failure. The consequences differ, so both metrics matter.

## 10. Why might recall matter for machine failure?

Missing a failure could mean downtime or equipment damage in a real application. However, excessive false alarms also have a cost. This project has no measured cost data, so it does not claim that maximum recall or any particular threshold is economically best.

## 11. What is F1-score?

`F1 = 2 × precision × recall / (precision + recall)`. It balances those two quantities and becomes low if either is low. It ignores true negatives and assumes a symmetric emphasis on precision and recall. We maximize validation F1 to demonstrate threshold selection, not to simulate an unknown business cost.

## 12. What is ROC-AUC?

The area under the true-positive-rate versus false-positive-rate curve across thresholds. It can be interpreted as ranking a randomly chosen positive above a randomly chosen negative, with appropriate treatment of ties. It is not the fraction of correct classifications and does not tell us the alert precision at the chosen threshold.

## 13. What is PR-AUC? What exactly did this project calculate?

A precision–recall curve shows precision against recall while varying the threshold. This project uses scikit-learn's **Average Precision**, a sum of precision values weighted by increases in recall. AP and the trapezoidal area under that curve are related but not identical. Always say AP when discussing this project's reported number. A no-skill reference is approximately the positive-class prevalence.

## 14. Why did we use stratification?

To maintain roughly the same failure proportion across partitions and CV folds. With a rare positive class, an unstratified split could produce poorly balanced evaluation samples. Stratification does not remove temporal dependence or guarantee identical operating conditions across partitions.

## 15. Why use cross-validation?

It compares candidates across five different held-out slices of fitting data rather than depending on one internal split. The final test set stays outside this process. The same CV folds also guide tuning, so the winning CV score is optimistic as a selected maximum. It is not an independent final evaluation, and fold SD is not a confidence interval.

## 16. Why use Pipeline and ColumnTransformer?

Pipeline ties preprocessing to the estimator. During CV, imputation and scaling are learned only from each training fold. ColumnTransformer applies numeric and categorical processing to the correct columns. The saved pipeline repeats those identical transformations for future predictions. This avoids training-serving differences and preprocessing leakage.

## 17. What is data leakage?

Information unavailable at the intended prediction point influences model building or evaluation. Examples include using outcome flags as features, fitting a scaler on the entire dataset before splitting, or choosing hyperparameters after looking at test scores. This project protects against all three.

## 18. Which columns could create leakage and why?

`TWF`, `HDF`, `PWF`, `OSF`, and `RNF` record failure outcomes. They are excluded even though their OR disagrees with the target on 27 rows. Imperfect outcome information is still leakage. `Machine failure` itself is never an input. `UDI` and `Product ID` are excluded as identifiers and possible shortcuts; the useful product-quality category is retained as `Type`.

## 19. Why Logistic Regression?

It is a familiar probabilistic linear classifier and a sensible interpretable reference. Numeric standardization helps its regularized optimization. Its decision boundary is linear in the transformed inputs, so it may miss nonlinear interactions among temperature, speed, torque, and wear. This is a reason to compare alternatives, not assume one wins.

## 20. Why Random Forest?

It averages many randomized decision trees and can model nonlinear relationships and interactions without scaling numeric inputs. It provides a different inductive bias from logistic regression. It is not automatically immune to overfitting or class imbalance. The random seed and tree settings are recorded.

## 21. Why HistGradientBoostingClassifier?

It sequentially builds trees that improve a loss function using binned numeric features. It is a practical scikit-learn option for tabular nonlinear patterns and complements bagged random forests. Dense one-hot encoding is inexpensive for only three product categories. We disable early stopping to make the fixed iteration budget explicit and comparable across fits.

## 22. How was the final model selected?

First, choose the non-dummy family with the highest mean fitting-data CV AP. Second, tune only that family with eight configurations, again by CV AP. Third, fit its best pipeline on the fitting partition. Fourth, select the decision threshold using the reserved validation partition. Finally, freeze both and evaluate the test partition. No test score changes those decisions. The delivered artifact is that same fitted pipeline; validation rows are not added in a later refit.

## 23. What hyperparameters were tuned?

The exact selected-family grid and best parameters are in answer 25 and `reports/results.json`.

| Family, if selected | Search parameters |
| --- | --- |
| Logistic Regression | `C` in 0.1, 1, 10, 100; class weights None or balanced |
| Random Forest | Maximum depth None or 12; minimum leaf size 1 or 3; class weights None or balanced |
| HistGradientBoosting | Maximum leaf nodes 15 or 31; L2 regularization 0 or 1; class weights None or balanced |

Each grid contains eight configurations. Only the winning family is searched. Histogram boosting keeps 160 iterations and learning rate 0.1; random forest keeps 200 trees. The grid includes the original comparison configuration.

## 24. What do the confusion-matrix values mean?

Rows are actual labels, columns are predicted labels, in order `[0, 1]`:

| | Predicted no failure | Predicted failure |
| --- | --- | --- |
| Actual no failure | True negative (TN) | False positive (FP) |
| Actual failure | False negative (FN) | True positive (TP) |

Answer 25 reports the actual matrix. Explain the false negatives and false positives in plain language, rather than mentioning accuracy alone.

## 25. What were the actual final results?

The following section is generated from the measured run. It is the same source of truth used by the README.

<!-- RUN_RESULTS_START -->

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

<!-- RUN_RESULTS_END -->

## 26. Which features were most important?

Use the actual leading inputs in answer 25 and the full table in `reports/feature_importance.csv`. Importance is the loss of validation AP when a feature is randomly shuffled. The ten-repeat standard deviation describes shuffling variability. A low value may mean a feature is less useful to this fitted model, or that correlated features supply similar information.

## 27. Does feature importance prove causality?

No. It describes a model's reliance on inputs under a perturbation experiment. It does not establish that changing a sensor value would prevent a failure. Shuffling correlated variables can also create unrealistic combinations and complicate interpretation.

## 28. What are the limitations?

Synthetic data; same-observation labels rather than a proven future horizon; limited failure counts; original flag/target inconsistencies; dependence in generated sensor sequences; one random split; reused CV for model selection; no true machine groups, maintenance cost model, calibration study, or production validation. The model returns an uncalibrated score, not a guaranteed real-world failure probability.

## 29. What would you improve next?

Define a useful forecast horizon and obtain real timestamped, machine-identified data. Use chronological/group splits, confirm label definitions, measure operating costs, and evaluate calibration. For benchmark model comparisons, repeated or nested CV can better assess close candidates. Do not add complex libraries just to lengthen a resume.

## 30. How would this differ in a real factory?

You would need to know when every feature becomes available, avoid using post-failure measurements, link readings to machines and maintenance actions, define the warning horizon, and separate future machines/time periods for evaluation. You would also need data-quality monitoring, handling of drift and new operating regimes, human maintenance decisions, and evidence that alerts are useful. None of that has been claimed as implemented here.

## Practical walkthrough before an interview

1. Run data verification, training, tests, and one prediction locally.
2. Explain why `prepare_xy()` selects only six named columns.
3. Find where `GridSearchCV` gets only fitting data.
4. Explain why the threshold plot uses validation labels and why the final model is not refit afterward.
5. Read the final confusion matrix out loud and relate it to precision and recall.
6. Change a separate input JSON file and explain the output score versus the chosen class.
7. Describe one limitation you would address with additional data rather than additional algorithms.
