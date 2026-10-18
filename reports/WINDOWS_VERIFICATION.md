# Windows Reproduction Verification

Verified on 2026-10-18 using the unchanged numerical source code and included UCI CSV.

## Environment

Windows; Python 3.12.14. Numerical dependencies were installed from `requirements-repro.txt`: NumPy 2.3.5, pandas 2.2.3, Matplotlib 3.10.8, scikit-learn 1.8.0, SciPy 1.17.0, joblib 1.5.3, threadpoolctl 3.6.0.

## Checks completed

- The complete source ZIP passed CRC validation; all 53 extracted files matched their archived bytes.
- A separate download of the official UCI CSV matched the included dataset checksum.
- The original test suite completed: **13 passed, 0 failed, 0 skipped** (4.56 seconds in this environment).
- The complete experiment was rerun into a separate output directory using `python -m src.train --output-dir ...`.
- The rerun reproduced the selected family, best parameters, threshold, and all reported test metrics. The generated model artifact had the same SHA-256 as the supplied model.
- All 13 generated report CSVs agreed numerically with the supplied reports; execution-time columns were excluded from the tuning-table comparison.
- Original and retrained models produced the same example inference result.
- All 22 notebook code cells completed with zero errors through nbclient using a separate ipykernel. The notebook's default mode displays saved results; full training was verified separately.
- All nine report PNGs decoded successfully and were visually inspected.
- `python -m pip check` reported no broken requirements; JupyterLab version 4.6.4 was available. Browser interaction with the JupyterLab UI was not tested.

A plotting FutureWarning was observed under scikit-learn 1.8.0; it did not affect training or the numerical results. Runtime durations are environment-specific.

## Interpretation and limits

The reproduced test results are AP 0.8430, F1 0.7626, precision 0.7465, recall 0.7794, ROC-AUC 0.9664 and accuracy 0.9835, with confusion matrix `[[1914, 18], [15, 53]]` at threshold 0.10.

An additional inspection of the fixed model found fitting F1 1.0000 versus validation F1 0.6916 and test F1 0.7626. This generalization gap merits further analysis; it does not invalidate the held-out measurement. No parameters, features, or threshold were changed during this verification.

This confirms reproduction of the declared benchmark workflow, not prospective factory performance, a future warning horizon, or independent verification of the project's authorship. The dataset is synthetic and the existing test split has already been exposed. Future modeling changes should use development-only evaluation and a new independent test when making fresh final-performance claims.

## Repository artifact policy

The generated `models/failure_model.joblib` is intentionally excluded by the existing `.gitignore`. Recreate it after cloning with `python -m src.train`. Source code, the attributed dataset, the executed explanatory notebook, example inputs, tests, dependency specifications, and measured reports are included.
