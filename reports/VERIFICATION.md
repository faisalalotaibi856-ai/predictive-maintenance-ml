# Execution and Verification Record

Verified on 2026-10-05. This record describes checks actually performed on the delivered project, not planned checks.

| Check | Observed outcome |
| --- | --- |
| Official dataset acquisition | Downloaded the UCI archive and extracted the original CSV |
| Dataset identity | SHA-256 matches the recorded original bytes |
| Schema and quality | 10,000 rows, 14 columns; missing values, duplicates, class counts, and flag/target mismatches audited |
| Leakage protection | Tests confirm changing outcome flags, target, and identifiers cannot change selected input features |
| Partition integrity | Disjoint, exhaustive, stratified fitting/validation/test partitions verified |
| Preprocessing | Fits successfully; eight finite transformed columns; held-out values cannot alter fitted imputation statistics |
| Complete experiment | `python -m src.train` executed; CV comparison, eight-configuration tuning, validation threshold selection, and final test prediction completed |
| Final result integrity | Saved test predictions reproduce all reported final metrics and confusion counts |
| Source imports | All source modules compile and import |
| Automated tests | **13 passed, 0 failed, 0 skipped** after training; also passed from a separately extracted package |
| Model persistence | Saved joblib model loaded successfully; prediction command produced a valid score/class using the stored threshold |
| Notebook | **22 code cells executed sequentially without errors** in an in-process IPython session |
| Notebook outputs | Saved tables and **9 embedded PNG figures**; nbformat schema validation passed |
| Figures | All nine PNG files decode; main evaluation, importance, threshold, and EDA plots visually inspected |
| Documentation | Result blocks agree with JSON/CSV artifacts; local links resolve |
| Paths and sensitive data | No build-machine absolute paths or private-key/API-token patterns found in distributable text files |
| Package portability | ZIP integrity checked, extracted into a different directory, data verification/tests/prediction succeeded there |
| JupyterLab module | `python -m jupyterlab --version` succeeded |

## Exact environment and test evidence

The measured run used Python 3.12.14. Numerical package versions, dataset/model/source hashes, split counts, best parameters, and final metrics are in `results.json`; `requirements-repro.txt` captures the numerical versions used. Required missing notebook/test dependencies were installed; the numerical libraries were already available.

The final test suite output in the separately extracted package was:

```text
.............                                                            [100%]
13 passed in 1.44s
```

Test duration is specific to this environment. The saved model, chosen threshold, and test score vector were not changed after observing final test performance. A presentation-only fix removed the aggregate accuracy scalar from per-class report columns; original and final source hashes plus the scope of that change are retained in `results.json`. Accuracy remains in the separate metrics table.

## Notebook execution limitation

The build environment disallows socket binding. Both ordinary and local-IPC Jupyter kernel startup were blocked before cell execution. Instead, every code cell was executed in order in a shared IPython session without using sockets, and the resulting streams, rich tables, and images were saved into the real `.ipynb` file. A final syntax/nbformat/output check also passed.

This validates the notebook's Python logic and saved outputs. It does **not** establish that a separate Jupyter server/UI was successfully launched in this restricted build environment. The standard local JupyterLab command is included in the README. Notebook metadata records the in-process execution method explicitly.

## Scope of the checks

Linux execution was verified. Windows commands are provided using standard Python/venv syntax but were not run on a Windows host here. The example prediction is an existing dataset row and is not independent performance evidence. No production service, future-warning horizon, factory deployment, or maintenance savings were validated.
