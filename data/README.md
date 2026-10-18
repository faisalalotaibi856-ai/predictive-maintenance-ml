# Dataset provenance and setup

This repository includes the unmodified official `ai4i2020.csv` from the UCI AI4I 2020 Predictive Maintenance Dataset. The approximately 510 KiB CSV is small enough to commit and can be redistributed with attribution under UCI's stated CC BY 4.0 license.

- Official dataset page: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Official download: https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip
- Dataset DOI: https://doi.org/10.24432/C5HS5C
- License: https://creativecommons.org/licenses/by/4.0/
- Attribution: *AI4I 2020 Predictive Maintenance Dataset* (2020), UCI Machine Learning Repository. Related work: Stephan Matzka, *Explainable Artificial Intelligence for Predictive Maintenance Applications* (2020), https://doi.org/10.1109/AI4I49448.2020.00023.
- Retrieved and verified: 2026-10-05.
- Modifications to distributed CSV: **none**. Analysis-derived tables and figures are separately stored in `reports/`.

## Automatic setup

From the repository root:

```bash
python -m src.data
```

An existing file is checked rather than downloaded again. If absent, the command downloads the official archive, reads only its named CSV member, verifies the checksum, and saves to `data/raw/ai4i2020.csv`. It never fabricates data or switches to a different benchmark.

## Manual setup

1. Open the official UCI dataset page linked above.
2. Select **Download**, or use the official ZIP link.
3. Extract `ai4i2020.csv`.
4. Put that file in `data/raw/ai4i2020.csv` beneath the repository root.
5. Run `python -m src.data` to verify it.

Expected SHA-256 of the original CSV bytes:

```text
dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e
```

Opening and resaving the CSV in Excel can alter its bytes. If verification fails, restore the unchanged UCI file. If UCI publishes a changed file in the future, inspect that change and record its new provenance before deliberately updating the code's checksum. Do not bypass a mismatch without understanding it.

## What this data can support

UCI describes AI4I as synthetic data reflecting industrial maintenance conditions. It is an authentic public dataset, but it is not collected live-factory data. Rows are operating observations, not guaranteed independent machines. The target records failure at an observation; no advance-warning time horizon is specified.

The observed CSV has 10,000 rows, 14 columns and 339 positive `Machine failure` labels. The five mode flags are outcomes and are never features. Their OR differs from the target on 27 rows; the original labels are preserved. See `reports/data_quality.json` for all audited counts and affected UDI values. The official page's narrative and the actual CSV do not agree on every count; this project's measurements always use the CSV.

The MIT license in the repository root applies to the project's original source code. It does not relicense the UCI dataset. Preserve this attribution and the CC BY 4.0 link when redistributing the CSV.
