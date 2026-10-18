# GitHub Upload and Suggested Commit Plan

This repository now contains the actual publication commits. The instructions below were included with the source package and remain as a reference for future uploads. Do not invent backdated work or claim the sequence below already occurred. A single honest initial commit is suitable for the complete starting project; use subsequent commits for changes you actually make and understand.

## Review before uploading

1. Read the README and run the commands locally.
2. Review the notebook and interview guide, and personalize the descriptions only to match your actual understanding and contribution.
3. Keep `data/README.md` and its attribution with the included official CSV.
4. Keep code, notebook, figures, and measured reports. The approximately 510 KiB raw CSV is intentionally committed.
5. Do not upload `.venv`, caches, or `models/failure_model.joblib`. The ZIP includes the recorded model for convenience, but Git ignores it because it is reproducible and library-version dependent.

## First upload with Git

Create an empty repository named `predictive-maintenance-ml` on your GitHub account. Avoid generating a separate remote README/license because this package already has them. Then, from this project's root:

```bash
git init
git add .
git status --short
git commit -m "Add reproducible predictive maintenance portfolio project"
git branch -M main
```

Review the staged file list before committing. Set the remote to the actual HTTPS or SSH repository URL GitHub gives you:

```bash
git remote add origin YOUR_ACTUAL_REPOSITORY_URL
git push -u origin main
```

`YOUR_ACTUAL_REPOSITORY_URL` is intentionally a user-specific value; no account name or remote URL has been invented. If Git asks for your identity, configure your own name and email. Authentication is handled through your usual GitHub method; the project needs no API keys.

If you upload through the GitHub website, manually omit virtual environments, cache directories, and the trained model. A browser upload does not apply `.gitignore` rules for you. Upload the project's contents, not the ZIP as the repository's only file. Check that the README images render and the notebook opens afterward.

## Suggested logical sequence for actual incremental work

If you work through the implementation incrementally, these are reasonable commit boundaries. This is a recommendation, not a fabricated history.

| Step | Suggested commit message | Contents to review |
| --- | --- | --- |
| 1 | Initialize project structure and source license | Dependencies, license, directories, ignore rules |
| 2 | Add official data loading and schema validation | Downloader, checksum, attribution, feature allowlist |
| 3 | Add training-only exploratory analysis | Quality checks, plots, descriptive tables |
| 4 | Build fold-local preprocessing pipelines | Imputation, encoding, optional scaling |
| 5 | Compare baseline and three classifiers | Dummy, logistic regression, random forest, boosting |
| 6 | Add imbalance-aware evaluation | AP, precision/recall/F1, ROC, confusion matrix |
| 7 | Tune selected model and validation threshold | Eight-configuration search and validation-only threshold grid |
| 8 | Add saved-model inference | Validated input JSON and stored threshold |
| 9 | Add pipeline and artifact-integrity tests | Leakage, splits, transformations, prediction, report consistency |
| 10 | Finalize executed notebook and documentation | Real reports, limitations, interview guide |

After future code changes, rerun the relevant checks and refresh artifacts. Do not keep optimizing against the already exposed test set while describing it as untouched.
