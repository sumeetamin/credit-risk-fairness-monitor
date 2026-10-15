# Model card: credit default risk baseline

## Summary
A local benchmark pipeline estimates a binary default label using scikit-learn logistic regression. It includes an untouched stratified holdout, probability and classification metrics, optional subgroup error rates, a decision threshold, and a CLI for scoring new CSV rows.

## Intended use
Portfolio demonstration, model-risk discussion, and controlled experimentation on data the operator is authorized to use. Outputs are a review aid only.

## Not intended for
Credit eligibility decisions, pricing, limit changes, collections, or any other real lending outcome. This sample does not establish legal compliance, fairness, calibration, or suitability for a specific institution.

## Data
Default of Credit Card Clients, UCI Machine Learning Repository, DOI 10.24432/C55S3H. The dataset is historical, geographically specific, and subject to the repository's CC BY 4.0 terms. The repository does not redistribute the data.

## Method
Numeric fields are median-imputed and standardized; categorical fields are mode-imputed and one-hot encoded. Logistic regression uses class weighting. A reproducible stratified holdout is reserved with seed 42. The selected positive label and threshold are written into the model bundle. If --group is supplied, held-out true-positive and false-positive rates are reported for sufficiently large groups.

## Evaluation
Run the CLI to create outputs/metrics.json, outputs/holdout_predictions.csv, and outputs/model.joblib. Record the actual run output before making any performance claim. A random holdout is not a substitute for out-of-time, external, or institution-specific validation. The threshold is a user-selected operating point, not a calibrated lending policy.

## Known limitations
- No subgroup uncertainty intervals or multiple-testing adjustment.
- No out-of-time validation or probability-calibration step.
- Missingness and target/feature definitions should be reviewed against the original dataset documentation.
- Group labels and disparities are diagnostic only and require context and expert review.
- New-data scoring rejects missing features but ignores extra columns.

## Reproduction
See the project README for environment setup and commands. Do not commit private application data, generated predictions, API credentials, or fitted models.
