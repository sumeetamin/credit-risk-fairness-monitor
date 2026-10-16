# Model card: credit-default risk baseline

## Summary

A class-weighted logistic-regression baseline estimates the probability of the positive default label. The workflow prepares the cited UCI dataset, retains a stratified random holdout, records aggregate metrics and threshold trade-offs, audits optional groups, and saves a local model bundle.

## Intended use

Portfolio demonstration, model-risk discussion, and controlled experiments on data the operator is authorized to use. Outputs are a review aid for analysis only.

## Not intended for

Credit eligibility, pricing, limit changes, collections, marketing, or any other consequential lending decision. This prototype does not establish compliance, fairness, calibrated probabilities, causal impact, or suitability for a particular institution.

## Data

The benchmark uses UCI's Default of Credit Card Clients dataset (30,000 historical customers from Taiwan), DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H), licensed CC BY 4.0. `prepare_data.py` downloads and converts the source workbook. The data and row-level outputs are ignored by Git and are not redistributed.

## Method and results

Numeric fields are median-imputed and standardized; categorical fields are mode-imputed and one-hot encoded. Logistic regression uses class weighting. ID and the optional audit group are excluded from model inputs. The reported evaluation uses a stratified random 75%/25% train/holdout split with seed 42 and a user-selected threshold of 0.50.

On the recorded run, holdout ROC-AUC was 0.715, average precision 0.499, Brier score 0.209, and balanced accuracy 0.660. For `SEX=1` and `SEX=2`, holdout TPR was 0.638 and 0.622; FPR was 0.337 and 0.291. These are descriptive results from one split and one historical dataset. Brier score alone does not establish calibration; group rates without uncertainty intervals do not establish fairness.

## Risks and limitations

- The data is historical and geographically specific; it is not representative of a current lender's applicants.
- No suitable application-time field is used for temporal validation, and the result is not an external replication.
- No confidence intervals, repeated split analysis, calibration study, subgroup uncertainty analysis, or mitigation evaluation are provided.
- The 1:5 false-positive/false-negative cost ratio used in the illustrative threshold sweep is not a measured business cost.
- `SEX` and other dataset attributes are coded categories; subgroup results need context and expert review.
- No causal claim can be made from predictive scores or this historical dataset.

## Reproduction

Run `python -m pip install -r requirements.txt`, then `python prepare_data.py` and the training command in the README. Aggregate metrics include the CSV and source-archive SHA-256 fingerprints and package versions. Do not commit private data, row-level predictions, credentials, or fitted models.

