# Model card: credit-default risk baseline

## Summary

A logistic-regression baseline estimates the probability of the positive default label. A validation slice selects unweighted versus class-weighted training by Brier score; the selected model is refit on the full training partition. The workflow prepares the cited UCI dataset, retains a stratified random holdout, records aggregate metrics and threshold trade-offs, audits optional groups, and saves a local model bundle.

## Intended use

Portfolio demonstration, model-risk discussion, and controlled experiments on data the operator is authorized to use. Outputs are a review aid for analysis only.

## Not intended for

Credit eligibility, pricing, limit changes, collections, marketing, or any other consequential lending decision. This prototype does not establish compliance, fairness, calibrated probabilities, causal impact, or suitability for a particular institution.

## Data

The benchmark uses UCI's Default of Credit Card Clients dataset (30,000 historical customers from Taiwan), DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H), licensed CC BY 4.0. `prepare_data.py` downloads and converts the source workbook. The data and row-level outputs are ignored by Git and are not redistributed.

## Method and results

Numeric fields are median-imputed and standardized; categorical fields are mode-imputed and one-hot encoded. ID and the optional audit group are excluded from model inputs. The reported evaluation uses a stratified random 75%/25% train/holdout split with seed 42; 20% of the training partition is reserved for model selection. The user-selected decision threshold defaults to 0.50.

On the recorded run, validation selected unweighted logistic regression (validation Brier 0.144 vs. 0.207 for class-weighted). Holdout ROC-AUC was 0.714, average precision 0.502, Brier score 0.146, log loss 0.468, and balanced accuracy 0.607. The train-prevalence constant baseline had Brier 0.172 and log loss 0.528. For `SEX=1`, holdout TPR was 0.244 (95% Wilson interval 0.213–0.277) and FPR 0.035 (0.028–0.043); for `SEX=2`, TPR was 0.243 (0.217–0.272) and FPR 0.026 (0.021–0.032). These are descriptive results from one split and one historical dataset. The Brier comparison does not establish calibration; subgroup intervals are unadjusted and do not establish fairness.

## Risks and limitations

- The data is historical and geographically specific; it is not representative of a current lender's applicants.
- No suitable application-time field is used for temporal validation, and the result is not an external replication.
- No repeated split analysis, calibration study, multiple-comparison adjustment, or mitigation evaluation is provided. Wilson intervals describe only the reported subgroup TPR/FPR rates on this one holdout.
- The 1:5 false-positive/false-negative cost ratio used in the illustrative threshold sweep is not a measured business cost.
- `SEX` and other dataset attributes are coded categories; subgroup results need context and expert review.
- No causal claim can be made from predictive scores or this historical dataset.

## Reproduction

Run `python -m pip install -r requirements.txt`, then `python prepare_data.py` and the training command in the README. Aggregate metrics include the CSV and source-archive SHA-256 fingerprints and package versions. Do not commit private data, row-level predictions, credentials, or fitted models.

