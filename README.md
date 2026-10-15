# Credit Risk Fairness Monitor

A reproducible credit-default classifier with calibrated probabilities, threshold trade-offs, subgroup error auditing, and a saved model artifact.

> **Portfolio prototype.** Code is provided; no benchmark results are claimed until you run it on the cited data and record the output.

## Use case

A reproducible credit-default classifier with calibrated probabilities, threshold trade-offs, subgroup error auditing, and a saved model artifact. The command below reads a local data export, applies a transparent baseline, writes holdout or decision-support outputs, and saves the fitted model where applicable.

## Data and provenance

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). A CSV with predictors plus a binary outcome (for example `default.payment.next.month`). Choose a protected/monitoring column such as `SEX` or `AGE` with `--group`. Dataset files are not bundled. Respect the source terms and cite the dataset when publishing results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download/prepare the source data locally, then run:
python main.py --input data/credit.csv --target default.payment.next.month --positive-label 1 --group SEX --out outputs
```

## Outputs

The `outputs/` directory contains the model-specific results (metrics and/or ranked CSV) and a serialized estimator where applicable. Outputs are reproducible with the same input, package versions, and random seed. The repository intentionally excludes downloaded data and generated artifacts.

## Evaluation and operating notes

Risk ranking is decision support only. The benchmark is historical and does not establish fair lending compliance, causal impact, or current creditworthiness. Review threshold and subgroup metrics with domain/legal experts.

For a production review, preserve a time-based, untouched holdout; compare against a simple business baseline; inspect calibration and subgroup error; document data lineage; and monitor drift. The included baseline is not a production service.

## License

No code license file is included yet. Third-party datasets remain subject to their original terms.
