# Credit Risk Fairness Monitor

A reproducible credit-default baseline with a public-data preparation command, a stratified holdout, probability and threshold reports, subgroup error auditing, and a saved scoring bundle.

> **Research and portfolio prototype only.** It is not a lending decision system and does not establish legal compliance, fairness, or suitability for any lender.

## Real-data benchmark

The pipeline was run on all 30,000 records of UCI's [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients) dataset. It used a stratified random 75%/25% split (seed 42), excluded the ID and `SEX` from model inputs, and audited `SEX` on the holdout. On this single holdout, the class-weighted logistic-regression baseline achieved:

| Measure | Holdout result |
|---|---:|
| ROC-AUC | 0.715 |
| Average precision | 0.499 |
| Brier score | 0.209 |
| Balanced accuracy at 0.50 | 0.660 |
| Confusion counts (TN / FP / FN / TP) | 4,036 / 1,805 / 616 / 1,043 |

At the 0.50 threshold, the `SEX=1` and `SEX=2` groups had TPRs of 0.638 and 0.622, and FPRs of 0.337 and 0.291. These are descriptive differences from one random split without uncertainty intervals or adjustment; they are not proof of fairness or unfairness. The data's historical Taiwan credit-card population and its coded attributes have important limitations.

The source archive SHA-256 for this run is `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`. The metrics also record the prepared CSV fingerprint and software versions. No raw dataset, row-level predictions, or fitted model are published.

Committed, aggregate-only artifacts: [`results/benchmark.json`](results/benchmark.json) and [`results/threshold_tradeoffs.csv`](results/threshold_tradeoffs.csv). They contain no customer-level rows.

## Reproduce

Requires Python 3.10 or newer. Install dependencies, download the cited UCI workbook, convert its second header row to a local CSV, then run the model:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py
python main.py --input data/credit.csv --target "default payment next month" --positive-label 1 --group SEX --out outputs
```

To score a new CSV, provide the same feature columns used at training time:

```bash
python predict.py --model outputs/model.joblib --input data/new_applications.csv --out outputs/scored.csv
```

`prepare_data.py` downloads from UCI, reads the workbook's actual column-name row, and writes the CSV under the ignored `data/` directory. The runner writes aggregate metrics, a threshold-cost table, holdout predictions, and a serialized model under the ignored `outputs/` directory. Choose and disclose false-positive/false-negative costs when interpreting the threshold sweep; the defaults (1 and 5) are illustrative.

## Data and attribution

Yeh, I. (2009). [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients). UCI Machine Learning Repository. DOI: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H). Dataset license: CC BY 4.0. The data is not redistributed here.

## Method and limits

- Numeric fields are median-imputed and standardized; categorical fields are mode-imputed and one-hot encoded. The model is class-weighted logistic regression.
- `ID` is excluded from model inputs. The selected group field is also excluded from training and used only for descriptive holdout auditing.
- The holdout is stratified and random because the source data does not provide a suitable application-time field for a temporal split. Results do not measure performance on later customers or another population.
- The reported Brier score does not establish calibration. No calibration step, confidence intervals, repeated-split estimate, or external validation is included.
- Threshold costs are user-supplied assumptions, not measured lender costs. The threshold sweep is not a lending policy.
- Group error rates are descriptive. They do not establish fairness, protected-class compliance, causality, or acceptable treatment.
- Predictions require human review and should not be used for eligibility, pricing, credit limits, collections, or other consequential decisions.

See [MODEL_CARD.md](MODEL_CARD.md) for intended-use and risk details.

