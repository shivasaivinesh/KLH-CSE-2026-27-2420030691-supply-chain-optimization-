# Data directory

The three benchmark datasets are **not** versioned in this repository — together
they are ~210 MB, and the project guidelines ask for large data to be kept out
of Git. Everything about them is reproducible:

```bash
python src/fetch_data.py     # downloads all three into data/raw/
python src/data_ingest.py    # writes the cleaned tables into data/processed/
```

Set `SC_DATA_ROOT` to relocate both folders (default: `./data`).

## Dataset 1 — DataCo Smart Supply Chain for Big Data Analysis (D1)

| | |
|---|---|
| Source | Kaggle — `shashwatwork/dataco-smart-supply-chain-for-big-data-analysis` |
| Raw size | 180,519 orders × 53 columns (~95 MB CSV) |
| Clean size | 180,519 orders × 49 columns |
| Contents | order lines, shipping mode, market/region, product category, order quantity, discount, price, profit, scheduled vs realised shipping days, late-delivery flag |
| Role in the project | supplier / logistics view: delay-risk classification, lead-time regression, and the **measured lead-time variability σ_L** that the safety-stock formula consumes |
| Notes | personally identifiable columns (customer name, e-mail, password, street, zip) are dropped at ingestion |

## Dataset 2 — Store Item Demand Forecasting (D2)

| | |
|---|---|
| Source | Kaggle — `demand-forecasting-kernels-only` (Store Item Demand Forecasting Challenge) |
| Raw size | 913,000 rows (train) + 450,000 rows (test) |
| Clean size | 913,000 rows × 9 columns |
| Contents | 10 stores × 50 items = 500 daily series, 2013-01-01 → 2017-12-31, unit sales per day |
| Role in the project | the demand-forecasting workhorse: multi-horizon (7/30/90-day) forecasting, inventory simulation, feature engineering, SHAP analysis |
| Notes | Review-I described this dataset as "42k rows"; the actual benchmark is 913,000 training rows, and that is what was used |

## Dataset 3 — UCI Online Retail II (D3)

| | |
|---|---|
| Source | UCI Machine Learning Repository, dataset ID 502 |
| Raw size | 1,067,371 transactions (xlsx, ~1 M+ rows) |
| Clean size | 1,002,800 completed sales lines across 4,899 SKUs and 43 countries |
| Contents | invoice, stock code, description, quantity, invoice timestamp, price, customer ID, country |
| Role in the project | high-cardinality SKU mix: ABC/Pareto analysis, RFM customer segmentation, price and basket behaviour |
| Notes | cancellations (invoice prefix `C`), non-positive quantities/prices and non-product stock codes are removed for the demand view |

## External signals

`src/features.py` builds the external-signal block from a documented calendar:
US holidays (New Year, 4 July, Thanksgiving, Christmas), a December promotion
window proxy, weekend and month-boundary flags, and Fourier terms for the weekly
and yearly cycles. D2 does not carry price or promotion columns, so the
promotion flag is an explicit **synthetic proxy** — this is stated as a
limitation in the Review-III report rather than presented as observed data.

## Layout after a run

```
data/
├── raw/            DataCoSupplyChainDataset.csv, store_item_demand_train.csv, online_retail_II.csv
└── processed/      dataco_clean.csv
                    store_demand_clean.csv.gz
                    store_demand_features.csv.gz      <- 913,000 x 85 feature table
                    online_retail_clean.csv.gz
                    forecast_test_predictions.csv.gz  <- held-out forecasts vs actual
                    delay_risk_model.joblib           <- trained risk classifier
```
