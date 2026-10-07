# Demo artefact bundle

These files are small, committed derivatives of the full pipeline output, so the
dashboard and the report can be opened without downloading the ~210 MB of raw data.

| File | Produced by | Purpose |
|---|---|---|
| `forecast_demo.csv` | `src/models_forecast.py` then `src/make_demo_bundle.py` | 11,040 rows / 60 SKUs of held-out predictions plus the actual demand |
| `delay_risk_model.joblib` | `src/models_risk.py` | trained delay-risk classifier used by the live risk calculator |
| `demo_prediction_sample.csv` | `src/data_ingest.py` | 250-row schema sample of the cleaned DataCo table |

Regenerate everything with `python src/run_pipeline.py`.
