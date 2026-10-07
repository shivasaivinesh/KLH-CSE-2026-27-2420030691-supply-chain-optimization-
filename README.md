# Machine Learning-Based Predictive Analytics for Supply Chain Optimization

**Course:** Engineering Capstone Project – 1 (`23IE4053R` / `23IE4053A`)
**Department:** Computer Science & Engineering (CSE)
**Academic Year:** 2026 – 2027 · **Batch 15 — Team 15**

## Team Members

| Name | ID Number |
|---|---|
| M. Shiva Sai Vinesh | 2420030691 |
| R. Harsha Vardhan Reddy | 2420030748 |
| D. Venkatesh | 2420030134 |
| A. Shashank Reddy | 2420030034 |

**Supervisor:** Dr. K. Swapnika, Dept. of CSE

---

## Current Phase Status

| Review | Phase | Deliverable | Status |
|---|---|---|---|
| Review-I | Phase 1 | Literature review (12 papers), research gaps, objectives, architecture, feasibility, plan | ✅ Completed — `docs/team 15 review.docx` |
| Review-III | Phases 2–7 | Working implementation, measured benchmark results, explainability, dashboard | ✅ Completed — `docs/Team 15 Review-III.docx` |

**Everything in this repository is reproducible end-to-end with one command:**

```bash
pip install -r requirements.txt
python src/fetch_data.py      # ~210 MB of public benchmark data
python src/run_pipeline.py    # ingest → EDA → features → forecast → risk → optimise → XAI → report
streamlit run dashboard/app.py
```

> The report is generated *from* the pipeline artefacts, so the documents can never
> drift out of sync with the experiments. Every number below is measured, not assumed.

---

## Abstract

Modern supply chains face volatile demand, inventory imbalance, supplier lead-time
variability and logistics disruptions. Traditional forecasting — ARIMA, exponential
smoothing — assumes linearity and stationarity and is unable to exploit promotions,
seasonality or external disruption signals. This project builds a **unified predictive
analytics framework** that carries historical and operational data all the way through
to an actionable prescription.

The system ingests three public benchmark datasets, performs automated cleaning,
exploratory analysis and feature engineering (lags, rolling statistics, Fourier
seasonality, holiday/promotion flags), and trains a model family spanning classical
statistics, gradient boosting and deep learning. Forecasts are not left as a chart on a
wall: the predicted demand distribution feeds an inventory optimisation layer that
computes dynamic safety stock, reorder points and EOQ, while a parallel classifier
predicts delivery-delay risk. Explainability is delivered through SHAP, and the whole
pipeline is presented in an interactive Streamlit dashboard.

---

## Measured Results

### Demand forecasting — 500 store–item series, 184-day recursive rollout

| Model | MAPE | MAE | R² |
|---|---|---|---|
| **Ensemble (proposed)** | **12.11 %** | **6.46** | **0.928** |
| LightGBM | 12.11 % | 6.61 | 0.925 |
| XGBoost | 13.19 % | 7.53 | 0.900 |
| Random Forest | 14.77 % | 7.13 | 0.915 |
| SARIMA (statistical baseline) | 19.19 % | 9.57 | 0.824 |
| Seasonal naive | 26.42 % | 13.30 | 0.677 |

Scored on **identical** SKUs and dates. Evaluation is a genuine recursive multi-step
rollout — each model feeds back its own predictions as the lag inputs, so no
ground-truth value from inside the test window is ever used. **36.9 % lower MAPE than
SARIMA**, against a Review-I target of ≥ 25 %.

Engineered lag/rolling features are worth a **32.8 % MAPE reduction** over
calendar-only features (ablation study). The LSTM reaches 11.75 % one-step MAPE with
17,861 parameters.

### Delay-risk and lead-time prediction (DataCo, temporal hold-out)

| Model | F1 | ROC-AUC | Recall |
|---|---|---|---|
| Random Forest | 0.694 → **0.722** at the tuned threshold | 0.769 | 0.539 → 0.671 |
| XGBoost | 0.665 | 0.764 | 0.541 |
| Logistic Regression | 0.657 | 0.743 | 0.539 |

Lead-time prediction: **MAE 0.96 days**; the measured residual variability
**σ_L = 1.27 days** is fed directly into the safety-stock formula. The Review-I target
of F1 ≥ 0.85 is **not met** and this is documented with a diagnosis in the report.

### Inventory optimisation — forecast to prescription

Three policies are compared under the **same realised demand** and the **same
lead-time random draws** (common random numbers), at a **service-matched** operating
point:

| Experiment | Question | Result |
|---|---|---|
| **E1** | Value of the forecast alone (same formula both sides) | total cost **−3.07 %**, stock-outs **−3.50 %** |
| **E2** | As-designed risk-aware policy vs traditional practice | total cost **−2.12 %**, holding cost +0.59 %, stock-outs −0.64 % |
| **E3** | **Lead-time disruption** (3× measured variability) | stock-outs **−25.0 %**, total cost **−13.5 %**, fill rate 95.14 % → 96.36 % |

**Honest negative result:** the Review-I target of a 15–20 % *holding-cost* reduction is
**not achieved**. At equal service level the forecast-driven policy's holding cost is
essentially unchanged (or slightly higher) — a better forecast gives leverage on
stock-outs and resilience, not on cycle stock. The report explains why rather than
quietly dropping the claim.

### Explainability

SHAP beeswarm/dependence plots for the forecasting model, global and **local**
attributions for the delay-risk classifier, and a what-if simulator. Notably the
what-if study shows the synthetic promotion flag moves the forecast by 0.0 % — the model
correctly ignores a signal that does not exist in the data, and this is reported as a
data limitation rather than a model failure.

---

## Repository Structure

```
KLH-CSE-2026-27-2420030691-supply-chain-optimization-/
├── README.md                       ← you are here
├── requirements.txt
├── .gitignore                      ← keeps the ~210 MB of raw data out of Git
│
├── src/                            ← the entire reproducible pipeline
│   ├── config.py                   ← central paths, seeds, cost parameters
│   ├── fetch_data.py               ← downloads the 3 public benchmark datasets
│   ├── data_ingest.py              ← M1: cleaning, validation, PII removal
│   ├── eda.py                      ← M2: 12 EDA techniques + ABC + STL + RFM
│   ├── features.py                 ← M2b: 57 engineered features
│   ├── models_forecast.py          ← M3: SARIMA, RF, XGB, LightGBM, ensemble + rollouts
│   ├── models_dl.py                ← M3b: global LSTM with SKU embeddings
│   ├── models_risk.py              ← M5: delay-risk classifier + lead-time regression
│   ├── optimization.py             ← M4: safety stock / ROP / EOQ + policy simulation
│   ├── explain.py                  ← M6: SHAP + what-if simulator
│   ├── make_demo_bundle.py         ← small committed artefacts for the dashboard
│   ├── generate_report.py          ← renders the DOCX / PPTX / Markdown deliverables
│   └── run_pipeline.py             ← one-command orchestrator
│
├── dashboard/app.py                ← M7: 5-tab Streamlit decision-support app
│
├── results/
│   ├── figures/                    ← 26 generated figures (f03 … f28)
│   ├── metrics/                    ← JSON metric files per module
│   ├── tables/                     ← 20 generated CSV tables
│   └── demo/                       ← small bundle so the dashboard runs without raw data
│
├── docs/                           ← submission documents
│   ├── team 15 review.docx         ← Review-I
│   ├── Team 15 Review-III.docx     ← Review-III report (generated)
│   ├── team 15 review-III ppt.pptx ← Review-III deck (generated)
│   ├── team 15 abstarct.docx       ← abstract submission
│   └── Team 15 Roadmap.docx        ← supervisor roadmap
│
├── reports/                        ← Review-III report copies + README
│   ├── Review_III_Report.docx
│   └── Review_III_Report.md
│
└── data/                           ← NOT versioned (see data/README.md)
    ├── raw/                        ← populated by fetch_data.py
    └── processed/                  ← intermediate cleaned / feature tables
```

---

## Dashboard

Five tabs, all driven by the pipeline artefacts:

1. **Executive overview** — integrated dataset KPIs, pipeline diagram, cross-dataset evidence
2. **Demand forecast** — SKU/horizon selector, forecast vs actual, per-SKU scoreboard
3. **Inventory optimization** — policy comparison, cost impact, filterable reorder recommendations
4. **Supply-chain risk** — classifier scoreboard, lead-time model, **live order risk calculator**
5. **Explainability** — SHAP attributions and the what-if simulator

---

## Datasets

| ID | Dataset | Size used | Role |
|---|---|---|---|
| D1 | DataCo Smart Supply Chain (Kaggle) | 180,519 orders | delay risk, lead-time σ_L |
| D2 | Store Item Demand Forecasting (Kaggle) | 913,000 daily rows, 500 series | demand forecasting, inventory simulation |
| D3 | UCI Online Retail II (UCI ML Repo #502) | 1,002,800 clean sales lines, 4,899 SKUs | ABC/Pareto, RFM, price behaviour |

Full provenance, cleaning decisions and layout: [`data/README.md`](data/README.md).

---

## Module Status

| Module | Description | Status |
|---|---|---|
| M1 | Data ingestion, validation, cleaning | ✅ 3 datasets integrated |
| M2 | EDA — 12 techniques + ABC + STL + RFM | ✅ 11 figures |
| M2b | Feature engineering | ✅ 57 features + 28-day sequence block |
| M3 | Demand forecasting (SARIMA/RF/XGB/LGBM/LSTM/ensemble) | ✅ 36.9 % better than SARIMA |
| M4 | Inventory optimisation (SS/ROP/EOQ + simulation) | ✅ with an honest negative result documented |
| M5 | Delay-risk + lead-time prediction | ✅ F1 0.722, σ_L measured |
| M6 | Explainability (SHAP + what-if) | ✅ global + local |
| M7 | Streamlit dashboard | ✅ 5 tabs, live risk calculator |
| — | TFT (Temporal Fusion Transformer) | ⏳ next phase |

---

## Known Limitations (carried openly into the report)

* **TFT not yet trained** — the LSTM covers the deep-learning branch.
* **Delay-risk F1 is 0.722, not ≥ 0.85** — the temporal split puts the most disrupted
  quarter in the test set; remedies are planned.
* **Holding-cost reduction of 15–20 % was not achieved** — the simulation shows the
  target was arithmetically optimistic for this demand regime.
* **Promotion flag is a synthetic proxy** — D2 has no promotion column, and the what-if
  study confirms it carries no signal.
* **Uniform cost parameters** across SKUs pending per-family calibration.
* **SARIMA benchmarked on 12 representative series**, with all models re-scored on that
  matched subset.

---

## Licence

Developed for academic and research purposes. All datasets are public benchmarks; no
proprietary or confidential data is stored in this repository.
