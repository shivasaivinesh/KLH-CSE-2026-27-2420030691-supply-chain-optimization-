"""
Builds the small, version-control friendly artefact bundle that the dashboard
and the report need, so they work even when the large raw/processed datasets
are not present on the machine.

Produces (all inside results/demo/):
    forecast_demo.csv        - forecast vs actual for a representative SKU subset
    delay_risk_model.joblib  - trained delay-risk classifier (few hundred KB)
    demo_prediction_sample.csv - a small head of the cleaned transaction table
    README.md                - what is in here and how it was produced

Usage:
    python src/make_demo_bundle.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DEMO = config.PROJECT_ROOT / "results" / "demo"
DEMO.mkdir(parents=True, exist_ok=True)
N_SKUS = 60


def main() -> None:
    src = config.PROCESSED_DIR / "forecast_test_predictions.csv.gz"
    if not src.exists():
        print("!! run src/models_forecast.py first")
        return
    p = pd.read_csv(src)
    p["date"] = pd.to_datetime(p["date"])

    # representative subset: spread across the demand range, plus the extremes
    vol = p.groupby("sku")["sales"].mean().sort_values()
    qs = np.linspace(0.02, 0.98, N_SKUS - 3)
    chosen = list(dict.fromkeys([vol.index[int(q * (len(vol) - 1))] for q in qs]))
    chosen = list(dict.fromkeys(chosen + list(vol.index[:1]) + list(vol.index[-2:])))
    sub = p[p["sku"].isin(chosen)].copy()
    sub.to_csv(DEMO / "forecast_demo.csv", index=False)
    print(f"    forecast_demo.csv        {len(sub):,} rows, {sub['sku'].nunique()} SKUs, "
          f"{sub['date'].nunique()} days")

    model = config.PROCESSED_DIR / "delay_risk_model.joblib"
    if model.exists():
        shutil.copy2(model, DEMO / "delay_risk_model.joblib")
        print(f"    delay_risk_model.joblib  {model.stat().st_size / 1024:.0f} KB")

    clean = config.PROCESSED_DIR / "dataco_clean.csv"
    if clean.exists():
        head = pd.read_csv(clean, nrows=250, low_memory=False)
        head.to_csv(DEMO / "demo_prediction_sample.csv", index=False)
        print(f"    demo_prediction_sample.csv {len(head)} rows (schema sample)")

    (DEMO / "README.md").write_text(
        "# Demo artefact bundle\n\n"
        "These files are small, committed derivatives of the full pipeline output, so the\n"
        "dashboard and the report can be opened without downloading the ~210 MB of raw data.\n\n"
        "| File | Produced by | Purpose |\n"
        "|---|---|---|\n"
        "| `forecast_demo.csv` | `src/models_forecast.py` then `src/make_demo_bundle.py` | "
        f"{len(sub):,} rows / {sub['sku'].nunique()} SKUs of held-out predictions plus the actual demand |\n"
        "| `delay_risk_model.joblib` | `src/models_risk.py` | trained delay-risk classifier used by the "
        "live risk calculator |\n"
        "| `demo_prediction_sample.csv` | `src/data_ingest.py` | 250-row schema sample of the cleaned "
        "DataCo table |\n\n"
        "Regenerate everything with `python src/run_pipeline.py`.\n")
    print(f"    README.md written")
    print(f"\n[demo] bundle ready in {DEMO}")


if __name__ == "__main__":
    main()
