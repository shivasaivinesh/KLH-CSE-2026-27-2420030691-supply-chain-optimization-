"""
End-to-end orchestrator.

Runs every stage of the project in dependency order and stops with a clear
message if a stage fails, so the whole experiment is reproducible with a single
command:

    python src/run_pipeline.py                # full run
    python src/run_pipeline.py --skip-fetch   # datasets already downloaded
    python src/run_pipeline.py --quick        # fast smoke test
    python src/run_pipeline.py --only eda     # a single stage

Stage order
-----------
fetch -> ingest -> features -> eda -> forecast -> risk -> optimization -> explain
-> report
(the dashboard is a long-running server and is launched separately with
 `streamlit run dashboard/app.py`)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGES = ["fetch", "ingest", "features", "eda", "forecast", "risk",
          "optimization", "explain", "report"]


def run_stage(name: str, quick: bool) -> None:
    print("\n" + "#" * 78)
    print(f"# STAGE: {name.upper()}")
    print("#" * 78)
    t0 = time.time()
    if name == "fetch":
        import fetch_data
        fetch_data.fetch()
    elif name == "ingest":
        import data_ingest
        data_ingest.run()
    elif name == "features":
        import features
        features.run()
    elif name == "eda":
        import eda
        eda.run()
    elif name == "forecast":
        import models_forecast
        models_forecast.run(quick=quick)
    elif name == "risk":
        import models_risk
        models_risk.run()
    elif name == "optimization":
        import optimization
        optimization.run()
    elif name == "explain":
        import explain
        explain.run()
    elif name == "report":
        import generate_report
        generate_report.main()
    else:
        raise ValueError(f"unknown stage {name}")
    print(f"[stage {name}] done in {time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the supply-chain analytics pipeline")
    ap.add_argument("--skip-fetch", action="store_true", help="assume raw data already present")
    ap.add_argument("--quick", action="store_true", help="reduced settings for a smoke test")
    ap.add_argument("--only", choices=STAGES, help="run a single stage")
    ap.add_argument("--from", dest="start_at", choices=STAGES, help="start from this stage")
    args = ap.parse_args()

    stages = [args.only] if args.only else list(STAGES)
    if not args.only and args.skip_fetch:
        stages = [s for s in stages if s != "fetch"]
    if not args.only and args.start_at:
        stages = stages[stages.index(args.start_at):]

    print("=" * 78)
    print("SUPPLY CHAIN PREDICTIVE ANALYTICS - FULL PIPELINE")
    print("stages:", " -> ".join(stages))
    print("=" * 78)
    t0 = time.time()
    for s in stages:
        run_stage(s, args.quick)
    print(f"\nPIPELINE COMPLETE in {time.time() - t0:.1f}s")
    print("Launch the dashboard with:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
