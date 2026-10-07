# Reports

| File | Stage | Contents |
|---|---|---|
| `Review_III_Report.docx` | Review-III | Full progress report — implementation, measured results, objective-by-objective assessment, limitations and next-phase plan |
| `Review_III_Report.md` | Review-III | Same report in Markdown, for convenient diffing and in-browser viewing |

The editable Word original is kept in `docs/Team 15 Review-III.docx` and the
presentation deck in `docs/team 15 review-III ppt.pptx`, matching the naming of
the Review-I deliverables (`docs/team 15 review.docx`, `docs/team 15 ppt.pptx`).

## How these are produced

Nothing here is written by hand. `src/generate_report.py` reads the pipeline
artefacts and renders the documents, so the report can never drift out of sync
with the experiments:

```
results/metrics/forecast_metrics.json   ┐
results/metrics/risk_metrics.json       │
results/metrics/optimization_metrics.json├─→ src/generate_report.py ─→ reports/ + docs/
results/metrics/eda_summary.json        │
results/metrics/xai_metrics.json        │
results/tables/*.csv                    │
results/figures/*.png                   ┘
```

Regenerate with:

```bash
python src/run_pipeline.py        # runs the whole pipeline, then the report
python src/generate_report.py     # or just rebuild the documents from existing artefacts
```

## Review history

| Review | Deliverable | Where |
|---|---|---|
| Review-I | Literature survey (12 papers), research gaps, objectives, architecture, feasibility, plan | `docs/team 15 review.docx` |
| Review-II | *(not produced as a separate document — the plan's Phase 7 milestones were folded into the Review-III build)* | — |
| Review-III | Working implementation + measured results + dashboard | `docs/Team 15 Review-III.docx`, this folder |
