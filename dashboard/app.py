"""
Module M7 - Supply Chain Predictive Analytics Dashboard (Streamlit).

Interactive decision-support front end for the whole pipeline:

    Tab 1  Executive overview      - dataset KPIs and the end-to-end pipeline
    Tab 2  Demand forecast         - SKU-level forecast vs actual, model scoreboard
    Tab 3  Inventory optimisation  - safety stock / ROP / EOQ and reorder alerts
    Tab 4  Supply-chain risk       - delay-risk model + live order risk calculator
    Tab 5  Explainability          - SHAP attributions and the what-if simulator

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "metrics"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
DEMO = ROOT / "results" / "demo"
PROC = Path(__import__("os").environ.get("SC_DATA_ROOT", ROOT / "data")) / "processed"

st.set_page_config(page_title="Supply Chain Predictive Analytics",
                   page_icon="📦", layout="wide")

CSS = """
<style>
    .block-container {padding-top: 1.4rem;}
    .kpi {background:linear-gradient(135deg,#154360,#2874a6);color:white;border-radius:10px;
          padding:.7rem .9rem;margin-bottom:.4rem;}
    .kpi h3{margin:0;font-size:1.35rem;} .kpi p{margin:0;font-size:.74rem;opacity:.85;}
    .note{background:#fef9e7;border-left:4px solid #f1c40f;padding:.55rem .8rem;
          border-radius:4px;font-size:.82rem;}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><h3>{value}</h3><p>{label}</p></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


fc = load_json(str(MET / "forecast_metrics.json"))
risk = load_json(str(MET / "risk_metrics.json"))
opt = load_json(str(MET / "optimization_metrics.json"))
eda = load_json(str(MET / "eda_summary.json"))
xai = load_json(str(MET / "xai_metrics.json"))
lstm = load_json(str(MET / "lstm_holdout.json"))

# --------------------------------------------------------------------------
st.title("📦 ML-Based Predictive Analytics for Supply Chain Optimization")
st.caption("Team 15 · KLH University, Bachupally · Review-III demonstration build · "
           "Datasets: DataCo Smart Supply Chain · Store Item Demand Forecasting · UCI Online Retail II")

tabs = st.tabs(["🏠 Executive overview", "📈 Demand forecast", "📦 Inventory optimization",
                "⚠️ Supply-chain risk", "🔍 Explainability"])

# ==========================================================================
# TAB 1 - overview
# ==========================================================================
with tabs[0]:
    c = st.columns(4)
    ov = pd.DataFrame(eda.get("overview", []))
    total_rows = int(ov["rows"].sum()) if not ov.empty else 0
    kpi(c[0], f"{total_rows:,}", "records across 3 datasets")
    kpi(c[1], f"{len(ov)}", "benchmark datasets integrated")
    match = pd.DataFrame(fc.get("matched_subset_comparison", []))
    if not match.empty:
        best = match.iloc[0]
        kpi(c[2], f"{best['MAPE']:.1f}%", f"best MAPE — {best['model']}")
        sarima = match[match["model"] == "SARIMA"]
        if not sarima.empty:
            gain = (sarima.iloc[0]["MAPE"] - best["MAPE"]) / sarima.iloc[0]["MAPE"] * 100
            kpi(c[3], f"{gain:.0f}%", "MAPE reduction vs SARIMA baseline")
        else:
            kpi(c[3], "n/a", "baseline")
    else:
        kpi(c[2], "—", "run the pipeline")
        kpi(c[3], "—", "run the pipeline")

    st.markdown("#### End-to-end pipeline")
    st.markdown(
        "```\n"
        "D1 DataCo (orders/logistics)   D2 Store-Item Demand   D3 Online Retail II\n"
        "            \\                        |                        /\n"
        "             \\________  M1 ingest / clean / validate  ______/\n"
        "                              |\n"
        "                    M2 EDA + feature engineering\n"
        "                     (lags, rolling stats, Fourier,\n"
        "                      holiday & promotion flags)\n"
        "                              |\n"
        "         M3 forecasting            M5 risk models\n"
        "   RF | XGB | LGBM | LSTM | ens.   delay classifier + lead-time reg.\n"
        "              |                          |\n"
        "              +----------> M4 inventory optimisation <---------+\n"
        "                safety stock / ROP / EOQ / reorder alerts\n"
        "                              |\n"
        "         M6 explainability (SHAP + what-if)  →  M7 dashboard\n"
        "```")

    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown("#### Dataset overview")
        if not ov.empty:
            st.dataframe(ov, width="stretch", hide_index=True)
    with c2:
        st.markdown("#### Cross-dataset evidence")
        bullets = []
        if risk:
            cr = pd.DataFrame(risk.get("classifiers", []))
            if not cr.empty:
                b = cr.sort_values("f1", ascending=False).iloc[0]
                bullets.append(f"Delay-risk classifier **{b['model']}**: F1 **{b['f1']:.3f}**, "
                               f"ROC-AUC **{b['roc_auc']:.3f}**")
            bullets.append(f"Lead-time σ measured on DataCo: "
                           f"**{risk.get('sigma_lead_time_days', float('nan')):.2f} days** "
                           f"→ feeds the safety-stock formula")
        if opt:
            i = opt.get("improvement_pct", {})
            bullets.append(f"ML-driven replenishment: holding cost "
                           f"**{i.get('holding_cost_reduction', 0):+.1f}%**, stock-outs "
                           f"**{i.get('stockout_units_reduction', 0):+.1f}%**, fill rate "
                           f"**{opt['policy_comparison']['proposed_ml']['fill_rate_pct']:.2f}%**")
        if fc:
            abl = fc.get("ablation", {})
            if abl.get("lag_feature_lift_pct") is not None:
                bullets.append(f"Lag/rolling features lift XGBoost by "
                               f"**{abl['lag_feature_lift_pct']:.1f}%** MAPE over calendar-only")
        if lstm:
            bullets.append(f"LSTM deep model: one-step hold-out MAPE **{lstm['one_step_MAPE']:.2f}%** "
                           f"({lstm['params']:,} parameters)")
        for b in bullets:
            st.markdown(f"- {b}")
        if not bullets:
            st.info("Run `python src/run_pipeline.py` to populate this dashboard.")
    st.markdown('<div class="note">All figures on this page are produced by the reproducible '
                'pipeline in <code>src/</code> from the three public benchmark datasets — '
                'no simulated or hand-entered numbers.</div>', unsafe_allow_html=True)


# ==========================================================================
# TAB 2 - demand forecast
# ==========================================================================
with tabs[1]:
    demo_path = DEMO / "forecast_demo.csv"
    if demo_path.exists():
        d = load_csv(str(demo_path))
        d["date"] = pd.to_datetime(d["date"])
        left, right = st.columns([1, 3])
        with left:
            sku = st.selectbox("SKU (store-item)", sorted(d["sku"].unique()))
            horizon = st.select_slider("Forecast horizon (days)", options=[7, 30, 90, 184], value=90)
            models = [m for m in ["Ensemble", "LightGBM", "XGBoost", "LSTM", "RandomForest",
                                  "ColdStart", "SeasonalNaive", "SARIMA"] if m in d.columns]
            chosen = st.multiselect("Models to display", models, default=[m for m in models if m in
                                    ("Ensemble", "LightGBM", "LSTM")][:3])
        sub = d[d["sku"] == sku].sort_values("date").head(horizon)
        with right:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub["date"], y=sub["sales"], name="actual",
                                     line=dict(color="black", width=2.4)))
            palette = px.colors.qualitative.D3
            for i, m in enumerate(chosen):
                fig.add_trace(go.Scatter(x=sub["date"], y=sub[m], name=m,
                                         line=dict(width=1.6, color=palette[i % len(palette)])))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                              title=f"{sku} — {horizon}-day ahead forecast vs actual",
                              legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fig, width="stretch")

        err = sub.dropna(subset=chosen)
        if not err.empty and chosen:
            rows = []
            for m in chosen:
                e = err["sales"] - err[m]
                mask = err["sales"] > 1e-6
                rows.append({"model": m, "MAE": round(e.abs().mean(), 3),
                             "RMSE": round(float(np.sqrt((e ** 2).mean())), 3),
                             "MAPE %": round(float((e[mask].abs() / err["sales"][mask]).mean() * 100), 2),
                             "bias": round(float(e.mean()), 3)})
            st.markdown("##### Scoreboard for this SKU / horizon")
            st.dataframe(pd.DataFrame(rows).sort_values("MAPE %"), width="stretch",
                         hide_index=True)
    else:
        st.info("Prediction file not found — run `python src/models_forecast.py`.")

    st.markdown("#### Model comparison across all 500 series")
    m1, m2 = st.columns(2)
    with m1:
        ht = load_csv(str(TAB / "t12_mape_by_horizon.csv"))
        if not ht.empty:
            piv = ht.pivot(index="model", columns="horizon", values="MAPE")
            st.dataframe(piv.round(2).style.background_gradient(cmap="RdYlGn_r", axis=None),
                         width="stretch")
    with m2:
        ms = load_csv(str(TAB / "t11_model_comparison_matched_subset.csv"))
        if not ms.empty:
            st.markdown("*Matched subset — identical SKUs and dates for every model*")
            st.dataframe(ms[["model", "MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "R2", "bias"]]
                         .style.background_gradient(cmap="RdYlGn_r", subset=["MAPE"]),
                         width="stretch", hide_index=True)
    if fc.get("improvements"):
        st.markdown("##### Measured improvement")
        st.json(fc["improvements"])
    for f in ["f14_model_comparison_mape", "f15_forecast_vs_actual", "f16_residual_diagnostics"]:
        p = FIG / f"{f}.png"
        if p.exists():
            st.image(str(p), caption=f.replace("_", " "))


# ==========================================================================
# TAB 3 - inventory optimisation
# ==========================================================================
with tabs[2]:
    if opt:
        pc = pd.DataFrame(opt["policy_comparison"]).T
        c = st.columns(4)
        imp = opt.get("improvement_pct", {})
        kpi(c[0], f"{imp.get('holding_cost_reduction', 0):+.1f}%", "holding-cost reduction")
        kpi(c[1], f"{imp.get('stockout_units_reduction', 0):+.1f}%", "stock-out reduction")
        kpi(c[2], f"{opt['policy_comparison']['proposed_ml']['fill_rate_pct']:.2f}%", "fill rate (service level)")
        kpi(c[3], f"{imp.get('total_cost_reduction', 0):+.1f}%", "total cost reduction")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Policy comparison (184-day simulation)")
            st.dataframe(pc, width="stretch")
            with st.expander("Assumptions"):
                st.json(opt.get("assumptions", {}))
        with c2:
            recs = load_csv(str(TAB / "t18_reorder_recommendations.csv"))
            if not recs.empty:
                st.markdown("#### Reorder recommendations")
                f1, f2 = st.columns(2)
                cls = f1.multiselect("ABC class", sorted(recs["abc_class"].unique()),
                                     default=sorted(recs["abc_class"].unique()))
                act = f2.multiselect("Action", sorted(recs["recommended_action"].unique()),
                                     default=sorted(recs["recommended_action"].unique()))
                sub = recs[recs["abc_class"].isin(cls) & recs["recommended_action"].isin(act)]
                st.dataframe(sub[["sku", "abc_class", "avg_daily_demand", "safety_stock",
                                  "reorder_point", "eoq_order_qty", "days_of_cover",
                                  "recommended_action"]].head(300),
                             width="stretch", hide_index=True, height=300)
                st.caption(f"{len(sub)} of {len(recs)} SKUs shown")
        for f in ["f20_inventory_policies", "f21_policy_cost_impact", "f22_abc_benefit"]:
            p = FIG / f"{f}.png"
            if p.exists():
                st.image(str(p))
    else:
        st.info("Run `python src/optimization.py` to populate this tab.")


# ==========================================================================
# TAB 4 - risk
# ==========================================================================
with tabs[3]:
    if risk:
        cr = pd.DataFrame(risk.get("classifiers", []))
        c = st.columns(4)
        if not cr.empty:
            b = cr.sort_values("f1", ascending=False).iloc[0]
            kpi(c[0], f"{b['f1']:.3f}", f"best F1 — {b['model']}")
            kpi(c[1], f"{b['roc_auc']:.3f}", "ROC-AUC")
            kpi(c[2], f"{b['recall']:.3f}", "recall (late orders caught)")
            kpi(c[3], f"{risk['operational_threshold']['value']:.2f}",
                "cost-optimal decision threshold")
        st.markdown("#### Classifier comparison (temporal hold-out)")
        st.dataframe(cr, width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Lead-time prediction")
            st.dataframe(pd.DataFrame(risk.get("lead_time_regression", [])),
                         width="stretch", hide_index=True)
            st.caption(f"σ(lead time) = {risk.get('sigma_lead_time_days', float('nan')):.2f} days "
                       f"→ consumed by the safety-stock formula")
        with c2:
            st.markdown("#### Live order risk calculator")
            bundle_path = next((p for p in (DEMO / "delay_risk_model.joblib",
                                      PROC / "delay_risk_model.joblib") if p.exists()), PROC / "delay_risk_model.joblib")
            if bundle_path.exists():
                import joblib
                bundle = joblib.load(bundle_path)
                feats = bundle["features"]
                st.caption("Move the levers that are plannable **at order time** and watch the "
                           "predicted probability of a late delivery.")
                cc = st.columns(3)
                sched = cc[0].slider("scheduled shipping days", 0, 4, 4)
                qty = cc[1].slider("order quantity", 1, 10, 2)
                price = cc[2].slider("product price", 10, 2000, 250)
                cc2 = st.columns(3)
                mode = cc2[0].selectbox("shipping mode", ["Standard Class", "Second Class",
                                                          "First Class", "Same Day"])
                price_ratio = cc2[1].slider("discount rate", 0.0, 0.3, 0.05)
                hour = cc2[2].slider("order hour", 0, 23, 12)
                row = dict.fromkeys(feats, 0.0)
                row["Days for shipment (scheduled)"] = sched
                row["Order Item Quantity"] = qty
                row["Product Price"] = price
                row["Order Item Discount Rate"] = price_ratio
                row["order_hour"] = hour
                for f in feats:
                    if f == f"Shipping Mode_{mode}":
                        row[f] = 1.0
                X = pd.DataFrame([row])[feats].astype(float)
                p_late = float(bundle["model"].predict_proba(X)[:, 1][0])
                thr = bundle["threshold"]
                st.metric("Predicted probability of a late delivery", f"{p_late * 100:.1f}%")
                if p_late >= thr:
                    st.error(f"⚠️ Above the action threshold ({thr:.2f}) — consider expediting, "
                             f"an alternative supplier or a buffer on the promised date.")
                else:
                    st.success(f"✅ Below the action threshold ({thr:.2f}) — standard routing.")
            else:
                st.info("Risk model artefact not found — run `python src/models_risk.py`.")
        for f in ["f17_delay_risk_models", "f18_risk_roc_pr_curves",
                  "f19_delay_risk_feature_importance"]:
            p = FIG / f"{f}.png"
            if p.exists():
                st.image(str(p))
    else:
        st.info("Run `python src/models_risk.py` to populate this tab.")


# ==========================================================================
# TAB 5 - explainability
# ==========================================================================
with tabs[4]:
    if xai:
        st.markdown("#### What-if simulator — how the forecast reacts to planner levers")
        wi = pd.DataFrame(xai.get("forecaster", {}).get("whatif", []))
        if not wi.empty:
            fig = go.Figure(go.Bar(x=wi["mean_delta_pct"], y=wi["scenario"], orientation="h",
                                   marker_color=["#7f8c8d", "#e67e22", "#8e44ad",
                                                 "#c0392b", "#1e8449"][:len(wi)]))
            fig.update_layout(height=280, xaxis_title="change in daily demand forecast (%)",
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(wi, width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Forecast model — global attribution")
            imp = pd.DataFrame(
                [{"feature": k, "mean_abs_SHAP": v}
                 for k, v in xai["forecaster"]["top_features"].items()])
            fig = px.bar(imp.sort_values("mean_abs_SHAP"), x="mean_abs_SHAP", y="feature",
                         orientation="h", color_discrete_sequence=["#2874a6"])
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("#### Delay-risk model — global attribution")
            imp2 = pd.DataFrame(
                [{"feature": k, "mean_abs_SHAP": v}
                 for k, v in xai["risk"]["top_features"].items()])
            fig = px.bar(imp2.sort_values("mean_abs_SHAP"), x="mean_abs_SHAP", y="feature",
                         orientation="h", color_discrete_sequence=["#c0392b"])
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        for f in ["f23_shap_forecast_beeswarm", "f25_shap_dependence",
                  "f27_shap_risk_beeswarm", "f28_shap_risk_local"]:
            p = FIG / f"{f}.png"
            if p.exists():
                st.image(str(p))
    else:
        st.info("Run `python src/explain.py` to populate this tab.")
        st.markdown("SHAP attributions and the what-if simulator appear here once generated.")

st.divider()
st.caption("Review-III build · pipeline entry point: `python src/run_pipeline.py` · "
           "all metrics regenerate deterministically from the raw public datasets.")
