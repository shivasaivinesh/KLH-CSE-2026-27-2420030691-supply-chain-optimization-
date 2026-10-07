"""
Module M6 - Explainable AI (SHAP).

Produces the interpretability artefacts promised in the Review-I plan:

  * global feature attribution for the demand-forecasting model
    (SHAP beeswarm + importance bar + dependence plots)
  * global and *local* attribution for the delay-risk classifier
    (who / what drove this particular order being flagged high-risk)
  * a what-if sensitivity table that lets a planner see how the forecast reacts
    to a promotion flag, a holiday flag or a demand shock

The forecasting explainer is fitted with the identical LightGBM configuration
and seed used by the production engine (LightGBM is deterministic for a fixed
configuration and data), so the attributions describe the deployed model.

Usage:
    python src/explain.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from features import FEATURE_COLS  # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})
FIG, MET, TAB = config.FIGURES_DIR, config.METRICS_DIR, config.TABLES_DIR
TEST_START = pd.Timestamp("2017-07-01")
N_EXPLAIN = 3000


def shap_forecaster() -> dict:
    import lightgbm as lgb
    import shap

    print("\n[SHAP-1] demand forecasting model")
    ft = pd.read_csv(config.PROCESSED_DIR / "store_demand_features.csv.gz")
    ft["date"] = pd.to_datetime(ft["date"])
    tr = ft[ft["date"] < TEST_START].dropna(subset=FEATURE_COLS)
    mdl = lgb.LGBMRegressor(
        n_estimators=900, learning_rate=0.05, num_leaves=63, min_child_samples=20,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        n_jobs=config.N_JOBS, random_state=config.RANDOM_STATE, verbose=-1)
    mdl.fit(tr[FEATURE_COLS].astype(float), tr["sales"].astype(float))
    print(f"    fitted interpretation model on {len(tr):,} rows")

    rng = np.random.default_rng(config.RANDOM_STATE)
    sample = tr.iloc[rng.choice(len(tr), size=min(N_EXPLAIN, len(tr)), replace=False)]
    Xs = sample[FEATURE_COLS].astype(float)

    expl = shap.TreeExplainer(mdl)
    sv = expl.shap_values(Xs, check_additivity=False)
    np.save(config.PROCESSED_DIR / "shap_forecast_values.npy", sv.astype(np.float32))
    Xs.iloc[:500].to_csv(config.PROCESSED_DIR / "shap_forecast_sample.csv", index=False)

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)
    mean_abs.round(4).to_csv(TAB / "t19_shap_forecast_global_importance.csv")

    fig = plt.figure(figsize=(8, 6.4))
    shap.summary_plot(sv, Xs, max_display=18, show=False, plot_size=None)
    plt.title("SHAP - demand forecasting model (global attribution)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "f23_shap_forecast_beeswarm.png")
    plt.close("all")
    print("    figure -> f23_shap_forecast_beeswarm.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    mean_abs.head(18).sort_values().plot(kind="barh", ax=ax, color="#2874a6")
    ax.set_xlabel("mean |SHAP value| (units/day)")
    ax.set_title("Forecast drivers - mean absolute SHAP importance")
    fig.savefig(FIG / "f24_shap_forecast_importance.png")
    plt.close(fig)
    print("    figure -> f24_shap_forecast_importance.png")

    # dependence plots for the three operational levers planners care about
    top = [f for f in ["lag_1", "roll_mean_7", "is_promo", "dow", "month", "cv_28"]
           if f in mean_abs.head(20).index]
    fig, axes = plt.subplots(1, len(top), figsize=(3.1 * len(top), 3.1))
    if len(top) == 1:
        axes = [axes]
    for ax, f in zip(axes, top):
        j = FEATURE_COLS.index(f)
        ax.scatter(Xs[f], sv[:, j], s=3, alpha=.28, color="#1a5276")
        ax.axhline(0, color="grey", lw=.8)
        ax.set_xlabel(f); ax.set_ylabel("SHAP value")
        ax.set_title(f"dependence: {f}", fontsize=9)
    fig.suptitle("SHAP dependence - how each lever moves the daily demand forecast", y=1.04)
    fig.savefig(FIG / "f25_shap_dependence.png")
    plt.close(fig)
    print("    figure -> f25_shap_dependence.png")

    # ---------------- what-if simulator ----------------
    base_rows = Xs.iloc[:300].copy()
    scen = {}
    scen["baseline"] = base_rows.copy()
    promo = base_rows.copy(); promo["is_promo"] = 1; scen["promotion ON"] = promo
    hol = base_rows.copy(); hol["is_holiday"] = 1; scen["holiday ON"] = hol
    shock = base_rows.copy()
    for c in ["lag_1", "lag_7", "roll_mean_7", "roll_mean_14", "roll_mean_28"]:
        shock[c] = shock[c] * 1.25
    scen["demand shock +25%"] = shock
    calm = base_rows.copy()
    for c in ["lag_1", "lag_7", "roll_mean_7", "roll_mean_14", "roll_mean_28"]:
        calm[c] = calm[c] * 0.80
    scen["demand drop -20%"] = calm

    whatif = []
    base_pred = mdl.predict(scen["baseline"].astype(float))
    for name, frame in scen.items():
        p = mdl.predict(frame.astype(float))
        whatif.append({
            "scenario": name,
            "mean_forecast_units": round(float(p.mean()), 3),
            "mean_delta_units": round(float((p - base_pred).mean()), 3),
            "mean_delta_pct": round(float((p - base_pred).mean() / max(base_pred.mean(), 1e-9) * 100), 3),
            "pct_skus_increase": round(float((p > base_pred).mean() * 100), 1),
        })
    wi = pd.DataFrame(whatif)
    wi.to_csv(TAB / "t20_whatif_scenarios.csv", index=False)
    print("\n    --- what-if simulator ---")
    print(wi.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    bars = ax.barh(wi["scenario"], wi["mean_delta_pct"],
                   color=["#7f8c8d", "#e67e22", "#8e44ad", "#c0392b", "#1e8449"])
    ax.axvline(0, color="black", lw=.8)
    for b, v in zip(bars, wi["mean_delta_pct"]):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:+.1f}%", va="center", fontsize=8)
    ax.set_xlabel("change in the daily demand forecast (%)")
    ax.set_title("What-if simulator - forecast sensitivity to operational levers")
    ax.invert_yaxis()
    fig.savefig(FIG / "f26_whatif_simulator.png")
    plt.close(fig)
    print("    figure -> f26_whatif_simulator.png")

    return {"top_features": mean_abs.head(15).round(4).to_dict(),
            "whatif": wi.to_dict("records"),
            "n_explained": int(len(Xs))}


def shap_risk() -> dict:
    import joblib
    import shap

    print("\n[SHAP-2] delay-risk classifier")
    bundle = joblib.load(config.PROCESSED_DIR / "delay_risk_model.joblib")
    mdl, feats = bundle["model"], bundle["features"]
    X = np.load(config.PROCESSED_DIR / "risk_X_test.npy")
    y = np.load(config.PROCESSED_DIR / "risk_y_test.npy")
    Xdf = pd.DataFrame(X[:4000], columns=feats)

    expl = shap.TreeExplainer(mdl)
    sv = expl.shap_values(Xdf, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
    mean_abs.round(5).to_csv(TAB / "t21_shap_risk_global_importance.csv")

    fig = plt.figure(figsize=(8, 5.6))
    shap.summary_plot(sv, Xdf, max_display=16, show=False, plot_size=None)
    plt.title("SHAP - delay-risk classifier (global attribution)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "f27_shap_risk_beeswarm.png")
    plt.close("all")
    print("    figure -> f27_shap_risk_beeswarm.png")

    # local explanation: a confidently-late order and a confidently-on-time order
    prob = mdl.predict_proba(Xdf)[:, 1] if hasattr(mdl, "predict_proba") else None
    note = ""
    if prob is not None:
        hi, lo = int(np.argmax(prob)), int(np.argmin(prob))
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        for ax, idx, tag in [(axes[0], hi, "highest-risk order"),
                             (axes[1], lo, "lowest-risk order")]:
            order = np.argsort(-np.abs(sv[idx]))[:9]
            cols = [feats[i].replace("_", " ") for i in order][::-1]
            vals = sv[idx][order][::-1]
            ax.barh(cols, vals, color=np.where(vals > 0, "#c0392b", "#1e8449"))
            ax.axvline(0, color="black", lw=.8)
            ax.set_title(f"{tag}  (predicted P(late)={prob[idx]:.3f})", fontsize=9)
            ax.set_xlabel("SHAP value (log-odds)")
            ax.tick_params(axis="y", labelsize=7)
        fig.suptitle("Local explanations - why these two orders were scored as they were", y=1.02)
        fig.savefig(FIG / "f28_shap_risk_local.png")
        plt.close(fig)
        print("    figure -> f28_shap_risk_local.png")
        note = f"local explanations for orders #{hi} (P_late={prob[hi]:.3f}) and #{lo} (P_late={prob[lo]:.3f})"

    return {"top_features": mean_abs.head(15).round(5).to_dict(), "local": note}


def run() -> dict:
    t0 = time.time()
    print("=" * 78)
    print("MODULE M6 - EXPLAINABLE AI (SHAP)")
    print("=" * 78)
    out = {"forecaster": shap_forecaster(), "risk": shap_risk(),
           "runtime_sec": round(time.time() - t0, 1)}
    Path(MET / "xai_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[M6] explainability complete in {out['runtime_sec']}s")
    return out


if __name__ == "__main__":
    run()
