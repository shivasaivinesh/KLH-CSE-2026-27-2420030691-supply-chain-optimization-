"""
Module M5 - Supplier / Logistics Risk Prediction.

Two supervised tasks on the DataCo Smart Supply Chain dataset (D1):

  T1  Shipment delay-risk classification  (target: Late_delivery_risk)
      Logistic Regression (baseline) vs Random Forest vs XGBoost
  T2  Lead-time regression                (target: realised shipping days)
      Random Forest vs XGBoost  -> provides the lead-time variability that the
      inventory optimisation module consumes as sigma_L

Validation protocol
-------------------
Temporal split: orders are sorted by date and the most recent 20 % are held
out (never randomly shuffled) because the operational use-case is forecasting
risk for *future* orders.

Leakage control
---------------
`Days for shipping (real)`, `Delivery Status`, `shipping_delay` and
`Late_delivery_risk` itself are removed from the feature set - they are only
known *after* delivery and would otherwise leak the target.

Usage:
    python src/models_risk.py
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
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})
FIG, MET, TAB = config.FIGURES_DIR, config.METRICS_DIR, config.TABLES_DIR

TARGET = "Late_delivery_risk"
LEAKY = ["Late_delivery_risk", "Delivery Status", "Days for shipping (real)",
         "shipping_delay", "Order Status"]

CAT_FEATURES = ["Shipping Mode", "Market", "Order Region", "Category Name",
                "Department Name", "Customer Segment", "Type", "Order Country"]
NUM_FEATURES = ["Days for shipment (scheduled)", "Order Item Quantity", "Product Price",
                "Order Item Discount Rate", "Order Item Profit Ratio", "Sales",
                "order_month", "order_dow", "order_hour", "Latitude", "Longitude"]


# --------------------------------------------------------------------------
def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    keep_cat = [c for c in CAT_FEATURES if c in df.columns]
    keep_num = [c for c in NUM_FEATURES if c in df.columns]
    X = df[keep_cat + keep_num].copy()
    for c in keep_cat:
        X[c] = X[c].astype("string").fillna("UNKNOWN").astype(str)
    for c in keep_num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = pd.get_dummies(X, columns=keep_cat, drop_first=False, dtype=np.int8)
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df[TARGET].astype(int)
    return X, y, list(X.columns)


def temporal_split(dates: pd.Series, test_fraction: float):
    order = dates.sort_values().index
    cut = int(len(order) * (1 - test_fraction))
    return order[:cut], order[cut:]


# --------------------------------------------------------------------------
def run() -> dict:
    t0 = time.time()
    print("=" * 78)
    print("MODULE M5 - DELAY-RISK AND LEAD-TIME PREDICTION (DataCo)")
    print("=" * 78)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score,
                                 mean_absolute_error, mean_squared_error, r2_score)
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    df = pd.read_csv(config.PROCESSED_DIR / "dataco_clean.csv", low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"])
    print(f"    orders {len(df):,} | period {df['order_date'].min().date()} -> {df['order_date'].max().date()}")

    X, y, feature_names = build_matrix(df)
    tr_idx, te_idx = temporal_split(df["order_date"], config.RISK_TEST_FRACTION)
    X_tr, X_te = X.loc[tr_idx], X.loc[te_idx]
    y_tr, y_te = y.loc[tr_idx], y.loc[te_idx]
    print(f"    features {X.shape[1]} (after one-hot) | train {len(tr_idx):,} / test {len(te_idx):,}")
    print(f"    late-delivery base rate: train {y_tr.mean():.3f} | test {y_te.mean():.3f}")

    scaler = StandardScaler().fit(X_tr)
    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    clfs = {
        "LogisticRegression": (LogisticRegression(max_iter=1500, class_weight="balanced",
                                                  random_state=config.RANDOM_STATE),
                               "scaled"),
        "RandomForest": (RandomForestClassifier(
            n_estimators=220, max_depth=20, min_samples_leaf=4, max_features=0.4,
            class_weight="balanced_subsample", n_jobs=config.N_JOBS,
            random_state=config.RANDOM_STATE), "raw"),
        "XGBoost": (xgb.XGBClassifier(
            n_estimators=450, learning_rate=0.08, max_depth=7, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
            scale_pos_weight=pos_w * 0.5, tree_method="hist", n_jobs=config.N_JOBS,
            random_state=config.RANDOM_STATE, eval_metric="logloss", verbosity=0), "raw"),
    }

    results, fitted, probas = [], {}, {}
    for name, (mdl, kind) in clfs.items():
        t = time.time()
        Xtr = scaler.transform(X_tr) if kind == "scaled" else X_tr
        Xte = scaler.transform(X_te) if kind == "scaled" else X_te
        mdl.fit(Xtr, y_tr)
        prob = mdl.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(y_te, pred)
        results.append({
            "model": name,
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_te, prob)), 4),
            "pr_auc": round(float(average_precision_score(y_te, prob)), 4),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            "fit_sec": round(time.time() - t, 1),
        })
        fitted[name] = (mdl, kind)
        probas[name] = prob
        print(f"    {name:<19} F1={results[-1]['f1']:.3f}  ROC-AUC={results[-1]['roc_auc']:.3f}  "
              f"recall={results[-1]['recall']:.3f}  acc={results[-1]['accuracy']:.3f}")

    res_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    res_df.to_csv(TAB / "t13_delay_risk_classifier_comparison.csv", index=False)

    # ---------------- threshold tuning for the operational use-case ----------------
    best_name = res_df.iloc[0]["model"]
    bp = probas[best_name]
    thr_rows = []
    for thr in np.arange(0.20, 0.81, 0.02):
        pr = (bp >= thr).astype(int)
        thr_rows.append({"threshold": round(float(thr), 2),
                         "precision": round(float(precision_score(y_te, pr, zero_division=0)), 4),
                         "recall": round(float(recall_score(y_te, pr, zero_division=0)), 4),
                         "f1": round(float(f1_score(y_te, pr, zero_division=0)), 4)})
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(TAB / "t14_delay_risk_threshold_tuning.csv", index=False)
    best_thr = float(thr_df.loc[thr_df["f1"].idxmax(), "threshold"])
    print(f"    cost-optimal decision threshold on {best_name}: {best_thr:.2f} "
          f"(F1={thr_df['f1'].max():.3f})")

    # ---------------- T2 lead-time regression ----------------
    target2 = "Days for shipping (real)"
    y2 = pd.to_numeric(df[target2], errors="coerce")
    reg = {
        "RandomForest": RandomForestRegressor(n_estimators=160, max_depth=18,
                                              min_samples_leaf=5, max_features=0.4,
                                              n_jobs=config.N_JOBS,
                                              random_state=config.RANDOM_STATE),
        "XGBoost": xgb.XGBRegressor(n_estimators=350, learning_rate=0.08, max_depth=7,
                                    subsample=0.85, colsample_bytree=0.8, tree_method="hist",
                                    n_jobs=config.N_JOBS, random_state=config.RANDOM_STATE,
                                    verbosity=0),
    }
    reg_rows, sigma_l = [], {}
    for name, mdl in reg.items():
        mdl.fit(X_tr, y2.loc[tr_idx])
        p = mdl.predict(X_te)
        resid = y2.loc[te_idx].to_numpy(float) - p
        reg_rows.append({"model": name,
                         "MAE_days": round(float(mean_absolute_error(y2.loc[te_idx], p)), 4),
                         "RMSE_days": round(float(np.sqrt(mean_squared_error(y2.loc[te_idx], p))), 4),
                         "R2": round(float(r2_score(y2.loc[te_idx], p)), 4),
                         "sigma_lead_time_days": round(float(np.std(resid, ddof=1)), 4),
                         "bias_days": round(float(np.mean(resid)), 4)})
        sigma_l[name] = float(np.std(resid, ddof=1))
        print(f"    lead-time {name:<13} MAE={reg_rows[-1]['MAE_days']:.3f} d  "
              f"R2={reg_rows[-1]['R2']:.3f}  sigma_L={reg_rows[-1]['sigma_lead_time_days']:.3f} d")
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(TAB / "t15_leadtime_regression.csv", index=False)

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    m = res_df.set_index("model")[["precision", "recall", "f1", "roc_auc"]]
    m.plot(kind="bar", ax=axes[0], width=.8)
    axes[0].set_title("Delay-risk classifier - metric comparison")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=7, ncol=2); axes[0].tick_params(axis="x", rotation=12)

    pred_best = (bp >= best_thr).astype(int)
    cm = confusion_matrix(y_te, pred_best)
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["on time", "late"], yticklabels=["on time", "late"], ax=axes[1])
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("actual")
    axes[1].set_title(f"Confusion matrix - {best_name} @ thr={best_thr:.2f}")

    axes[2].plot(thr_df["threshold"], thr_df["precision"], label="precision", color="#2874a6")
    axes[2].plot(thr_df["threshold"], thr_df["recall"], label="recall", color="#c0392b")
    axes[2].plot(thr_df["threshold"], thr_df["f1"], label="F1", color="#27ae60", lw=2)
    axes[2].axvline(best_thr, ls=":", color="grey")
    axes[2].set_xlabel("decision threshold"); axes[2].set_title("Threshold tuning")
    axes[2].legend(fontsize=7)
    fig.savefig(FIG / "f17_delay_risk_models.png")
    plt.close(fig)
    print("    figure -> f17_delay_risk_models.png")

    # ROC + PR curves
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for name, prob in probas.items():
        fpr, tpr, _ = roc_curve(y_te, prob)
        axes[0].plot(fpr, tpr, lw=1.4,
                     label=f"{name} (AUC={roc_auc_score(y_te, prob):.3f})")
        pr, rc, _ = precision_recall_curve(y_te, prob)
        axes[1].plot(rc, pr, lw=1.4,
                     label=f"{name} (AP={average_precision_score(y_te, prob):.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    axes[0].set_xlabel("false-positive rate"); axes[0].set_ylabel("true-positive rate")
    axes[0].set_title("ROC curves - delay risk"); axes[0].legend(fontsize=7)
    axes[1].axhline(y_te.mean(), ls="--", color="grey", lw=1, label="base rate")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall curves"); axes[1].legend(fontsize=7)
    fig.savefig(FIG / "f18_risk_roc_pr_curves.png")
    plt.close(fig)
    print("    figure -> f18_risk_roc_pr_curves.png")

    # feature importance of the best tree model
    imp_model = "XGBoost" if "XGBoost" in fitted else best_name
    mdl = fitted[imp_model][0]
    if hasattr(mdl, "feature_importances_"):
        fi = pd.Series(mdl.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
        fi.round(5).to_csv(TAB / "t16_delay_risk_feature_importance.csv")
        fig, ax = plt.subplots(figsize=(7, 4.6))
        fi.sort_values().plot(kind="barh", ax=ax, color="#2874a6")
        ax.set_title(f"Delay-risk drivers - {imp_model} gain importance (top 20)")
        ax.set_xlabel("importance")
        fig.savefig(FIG / "f19_delay_risk_feature_importance.png")
        plt.close(fig)
        print("    figure -> f19_delay_risk_feature_importance.png")

    # persist artefacts for SHAP / optimisation modules
    np.save(config.PROCESSED_DIR / "risk_X_test.npy", X_te.to_numpy(dtype=np.float32))
    np.save(config.PROCESSED_DIR / "risk_y_test.npy", y_te.to_numpy())
    np.save(config.PROCESSED_DIR / "risk_proba_test.npy", bp)
    import joblib
    joblib.dump({"model": fitted[imp_model][0], "features": feature_names,
                 "threshold": best_thr, "model_name": imp_model},
                config.PROCESSED_DIR / "delay_risk_model.joblib")

    report = {
        "protocol": {
            "split": f"temporal {int((1 - config.RISK_TEST_FRACTION) * 100)}/"
                     f"{int(config.RISK_TEST_FRACTION * 100)} by order date",
            "train_orders": int(len(tr_idx)), "test_orders": int(len(te_idx)),
            "features": int(X.shape[1]),
            "excluded_for_leakage": LEAKY[1:],
            "late_rate_train": round(float(y_tr.mean()), 4),
            "late_rate_test": round(float(y_te.mean()), 4),
        },
        "classifiers": res_df.to_dict("records"),
        "best_classifier": best_name,
        "operational_threshold": {"value": best_thr,
                                  "f1": round(float(thr_df["f1"].max()), 4),
                                  "precision": round(float(thr_df.loc[thr_df["f1"].idxmax(), "precision"]), 4),
                                  "recall": round(float(thr_df.loc[thr_df["f1"].idxmax(), "recall"]), 4)},
        "lead_time_regression": reg_df.to_dict("records"),
        "sigma_lead_time_days": round(float(np.mean(list(sigma_l.values()))), 4),
        "runtime_sec": round(time.time() - t0, 1),
    }
    Path(MET / "risk_metrics.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[M5] risk module complete in {report['runtime_sec']}s")
    return report


if __name__ == "__main__":
    run()
