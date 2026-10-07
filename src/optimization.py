"""
Module M4 - Forecast-to-Prescription Inventory Optimisation.

Turns the demand forecasts produced by Module M3 into inventory decisions and
quantifies the operational benefit with a day-by-day inventory simulation.

Policies compared
-----------------
BASELINE  "traditional / static rule" policy - the practice criticised in the
          problem statement: reorder point and order quantity are computed once
          from *historical averages* and then held constant
              ROP = mu_hist * L + Z * sigma_hist * sqrt(L)
              Q   = EOQ(mu_hist)

PROPOSED  "ML-driven dynamic" policy - the reorder point is re-derived every
          day from the model forecast distribution
              SS_t = Z * sqrt(L * sigma_hat^2 + mu_hat_t^2 * sigma_L^2)
              ROP_t = mu_hat_t * L + SS_t
              Q_t   = EOQ(mu_hat_t, annualised)   (rounded, with a floor)
          sigma_L (lead-time variability) is *measured* on the DataCo logistics
          dataset, which is how the three data sources are coupled into one
          decision loop.

Both policies face the **same realised demand** and the **same lead-time
random stream** (common random numbers), so the comparison isolates the value
of the forecast rather than simulation noise.

Usage:
    python src/optimization.py
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

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})
FIG, MET, TAB = config.FIGURES_DIR, config.METRICS_DIR, config.TABLES_DIR

L = config.LEAD_TIME_DAYS
Z = config.Z_SCORE
S_ORDER = config.ORDERING_COST
C_UNIT = config.UNIT_COST
H_UNIT_YEAR = config.ANNUAL_HOLDING_RATE * C_UNIT      # ₹ / unit / year
H_UNIT_DAY = H_UNIT_YEAR / 365.0
STOCKOUT_PENALTY = 0.35 * C_UNIT                       # lost margin per unmet unit


# --------------------------------------------------------------------------
def measure_lead_time_variability() -> float:
    """
    sigma_L - the *measured* residual standard deviation of realised shipping
    days on the DataCo logistics dataset (produced by src/models_risk.py).

    Using the company's own measured logistics variability rather than a
    textbook constant is what couples the logistics dataset into the inventory
    decision, and is the concrete realisation of the 'unified framework'.
    """
    p = MET / "risk_metrics.json"
    if p.exists():
        rep = json.loads(p.read_text())
        sig = rep.get("sigma_lead_time_days")
        if sig:
            return float(sig)
    print("    !! risk_metrics.json missing - falling back to sigma_L = 1.0 day")
    return 1.0


# --------------------------------------------------------------------------
def eoq(annual_demand: np.ndarray, order_cost: float = S_ORDER,
        hold_cost_unit_year: float = H_UNIT_YEAR) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.sqrt(2 * np.maximum(annual_demand, 0) * order_cost / hold_cost_unit_year)
    return np.nan_to_num(q, nan=1.0)


def simulate(actual: np.ndarray, forecast: np.ndarray, hist_mu: np.ndarray,
             hist_sigma: np.ndarray, sigma_hat: np.ndarray, sigma_l: float,
             mode: str, seed: int = 42, z: float | None = None,
             sigma_scale: float = 1.0, lead_time_term: bool = True,
             lead_draw_sigma: float | None = None) -> dict:
    """
    Vectorised (day x SKU) periodic-review inventory simulation.

    Parameters
    ----------
    actual, forecast : (T, n_sku) realised demand and one-day-ahead ML forecast
    mode : 'static_demand' | 'static_full' | 'ml'
        static_demand - traditional practice: ROP = mu_hist*L + Z*sigma_hist*sqrt(L)
                        (demand variability only, parameters frozen from history)
        static_full   - fair control: the same full safety-stock formula as the
                        proposed policy, but with *historical averages* instead of
                        a forecast: ROP = mu_hist*L + Z*sqrt(L*sigma_hist^2 + mu_hist^2*sigma_L^2)
        ml            - proposed: ROP_t = mu_hat_t*L + Z*sqrt(L*sigma_hat^2 + mu_hat_t^2*sigma_L^2)
                        with a stable EOQ derived from the mean forecast

    Both policies share the realised demand, the lead-time random draws, the
    initial stock, the ordering rule (one outstanding order per SKU) and the
    order-quantity rule, so the measured difference isolates the value of the
    forecast itself.
    """
    T, n = actual.shape
    rng = np.random.default_rng(seed)

    # common random numbers: identical lead-time draws for every policy
    draw_sigma = max(sigma_l if lead_draw_sigma is None else lead_draw_sigma, 0.3)
    lead_draws = np.clip(np.rint(rng.normal(loc=L, scale=draw_sigma,
                                            size=(T, n))), 1, 3 * L).astype(int)

    horizon = L + 2 * L + 2
    on_hand = hist_mu * L            # identical starting inventory for all policies
    pipeline = np.zeros((horizon, n))
    outstanding = np.zeros(n, dtype=bool)
    backorder = np.zeros(n)

    hold_cost = order_cost_total = stockout_units = demand_total = 0.0
    orders = 0
    inv_track = np.zeros((T, n))
    stockout_track = np.zeros((T, n))

    z_used = Z if z is None else float(z)
    static_rop = {
        "static_demand": hist_mu * L + Z * hist_sigma * np.sqrt(L),
        "static_full": hist_mu * L + Z * np.sqrt(L * hist_sigma ** 2 + hist_mu ** 2 * sigma_l ** 2),
    }
    static_q = np.maximum(eoq(hist_mu * 365.0), 1.0)
    ml_q = np.maximum(eoq(np.maximum(forecast.mean(axis=0), 0) * 365.0), 1.0)

    for t in range(T):
        # ---- receive today's arrivals ----
        on_hand += pipeline[0]
        outstanding[pipeline[0] > 0] = False
        pipeline = np.roll(pipeline, -1, axis=0)
        pipeline[-1] = 0

        # ---- serve demand (unmet demand is lost / backordered at a penalty) ----
        demand = actual[t]
        demand_total += float(demand.sum())
        served = np.minimum(on_hand, demand)
        unmet = demand - served
        on_hand -= served
        backorder += unmet
        stockout_units += float(unmet.sum())
        stockout_track[t] = unmet

        # ---- replenishment decision ----
        ip = on_hand + pipeline.sum(axis=0) - backorder
        if mode == "ml":
            mu_hat = np.maximum(forecast[t], 0)
            sig = np.maximum(sigma_hat, 0) * sigma_scale
            if lead_time_term:
                ss = z_used * np.sqrt(L * sig ** 2 + mu_hat ** 2 * sigma_l ** 2)
            else:
                # forecast only, identical formula shape to the traditional policy
                ss = z_used * sig * np.sqrt(L)
            rop = mu_hat * L + ss
            q = ml_q
        else:
            rop = static_rop[mode]
            q = static_q
        trigger = (ip <= rop) & (~outstanding)
        n_trig = int(trigger.sum())
        if n_trig:
            orders += n_trig
            order_cost_total += n_trig * S_ORDER
            for i in np.where(trigger)[0]:
                pipeline[lead_draws[t, i], i] += q[i]
                outstanding[i] = True

        hold_cost += float(on_hand.sum()) * H_UNIT_DAY
        inv_track[t] = on_hand

    filled = demand_total - stockout_units
    fill_rate = filled / max(demand_total, 1e-9)
    avg_inv = float(inv_track.mean())
    total_cost = hold_cost + order_cost_total + stockout_units * STOCKOUT_PENALTY
    turns = float(demand_total) / max(avg_inv * T, 1e-9) * 365.0
    return {
        "holding_cost": round(hold_cost, 2),
        "ordering_cost": round(order_cost_total, 2),
        "stockout_cost": round(stockout_units * STOCKOUT_PENALTY, 2),
        "total_cost": round(total_cost, 2),
        "orders": orders,
        "stockout_units": round(float(stockout_units), 1),
        "fill_rate_pct": round(float(fill_rate) * 100, 3),
        "avg_inventory_units": round(avg_inv, 2),
        "inventory_turns": round(turns, 2),
        "demand_units": round(float(demand_total), 1),
        "z_used": z_used,
        "_inv_track": inv_track,
        "_stockout_track": stockout_track,
    }


# --------------------------------------------------------------------------
def run() -> dict:
    t0 = time.time()
    print("=" * 78)
    print("MODULE M4 - INVENTORY OPTIMISATION (forecast -> prescription)")
    print("=" * 78)

    pred_path = config.PROCESSED_DIR / "forecast_test_predictions.csv.gz"
    if not pred_path.exists():
        raise FileNotFoundError("Run src/models_forecast.py first.")
    p = pd.read_csv(pred_path)
    p["date"] = pd.to_datetime(p["date"])
    print(f"    forecast rows {len(p):,} | SKUs {p['sku'].nunique()} | days {p['date'].nunique()}")

    # history statistics for the static baseline
    ft = pd.read_csv(config.PROCESSED_DIR / "store_demand_features.csv.gz",
                     usecols=["date", "sku", "sales"])
    ft["date"] = pd.to_datetime(ft["date"])
    hist = ft[ft["date"] < p["date"].min()]
    g = hist.groupby("sku")["sales"]
    hist_mu = g.mean()
    hist_sigma = g.std().fillna(0.0)

    skus = sorted(p["sku"].unique())
    dates = np.array(sorted(p["date"].unique()))
    T, n = len(dates), len(skus)
    key = pd.MultiIndex.from_product([dates, skus], names=["date", "sku"])
    actual = p.set_index(["date", "sku"])["sales"].reindex(key).to_numpy(float).reshape(T, n)
    fc = p.set_index(["date", "sku"])["Ensemble"].reindex(key).to_numpy(float).reshape(T, n)
    mu_vec = hist_mu.reindex(skus).to_numpy(float)
    sd_vec = hist_sigma.reindex(skus).to_numpy(float)

    # per-SKU forecast uncertainty sigma_hat is measured on the *validation*
    # window that precedes the test window. Using the test window itself would
    # be look-ahead bias in the replenishment policy.
    val_path = config.PROCESSED_DIR / "forecast_validation_predictions.csv.gz"
    if val_path.exists():
        v = pd.read_csv(val_path)
        vsk = (v.groupby("sku")
               .apply(lambda g: float(np.std(g["sales"] - g["Ensemble"])))
               .reindex(skus).fillna(1.0))
        err_sigma = np.maximum(vsk.to_numpy(float), 0.3)
        sigma_source = "validation window (90 days, no look-ahead)"
    else:
        err_sigma = np.maximum(np.nanstd(fc - actual, axis=0), 0.3)
        sigma_source = "test window (fallback - run models_forecast.py to regenerate)"
    print(f"    sigma_hat per SKU: mean {err_sigma.mean():.2f} units/day "
          f"[{sigma_source}]")

    sigma_l = measure_lead_time_variability()
    print(f"    lead time  : L={L} d | sigma_L={sigma_l:.3f} d (measured on DataCo logistics)")
    print(f"    service    : Z={Z} (95%) | ordering cost {S_ORDER}/order | "
          f"holding {H_UNIT_YEAR}/unit/year")
    print(f"    simulation : {T} days x {n} SKUs, common random numbers")

    t = time.time()
    trad = simulate(actual, fc, mu_vec, sd_vec, err_sigma, sigma_l, "static_demand")
    ctrl = simulate(actual, fc, mu_vec, sd_vec, err_sigma, sigma_l, "static_full")
    prop = simulate(actual, fc, mu_vec, sd_vec, err_sigma, sigma_l, "ml")
    print(f"    three policies simulated in {time.time() - t:.1f}s")
    # the fair control is the strongest of the two static policies
    base = ctrl if ctrl["holding_cost"] <= trad["holding_cost"] else trad

    def delta(a, b):
        return round((a - b) / a * 100, 2) if a else 0.0

    summary = {
        "policy_comparison": {
            "traditional_static": {k: v for k, v in trad.items() if not k.startswith("_")},
            "static_full_control": {k: v for k, v in ctrl.items() if not k.startswith("_")},
            "proposed_ml": {k: v for k, v in prop.items() if not k.startswith("_")},
        },
        "improvement_vs_traditional": {
            "holding_cost_reduction": delta(trad["holding_cost"], prop["holding_cost"]),
            "total_cost_reduction": delta(trad["total_cost"], prop["total_cost"]),
            "stockout_units_reduction": delta(trad["stockout_units"], prop["stockout_units"]),
            "avg_inventory_reduction": delta(trad["avg_inventory_units"], prop["avg_inventory_units"]),
            "fill_rate_gain_pp": round(prop["fill_rate_pct"] - trad["fill_rate_pct"], 3),
            "inventory_turns_gain_pct": delta(prop["inventory_turns"], trad["inventory_turns"]),
        },
        "improvement_vs_control": {
            "holding_cost_reduction": delta(ctrl["holding_cost"], prop["holding_cost"]),
            "total_cost_reduction": delta(ctrl["total_cost"], prop["total_cost"]),
            "stockout_units_reduction": delta(ctrl["stockout_units"], prop["stockout_units"]),
            "avg_inventory_reduction": delta(ctrl["avg_inventory_units"], prop["avg_inventory_units"]),
            "fill_rate_gain_pp": round(prop["fill_rate_pct"] - ctrl["fill_rate_pct"], 3),
            "inventory_turns_gain_pct": delta(ctrl["inventory_turns"], ctrl["inventory_turns"]),
        },
        "improvement_pct": {   # headline: proposed policy vs traditional practice
            "holding_cost_reduction": delta(trad["holding_cost"], prop["holding_cost"]),
            "total_cost_reduction": delta(trad["total_cost"], prop["total_cost"]),
            "stockout_units_reduction": delta(trad["stockout_units"], prop["stockout_units"]),
            "avg_inventory_reduction": delta(trad["avg_inventory_units"], prop["avg_inventory_units"]),
            "fill_rate_gain_pp": round(prop["fill_rate_pct"] - trad["fill_rate_pct"], 3),
            "inventory_turns_gain_pct": delta(prop["inventory_turns"], trad["inventory_turns"]),
        },
        "assumptions": {
            "lead_time_days": L, "sigma_lead_time_days": round(sigma_l, 3),
            "service_level": config.SERVICE_LEVEL, "z": Z,
            "ordering_cost": S_ORDER, "unit_cost": C_UNIT,
            "annual_holding_rate": config.ANNUAL_HOLDING_RATE,
            "stockout_penalty_per_unit": round(STOCKOUT_PENALTY, 3),
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    print("\n    --- policy comparison (184-day simulation, 500 SKUs) ---")
    comp = pd.DataFrame(summary["policy_comparison"]).T
    print(comp.to_string())
    print("\n    proposed vs TRADITIONAL practice:",
          json.dumps(summary["improvement_vs_traditional"]))
    print("    proposed vs FAIR CONTROL (same formula, historical averages):",
          json.dumps(summary["improvement_vs_control"]))
    print(f"\n    holding cost  : {summary['improvement_pct']['holding_cost_reduction']:+.2f} %")
    print(f"    stockouts     : {summary['improvement_pct']['stockout_units_reduction']:+.2f} %")
    print(f"    total cost    : {summary['improvement_pct']['total_cost_reduction']:+.2f} %")
    print(f"    fill rate     : {prop['fill_rate_pct']:.2f} % vs {trad['fill_rate_pct']:.2f} % "
          f"({summary['improvement_pct']['fill_rate_gain_pp']:+.3f} pp)")


    # =====================================================================
    # Controlled study.  Three questions, three experiments:
    #   E1  does the forecast help when both policies use the SAME formula?
    #   E2  does the as-designed risk-aware policy beat traditional practice?
    #   E3  what happens when lead times actually get disrupted?
    # Every comparison is made at a SERVICE-MATCHED operating point, because
    # otherwise the comparison only measures which policy holds more stock.
    # =====================================================================

    def calibrate(lead_time_term: bool, lead_draw_sigma: float | None = None,
                  ref_fill: float | None = None, lo=-1.5, hi=Z, step=0.05):
        """Smallest safety factor z whose fill rate matches the reference policy."""
        ref = trad["fill_rate_pct"] if ref_fill is None else ref_fill
        best_z, best_run = hi, None
        z = lo
        while z <= hi + 1e-9:
            r = simulate(actual, fc, mu_vec, sd_vec, err_sigma, sigma_l, "ml", seed=42,
                         z=z, lead_time_term=lead_time_term,
                         lead_draw_sigma=lead_draw_sigma)
            if r["fill_rate_pct"] >= ref:
                best_z, best_run = z, r
                break
            z += step
        if best_run is None:
            best_run = simulate(actual, fc, mu_vec, sd_vec, err_sigma, sigma_l, "ml", seed=42,
                                z=hi, lead_time_term=lead_time_term,
                                lead_draw_sigma=lead_draw_sigma)
            best_z = hi
        return round(float(best_z), 3), best_run

    print("\n    --- E1: same formula, forecast vs historical averages (isolates the forecast) ---")
    z_f, run_f = calibrate(lead_time_term=False)
    e1 = {
        "policy": "proposed (forecast, demand-variability formula only)",
        "z_matched": z_f, "fill_rate_pct": run_f["fill_rate_pct"],
        "holding_cost_reduction_pct": delta(trad["holding_cost"], run_f["holding_cost"]),
        "stockout_units_reduction_pct": delta(trad["stockout_units"], run_f["stockout_units"]),
        "total_cost_reduction_pct": delta(trad["total_cost"], run_f["total_cost"]),
        "avg_inventory_reduction_pct": delta(trad["avg_inventory_units"], run_f["avg_inventory_units"]),
    }
    print("      " + json.dumps(e1))

    print("\n    --- E2: as-designed risk-aware policy vs traditional practice ---")
    z_r, run_r = calibrate(lead_time_term=True)
    e2 = {
        "policy": "proposed (forecast + measured lead-time variability)",
        "z_matched": z_r, "fill_rate_pct": run_r["fill_rate_pct"],
        "holding_cost_reduction_pct": delta(trad["holding_cost"], run_r["holding_cost"]),
        "stockout_units_reduction_pct": delta(trad["stockout_units"], run_r["stockout_units"]),
        "total_cost_reduction_pct": delta(trad["total_cost"], run_r["total_cost"]),
        "avg_inventory_reduction_pct": delta(trad["avg_inventory_units"], run_r["avg_inventory_units"]),
    }
    print("      " + json.dumps(e2))

    print("\n    --- E3: lead-time disruption scenario (3x measured variability) ---")
    dis_sigma = 3 * sigma_l
    trad_dis = simulate(actual, fc, mu_vec, sd_vec, err_sigma, sigma_l, "static_demand",
                        seed=42, lead_draw_sigma=dis_sigma)
    ctrl_dis = simulate(actual, fc, mu_vec, sd_vec, err_sigma, sigma_l, "static_full",
                        seed=42, lead_draw_sigma=dis_sigma)
    ml_dis = simulate(actual, fc, mu_vec, sd_vec, err_sigma, sigma_l, "ml", seed=42,
                      z=Z, lead_time_term=True, lead_draw_sigma=dis_sigma)
    e3 = {
        "disruption_sigma_L_days": round(dis_sigma, 3),
        "traditional_fill_rate_pct": trad_dis["fill_rate_pct"],
        "traditional_stockout_units": trad_dis["stockout_units"],
        "risk_aware_fill_rate_pct": ml_dis["fill_rate_pct"],
        "risk_aware_stockout_units": ml_dis["stockout_units"],
        "stockout_reduction_pct": delta(trad_dis["stockout_units"], ml_dis["stockout_units"]),
        "total_cost_reduction_pct": delta(trad_dis["total_cost"], ml_dis["total_cost"]),
        "traditional_total_cost": trad_dis["total_cost"],
        "risk_aware_total_cost": ml_dis["total_cost"],
    }
    print("      " + json.dumps(e3))

    # headline = E2 (the as-designed policy), matched to the traditional service level
    matched = run_r
    z_match = z_r
    matched_summary = {
        "note": "proposed policy re-calibrated so that its fill rate matches the "
                "traditional policy; comparison is therefore at equal service",
        "z_used": z_match, "traditional_fill_rate_pct": trad["fill_rate_pct"],
        "proposed_fill_rate_pct": matched["fill_rate_pct"],
        "holding_cost_reduction": e2["holding_cost_reduction_pct"],
        "stockout_units_reduction": e2["stockout_units_reduction_pct"],
        "total_cost_reduction": e2["total_cost_reduction_pct"],
        "avg_inventory_reduction": e2["avg_inventory_reduction_pct"],
        "detail": {k: v for k, v in matched.items() if not k.startswith("_")},
    }
    summary["matched_service_comparison"] = matched_summary
    summary["experiment_E1_forecast_only"] = e1
    summary["experiment_E2_risk_aware"] = e2
    summary["experiment_E3_lead_time_disruption"] = e3

    # service-level sweep table for the as-designed policy
    sweep = []
    for z_try in [-0.5, 0.0, 0.25, 0.5, 0.84, 1.0, 1.28, Z]:
        r = simulate(actual, fc, mu_vec, sd_vec, err_sigma, sigma_l, "ml", seed=42, z=z_try)
        sweep.append({"z": z_try, "holding_cost": r["holding_cost"],
                      "stockout_units": r["stockout_units"], "fill_rate_pct": r["fill_rate_pct"],
                      "total_cost": r["total_cost"], "avg_inventory_units": r["avg_inventory_units"],
                      "orders": r["orders"]})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(TAB / "t19_service_level_sweep.csv", index=False)

    # sensitivity to forecast accuracy, evaluated at the design service factor
    print("\n    --- sensitivity of the design point to forecast accuracy ---")
    sens = []
    for scale, label in [(1.00, "current model"), (0.85, "15% lower forecast error"),
                         (0.70, "30% lower forecast error"), (0.55, "45% lower forecast error")]:
        r = simulate(actual, fc, mu_vec, sd_vec, err_sigma, sigma_l, "ml", seed=42,
                     z=Z, sigma_scale=scale)
        sens.append({"scenario": label, "sigma_scale": scale,
                     "holding_cost": r["holding_cost"], "stockout_units": r["stockout_units"],
                     "fill_rate_pct": r["fill_rate_pct"], "total_cost": r["total_cost"],
                     "holding_cost_reduction_vs_traditional_pct":
                         delta(trad["holding_cost"], r["holding_cost"]),
                     "total_cost_reduction_vs_traditional_pct": delta(trad["total_cost"], r["total_cost"])})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(TAB / "t20_forecast_accuracy_sensitivity.csv", index=False)
    print(sens_df.to_string(index=False))
    summary["forecast_accuracy_sensitivity"] = sens_df.to_dict("records")

    # ---------------- ABC segmentation of the benefit ----------------
    annual_demand = actual.sum(axis=0) * (365 / T)
    value = annual_demand * C_UNIT
    order = np.argsort(-value)
    cum = np.cumsum(value[order]) / value.sum()
    abc = np.empty(n, dtype="U1")
    abc[order] = np.where(cum <= .8, "A", np.where(cum <= .95, "B", "C"))
    rows = []
    for cls in ["A", "B", "C"]:
        m = abc == cls
        if not m.any():
            continue
        rows.append({
            "class": cls, "skus": int(m.sum()),
            "baseline_holding": round(float(trad["_inv_track"][:, m].sum()) * H_UNIT_DAY, 2),
            "proposed_holding": round(float(matched["_inv_track"][:, m].sum()) * H_UNIT_DAY, 2),
            "baseline_stockouts": round(float(trad["_stockout_track"][:, m].sum()), 1),
            "proposed_stockouts": round(float(matched["_stockout_track"][:, m].sum()), 1),
            "demand_units": round(float(actual[:, m].sum()), 1),
        })
    abc_df = pd.DataFrame(rows)
    abc_df["holding_reduction_pct"] = (
        (abc_df["baseline_holding"] - abc_df["proposed_holding"]) / abc_df["baseline_holding"] * 100).round(2)
    abc_df.to_csv(TAB / "t17_benefit_by_abc_class.csv", index=False)
    print("\n    --- benefit by ABC class ---")
    print(abc_df.to_string(index=False))

    # ---------------- reorder recommendation table (decision support) ----------------
    mu_hat = np.maximum(fc.mean(axis=0), 0)
    ss = Z * np.sqrt(L * np.maximum(err_sigma, 0) ** 2 + mu_hat ** 2 * sigma_l ** 2)
    rop = mu_hat * L + ss
    q = np.maximum(eoq(mu_hat * 365.0), 1.0)
    recs = pd.DataFrame({
        "sku": skus,
        "avg_daily_demand": mu_hat.round(2),
        "forecast_error_sd": err_sigma.round(2),
        "safety_stock": ss.round(1),
        "reorder_point": rop.round(1),
        "eoq_order_qty": q.round(1),
        "days_of_cover": (rop / np.maximum(mu_hat, 1e-6)).round(2),
        "abc_class": abc,
        "annual_value": (value).round(0),
        "current_on_hand_avg": trad["_inv_track"].mean(axis=0).round(1),
        "recommended_action": np.where(
            trad["_inv_track"].mean(axis=0) < rop * 0.8, "REORDER NOW",
            np.where(trad["_inv_track"].mean(axis=0) < rop * 1.2, "REORDER SOON", "HOLD")),
    }).sort_values(["abc_class", "annual_value"], ascending=[True, False])
    recs.to_csv(TAB / "t18_reorder_recommendations.csv", index=False)
    print(f"\n    reorder recommendation table written for {len(recs)} SKUs "
          f"(REORDER NOW: {(recs['recommended_action'] == 'REORDER NOW').sum()})")

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(dates, trad["_inv_track"].sum(axis=1), label="traditional static policy",
            color="#c0392b", lw=1.3)
    ax.plot(dates, prop["_inv_track"].sum(axis=1),
            label="proposed ML-driven policy (as designed, z=1.645)", color="#2471a3", lw=1.2, ls="--")
    ax.plot(dates, matched["_inv_track"].sum(axis=1),
            label=f"proposed policy re-calibrated to the same service level (z={z_match:.2f})",
            color="#1e8449", lw=1.4)
    ax.set_ylabel("total inventory (units)")
    ax.set_title("Aggregate inventory position - static vs ML-driven replenishment (184 days)")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "f20_inventory_policies.png")
    plt.close(fig)
    print("    figure -> f20_inventory_policies.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    metrics = [("holding_cost", "Holding cost (₹)"), ("stockout_units", "Stock-out units"),
               ("total_cost", "Total cost (₹)")]
    labels = ["traditional", "proposed\n(design z=1.645)", "proposed\n(service-matched)"]
    for ax, (k, label) in zip(axes, metrics):
        vals = [trad[k], prop[k], matched[k]]
        b = ax.bar(labels, vals, color=["#c0392b", "#2471a3", "#1e8449"], width=.62)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(label)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Operational impact of the ML-driven policy (184 days, 500 SKUs)", y=1.04)
    fig.savefig(FIG / "f21_policy_cost_impact.png")
    plt.close(fig)
    print("    figure -> f21_policy_cost_impact.png")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(abc_df))
    ax.bar(x - .18, abc_df["baseline_holding"], width=.36, label="traditional", color="#c0392b")
    ax.bar(x + .18, abc_df["proposed_holding"], width=.36, label="proposed", color="#1e8449")
    ax.set_xticks(x); ax.set_xticklabels(abc_df["class"] + "\n(" + abc_df["skus"].astype(str) + " SKUs)")
    ax.set_ylabel("holding cost (₹)"); ax.legend(fontsize=8)
    ax.set_title("Holding-cost benefit by ABC class")
    fig.savefig(FIG / "f22_abc_benefit.png")
    plt.close(fig)
    print("    figure -> f22_abc_benefit.png")

    Path(MET / "optimization_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[M4] optimisation complete in {summary['runtime_sec']}s")
    return summary


if __name__ == "__main__":
    run()
