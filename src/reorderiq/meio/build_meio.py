"""Two-echelon Multi-Echelon Inventory Optimization (MEIO) on top of the
forecasted store x product demand.

Echelon 2 (downstream): synthetic store, replenished frequently from its
category DC.
Echelon 1 (upstream): department-level "category DC", serving all stores in
the network for that department -- this is where risk pooling shows up,
because aggregate demand across many stores is proportionally less variable
than any single store's demand (coefficient of variation shrinks as you pool
independent-ish demand streams).

Assumptions stated explicitly (we have no real stock or lead-time data):
  - Store replenishment lead time: 1 day (DC -> store, fast/local).
  - DC replenishment lead time: 3 days (supplier -> DC).
  - Store demand streams within a department are treated as independent for
    pooling purposes (a simplification -- in reality regional demand is
    correlated, which would shrink the pooling benefit below what's shown here).
  - Service level: 95% (z=1.645) network-wide, single target for all SKUs.
Safety stock formula (standard periodic demand model):
    SS = z * sigma_period * sqrt(lead_time / period_days)
    ROP = mean_daily_demand * lead_time + SS
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"
REPORT_PATH = ROOT / "reports" / "meio_summary.json"

SERVICE_LEVEL = 0.95
Z = norm.ppf(SERVICE_LEVEL)
STORE_LEAD_TIME_DAYS = 1
DC_LEAD_TIME_DAYS = 3
PERIOD_DAYS = 7  # our demand is aggregated weekly


def main():
    con = duckdb.connect(str(DB_PATH))
    forecast = con.execute("SELECT * FROM forecast_store_product").fetchdf()

    # ---- Echelon 2: store-level safety stock / ROP ----
    forecast["mu_daily"] = forecast["forecast_units_next_week"] / PERIOD_DAYS
    forecast["safety_stock_store"] = Z * forecast["hist_std"] * np.sqrt(STORE_LEAD_TIME_DAYS / PERIOD_DAYS)
    forecast["reorder_point_store"] = forecast["mu_daily"] * STORE_LEAD_TIME_DAYS + forecast["safety_stock_store"]
    forecast["order_up_to_store"] = forecast["reorder_point_store"] + forecast["forecast_units_next_week"]

    # ---- Echelon 1: department-DC level, pooled across stores ----
    dc = (
        forecast.groupby(["department_id", "product_id"])
        .agg(
            mu_week_dc=("forecast_units_next_week", "sum"),
            sigma_sq_sum=("hist_std", lambda s: float(np.sum(s.values ** 2))),  # pooled variance (independence assumption)
            n_stores_serving=("store_id", "nunique"),
            sum_of_store_safety_stock=("safety_stock_store", "sum"),  # naive "no pooling" baseline
        )
        .reset_index()
    )
    dc["sigma_pooled"] = np.sqrt(dc["sigma_sq_sum"])
    dc["mu_daily_dc"] = dc["mu_week_dc"] / PERIOD_DAYS
    dc["safety_stock_dc"] = Z * dc["sigma_pooled"] * np.sqrt(DC_LEAD_TIME_DAYS / PERIOD_DAYS)
    dc["reorder_point_dc"] = dc["mu_daily_dc"] * DC_LEAD_TIME_DAYS + dc["safety_stock_dc"]
    dc["pooling_reduction_pct"] = 1 - (dc["safety_stock_dc"] / dc["sum_of_store_safety_stock"].replace(0, np.nan))

    con.execute("CREATE OR REPLACE TABLE meio_store AS SELECT * FROM forecast")
    con.execute("CREATE OR REPLACE TABLE meio_dc AS SELECT * FROM dc")

    total_ss_no_pooling = dc["sum_of_store_safety_stock"].sum()
    total_ss_pooled = dc["safety_stock_dc"].sum()
    network_reduction = 1 - total_ss_pooled / total_ss_no_pooling

    fill_rate_at_current = SERVICE_LEVEL  # by construction of the z-target; reported for the dashboard

    summary = {
        "service_level_target": SERVICE_LEVEL,
        "z": float(Z),
        "store_lead_time_days": STORE_LEAD_TIME_DAYS,
        "dc_lead_time_days": DC_LEAD_TIME_DAYS,
        "n_store_product_pairs": len(forecast),
        "n_dc_product_pairs": len(dc),
        "total_safety_stock_units_no_pooling": float(total_ss_no_pooling),
        "total_safety_stock_units_pooled_at_dc": float(total_ss_pooled),
        "network_safety_stock_reduction_from_pooling_pct": float(network_reduction * 100),
        "target_fill_rate_pct": float(fill_rate_at_current * 100),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Store-level (store, product) pairs: {len(forecast):,}")
    print(f"DC-level (department, product) pairs: {len(dc):,}")
    print(f"\nSafety stock if every store held its own buffer (no pooling): {total_ss_no_pooling:,.0f} units")
    print(f"Safety stock if pooled at the department-DC echelon:          {total_ss_pooled:,.0f} units")
    print(f"Network-wide safety stock reduction from pooling: {network_reduction:.1%}")
    print(f"\nSaved -> {REPORT_PATH}")
    con.close()


if __name__ == "__main__":
    main()
