"""Build the DC -> store -> SKU demand hierarchy and a bottom-up reconciled
weekly forecast.

Instacart gives no calendar dates, only `days_since_prior_order` (the gap
before each order) and `order_number` (sequence). We reconstruct a *relative*
timeline per user by cumulatively summing those gaps, starting each user at
day 0 -- this recovers relative spacing, not real calendar dates, and users
are NOT aligned to a shared clock (user A's "week 3" is not user B's "week
3"). That's fine for our purpose: we only need a consistent week index to
bucket a store's *own* demand history for a moving-average forecast, not
cross-user calendar alignment.

Hierarchy: product (SKU) -> department (synthetic DC) -> synthetic store ->
network. Reconciliation method: bottom-up (forecast at the SKU x store level,
sum upward) -- the simplest and most robust reconciliation approach, and the
right choice given we're not modeling top-down seasonal splits.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"
TOP_N_PRODUCTS = 3000  # keep the forecast/MEIO grid tractable; covers the vast majority of volume


def main():
    con = duckdb.connect(str(DB_PATH))

    print("Reconstructing relative order timeline per user...")
    con.execute("""
        CREATE OR REPLACE TABLE order_timeline AS
        SELECT
            order_id, user_id, order_number,
            sum(coalesce(days_since_prior_order, 0)) OVER (
                PARTITION BY user_id ORDER BY order_number
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS rel_day
        FROM orders
        WHERE eval_set IN ('prior', 'train')
    """)
    con.execute("ALTER TABLE order_timeline ADD COLUMN rel_week INTEGER")
    con.execute("UPDATE order_timeline SET rel_week = CAST(rel_day / 7 AS INTEGER)")

    print(f"Selecting top {TOP_N_PRODUCTS} products by volume...")
    con.execute(f"""
        CREATE OR REPLACE TABLE top_products AS
        SELECT product_id FROM (
            SELECT product_id, count(*) AS n FROM order_products GROUP BY product_id
            ORDER BY n DESC LIMIT {TOP_N_PRODUCTS}
        )
    """)
    covered = con.execute("""
        SELECT sum(CASE WHEN op.product_id IN (SELECT product_id FROM top_products) THEN 1 ELSE 0 END)::DOUBLE
               / count(*) FROM order_products op
    """).fetchone()[0]
    print(f"  top {TOP_N_PRODUCTS} products cover {covered:.1%} of all order-product volume")

    print("Building store x product x week demand fact table...")
    con.execute("""
        CREATE OR REPLACE TABLE demand_store_product_week AS
        SELECT
            usm.store_id,
            p.department_id,
            op.product_id,
            ot.rel_week,
            count(*) AS units
        FROM order_products op
        JOIN order_timeline ot USING (order_id)
        JOIN user_store_map usm ON usm.user_id = ot.user_id
        JOIN products p ON p.product_id = op.product_id
        WHERE op.product_id IN (SELECT product_id FROM top_products)
        GROUP BY usm.store_id, p.department_id, op.product_id, ot.rel_week
    """)
    n_rows = con.execute("SELECT count(*) FROM demand_store_product_week").fetchone()[0]
    print(f"  {n_rows:,} store-product-week demand rows")

    # ---- Bottom-up forecast: next-week demand per (store, product) via weighted moving average ----
    print("Forecasting next-week demand per (store, product) [weighted moving average, last 6 weeks]...")
    df = con.execute("""
        SELECT store_id, department_id, product_id, rel_week, units
        FROM demand_store_product_week
    """).fetchdf()

    def forecast_group(g: pd.DataFrame) -> float:
        g = g.sort_values("rel_week").tail(6)
        if len(g) == 0:
            return 0.0
        weights = np.arange(1, len(g) + 1)
        return float(np.average(g["units"].values, weights=weights))

    forecast = (
        df.groupby(["store_id", "department_id", "product_id"])
        .apply(forecast_group, include_groups=False)
        .rename("forecast_units_next_week")
        .reset_index()
    )
    hist_stats = (
        df.groupby(["store_id", "department_id", "product_id"])["units"]
        .agg(hist_mean="mean", hist_std="std", n_weeks_observed="count")
        .reset_index()
    )
    forecast = forecast.merge(hist_stats, on=["store_id", "department_id", "product_id"])
    forecast["hist_std"] = forecast["hist_std"].fillna(0)

    con.execute("CREATE OR REPLACE TABLE forecast_store_product AS SELECT * FROM forecast")
    con.execute("CREATE INDEX IF NOT EXISTS idx_fsp_store ON forecast_store_product(store_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_fsp_product ON forecast_store_product(product_id)")

    # ---- Reconciliation check: bottom-up sums must equal top-level totals by construction ----
    sku_total = forecast["forecast_units_next_week"].sum()
    store_total = forecast.groupby("store_id")["forecast_units_next_week"].sum().sum()
    dept_total = forecast.groupby("department_id")["forecast_units_next_week"].sum().sum()
    print(f"\nReconciliation check (bottom-up, must match exactly):")
    print(f"  SKU-level total:        {sku_total:,.1f}")
    print(f"  Sum via store rollup:   {store_total:,.1f}")
    print(f"  Sum via department rollup: {dept_total:,.1f}")
    assert abs(sku_total - store_total) < 1e-6 and abs(sku_total - dept_total) < 1e-6, "reconciliation broke"
    print("  OK: network total reconciles exactly across all rollup levels.")

    con.close()


if __name__ == "__main__":
    main()
