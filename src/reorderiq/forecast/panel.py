"""Build the dense, calendar-aligned demand panel the forecasting models train on.

Why this file exists at all: `demand_store_product_week` is NOT a time series.
It is a table of weeks in which demand *happened* -- 82.4% of (store, product)
series contain gaps, and `min(units) = 1`, so a zero-demand week is a MISSING
ROW, not a zero. Three consequences the original forecast did not handle:

 1. `.tail(6)` in build_hierarchical_demand.py takes the last 6 *rows*, not the
    last 6 *weeks*. Measured over the real data, those 6 rows span more than 6
    weeks in 80.4% of series (median 11 weeks, p90 25 weeks). The "most recent"
    weight therefore often lands on a months-old observation.

 2. Averaging only the weeks that HAD demand estimates E[units | demand > 0],
    not E[units]. That is biased upward everywhere, and it propagates straight
    into MEIO safety stock and reorder points.

 3. Total units decay from ~750k (week 1) to 57k (week 52) -- not real demand
    decline, but survivorship: the relative timeline starts each user at their
    own day 0, so fewer users survive to week 52. A trend term fit naively on
    rel_week learns a spurious ~-85% slope.

This module fixes (1) and (2) by densifying each series onto a contiguous weekly
grid with explicit zeros, and gives models the raw material to fix (3) by
attaching per-(store, week) EXPOSURE -- the number of distinct users actually
active behind that cell. Poisson models use exposure as an offset so they learn
demand *per active user* (stable) instead of demand *per week* (survivorship-
contaminated).

Week 0 is excluded throughout: every user's timeline starts at their own day 0,
so week 0 absorbs every user's first order and carries 1.6M units against ~750k
for week 1. It is a construction artifact, not a demand signal.

Densification runs from each series' first observed week through LAST_WEEK. Weeks
before a series' first sale are treated as "not yet active" (excluded) rather
than zero, so we do not fabricate leading zeros for a SKU a store had not yet
begun selling. Weeks after its last sale ARE filled with zeros -- those are
genuine no-demand weeks, and letting exposure explain the thinning is the whole
point of carrying an exposure column.
"""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"

FIRST_WEEK = 1    # week 0 excluded: absorbs every user's first order (see module docstring)
LAST_WEEK = 52


def build_exposure(con: duckdb.DuckDBPyConnection) -> None:
    """Active users and orders per (store, week) -- the denominator that makes
    demand comparable across a survivorship-thinned relative timeline."""
    con.execute(f"""
        CREATE OR REPLACE TABLE store_week_exposure AS
        SELECT
            usm.store_id,
            ot.rel_week,
            count(DISTINCT ot.user_id) AS active_users,
            count(DISTINCT ot.order_id) AS n_orders
        FROM order_timeline ot
        JOIN user_store_map usm USING (user_id)
        WHERE ot.rel_week BETWEEN {FIRST_WEEK} AND {LAST_WEEK}
        GROUP BY 1, 2
    """)


def build_panel(con: duckdb.DuckDBPyConnection) -> None:
    """Densify every (store, product) series onto a contiguous weekly grid,
    making zero-demand weeks explicit, and attach store-week exposure."""
    con.execute(f"""
        CREATE OR REPLACE TABLE demand_panel AS
        WITH series AS (
            SELECT store_id, department_id, product_id, min(rel_week) AS first_week
            FROM demand_store_product_week
            WHERE rel_week BETWEEN {FIRST_WEEK} AND {LAST_WEEK}
            GROUP BY 1, 2, 3
        ),
        weeks AS (
            SELECT UNNEST(generate_series({FIRST_WEEK}, {LAST_WEEK})) AS rel_week
        )
        SELECT
            s.store_id,
            s.department_id,
            s.product_id,
            w.rel_week,
            COALESCE(d.units, 0)::DOUBLE AS units,
            COALESCE(e.active_users, 0)::DOUBLE AS active_users
        FROM series s
        JOIN weeks w ON w.rel_week >= s.first_week
        LEFT JOIN demand_store_product_week d
               ON d.store_id = s.store_id
              AND d.product_id = s.product_id
              AND d.rel_week = w.rel_week
        LEFT JOIN store_week_exposure e
               ON e.store_id = s.store_id
              AND e.rel_week = w.rel_week
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_panel_sp ON demand_panel(store_id, product_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_panel_wk ON demand_panel(rel_week)")


def load_panel(con: duckdb.DuckDBPyConnection, min_weeks: int = 12) -> pd.DataFrame:
    """Load the dense panel, keeping series with enough history to backtest.

    `min_weeks` counts grid weeks present (not weeks with demand), so a sparse
    but long-lived series still qualifies -- intermittency is a pattern to model,
    not a reason to drop a SKU.
    """
    return con.execute(f"""
        SELECT p.*
        FROM demand_panel p
        JOIN (
            SELECT store_id, product_id
            FROM demand_panel
            GROUP BY 1, 2
            HAVING count(*) >= {min_weeks}
        ) k USING (store_id, product_id)
        ORDER BY store_id, product_id, rel_week
    """).fetchdf()


def main():
    con = duckdb.connect(str(DB_PATH))
    print("Building store-week exposure...")
    build_exposure(con)
    n_exp = con.execute("SELECT count(*) FROM store_week_exposure").fetchone()[0]
    print(f"  {n_exp:,} store-week exposure cells")

    print("Densifying demand panel (explicit zero-weeks)...")
    build_panel(con)

    stats = con.execute("""
        SELECT count(*) AS rows,
               count(DISTINCT store_id || '-' || product_id) AS series,
               avg(units) AS mean_units,
               avg(CASE WHEN units = 0 THEN 1.0 ELSE 0.0 END) AS zero_share
        FROM demand_panel
    """).fetchdf()
    print(f"  {int(stats['rows'][0]):,} panel rows across {int(stats['series'][0]):,} series")
    print(f"  mean units/week: {stats['mean_units'][0]:.2f}   zero-weeks: {stats['zero_share'][0]:.1%}")

    sparse_mean = con.execute(f"""
        SELECT avg(units) FROM demand_store_product_week
        WHERE rel_week BETWEEN {FIRST_WEEK} AND {LAST_WEEK}
    """).fetchone()[0]
    dense_mean = float(stats["mean_units"][0])
    print(f"\n  Upward bias from ignoring zero-weeks: sparse mean {sparse_mean:.2f} "
          f"vs true mean {dense_mean:.2f}  ({sparse_mean / dense_mean - 1:+.1%})")
    con.close()


if __name__ == "__main__":
    main()
