"""Feature store for the reorder-prediction task.

Task definition (matches the Kaggle competition exactly): for each user, using
only their *prior* order history, predict which previously-purchased products
will appear again in that user's next order. We only ever generate candidate
(user, product) pairs the user has bought at least once before -- this is a
"reorder" model by construction, not a first-purchase model.

Every feature below is computed strictly from `eval_set = 'prior'` rows, so
none of it can see the held-out order it's trying to predict (see the leakage
note in reports/data_assumptions.md).
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"


def build_features(con: duckdb.DuckDBPyConnection):
    # ---- User-level features (history only) ----
    con.execute("""
        CREATE OR REPLACE TABLE feat_user AS
        WITH prior_orders AS (
            SELECT * FROM orders WHERE eval_set = 'prior'
        ),
        basket_size AS (
            SELECT o.user_id, o.order_id, count(*) AS n_items
            FROM prior_orders o
            JOIN order_products op USING (order_id)
            GROUP BY 1, 2
        )
        SELECT
            po.user_id,
            count(DISTINCT po.order_id)                    AS user_total_orders,
            avg(po.days_since_prior_order)                  AS user_avg_days_between_orders,
            avg(bs.n_items)                                  AS user_avg_basket_size,
            sum(CASE WHEN fo.reordered = 1 THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(sum(CASE WHEN po.order_number > 1 THEN 1 ELSE 0 END), 0) AS user_reorder_ratio,
            count(DISTINCT fo.product_id)                    AS user_distinct_products
        FROM prior_orders po
        JOIN fact_order_products fo ON fo.order_id = po.order_id
        JOIN basket_size bs ON bs.order_id = po.order_id
        GROUP BY po.user_id
    """)

    # ---- Product-level features (history only) ----
    con.execute("""
        CREATE OR REPLACE TABLE feat_product AS
        SELECT
            product_id,
            count(*)                                          AS product_total_orders,
            avg(reordered::DOUBLE)                             AS product_reorder_rate,
            avg(add_to_cart_order::DOUBLE)                     AS product_avg_cart_position,
            count(DISTINCT user_id)                            AS product_distinct_users
        FROM fact_order_products
        WHERE eval_set = 'prior'
        GROUP BY product_id
    """)

    # ---- User x Product interaction features (history only) ----
    con.execute("""
        CREATE OR REPLACE TABLE feat_user_product AS
        WITH up AS (
            SELECT
                user_id, product_id,
                count(*)                                       AS up_order_count,
                avg(add_to_cart_order::DOUBLE)                  AS up_avg_cart_position,
                min(order_number)                               AS up_first_order_number,
                max(order_number)                               AS up_last_order_number,
                avg(reordered::DOUBLE)                          AS up_reorder_rate
            FROM fact_order_products
            WHERE eval_set = 'prior'
            GROUP BY user_id, product_id
        )
        SELECT
            up.*,
            fu.user_total_orders,
            up.up_order_count::DOUBLE / NULLIF(fu.user_total_orders, 0)      AS up_order_rate,
            (fu.user_total_orders - up.up_last_order_number)                  AS up_orders_since_last
        FROM up
        JOIN feat_user fu USING (user_id)
    """)

    # ---- Labeled dataset: candidates x label, joined with all features ----
    # Target order per user = the single held-out order tagged 'train' (labels visible)
    # or 'test' (labels hidden by Kaggle, unusable -- excluded here).
    con.execute("""
        CREATE OR REPLACE TABLE dataset AS
        WITH target_orders AS (
            SELECT order_id, user_id FROM orders WHERE eval_set = 'train'
        ),
        target_items AS (
            SELECT t.user_id, op.product_id
            FROM target_orders t
            JOIN order_products op ON op.order_id = t.order_id
        )
        SELECT
            fup.user_id,
            fup.product_id,
            fup.up_order_count, fup.up_avg_cart_position, fup.up_reorder_rate,
            fup.up_order_rate, fup.up_orders_since_last,
            fu.user_total_orders, fu.user_avg_days_between_orders,
            fu.user_avg_basket_size, fu.user_reorder_ratio, fu.user_distinct_products,
            fp.product_total_orders, fp.product_reorder_rate,
            fp.product_avg_cart_position, fp.product_distinct_users,
            p.department_id, p.aisle_id,
            CASE WHEN ti.product_id IS NOT NULL THEN 1 ELSE 0 END AS label
        FROM feat_user_product fup
        JOIN feat_user fu USING (user_id)
        JOIN feat_product fp USING (product_id)
        JOIN products p USING (product_id)
        JOIN target_orders t ON t.user_id = fup.user_id
        LEFT JOIN target_items ti ON ti.user_id = fup.user_id AND ti.product_id = fup.product_id
    """)

    n = con.execute("SELECT count(*) FROM dataset").fetchone()[0]
    pos = con.execute("SELECT sum(label) FROM dataset").fetchone()[0]
    n_users = con.execute("SELECT count(DISTINCT user_id) FROM dataset").fetchone()[0]
    print(f"dataset: {n:,} candidate (user,product) rows, {n_users:,} users, "
          f"{pos:,} positive ({pos/n:.1%})")


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    build_features(con)
    con.close()
