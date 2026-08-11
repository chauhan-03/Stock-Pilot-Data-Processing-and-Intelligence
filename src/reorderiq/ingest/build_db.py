"""Load raw Instacart CSVs into a single DuckDB file for fast columnar querying.

Instacart's `eval_set` column already splits orders into prior / train / test:
  - prior: full order history, used to build features
  - train: each user's held-out *next* order, with known reordered labels
  - test:  each user's held-out next order, labels withheld by Kaggle (unusable for us)
"""
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"


def build():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    t0 = time.time()

    con.execute(f"""
        CREATE TABLE orders AS
        SELECT
            order_id::INTEGER AS order_id,
            user_id::INTEGER AS user_id,
            eval_set::VARCHAR AS eval_set,
            order_number::INTEGER AS order_number,
            order_dow::TINYINT AS order_dow,
            order_hour_of_day::TINYINT AS order_hour_of_day,
            days_since_prior_order::DOUBLE AS days_since_prior_order
        FROM read_csv_auto('{RAW / "orders.csv"}')
    """)

    con.execute(f"""
        CREATE TABLE order_products AS
        SELECT order_id::INTEGER AS order_id, product_id::INTEGER AS product_id,
               add_to_cart_order::INTEGER AS add_to_cart_order, reordered::TINYINT AS reordered
        FROM read_csv_auto('{RAW / "order_products__prior.csv"}')
        UNION ALL
        SELECT order_id::INTEGER, product_id::INTEGER, add_to_cart_order::INTEGER, reordered::TINYINT
        FROM read_csv_auto('{RAW / "order_products__train.csv"}')
    """)

    con.execute(f"""
        CREATE TABLE products AS
        SELECT product_id::INTEGER AS product_id, product_name::VARCHAR AS product_name,
               aisle_id::INTEGER AS aisle_id, department_id::INTEGER AS department_id
        FROM read_csv_auto('{RAW / "products.csv"}')
    """)
    con.execute(f"""
        CREATE TABLE aisles AS
        SELECT aisle_id::INTEGER AS aisle_id, aisle::VARCHAR AS aisle
        FROM read_csv_auto('{RAW / "aisles.csv"}')
    """)
    con.execute(f"""
        CREATE TABLE departments AS
        SELECT department_id::INTEGER AS department_id, department::VARCHAR AS department
        FROM read_csv_auto('{RAW / "departments.csv"}')
    """)

    for col, tbl in [("order_id", "orders"), ("user_id", "orders"), ("order_id", "order_products"),
                      ("product_id", "order_products"), ("product_id", "products")]:
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_{col} ON {tbl}({col})")

    # Denormalized fact table: one row per (order, product), with order + product context.
    con.execute("""
        CREATE TABLE fact_order_products AS
        SELECT
            op.order_id, o.user_id, o.eval_set, o.order_number, o.order_dow,
            o.order_hour_of_day, o.days_since_prior_order,
            op.product_id, op.add_to_cart_order, op.reordered,
            p.product_name, p.aisle_id, p.department_id
        FROM order_products op
        JOIN orders o USING (order_id)
        JOIN products p USING (product_id)
    """)
    con.execute("CREATE INDEX idx_fact_user ON fact_order_products(user_id)")
    con.execute("CREATE INDEX idx_fact_product ON fact_order_products(product_id)")
    con.execute("CREATE INDEX idx_fact_dept ON fact_order_products(department_id)")

    counts = con.execute("""
        SELECT 'orders', count(*) FROM orders
        UNION ALL SELECT 'order_products', count(*) FROM order_products
        UNION ALL SELECT 'products', count(*) FROM products
        UNION ALL SELECT 'fact_order_products', count(*) FROM fact_order_products
    """).fetchall()

    print(f"Built {DB_PATH} in {time.time() - t0:.1f}s")
    for name, n in counts:
        print(f"  {name}: {n:,} rows")
    con.close()


if __name__ == "__main__":
    build()
