"""Basket association mining -- the one layer of this project that is 100%
real, end to end, with no simulated topology involved.

Two outputs:
  1. Pairwise complement/substitute rules computed exactly across the full
     33M-row order history via SQL self-join (scales far better than
     one-hot FP-Growth at this size, and every number is exact, not sampled).
       - "frequently bought together": high lift, likely complements.
       - "substitute candidates": same aisle, lift well below 1 -- shoppers
         pick one or the other rather than both, which is exactly the signal
         the Control Tower needs when a SKU is predicted to stock out.
  2. A genuine FP-Growth run (mlxtend) on a sampled subset of orders, to
     surface multi-item (3+) frequent itemsets that pairwise analysis can't
     see. Sampling is necessary here because one-hot encoding all 3.4M
     orders x 49.7k products is not memory-feasible; the pairwise rules
     above do NOT sample and are the primary, trustworthy output.
"""
import json
from pathlib import Path

import duckdb
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"
REPORT_DIR = ROOT / "reports"
TOP_N_PRODUCTS_FOR_PAIRS = 2000
FPGROWTH_SAMPLE_ORDERS = 120_000
FPGROWTH_TOP_PRODUCTS = 250


def build_pairwise_rules(con: duckdb.DuckDBPyConnection):
    print(f"Computing exact pairwise co-occurrence over full order history "
          f"(top {TOP_N_PRODUCTS_FOR_PAIRS} products)...")
    con.execute(f"""
        CREATE OR REPLACE TABLE top_products_pairs AS
        SELECT product_id FROM (
            SELECT product_id, count(*) n FROM order_products GROUP BY product_id
            ORDER BY n DESC LIMIT {TOP_N_PRODUCTS_FOR_PAIRS}
        )
    """)
    total_orders = con.execute("SELECT count(DISTINCT order_id) FROM order_products").fetchone()[0]

    con.execute("""
        CREATE OR REPLACE TABLE product_support AS
        SELECT product_id, count(DISTINCT order_id) AS n_orders
        FROM order_products
        WHERE product_id IN (SELECT product_id FROM top_products_pairs)
        GROUP BY product_id
    """)

    con.execute("""
        CREATE OR REPLACE TABLE pair_cooccurrence AS
        SELECT a.product_id AS product_a, b.product_id AS product_b, count(*) AS n_together
        FROM order_products a
        JOIN order_products b ON a.order_id = b.order_id AND a.product_id < b.product_id
        WHERE a.product_id IN (SELECT product_id FROM top_products_pairs)
          AND b.product_id IN (SELECT product_id FROM top_products_pairs)
        GROUP BY 1, 2
        HAVING count(*) >= 50
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE pair_rules AS
        SELECT
            pc.product_a, pc.product_b, pc.n_together,
            pc.n_together::DOUBLE / {total_orders} AS support,
            pc.n_together::DOUBLE / sa.n_orders AS confidence_a_to_b,
            pc.n_together::DOUBLE / sb.n_orders AS confidence_b_to_a,
            (pc.n_together::DOUBLE / {total_orders})
                / ((sa.n_orders::DOUBLE / {total_orders}) * (sb.n_orders::DOUBLE / {total_orders})) AS lift,
            pa.product_name AS name_a, pb.product_name AS name_b,
            pa.aisle_id AS aisle_a, pb.aisle_id AS aisle_b
        FROM pair_cooccurrence pc
        JOIN product_support sa ON sa.product_id = pc.product_a
        JOIN product_support sb ON sb.product_id = pc.product_b
        JOIN products pa ON pa.product_id = pc.product_a
        JOIN products pb ON pb.product_id = pc.product_b
    """)

    n_rules = con.execute("SELECT count(*) FROM pair_rules").fetchone()[0]
    print(f"  {n_rules:,} candidate product pairs scored (exact, full data)")

    complements = con.execute("""
        SELECT name_a, name_b, n_together, round(support,5) support, round(lift,2) lift,
               round(confidence_a_to_b,3) conf_a_to_b
        FROM pair_rules
        WHERE lift > 1.5 AND confidence_a_to_b > 0.05
        ORDER BY lift DESC LIMIT 25
    """).fetchdf()

    substitutes = con.execute("""
        SELECT name_a, name_b, aisle_a, n_together, round(lift,2) lift
        FROM pair_rules
        WHERE aisle_a = aisle_b AND lift < 0.8
        ORDER BY n_together DESC LIMIT 25
    """).fetchdf()

    con.execute("CREATE OR REPLACE TABLE substitution_candidates AS SELECT * FROM pair_rules WHERE aisle_a = aisle_b AND lift < 0.8")

    complements.to_csv(REPORT_DIR / "top_complement_rules.csv", index=False)
    substitutes.to_csv(REPORT_DIR / "top_substitute_candidates.csv", index=False)
    print(f"  saved top_complement_rules.csv ({len(complements)} rows), "
          f"top_substitute_candidates.csv ({len(substitutes)} rows)")
    return {"n_pairs_scored": int(n_rules), "n_substitution_candidates": int(
        con.execute("SELECT count(*) FROM substitution_candidates").fetchone()[0])}


def build_fpgrowth_sample(con: duckdb.DuckDBPyConnection):
    print(f"\nRunning real FP-Growth on a {FPGROWTH_SAMPLE_ORDERS:,}-order sample "
          f"(top {FPGROWTH_TOP_PRODUCTS} products, for multi-item itemsets)...")
    con.execute(f"""
        CREATE OR REPLACE TABLE fpg_top_products AS
        SELECT product_id, product_name FROM products
        WHERE product_id IN (
            SELECT product_id FROM order_products GROUP BY product_id
            ORDER BY count(*) DESC LIMIT {FPGROWTH_TOP_PRODUCTS}
        )
    """)
    sample_orders = con.execute(f"""
        SELECT DISTINCT order_id FROM order_products USING SAMPLE {FPGROWTH_SAMPLE_ORDERS} ROWS
    """).fetchdf()["order_id"].tolist()

    baskets_df = con.execute("""
        SELECT op.order_id, p.product_name
        FROM order_products op
        JOIN fpg_top_products p ON p.product_id = op.product_id
        WHERE op.order_id IN (SELECT UNNEST($orders))
    """, {"orders": sample_orders}).fetchdf()

    baskets = baskets_df.groupby("order_id")["product_name"].apply(list).tolist()
    print(f"  {len(baskets):,} baskets, {baskets_df['product_name'].nunique()} distinct products")

    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets, sparse=True)
    onehot = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    frequent = fpgrowth(onehot, min_support=0.003, use_colnames=True, max_len=4)
    print(f"  {len(frequent):,} frequent itemsets found (min_support=0.003)")
    rules = association_rules(frequent, metric="lift", min_threshold=1.2, num_itemsets=len(frequent))
    rules = rules.sort_values("lift", ascending=False).head(30)
    rules_out = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    rules_out["antecedents"] = rules_out["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules_out["consequents"] = rules_out["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules_out.to_csv(REPORT_DIR / "fpgrowth_multi_item_rules.csv", index=False)
    print(f"  saved fpgrowth_multi_item_rules.csv ({len(rules_out)} rows)")
    return {"n_frequent_itemsets": int(len(frequent)), "n_multi_item_rules": int(len(rules_out)),
            "sample_orders": len(baskets)}


def main():
    con = duckdb.connect(str(DB_PATH))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"pairwise_exact": build_pairwise_rules(con)}
    summary["fpgrowth_sample"] = build_fpgrowth_sample(con)
    with open(REPORT_DIR / "association_rules_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {REPORT_DIR / 'association_rules_summary.json'}")
    con.close()


if __name__ == "__main__":
    main()
