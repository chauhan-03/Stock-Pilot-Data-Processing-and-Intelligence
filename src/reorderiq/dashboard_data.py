"""Consolidate all report artifacts into a single JSON blob for the dashboard."""
import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"
REPORT_DIR = ROOT / "reports"
OUT_PATH = REPORT_DIR / "dashboard_data.json"


def load_json(name):
    with open(REPORT_DIR / name) as f:
        return json.load(f)


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    departments = con.execute("SELECT department_id, department FROM departments").fetchdf()
    dept_name = dict(zip(departments["department_id"], departments["department"]))

    store_top_dept = con.execute("""
        WITH sd AS (
            SELECT usm.store_id, p.department_id, count(*) n
            FROM fact_order_products fo
            JOIN user_store_map usm ON usm.user_id = fo.user_id
            JOIN products p ON p.product_id = fo.product_id
            WHERE fo.eval_set = 'prior'
            GROUP BY usm.store_id, p.department_id
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY store_id ORDER BY n DESC) rn
            FROM sd
        )
        SELECT store_id, department_id, n FROM ranked WHERE rn <= 3 ORDER BY store_id, rn
    """).fetchdf()

    top_dept_by_store = {}
    for store_id, g in store_top_dept.groupby("store_id"):
        total = g["n"].sum()
        top_dept_by_store[str(int(store_id))] = [
            {"department": dept_name.get(int(r.department_id), "?"), "share": round(float(r.n) / float(total), 3)}
            for r in g.itertuples()
        ]

    dc_dept_totals = con.execute("""
        SELECT department_id, sum(mu_week_dc) total_weekly_demand, sum(safety_stock_dc) total_safety_stock,
               avg(pooling_reduction_pct) avg_pooling_reduction
        FROM meio_dc GROUP BY department_id ORDER BY total_weekly_demand DESC
    """).fetchdf()
    dc_dept_totals["department"] = dc_dept_totals["department_id"].map(dept_name)

    store_sizes = con.execute("SELECT store_id, count(*) n_users FROM user_store_map GROUP BY store_id ORDER BY store_id").fetchdf()

    exceptions = load_json("control_tower_exceptions.json")
    top_exceptions = sorted(exceptions, key=lambda r: -r["ml_expected_demand"])[:20]

    complements = pd.read_csv(REPORT_DIR / "top_complement_rules.csv").head(15).to_dict(orient="records")
    substitutes = pd.read_csv(REPORT_DIR / "top_substitute_candidates.csv").head(15).to_dict(orient="records")
    fpgrowth = pd.read_csv(REPORT_DIR / "fpgrowth_multi_item_rules.csv").head(15).to_dict(orient="records")

    blob = {
        "reorder_model": load_json("reorder_model_metrics.json"),
        "clustering": load_json("store_clustering_validation.json"),
        "meio": load_json("meio_summary.json"),
        "control_tower": load_json("control_tower_summary.json"),
        "association": load_json("association_rules_summary.json"),
        "store_sizes": store_sizes.to_dict(orient="records"),
        "top_dept_by_store": top_dept_by_store,
        "dc_dept_totals": dc_dept_totals[["department", "total_weekly_demand", "total_safety_stock", "avg_pooling_reduction"]].to_dict(orient="records"),
        "top_exceptions": top_exceptions,
        "top_complement_rules": complements,
        "top_substitute_rules": substitutes,
        "fpgrowth_rules": fpgrowth,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(blob, f, indent=2, default=str)
    print(f"Saved -> {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.0f} KB)")
    con.close()


if __name__ == "__main__":
    main()
