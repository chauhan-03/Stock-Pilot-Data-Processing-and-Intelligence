"""Exception / Control Tower layer: rolls up demand sensing + MEIO +
substitution rules into a single flagged-exceptions feed.

Design choice: this is a deterministic rule-based orchestrator, not an LLM
agent graph. Every decision here (is this an exception? which store covers
the deficit? which SKU substitutes?) is fully computable from the data --
there's no ambiguous judgment call that benefits from an LLM in the loop, and
adding one would just add latency/cost/an API-key dependency for no gain. A
thin LangGraph wrapper could sit on top of these functions as callable tools
if a future iteration needs natural-language exception summaries or
free-form "what should I do about X" queries -- noted as an extension, not
implemented here since it wouldn't change what the system actually decides.

Exception logic: a naive moving-average forecast (what MEIO's order-up-to
level was sized against) can't disagree with its own history -- comparing it
to itself flags nothing. The genuine, non-tautological signal available here
is the trained reorder-probability model (models.train_reorder_model), which
predicts each user's *next* order from real behavioral features (recency,
frequency, cart position, etc.), independent of the trailing-average
baseline. We roll those per-user probabilities up to expected demand per
(store, product) and compare against the historical trailing-average
baseline the safety stock was sized on. Where the two disagree by more than
SPIKE_RATIO, the ML demand signal is telling us something the naive trend
forecast would miss -- that's the exception worth surfacing.
"""
import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"
MODEL_PATH = ROOT / "data" / "processed" / "reorder_model.joblib"
REPORT_DIR = ROOT / "reports"
SPIKE_RATIO_HIGH = 1.5  # ML-predicted demand *share* this far above its historical share -> understock risk
SPIKE_RATIO_LOW = 0.67  # ML-predicted demand *share* this far below its historical share -> overstock risk
MIN_ML_DEMAND = 5.0     # ignore low-volume SKUs where small absolute noise produces huge ratios
SURPLUS_UTIL_THRESHOLD = 0.5  # store considered a transfer donor if forecast/order_up_to < this


def score_ml_demand(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Roll the trained reorder-probability model up to expected demand per (store, product)."""
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]

    cand = con.execute(f"""
        SELECT d.user_id, d.product_id, usm.store_id, {", ".join("d." + f for f in features)}
        FROM dataset d
        JOIN user_store_map usm ON usm.user_id = d.user_id
        WHERE d.product_id IN (SELECT product_id FROM top_products)
    """).fetchdf()
    cand[features] = cand[features].fillna(0)
    cand["reorder_prob"] = model.predict_proba(cand[features])[:, 1]

    ml_demand = (
        cand.groupby(["store_id", "product_id"])["reorder_prob"]
        .sum()
        .rename("ml_expected_demand")
        .reset_index()
    )
    return ml_demand


def detect_exceptions(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT store_id, department_id, product_id, forecast_units_next_week,
               hist_mean, hist_std, reorder_point_store, order_up_to_store
        FROM meio_store
        WHERE n_weeks_observed >= 3
    """).fetchdf()

    ml_demand = score_ml_demand(con)
    df = df.merge(ml_demand, on=["store_id", "product_id"], how="inner")

    # The ML model scores "probability of appearing in the user's next order" while hist_mean
    # is a weekly aggregate -- the two live on different time units. Rather than guess a
    # units-per-week conversion factor, compare each SKU's *share* of its store's total demand
    # under each method. That cancels the scale mismatch and turns this into the question that
    # actually matters operationally: is the ML model predicting a materially different demand
    # mix than the trailing average would suggest?
    store_ml_total = df.groupby("store_id")["ml_expected_demand"].transform("sum")
    store_hist_total = df.groupby("store_id")["hist_mean"].transform("sum")
    df["ml_share"] = df["ml_expected_demand"] / store_ml_total
    df["hist_share"] = df["hist_mean"] / store_hist_total
    df["baseline_demand"] = df["hist_mean"].clip(lower=0.5)  # floor to avoid divide-by-near-zero, reported for context
    df["ml_vs_baseline_ratio"] = df["ml_share"] / df["hist_share"].clip(lower=1e-6)
    df["is_exception"] = (
        ((df["ml_vs_baseline_ratio"] > SPIKE_RATIO_HIGH) | (df["ml_vs_baseline_ratio"] < SPIKE_RATIO_LOW))
        & (df["ml_expected_demand"] >= MIN_ML_DEMAND)
    )
    df["direction"] = df["ml_vs_baseline_ratio"].apply(lambda r: "understock_risk" if r > SPIKE_RATIO_HIGH else "overstock_risk")
    df["deficit_units"] = (df["ml_expected_demand"] - df["order_up_to_store"]).clip(lower=0)
    df["utilization"] = df["forecast_units_next_week"] / df["order_up_to_store"].replace(0, pd.NA)
    return df


def find_transfer_donor(all_stores: pd.DataFrame, product_id: int, exclude_store: int) -> dict | None:
    candidates = all_stores[
        (all_stores["product_id"] == product_id)
        & (all_stores["store_id"] != exclude_store)
        & (all_stores["utilization"] < SURPLUS_UTIL_THRESHOLD)
    ].copy()
    if candidates.empty:
        return None
    candidates["surplus_units"] = candidates["order_up_to_store"] - candidates["forecast_units_next_week"]
    best = candidates.sort_values("surplus_units", ascending=False).iloc[0]
    return {"donor_store_id": int(best["store_id"]), "surplus_units": round(float(best["surplus_units"]), 1)}


def find_substitutes(con: duckdb.DuckDBPyConnection, product_id: int) -> list:
    rows = con.execute("""
        SELECT product_a, product_b, name_a, name_b, lift
        FROM substitution_candidates
        WHERE product_a = ? OR product_b = ?
        ORDER BY lift ASC LIMIT 3
    """, [product_id, product_id]).fetchall()
    subs = []
    for pa, pb, na, nb, lift in rows:
        other_id, other_name = (pb, nb) if pa == product_id else (pa, na)
        subs.append({"product_id": int(other_id), "product_name": other_name, "lift": round(lift, 2)})
    return subs


def main():
    con = duckdb.connect(str(DB_PATH))
    all_positions = detect_exceptions(con)
    exceptions = all_positions[all_positions["is_exception"]].copy()
    exceptions = exceptions.sort_values("ml_expected_demand", ascending=False)

    products = con.execute("SELECT product_id, product_name FROM products").fetchdf()
    name_lookup = dict(zip(products["product_id"], products["product_name"]))

    print(f"Flagged {len(exceptions):,} store-SKU exceptions (ML demand signal vs. "
          f"historical baseline) out of {len(all_positions):,} store-SKU positions "
          f"({len(exceptions)/len(all_positions):.1%})")

    records = []
    for _, row in exceptions.iterrows():
        donor = None
        if row["direction"] == "understock_risk":
            donor = find_transfer_donor(all_positions, int(row["product_id"]), int(row["store_id"]))
        subs = find_substitutes(con, int(row["product_id"]))
        records.append({
            "store_id": int(row["store_id"]),
            "department_id": int(row["department_id"]),
            "product_id": int(row["product_id"]),
            "product_name": name_lookup.get(row["product_id"], "?"),
            "direction": row["direction"],
            "ml_expected_demand": round(float(row["ml_expected_demand"]), 1),
            "historical_baseline_demand": round(float(row["baseline_demand"]), 1),
            "ml_vs_baseline_ratio": round(float(row["ml_vs_baseline_ratio"]), 2),
            "order_up_to_store": round(float(row["order_up_to_store"]), 1),
            "recommended_transfer": donor,
            "substitute_recommendations": subs,
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "control_tower_exceptions.json", "w") as f:
        json.dump(records, f, indent=2)

    n_with_transfer = sum(1 for r in records if r["recommended_transfer"] is not None)
    n_with_sub = sum(1 for r in records if r["substitute_recommendations"])
    summary = {
        "total_store_sku_positions": len(all_positions),
        "n_exceptions": len(exceptions),
        "exception_rate_pct": round(len(exceptions) / len(all_positions) * 100, 2),
        "n_exceptions_with_transfer_recommendation": n_with_transfer,
        "n_exceptions_with_substitute_recommendation": n_with_sub,
    }
    with open(REPORT_DIR / "control_tower_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  {n_with_transfer:,} exceptions have a same-network transfer donor")
    print(f"  {n_with_sub:,} exceptions have a substitute-product recommendation")
    print(f"Saved -> control_tower_exceptions.json, control_tower_summary.json")
    con.close()


if __name__ == "__main__":
    main()
