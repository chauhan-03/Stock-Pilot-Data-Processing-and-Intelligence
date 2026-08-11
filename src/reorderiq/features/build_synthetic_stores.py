"""Construct the missing network layer: cluster Instacart *users* into N
synthetic "stores" so we have a DC -> store -> SKU hierarchy to run
hierarchical forecasting / MEIO on. Instacart has no real store dimension --
this is a simulated topology laid on top of real demand signal, and it only
means anything if the clusters actually behave differently. We therefore
cluster on each user's *department purchase-share profile* (what fraction of
their spend goes to produce vs. dairy vs. snacks, etc. -- a proxy for store
format / demographic mix) plus basic shopping-cadence behavior, then run a
statistical check that the resulting store nodes have distinct demand mixes
before anything downstream is allowed to trust them.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"
REPORT_PATH = ROOT / "reports" / "store_clustering_validation.json"
N_STORES = 18
RANDOM_STATE = 42


def build_user_profiles(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    dept_share = con.execute("""
        WITH user_dept AS (
            SELECT user_id, department_id, count(*) AS n
            FROM fact_order_products
            WHERE eval_set = 'prior'
            GROUP BY user_id, department_id
        ),
        user_total AS (
            SELECT user_id, sum(n) AS total FROM user_dept GROUP BY user_id
        )
        SELECT ud.user_id, ud.department_id, ud.n::DOUBLE / ut.total AS share
        FROM user_dept ud JOIN user_total ut USING (user_id)
    """).fetchdf()
    dept_pivot = dept_share.pivot(index="user_id", columns="department_id", values="share").fillna(0)
    dept_pivot.columns = [f"dept_share_{c}" for c in dept_pivot.columns]

    behavior = con.execute("""
        SELECT
            user_id,
            count(DISTINCT order_id)            AS n_orders,
            avg(days_since_prior_order)          AS avg_days_between,
            mode() WITHIN GROUP (ORDER BY order_dow) AS modal_dow,
            avg(order_hour_of_day)               AS avg_hour,
        FROM orders WHERE eval_set = 'prior'
        GROUP BY user_id
    """).fetchdf().set_index("user_id")

    basket = con.execute("""
        SELECT user_id, avg(n_items) AS avg_basket_size
        FROM (
            SELECT o.user_id, o.order_id, count(*) AS n_items
            FROM orders o JOIN order_products op USING (order_id)
            WHERE o.eval_set = 'prior'
            GROUP BY o.user_id, o.order_id
        )
        GROUP BY user_id
    """).fetchdf().set_index("user_id")

    profile = dept_pivot.join(behavior, how="inner").join(basket, how="inner")
    return profile.fillna(profile.mean())


def choose_k(X: np.ndarray, candidates=(12, 15, 18, 20)) -> dict:
    scores = {}
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X), size=min(8000, len(X)), replace=False)
    for k in candidates:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        sil = silhouette_score(X[sample_idx], km.labels_[sample_idx])
        scores[k] = sil
        print(f"  k={k}: silhouette={sil:.4f}")
    return scores


def validate_clusters(profile: pd.DataFrame, labels: np.ndarray, dept_cols: list) -> dict:
    """Confirm store nodes actually differ in demand mix, not just noise."""
    profile = profile.copy()
    profile["store_id"] = labels

    # Kruskal-Wallis per department share: are distributions different across stores?
    kw_results = {}
    for col in dept_cols:
        groups = [g[col].values for _, g in profile.groupby("store_id")]
        stat, p = stats.kruskal(*groups)
        kw_results[col] = {"H": float(stat), "p": float(p)}
    n_significant = sum(1 for v in kw_results.values() if v["p"] < 0.01)

    # Between-cluster vs within-cluster variance ratio on the department-share space (pseudo-F)
    X = profile[dept_cols].values
    overall_mean = X.mean(axis=0)
    ss_between, ss_within = 0.0, 0.0
    for sid, g in profile.groupby("store_id"):
        gm = g[dept_cols].values.mean(axis=0)
        ss_between += len(g) * np.sum((gm - overall_mean) ** 2)
        ss_within += np.sum((g[dept_cols].values - gm) ** 2)
    pseudo_f = (ss_between / (N_STORES - 1)) / (ss_within / (len(profile) - N_STORES))

    cluster_summary = (
        profile.groupby("store_id")
        .agg(n_users=("n_orders", "size"), avg_basket_size=("avg_basket_size", "mean"),
             avg_orders=("n_orders", "mean"), avg_days_between=("avg_days_between", "mean"))
        .round(2)
        .to_dict(orient="index")
    )

    return {
        "n_stores": N_STORES,
        "n_departments_significant_at_p01": n_significant,
        "n_departments_total": len(dept_cols),
        "pseudo_f_stat": float(pseudo_f),
        "kruskal_wallis_by_department": kw_results,
        "cluster_summary": cluster_summary,
    }


def main():
    con = duckdb.connect(str(DB_PATH))
    print("Building per-user demand profiles...")
    profile = build_user_profiles(con)
    dept_cols = [c for c in profile.columns if c.startswith("dept_share_")]
    behavior_cols = ["n_orders", "avg_days_between", "modal_dow", "avg_hour", "avg_basket_size"]

    scaler = StandardScaler()
    X = scaler.fit_transform(profile[dept_cols + behavior_cols].values)

    print(f"Evaluating candidate store counts on {len(profile):,} users...")
    k_scores = choose_k(X)

    print(f"\nFitting final KMeans with k={N_STORES}...")
    km = KMeans(n_clusters=N_STORES, random_state=RANDOM_STATE, n_init=10).fit(X)
    profile["store_id"] = km.labels_

    print("Validating cluster distinctness (Kruskal-Wallis per department, pseudo-F)...")
    validation = validate_clusters(profile, km.labels_, dept_cols)
    validation["k_selection_silhouette_scores"] = k_scores
    print(f"  {validation['n_departments_significant_at_p01']}/{validation['n_departments_total']} "
          f"departments differ significantly across stores (p<0.01)")
    print(f"  pseudo-F statistic: {validation['pseudo_f_stat']:.1f} (higher = more distinct nodes)")

    store_map = profile[["store_id"]].reset_index()
    con.execute("CREATE OR REPLACE TABLE user_store_map AS SELECT * FROM store_map")
    con.execute("CREATE INDEX IF NOT EXISTS idx_usm_user ON user_store_map(user_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_usm_store ON user_store_map(store_id)")

    store_sizes = con.execute(
        "SELECT store_id, count(*) n_users FROM user_store_map GROUP BY store_id ORDER BY store_id"
    ).fetchdf()
    print("\nStore sizes:")
    print(store_sizes.to_string(index=False))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    print(f"\nSaved validation report -> {REPORT_PATH}")
    con.close()


if __name__ == "__main__":
    main()
