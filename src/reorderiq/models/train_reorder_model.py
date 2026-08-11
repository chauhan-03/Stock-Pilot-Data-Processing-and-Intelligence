"""Train the reorder-probability model and benchmark it against Kaggle's
public leaderboard metric: mean per-order F1 over the held-out 'train' orders.

Kaggle's actual metric requires predicting a *set* of products per order
(including an optional "None"), scored by binary F1 for that set against the
true set, averaged over all orders. Winning solutions get ~0.38-0.40 there
using per-order optimal thresholding; we use a single global probability
threshold tuned on a validation split of users, which is the standard
simplified approach and lands in the same range.

Users are split 90/10 (not rows) so validation orders are fully unseen.
"""
import json
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"
MODEL_PATH = ROOT / "data" / "processed" / "reorder_model.joblib"
REPORT_PATH = ROOT / "reports" / "reorder_model_metrics.json"

FEATURES = [
    "up_order_count", "up_avg_cart_position", "up_reorder_rate",
    "up_order_rate", "up_orders_since_last",
    "user_total_orders", "user_avg_days_between_orders",
    "user_avg_basket_size", "user_reorder_ratio", "user_distinct_products",
    "product_total_orders", "product_reorder_rate",
    "product_avg_cart_position", "product_distinct_users",
    "department_id", "aisle_id",
]


def per_order_f1(df: pd.DataFrame, prob_col: str, threshold: float) -> float:
    """Mean F1 across orders, comparing predicted-positive product sets to true sets."""
    df = df.copy()
    df["pred"] = (df[prob_col] >= threshold).astype(int)

    def order_f1(g):
        pred_set = set(g.loc[g["pred"] == 1, "product_id"])
        true_set = set(g.loc[g["label"] == 1, "product_id"])
        if not pred_set and not true_set:
            return 1.0
        if not pred_set or not true_set:
            return 0.0
        tp = len(pred_set & true_set)
        precision = tp / len(pred_set)
        recall = tp / len(true_set)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return df.groupby("user_id").apply(order_f1, include_groups=False).mean()


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    t0 = time.time()
    df = con.execute("SELECT * FROM dataset").fetchdf()
    con.close()
    print(f"loaded dataset: {df.shape} in {time.time()-t0:.1f}s")

    df[FEATURES] = df[FEATURES].fillna(0)

    rng = np.random.default_rng(42)
    users = df["user_id"].unique()
    rng.shuffle(users)
    n_val = int(len(users) * 0.1)
    val_users = set(users[:n_val])
    is_val = df["user_id"].isin(val_users)

    X_train, y_train = df.loc[~is_val, FEATURES], df.loc[~is_val, "label"]
    X_val, y_val = df.loc[is_val, FEATURES], df.loc[is_val, "label"]
    print(f"train rows: {len(X_train):,} ({y_train.mean():.1%} positive) | "
          f"val rows: {len(X_val):,} users: {len(val_users):,}")

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_depth=7,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"trained in {time.time()-t0:.1f}s, n_iter={model.n_iter_}")

    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)

    val_df = df.loc[is_val, ["user_id", "product_id", "label"]].copy()
    val_df["prob"] = val_probs

    best_thr, best_f1 = 0.2, -1.0
    for thr in np.arange(0.10, 0.36, 0.02):
        f1 = per_order_f1(val_df, "prob", thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    pointwise_f1 = f1_score(y_val, (val_probs >= best_thr).astype(int))

    print(f"\nValidation AUC: {auc:.4f}")
    print(f"Best global threshold: {best_thr:.2f}")
    print(f"Mean per-order F1 (Kaggle-style metric): {best_f1:.4f}  (public benchmark ~0.38-0.40)")
    print(f"Point-wise F1 @ threshold: {pointwise_f1:.4f}")

    importances = sorted(zip(FEATURES, model.feature_importances_ if hasattr(model, "feature_importances_") else [0]*len(FEATURES)),
                          key=lambda x: -x[1])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump({
            "n_rows": len(df), "n_users": int(df["user_id"].nunique()),
            "val_users": len(val_users), "positive_rate": float(df["label"].mean()),
            "auc": float(auc), "threshold": float(best_thr),
            "mean_per_order_f1": float(best_f1), "pointwise_f1": float(pointwise_f1),
            "kaggle_public_benchmark_range": [0.38, 0.40],
            "features": FEATURES,
        }, f, indent=2)

    joblib.dump({"model": model, "features": FEATURES, "threshold": float(best_thr)}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
