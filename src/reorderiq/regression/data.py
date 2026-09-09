"""Assemble design matrices from the existing feature pipeline.

Reuses the 16 engineered features from build_features.py.
Applies the identical user-level 90/10 split as the HistGradientBoostingClassifier
for fair comparison.
"""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"

logger = logging.getLogger(__name__)

# The 16 features used by the HGB model
FEATURES = [
    "up_order_count", "up_avg_cart_position", "up_reorder_rate",
    "up_order_rate", "up_orders_since_last",
    "user_total_orders", "user_avg_days_between_orders",
    "user_avg_basket_size", "user_reorder_ratio", "user_distinct_products",
    "product_total_orders", "product_reorder_rate",
    "product_avg_cart_position", "product_distinct_users",
    "department_id", "aisle_id",
]

CONTINUOUS_FEATURES = [
    "up_order_count", "up_avg_cart_position", "up_reorder_rate",
    "up_order_rate", "up_orders_since_last",
    "user_total_orders", "user_avg_days_between_orders",
    "user_avg_basket_size", "user_reorder_ratio", "user_distinct_products",
    "product_total_orders", "product_reorder_rate",
    "product_avg_cart_position", "product_distinct_users",
]

CATEGORICAL_FEATURES = ["department_id", "aisle_id"]

RANDOM_STATE = 42


def load_dataset(sample_size=None, random_seed=RANDOM_STATE):
    """Load the full dataset from DuckDB.
    
    Args:
        sample_size: If set, return a reproducible random sample of this size.
        random_seed: Random seed for sampling.
    
    Returns:
        DataFrame with columns: user_id, product_id, label, and all 16 features.
        Index: not reset (row numbers preserved).
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM dataset").fetchdf()
    con.close()
    
    logger.info(f"Loaded dataset: {df.shape[0]:,} rows, {df['user_id'].nunique():,} users")
    
    if sample_size:
        if sample_size > len(df):
            logger.warning(f"sample_size {sample_size} > dataset size {len(df)}; using full dataset")
        else:
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(len(df), size=sample_size, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
            logger.info(f"Sampled {sample_size:,} rows (seed={random_seed})")
    
    return df


def train_val_split(df, val_fraction=0.1, random_seed=RANDOM_STATE):
    """Split data into train and validation sets at the user level.
    
    This matches the exact logic from train_reorder_model.py to ensure
    fair comparison between logistic regression and HGB.
    
    Args:
        df: DataFrame with user_id column.
        val_fraction: Fraction of unique users to hold for validation (default 0.1).
        random_seed: Random seed for reproducibility.
    
    Returns:
        (X_train, X_val, y_train, y_val, val_users, (n_train_rows, n_val_rows))
    """
    # Identify validation users
    rng = np.random.default_rng(random_seed)
    users = df["user_id"].unique()
    rng.shuffle(users)
    n_val = int(len(users) * val_fraction)
    val_users = set(users[:n_val])
    is_val = df["user_id"].isin(val_users)
    
    X_train = df.loc[~is_val, FEATURES].copy()
    y_train = df.loc[~is_val, "label"].copy()
    X_val = df.loc[is_val, FEATURES].copy()
    y_val = df.loc[is_val, "label"].copy()
    
    # Fill NaN with 0 for numeric features
    for feat in CONTINUOUS_FEATURES:
        if feat in X_train.columns:
            X_train[feat] = X_train[feat].fillna(0)
            X_val[feat] = X_val[feat].fillna(0)
    
    logger.info(
        f"Train: {len(X_train):,} rows ({y_train.mean():.1%} positive) | "
        f"Val: {len(X_val):,} rows ({y_val.mean():.1%} positive) | "
        f"Val users: {len(val_users):,}"
    )
    
    return X_train, X_val, y_train, y_val, val_users, (len(X_train), len(X_val))


def get_feature_groups():
    """Return feature categorization for diagnostics.
    
    Returns:
        dict with keys: 'continuous', 'categorical', 'user_level', 'product_level', 'interaction'
    """
    return {
        "continuous": CONTINUOUS_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "user_level": [
            "user_total_orders", "user_avg_days_between_orders",
            "user_avg_basket_size", "user_reorder_ratio", "user_distinct_products"
        ],
        "product_level": [
            "product_total_orders", "product_reorder_rate",
            "product_avg_cart_position", "product_distinct_users"
        ],
        "interaction": [
            "up_order_count", "up_avg_cart_position", "up_reorder_rate",
            "up_order_rate", "up_orders_since_last"
        ],
    }
