# ReorderIQ — A Multi-Node Inventory Control Tower on Instacart Data

**Dashboard:** [`dashboard/index.html`](dashboard/index.html) (open directly in a browser) — also published at
https://claude.ai/code/artifact/856050bb-7063-4cdb-8955-0186c250ebe6

## The framing

Instacart's Market Basket dataset gives real demand and reorder behavior at genuine scale (3.42M orders,
33.8M order-product rows, 206K users) — but it is one retailer's order history across *users*, not a
multi-location inventory feed. There is no store, no DC, no warehouse dimension, and no stock levels.

So this project is explicit about a split that most retail-AI case studies gloss over:

- **The demand layer is real, end to end.** The reorder-prediction model and the basket association
  rules are trained and computed directly on unmodified Instacart data — nothing simulated.
- **The network layer is constructed.** Since Instacart has no store/DC topology, one is built on top
  of the real demand signal by clustering users into synthetic "stores," and that construction is
  validated statistically rather than assumed to be meaningful.
- **Inventory itself is inferred, not observed.** There are no real stock levels anywhere in this
  dataset. "Inventory" here means demand-coverage targets computed from forecasted demand and safety
  stock — a modeling choice, stated up front, not hidden downstream.

Every number in the [dashboard](dashboard/index.html) is labeled real or simulated for exactly this reason.

## Architecture

| Layer | What it does | Built from | Code |
|---|---|---|---|
| **Demand Sensing** | Reorder probability per (user, product), rolled up to store × product demand | Reorder/cart features from real order history | [`models/train_reorder_model.py`](src/reorderiq/models/train_reorder_model.py) |
| **Network Construction** | 18 synthetic stores from user clustering, statistically validated as distinct | Department purchase-share + shopping cadence per user | [`features/build_synthetic_stores.py`](src/reorderiq/features/build_synthetic_stores.py) |
| **Hierarchical Rollup** | SKU → department-DC → store → network, bottom-up reconciled weekly forecast | Reconstructed relative calendar from `days_since_prior_order` | [`features/build_hierarchical_demand.py`](src/reorderiq/features/build_hierarchical_demand.py) |
| **MEIO** | Two-echelon safety stock & reorder points, risk-pooled at the DC | Demand variance per store, 95% service-level target | [`meio/build_meio.py`](src/reorderiq/meio/build_meio.py) |
| **Basket / Substitution** | Exact pairwise co-occurrence (complements) + FP-Growth (multi-item) + same-aisle low-lift pairs (substitutes) | `order_products` co-purchase data, full history, no sampling for pairwise rules | [`rules/build_association_rules.py`](src/reorderiq/rules/build_association_rules.py) |
| **Control Tower** | Flags store-SKU exceptions where the ML demand signal diverges from the historical baseline; recommends transfers and substitutes | Rollup of all layers above, rule-based orchestrator | [`agents/control_tower.py`](src/reorderiq/agents/control_tower.py) |

**Why no LangGraph/LLM agent graph:** every decision in the control tower (is this an exception? which
store can cover the deficit? which SKU substitutes?) is fully computable from the data. There's no
ambiguous judgment call an LLM would improve, and adding one would only add latency, cost, and an
API-key dependency. A thin LangGraph wrapper could sit on top of these functions as callable tools if a
future iteration needs natural-language exception summaries — noted as an extension, not required here.

## Results

| Metric | Value | Context |
|---|---|---|
| Reorder model — mean per-order F1 | **0.371** | Kaggle public leaderboard range: 0.38–0.40, with 16 hand-built features and no hyperparameter tuning |
| Reorder model — validation AUC | 0.831 | 90/10 user-level split, 853K held-out (user, product) rows |
| Synthetic store validation | 21/21 departments differ significantly (Kruskal-Wallis, p<0.01) | pseudo-F statistic 9,397 |
| MEIO — network safety stock reduction from DC-level pooling | **30.1%** | 190,832 → 133,325 units, 95% service level, 3,000 SKUs × 18 stores |
| Association rules scored (exact, full data) | 415,479 pairs | 700 same-aisle substitute candidates found |
| FP-Growth multi-item itemsets | 1,475 (min support 0.003) | 120K-order sample, top 250 products |
| Control tower exception rate | 4.9% | 2,397 of 49,256 store-SKU positions; 1,079 have a transfer donor, 821 have a substitute |

Full metrics, including per-department and per-store breakdowns, are in [`reports/`](reports/) as
JSON/CSV, and consolidated in [`reports/dashboard_data.json`](reports/dashboard_data.json).

## What's real vs. simulated (read this before trusting any number above)

1. **Real, unmodified:** all 3.42M orders, 33.8M order-product rows, reorder labels, product/aisle/department
   taxonomy. The reorder model and association rules are trained/computed directly on this.
2. **Simulated network topology:** the 18 "stores" are K-means clusters of users on department
   purchase-share + cadence — a defensible proxy for store format/demographic mix, statistically
   validated, but not real geography or real store operations. Store sizes range from 798 to 35,588
   users — some imbalance is expected from behavioral (not enforced) clustering.
3. **Simulated calendar:** Instacart gives only `days_since_prior_order` (a gap), not real dates. Weekly
   buckets are reconstructed via cumulative sum per user, starting each user at day 0 — relative spacing
   within a user's history is real; calendar alignment *across* users is not.
4. **Simulated lead times & service level:** 1-day store lead time, 3-day DC lead time, and a 95% target
   service level are stated modeling assumptions — Instacart provides no supplier or fulfillment data.
5. **Inferred, not observed, inventory:** "inventory" means demand-coverage targets (order-up-to levels),
   not observed shelf or warehouse quantities.
6. **Held-out test set is unusable:** Kaggle's `eval_set='test'` orders have labels withheld by design.
   All validation uses a 90/10 user-level split of the `train` eval set, so held-out orders are genuinely
   unseen by the model — this was checked explicitly to avoid the #1 way these projects silently leak.

## Project layout

```
data/raw/               Original Instacart CSVs (orders, order_products__{prior,train}, products, aisles, departments)
data/processed/         reorderiq.duckdb (all derived tables), reorder_model.joblib
src/reorderiq/
  ingest/                build_db.py                    Raw CSVs -> DuckDB
  features/               build_features.py              Reorder-prediction feature store
                          build_synthetic_stores.py      User clustering -> 18 synthetic stores
                          build_hierarchical_demand.py   Weekly demand hierarchy + bottom-up forecast
  models/                 train_reorder_model.py         Reorder model + Kaggle-style benchmark
  meio/                   build_meio.py                  Two-echelon safety stock / ROP
  rules/                  build_association_rules.py     Pairwise + FP-Growth association rules
  agents/                 control_tower.py               Exception detection + recommendations
  dashboard_data.py       Consolidates all reports/ into one JSON for the dashboard
dashboard/index.html    Self-contained static dashboard (no external dependencies)
reports/                All metrics, validation results, and rule tables (JSON/CSV)
run_pipeline.sh         Runs the full pipeline end to end, in order
```

## Reproducing this

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_pipeline.sh
```

Requires `data/raw/{orders,order_products__prior,order_products__train,products,aisles,departments}.csv`
from the [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)
Kaggle competition (free login required — the dataset is no longer available outside Kaggle).

## Known limitations / honest gaps

- **LightGBM was dropped** in favor of scikit-learn's `HistGradientBoostingClassifier` — this machine has
  no Homebrew/`libomp`, and LightGBM's compiled extension won't load without it. Architecturally the same
  gradient-boosted-trees approach; the F1 gap to the Kaggle benchmark (0.371 vs 0.38–0.40) is more likely
  attributable to minimal feature engineering (16 features, no tuning) than the model swap.
- **Risk-pooling assumes independence** across stores within a department for the DC-level pooled variance
  calculation. Real regional demand is correlated, which would shrink the 30.1% pooling benefit shown here.
- **The FP-Growth run samples** 120K of 3.42M orders and the top 250 of 49,688 products, purely for memory
  tractability (one-hot encoding the full dataset is not feasible). The pairwise association rules, which
  are the primary output, do not sample — they're computed exactly across the full history via SQL.
- **Store clustering silhouette scores are low** (0.05–0.08) — expected for behavioral/purchase-share
  clustering rather than spatially separated blobs, and it's why cluster distinctness was validated via
  Kruskal-Wallis and pseudo-F rather than trusting silhouette alone.
