#!/usr/bin/env bash
# Runs the full ReorderIQ pipeline end to end, in dependency order.
# Usage: ./run_pipeline.sh   (from the project root, with data/raw/*.csv present)
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

echo "== 1/8 Ingest raw CSVs into DuckDB =="
python3 src/reorderiq/ingest/build_db.py

echo "== 2/8 Build reorder-prediction feature store =="
python3 src/reorderiq/features/build_features.py

echo "== 3/8 Train reorder model =="
python3 src/reorderiq/models/train_reorder_model.py

echo "== 4/8 Cluster users into synthetic stores =="
python3 src/reorderiq/features/build_synthetic_stores.py

echo "== 5/8 Build hierarchical demand + forecast =="
python3 src/reorderiq/features/build_hierarchical_demand.py

echo "== 6/8 Compute MEIO safety stock / reorder points =="
python3 src/reorderiq/meio/build_meio.py

echo "== 7/8 Mine association rules =="
python3 src/reorderiq/rules/build_association_rules.py

echo "== 8/8 Run control tower + consolidate dashboard data =="
python3 src/reorderiq/agents/control_tower.py
python3 src/reorderiq/dashboard_data.py

echo
echo "Done. Open dashboard/index.html in a browser to view the results."
