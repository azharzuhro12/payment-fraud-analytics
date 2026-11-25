#!/bin/bash
set -e

echo "=== [1/5] Running Payment ETL Data Pipeline ==="
python3 src/etl.py --rows 50000

echo "=== [2/5] Running Feature Engineering Engine ==="
python3 src/features_pandas.py

echo "=== [3/5] Running Merchant Behavioral Clustering (K-Means) ==="
python3 src/cluster.py

echo "=== [4/5] Training Models & Stacking Ensemble ==="
python3 src/train.py

echo "=== [5/5] Exporting Tableau BI Data & Running Tests ==="
python3 src/export_tableau.py
PYTHONPATH=. pytest tests/ -v

echo "=== Pipeline Completed Successfully! ==="
