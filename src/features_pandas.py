"""
Pandas-based Feature Engineering Engine.
Provides identical mathematical feature transformations, cyclical encoding,
rolling window aggregations, and unstructured text signal processing as PySpark.
Ensures zero-dependency execution while keeping PySpark available for big-data deployments.
"""

import os
import pandas as pd
import numpy as np

def engineer_features(input_parquet="data/raw/transactions.parquet", output_parquet="data/processed/features.parquet"):
    print(f"Loading data from {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df):,} transactions.")
    
    # 1. Balance error & account draining logic
    print("Calculating balance error metrics...")
    df["error_balance_orig"] = np.round(df["newbalance_orig"] + df["amount"] - df["oldbalance_orig"], 2)
    df["error_balance_dest"] = np.round(df["oldbalance_dest"] + df["amount"] - df["newbalance_dest"], 2)
    df["is_account_emptied"] = ((df["newbalance_orig"] == 0) & (df["oldbalance_orig"] > 0)).astype(int)
    
    df["amount_to_oldbalance_ratio"] = np.where(
        df["oldbalance_orig"] > 0,
        df["amount"] / df["oldbalance_orig"],
        999.0
    )
    
    # 2. Temporal & cyclical features
    print("Computing cyclical time features...")
    df["hour_of_day"] = df["step"] % 24
    df["day_of_week"] = (df["step"] // 24) % 7
    df["is_night_transaction"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 5)).astype(int)
    
    pi = np.pi
    df["hour_sin"] = np.sin(2 * pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * pi * df["hour_of_day"] / 24.0)
    
    # 3. Rolling merchant window aggregations (Velocity & Volume)
    print("Computing rolling merchant velocity metrics...")
    df = df.sort_values(by=["name_dest", "step"]).reset_index(drop=True)
    
    # Grouped rolling 24-step window per merchant
    df["mch_tx_count_24h"] = df.groupby("name_dest")["amount"].rolling(window=24, min_periods=1).count().values
    df["mch_tx_volume_24h"] = np.round(df.groupby("name_dest")["amount"].rolling(window=24, min_periods=1).sum().values, 2)
    
    # 4. Unstructured text log processing (NLP keyword extraction)
    print("Extracting NLP risk indicators from unstructured device remarks...")
    suspicious_keywords = ["fallback", "repeat", "spike", "emptied", "anomaly", "flagged"]
    remarks_lower = df["device_remarks"].str.lower().fillna("")
    
    flags = np.zeros(len(df), dtype=int)
    for kw in suspicious_keywords:
        flags += remarks_lower.str.contains(kw).astype(int)
    df["device_risk_flag_count"] = flags
    
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Feature engineering complete! Saved {len(df):,} records with {df.shape[1]} columns to {output_parquet}")

if __name__ == "__main__":
    engineer_features()
