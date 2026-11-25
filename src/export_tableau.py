"""
Export Aggregated Analytics Datasets for Tableau Public:
- Geographic & Channel Fraud Summary
- Hourly Temporal Heatmap
- High-risk Merchant Watchlist
"""

import sqlite3
import pandas as pd
import os

def export_all():
    os.makedirs("data/tableau", exist_ok=True)
    conn = sqlite3.connect("data/payments.db")
    
    # 1. Channel & Terminal Risk
    df_channel = pd.read_sql_query("""
        SELECT 
            terminal_type,
            merchant_category,
            location,
            COUNT(*) AS total_transactions,
            ROUND(SUM(amount), 2) AS total_volume_idr,
            SUM(is_fraud) AS fraud_cases,
            ROUND(AVG(is_fraud) * 100.0, 3) AS fraud_rate_pct,
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_loss_idr
        FROM transactions
        GROUP BY terminal_type, merchant_category, location
    """, conn)
    df_channel.to_csv("data/tableau/channel_location_risk.csv", index=False)
    
    # 2. Hourly Risk Cycle
    df_hourly = pd.read_sql_query("""
        SELECT 
            (step % 24) AS hour_of_day,
            type,
            COUNT(*) AS total_tx,
            SUM(is_fraud) AS fraud_cases,
            ROUND(AVG(is_fraud) * 100.0, 3) AS fraud_rate_pct,
            ROUND(SUM(amount), 2) AS total_volume_idr
        FROM transactions
        GROUP BY (step % 24), type
    """, conn)
    df_hourly.to_csv("data/tableau/hourly_fraud_risk.csv", index=False)
    
    conn.close()
    print("Exported Tableau CSVs to data/tableau/")

if __name__ == "__main__":
    export_all()
