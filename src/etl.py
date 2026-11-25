"""
Payment Transaction & Fraud Detection ETL Pipeline.
Generates / streams realistic transactional logs mimicking EDC, QRIS & Online Payment Gateways,
with distinct merchant behavioral archetypes and realistic non-linear fraud signatures.
Stores raw transactions into SQLite & Parquet for downstream Big Data & ML pipelines.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse

RAW_DATA_PATH = "data/raw/transactions.csv"
DB_PATH = "data/payments.db"

MERCHANT_CATEGORIES = [
    "retail_grocery", "food_beverage", "electronics", "fashion_apparel",
    "fuel_gas_station", "digital_services", "travel_hospitality"
]

LOCATIONS = [
    "Jakarta Selatan", "Jakarta Pusat", "Surabaya", "Bandung",
    "Medan", "Denpasar", "Makassar", "Semarang"
]

TERMINAL_TYPES = ["EDC_STATIC", "EDC_MOBILE", "QRIS_STATIC", "QRIS_DYNAMIC", "ONLINE_PG"]


def generate_synthetic_transactions(n_rows: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic payment transaction dataset with:
    1. Four latent merchant behavioral archetypes (Enterprise, Night-Active Hospitality,
       Mobile POS / High-Velocity, and Standard Everyday Retail) so unsupervised K-Means
       discovers meaningful, well-separated clusters.
    2. Subtle, non-linear multi-factor fraud patterns without trivial 1-to-1 target leakage,
       producing realistic industry ROC-AUC (~0.94-0.97) and PR-AUC (~0.78-0.88) where
       tree ensembles & stacking outperform linear baselines.
    """
    rng = np.random.default_rng(random_state)

    n_merchants = 1_500
    merchant_ids = np.array([f"MCH_{i:05d}" for i in range(1, n_merchants + 1)])

    # Assign each merchant to one of 4 behavioral archetypes:
    # 0: High-Volume Enterprise Retailers (22%)
    # 1: Night-Active Hospitality & Entertainment (20%)
    # 2: Mobile POS & E-Commerce High-Velocity Merchants (23%)
    # 3: Standard Everyday Retail POS Merchants (35%)
    archetypes = rng.choice([0, 1, 2, 3], size=n_merchants, p=[0.22, 0.20, 0.23, 0.35])
    mch_archetype_map = dict(zip(merchant_ids, archetypes))

    # Sampling weights per merchant so Enterprise & Mobile POS have higher transaction frequency
    mch_weights = np.where(archetypes == 0, 2.2,
                  np.where(archetypes == 2, 1.4,
                  np.where(archetypes == 1, 0.9, 0.7)))
    mch_weights = mch_weights / mch_weights.sum()

    name_dest = rng.choice(merchant_ids, size=n_rows, p=mch_weights)
    tx_archetype = np.array([mch_archetype_map[m] for m in name_dest])

    # Customers
    n_customers = max(500, int(n_rows * 0.4))
    customer_ids = np.array([f"CUST_{i:07d}" for i in range(1, n_customers + 1)])
    name_orig = rng.choice(customer_ids, size=n_rows)

    # Generate hour of day and step (30 days = 720 hours) conditioned on archetype
    days = rng.integers(0, 30, size=n_rows)
    hours = np.zeros(n_rows, dtype=int)

    for arch in range(4):
        mask = tx_archetype == arch
        cnt = mask.sum()
        if cnt == 0:
            continue
        if arch == 1:
            # Night-Active Hospitality: 55% nocturnal (00:00 - 05:00), 45% evening/day
            h_probs = np.array([0.11, 0.11, 0.10, 0.09, 0.08, 0.06] + [0.015]*12 + [0.045]*6)
        elif arch == 0:
            # Enterprise Retailers: mostly business hours (08:00 - 21:00)
            h_probs = np.array([0.01]*6 + [0.03, 0.04] + [0.065]*12 + [0.03, 0.02, 0.015, 0.015])
        elif arch == 2:
            # Mobile POS / Online: broad distribution with moderate night activity
            h_probs = np.array([0.035]*6 + [0.04]*12 + [0.0516]*6)
        else:
            # Standard Everyday Retail: daytime peak
            h_probs = np.array([0.008]*6 + [0.035, 0.045] + [0.068]*12 + [0.025, 0.015, 0.008, 0.008])
        h_probs = h_probs / h_probs.sum()
        hours[mask] = rng.choice(24, size=cnt, p=h_probs)

    steps = np.clip(days * 24 + hours + 1, 1, 720)
    order = np.argsort(steps)
    steps = steps[order]
    hours = hours[order]
    name_dest = name_dest[order]
    name_orig = name_orig[order]
    tx_archetype = tx_archetype[order]

    # Terminal types and amounts conditioned on merchant archetype
    terminals = np.empty(n_rows, dtype=object)
    merchant_cats = np.empty(n_rows, dtype=object)
    amounts = np.zeros(n_rows, dtype=float)

    for arch in range(4):
        mask = tx_archetype == arch
        cnt = mask.sum()
        if cnt == 0:
            continue
        if arch == 0:  # High-Volume Enterprise
            terminals[mask] = rng.choice(TERMINAL_TYPES, size=cnt, p=[0.62, 0.10, 0.12, 0.10, 0.06])
            merchant_cats[mask] = rng.choice(["electronics", "fashion_apparel", "retail_grocery"], size=cnt, p=[0.45, 0.35, 0.20])
            amounts[mask] = np.round(rng.lognormal(mean=14.6, sigma=1.1, size=cnt), 2)
        elif arch == 1:  # Night-Active Hospitality & Entertainment
            terminals[mask] = rng.choice(TERMINAL_TYPES, size=cnt, p=[0.30, 0.22, 0.18, 0.22, 0.08])
            merchant_cats[mask] = rng.choice(["food_beverage", "travel_hospitality", "digital_services"], size=cnt, p=[0.50, 0.35, 0.15])
            amounts[mask] = np.round(rng.lognormal(mean=12.8, sigma=1.15, size=cnt), 2)
        elif arch == 2:  # Mobile POS & Online High-Risk/Velocity
            terminals[mask] = rng.choice(TERMINAL_TYPES, size=cnt, p=[0.06, 0.52, 0.07, 0.07, 0.28])
            merchant_cats[mask] = rng.choice(["electronics", "digital_services", "fashion_apparel"], size=cnt, p=[0.40, 0.35, 0.25])
            amounts[mask] = np.round(rng.lognormal(mean=13.3, sigma=1.35, size=cnt), 2)
        else:  # Standard Everyday Retail
            terminals[mask] = rng.choice(TERMINAL_TYPES, size=cnt, p=[0.50, 0.08, 0.32, 0.08, 0.02])
            merchant_cats[mask] = rng.choice(["retail_grocery", "food_beverage", "fuel_gas_station"], size=cnt, p=[0.50, 0.30, 0.20])
            amounts[mask] = np.round(rng.lognormal(mean=11.7, sigma=0.95, size=cnt), 2)

    # Transaction types & locations
    types = rng.choice(
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
        size=n_rows,
        p=[0.38, 0.22, 0.20, 0.12, 0.08]
    )
    locations = rng.choice(LOCATIONS, size=n_rows, p=[0.24, 0.18, 0.15, 0.13, 0.09, 0.08, 0.07, 0.06])

    # Customer & merchant balances (note: some legitimate customers also spend 90-100% of wallet balance)
    balance_multiplier = rng.uniform(0.92, 4.8, size=n_rows)
    oldbalance_orig = np.round(amounts * balance_multiplier, 2)
    newbalance_orig = np.round(np.maximum(0.0, oldbalance_orig - amounts), 2)

    oldbalance_dest = np.round(rng.lognormal(mean=15.0, sigma=1.4, size=n_rows), 2)
    # Introduce small realistic settlement delay / fee discrepancy on both normal and fraud
    settlement_delta = rng.normal(0, 0.015, size=n_rows) * amounts
    newbalance_dest = np.round(np.maximum(0.0, oldbalance_dest + amounts + settlement_delta), 2)

    # Non-linear multi-factor fraud probability (target ~1.1% - 1.5% fraud rate)
    # Combines interaction terms: (High Ratio + Transfer/CashOut), (Nocturnal + Mobile/Online), (High Amount + Balance Error)
    ratio = amounts / np.maximum(1.0, oldbalance_orig)
    is_night = (hours >= 0) & (hours <= 5)
    is_risky_type = np.isin(types, ["TRANSFER", "CASH_OUT"])
    is_risky_term = np.isin(terminals, ["EDC_MOBILE", "ONLINE_PG"])
    is_high_amt = amounts > np.percentile(amounts, 88)

    logit = np.full(n_rows, -6.3)
    logit += np.where(is_risky_type, 1.35, -0.4)
    logit += np.where(is_risky_term, 1.15, -0.2)
    logit += np.where(is_night, 1.10, 0.0)
    logit += np.where(is_high_amt, 1.25, 0.0)
    logit += np.clip((ratio - 0.65) * 3.2, -0.5, 1.8)
    # Non-linear interaction boosts (tree models excel at capturing these over linear models)
    logit += np.where(is_risky_type & is_night & is_risky_term, 1.45, 0.0)
    logit += np.where(is_high_amt & (ratio > 0.88), 1.30, 0.0)

    prob_fraud = 1.0 / (1.0 + np.exp(-logit))
    is_fraud = (rng.random(n_rows) < prob_fraud).astype(int)

    # For fraud transactions, only *some* (~55%) drain the balance completely and some introduce accounting discrepancy
    for i in np.where(is_fraud == 1)[0]:
        if rng.random() < 0.55:
            newbalance_orig[i] = 0.0
        if rng.random() < 0.45:
            # Unsettled / rerouted destination balance discrepancy
            newbalance_dest[i] = np.round(oldbalance_dest[i] + amounts[i] * rng.uniform(0.1, 0.65), 2)

    # Unstructured device telemetry remarks with realistic overlap (not 100% deterministic)
    normal_remarks = [
        "Normal purchase authorization OK",
        "Chip read successful PIN verified",
        "Contactless card wave verified via NFC",
        "QRIS static scan success settlement pending",
        "Regular customer monthly bill payment settlement",
        "Standard retail POS swipe approved"
    ]
    ambiguous_remarks = [
        "Terminal retry after network timeout normal settlement",
        "Manual key-in fallback approved with customer PIN",
        "High traffic batch settlement queue delay"
    ]
    suspicious_remarks = [
        "Multiple fallback attempts detected before EMV fallback success",
        "Rapid repeat authorization attempt within 60 seconds",
        "Unusual transaction spike outside merchant business hours",
        "High value card-not-present transfer flagged by fraud rule engine",
        "Account balance emptied in single transaction attempt",
        "Geo-location anomaly between terminal IP and cardholder profile"
    ]

    remarks = []
    for f in is_fraud:
        r = rng.random()
        if f == 1:
            if r < 0.62:
                remarks.append(rng.choice(suspicious_remarks))
            elif r < 0.80:
                remarks.append(rng.choice(ambiguous_remarks))
            else:
                remarks.append(rng.choice(normal_remarks))
        else:
            if r < 0.045:
                remarks.append(rng.choice(suspicious_remarks))
            elif r < 0.14:
                remarks.append(rng.choice(ambiguous_remarks))
            else:
                remarks.append(rng.choice(normal_remarks))

    start_date = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(hours=int(s)) for s in steps]

    df = pd.DataFrame({
        "step": steps,
        "timestamp": timestamps,
        "type": types,
        "amount": amounts,
        "name_orig": name_orig,
        "oldbalance_orig": oldbalance_orig,
        "newbalance_orig": newbalance_orig,
        "name_dest": name_dest,
        "oldbalance_dest": oldbalance_dest,
        "newbalance_dest": newbalance_dest,
        "merchant_category": merchant_cats,
        "terminal_type": terminals,
        "location": locations,
        "device_remarks": remarks,
        "is_fraud": is_fraud
    })

    return df


def run_etl(n_rows: int = 50_000):
    print(f"Generating {n_rows:,} transactions...")
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = generate_synthetic_transactions(n_rows=n_rows)
    print(f"Dataset generated. Rows: {len(df):,}, Fraud cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")

    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved raw dataset to {RAW_DATA_PATH}")

    df.to_parquet("data/raw/transactions.parquet", index=False)
    print("Saved raw parquet to data/raw/transactions.parquet")

    print(f"Ingesting into database {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("transactions", conn, if_exists="replace", index=False)

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_step ON transactions(step);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_type ON transactions(type);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_mch ON transactions(name_dest);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_fraud ON transactions(is_fraud);")
    conn.commit()
    conn.close()
    print("Database ingestion & index creation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000, help="Number of rows to generate")
    args = parser.parse_args()
    run_etl(n_rows=args.rows)
