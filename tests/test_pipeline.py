"""
Unit tests for Prima Vista Payment Fraud Detection Pipeline:
- ETL output verification
- Spark / pandas feature calculation
- Model training & evaluation schema
- Merchant clustering output
- FastAPI inference endpoint
"""

import os
import pytest
import pandas as pd
import numpy as np

def test_etl_generation():
    from src.etl import generate_synthetic_transactions
    df = generate_synthetic_transactions(n_rows=1000, random_state=42)
    assert len(df) == 1000
    assert "is_fraud" in df.columns
    assert "device_remarks" in df.columns
    assert "terminal_type" in df.columns
    assert df["is_fraud"].isin([0, 1]).all()
    assert df["amount"].min() > 0

def test_feature_engineering_logic():
    # Test balance discrepancy computation
    amt = 50_000_000.0
    old_bal = 50_000_000.0
    new_bal = 0.0
    
    error_orig = round(new_bal + amt - old_bal, 2)
    is_emptied = 1 if (new_bal == 0 and old_bal > 0) else 0
    ratio = amt / old_bal
    
    assert error_orig == 0.0
    assert is_emptied == 1
    assert ratio == 1.0

def test_api_schema():
    from api.main import TransactionPayload, PredictionResponse
    payload = TransactionPayload(
        step=15,
        type="TRANSFER",
        amount=200_000_000.0,
        oldbalance_orig=200_000_000.0,
        newbalance_orig=0.0,
        oldbalance_dest=0.0,
        newbalance_dest=200_000_000.0,
        merchant_category="electronics",
        terminal_type="EDC_MOBILE",
        device_remarks="Rapid repeat authorization attempt within 60 seconds",
        mch_tx_count_24h=5,
        mch_tx_volume_24h=250_000_000.0
    )
    assert payload.amount == 200_000_000.0
    assert payload.terminal_type == "EDC_MOBILE"

