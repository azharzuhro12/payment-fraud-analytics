"""
FastAPI Microservice for Real-Time Payment Fraud Scoring.
Enables instant integration with Payment Gateways & EDC Terminal Controllers.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os
import json

app = FastAPI(
    title="Real-Time Payment Fraud Detection Engine",
    description="Real-time transaction risk evaluation API powered by Stacking Ensemble ML",
    version="1.0.0"
)

MODEL_PATH = "models/fraud_detection_pipeline.joblib"
METRICS_PATH = "reports/metrics.json"

pipeline = None
optimal_threshold = 0.45
block_threshold = 0.70


@app.on_event("startup")
def load_model():
    global pipeline, optimal_threshold, block_threshold
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print("Model pipeline loaded successfully.")
    else:
        print("Model file not found. Running in mock/development mode.")

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                m = json.load(f)
                optimal_threshold = float(m.get("optimal_threshold", 0.45))
                block_threshold = float(m.get("block_threshold", 0.70))
        except Exception:
            pass


class TransactionPayload(BaseModel):
    step: int = Field(..., example=122, description="Hour step (1-720)")
    type: str = Field(..., example="TRANSFER", description="PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN")
    amount: float = Field(..., example=150000000.0, description="Transaction amount in IDR")
    oldbalance_orig: float = Field(..., example=150000000.0, description="Customer initial balance")
    newbalance_orig: float = Field(..., example=0.0, description="Customer balance after transaction")
    oldbalance_dest: float = Field(..., example=10000000.0, description="Merchant initial balance")
    newbalance_dest: float = Field(..., example=160000000.0, description="Merchant balance after transaction")
    merchant_category: str = Field(..., example="electronics", description="Merchant business category")
    terminal_type: str = Field(..., example="EDC_MOBILE", description="Terminal device type")
    device_remarks: Optional[str] = Field("Normal POS transaction", description="Terminal device telemetry log")
    mch_tx_count_24h: Optional[int] = Field(15, description="Count of merchant transactions in last 24h")
    mch_tx_volume_24h: Optional[float] = Field(250000000.0, description="Sum volume of merchant transactions in last 24h")


class PredictionResponse(BaseModel):
    fraud_risk_score: float
    decision: str
    recommended_action: str
    threshold_applied: float
    risk_factors: list[str]


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Payment Fraud Detection API",
        "model_loaded": pipeline is not None,
        "optimal_threshold": optimal_threshold,
        "block_threshold": block_threshold
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(payload: TransactionPayload):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not yet trained or loaded.")

    error_balance_orig = round(payload.newbalance_orig + payload.amount - payload.oldbalance_orig, 2)
    error_balance_dest = round(payload.oldbalance_dest + payload.amount - payload.newbalance_dest, 2)
    is_account_emptied = 1 if (payload.newbalance_orig == 0 and payload.oldbalance_orig > 0) else 0
    amount_to_oldbalance_ratio = payload.amount / payload.oldbalance_orig if payload.oldbalance_orig > 0 else 999.0
    hour_of_day = payload.step % 24
    day_of_week = (payload.step // 24) % 7
    is_night_transaction = 1 if (0 <= hour_of_day <= 5) else 0
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)

    suspicious_keywords = ["fallback", "repeat", "spike", "emptied", "anomaly", "flagged"]
    remarks_lower = (payload.device_remarks or "").lower()
    device_risk_flag_count = sum(1 for kw in suspicious_keywords if kw in remarks_lower)

    input_df = pd.DataFrame([{
        "amount": payload.amount,
        "oldbalance_orig": payload.oldbalance_orig,
        "newbalance_orig": payload.newbalance_orig,
        "oldbalance_dest": payload.oldbalance_dest,
        "newbalance_dest": payload.newbalance_dest,
        "error_balance_orig": error_balance_orig,
        "error_balance_dest": error_balance_dest,
        "is_account_emptied": is_account_emptied,
        "amount_to_oldbalance_ratio": amount_to_oldbalance_ratio,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_night_transaction": is_night_transaction,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "mch_tx_count_24h": payload.mch_tx_count_24h or 1,
        "mch_tx_volume_24h": payload.mch_tx_volume_24h or payload.amount,
        "device_risk_flag_count": device_risk_flag_count,
        "type": payload.type,
        "merchant_category": payload.merchant_category,
        "terminal_type": payload.terminal_type
    }])

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["stacking_clf"]

        X_trans = preprocessor.transform(input_df)
        prob = float(clf.predict_proba(X_trans)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    risk_factors = []
    if is_account_emptied or amount_to_oldbalance_ratio >= 0.90:
        risk_factors.append(f"High balance depletion ratio ({amount_to_oldbalance_ratio*100:.1f}% of origin balance)")
    if is_night_transaction:
        risk_factors.append("Transaction occurred during high-risk nocturnal window (00:00-05:00 WIB)")
    if device_risk_flag_count > 0:
        risk_factors.append(f"Device telemetry log contains {device_risk_flag_count} suspicious security keywords")
    if payload.amount > 100_000_000:
        risk_factors.append("High monetary value transaction (> IDR 100,000,000)")
    if payload.terminal_type in ["EDC_MOBILE", "ONLINE_PG"]:
        risk_factors.append(f"Elevated channel risk profile: {payload.terminal_type}")
    if abs(error_balance_dest) > 100_000:
        risk_factors.append("Destination settlement balance discrepancy detected")

    if prob >= block_threshold:
        decision = "BLOCK"
        action = "Immediately decline transaction, temporarily freeze terminal token, and notify risk monitoring desk."
    elif prob >= optimal_threshold:
        decision = "STEP_UP_CHALLENGE"
        action = "Trigger mandatory 2FA OTP / biometric confirmation on cardholder device before settlement."
    else:
        decision = "APPROVE"
        action = "Standard instant settlement authorization."

    return PredictionResponse(
        fraud_risk_score=round(prob, 4),
        decision=decision,
        recommended_action=action,
        threshold_applied=optimal_threshold,
        risk_factors=risk_factors
    )
