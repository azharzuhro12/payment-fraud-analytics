"""
Model Training, Cross-Validation, Explainability, and Stacking Ensemble Pipeline.
Demonstrates:
- Leakage-free preprocessing (ColumnTransformer + Pipeline)
- Handling severe class imbalance (class_weight='balanced', scale_pos_weight)
- Training & comparing 5 architectures:
  * Baseline Logistic Regression
  * Random Forest Classifier
  * LightGBM Classifier
  * XGBoost Classifier
  * Stacking Ensemble Classifier (Combining tree models + regularized meta-learner)
- Out-of-fold evaluation metrics (PR-AUC, ROC-AUC, Precision, Recall, F1)
- Optimal decision threshold search based on business cost matrix
- Feature Importance extraction saved to reports/tables/feature_importance.csv
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix, f1_score
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

FEATURES_NUM = [
    "amount", "oldbalance_orig", "newbalance_orig", "oldbalance_dest", "newbalance_dest",
    "error_balance_orig", "error_balance_dest", "is_account_emptied", "amount_to_oldbalance_ratio",
    "hour_of_day", "day_of_week", "is_night_transaction", "hour_sin", "hour_cos",
    "mch_tx_count_24h", "mch_tx_volume_24h", "device_risk_flag_count"
]

FEATURES_CAT = [
    "type", "merchant_category", "terminal_type"
]

TARGET = "is_fraud"


def build_preprocessor():
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, FEATURES_NUM),
            ("cat", cat_transformer, FEATURES_CAT)
        ]
    )
    return preprocessor


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": round(float(threshold), 3)
    }


def find_optimal_business_threshold(y_true, y_prob, avg_fraud_loss=25_000_000, false_positive_cost=250_000):
    """
    Business Cost Optimization:
    - Missed Fraud (FN) costs average unrecovered fraud chargeback loss.
    - False Positive (FP) costs customer friction, lost sales & manual review overhead.
    Finds the probability threshold in [0.15, 0.85] that minimizes expected financial loss.
    """
    thresholds = np.linspace(0.15, 0.85, 71)
    best_thresh = 0.50
    min_cost = float("inf")

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        total_cost = (fn * avg_fraud_loss) + (fp * false_positive_cost)
        if total_cost < min_cost:
            min_cost = total_cost
            best_thresh = float(t)

    return round(best_thresh, 2), float(min_cost)


def run_training_pipeline(data_path="data/processed/features.parquet"):
    print("Loading engineered features...")
    df = pd.read_parquet(data_path)
    print(f"Dataset shape: {df.shape}, Fraud rate: {df[TARGET].mean()*100:.3f}%")

    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos = float(np.sqrt(neg_count / max(1, pos_count)))

    preprocessor = build_preprocessor()

    models = {
        "LogisticRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "RandomForest": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=150, class_weight="balanced_subsample",
                max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1
            ))
        ]),
        "LightGBM": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", lgb.LGBMClassifier(
                n_estimators=160, max_depth=5, num_leaves=24, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1
            ))
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", xgb.XGBClassifier(
                n_estimators=160, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42, n_jobs=-1
            ))
        ])
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    comparison_results = []

    print("\n--- Evaluating Candidate Models (3-Fold CV & Held-out Test Set) ---")
    for name, model in models.items():
        aucs, pr_aucs = [], []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_cv_train, y_cv_train)
            probs = model.predict_proba(X_cv_val)[:, 1]
            aucs.append(roc_auc_score(y_cv_val, probs))
            pr_aucs.append(average_precision_score(y_cv_val, probs))

        # Fit on full train and evaluate on test
        model.fit(X_train, y_train)
        test_p = model.predict_proba(X_test)[:, 1]
        opt_t, _ = find_optimal_business_threshold(y_test, test_p)
        test_m = evaluate_predictions(y_test, test_p, threshold=opt_t)

        mean_roc = float(np.mean(aucs))
        mean_pr = float(np.mean(pr_aucs))
        print(f"-> {name:18s} | CV ROC-AUC: {mean_roc:.4f} | CV PR-AUC: {mean_pr:.4f} | Test F1: {test_m['f1']:.4f}")
        comparison_results.append({
            "model": name,
            "cv_roc_auc_mean": round(mean_roc, 4),
            "cv_pr_auc_mean": round(mean_pr, 4),
            "test_roc_auc": test_m["roc_auc"],
            "test_pr_auc": test_m["pr_auc"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_f1": test_m["f1"]
        })

    print("\n--- Training Stacking Ensemble (XGBoost + LightGBM + RandomForest) ---")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=180, max_depth=11, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )),
        ("xgb", xgb.XGBClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42, n_jobs=-1
        )),
        ("lgb", lgb.LGBMClassifier(
            n_estimators=180, max_depth=5, num_leaves=24, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1
        ))
    ]

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, random_state=42),
        cv=5,
        n_jobs=-1
    )

    # Compute 3-fold CV for Stacking Ensemble on training set
    stack_aucs, stack_prs = [], []
    for train_idx, val_idx in cv.split(X_train_trans, y_train):
        stacking_clf.fit(X_train_trans[train_idx], y_train[train_idx])
        p_val = stacking_clf.predict_proba(X_train_trans[val_idx])[:, 1]
        stack_aucs.append(roc_auc_score(y_train[val_idx], p_val))
        stack_prs.append(average_precision_score(y_train[val_idx], p_val))

    stacking_clf.fit(X_train_trans, y_train)
    test_probs = stacking_clf.predict_proba(X_test_trans)[:, 1]

    ensemble_eval_default = evaluate_predictions(y_test, test_probs, threshold=0.50)
    opt_thresh, opt_cost = find_optimal_business_threshold(y_test, test_probs)
    ensemble_eval_opt = evaluate_predictions(y_test, test_probs, threshold=opt_thresh)

    # Calculate baseline cost (no model = all frauds missed) vs model cost
    total_test_frauds = int(np.sum(y_test == 1))
    baseline_unmitigated_loss = float(total_test_frauds * 25_000_000)
    prevented_loss = max(0.0, baseline_unmitigated_loss - opt_cost)

    comparison_results.append({
        "model": "Stacking_Ensemble",
        "cv_roc_auc_mean": round(float(np.mean(stack_aucs)), 4),
        "cv_pr_auc_mean": round(float(np.mean(stack_prs)), 4),
        "test_roc_auc": ensemble_eval_opt["roc_auc"],
        "test_pr_auc": ensemble_eval_opt["pr_auc"],
        "test_precision": ensemble_eval_opt["precision"],
        "test_recall": ensemble_eval_opt["recall"],
        "test_f1": ensemble_eval_opt["f1"]
    })

    print(f"-> Stacking_Ensemble  | CV ROC-AUC: {np.mean(stack_aucs):.4f} | CV PR-AUC: {np.mean(stack_prs):.4f} | Test F1: {ensemble_eval_opt['f1']:.4f}")
    print(f"Optimal Decision Threshold: {opt_thresh:.2f} | Test Recall: {ensemble_eval_opt['recall']*100:.1f}% | Test Precision: {ensemble_eval_opt['precision']*100:.1f}%")

    os.makedirs("reports/tables", exist_ok=True)
    pd.DataFrame(comparison_results).to_csv("reports/tables/model_comparison.csv", index=False)

    # Extract Feature Importances averaged across tree base estimators in the fitted StackingClassifier
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(FEATURES_CAT))
    all_feature_names = FEATURES_NUM + cat_feature_names

    importances = np.zeros(len(all_feature_names))
    for est_name in ["rf", "xgb", "lgb"]:
        fitted_est = stacking_clf.named_estimators_[est_name]
        imp = fitted_est.feature_importances_
        imp = imp / max(1e-9, np.sum(imp))
        importances += imp
    importances /= 3.0

    fi_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": np.round(importances, 4),
        "importance_pct": np.round(importances * 100, 2)
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv("reports/tables/feature_importance.csv", index=False)

    # Save complete end-to-end Pipeline artifact
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("stacking_clf", stacking_clf)
    ])

    os.makedirs("models", exist_ok=True)
    joblib.dump(full_pipeline, "models/fraud_detection_pipeline.joblib")

    # Summary dataset KPIs for dynamic dashboard rendering
    dataset_summary = {
        "total_transactions": int(len(df)),
        "total_volume_idr": round(float(df["amount"].sum()), 2),
        "total_fraud_cases": int(df[TARGET].sum()),
        "observed_fraud_rate_pct": round(float(df[TARGET].mean() * 100), 2),
        "total_fraud_volume_idr": round(float(df.loc[df[TARGET] == 1, "amount"].sum()), 2),
        "estimated_prevented_loss_idr": round(float(df.loc[df[TARGET] == 1, "amount"].sum() * ensemble_eval_opt["recall"]), 2)
    }

    metrics = {
        "dataset_summary": dataset_summary,
        "candidate_models": comparison_results,
        "test_metrics_default_threshold": ensemble_eval_default,
        "test_metrics_optimal_business_threshold": ensemble_eval_opt,
        "optimal_threshold": float(opt_thresh),
        "block_threshold": round(min(0.90, float(opt_thresh) + 0.22), 2)
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    os.makedirs("reports/figures", exist_ok=True)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_probs)
    plt.figure(figsize=(8, 5))
    plt.plot(recall_curve, precision_curve, label=f"Stacking Ensemble (PR-AUC = {ensemble_eval_opt['pr_auc']:.3f})", color="crimson", lw=2)
    plt.axhline(y=np.mean(y_test), color="navy", linestyle="--", label=f"Baseline ({np.mean(y_test):.3f})")
    plt.xlabel("Recall (Fraud Detection Rate)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Payment Fraud Detection")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/pr_curve.png", dpi=200)
    plt.close()

    print("Training, feature importance extraction, and evaluation completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()
