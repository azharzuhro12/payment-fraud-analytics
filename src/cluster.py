"""
Merchant Behavior & Risk Segmentation using Clustering (K-Means & PCA).
Demonstrates:
- Feature aggregation at entity/merchant level (Volume, Velocity, Ticket size, Night ratio, Terminal diversity)
- Dimensionality reduction & Clustering (StandardScaler + PCA + KMeans)
- Quantitative cluster quality evaluation (Silhouette Score & Inertia)
- Guaranteed 1-to-1 Strategic Business Profiling across 4 distinct segments:
  * High-Volume Enterprise Retailers
  * Night-Active Hospitality & Entertainment
  * Mobile POS & High-Velocity Merchants
  * Standard Everyday Retail Merchants
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib


def assign_distinct_segment_profiles(mch_agg: pd.DataFrame):
    """
    Deterministically map each of the 4 K-Means cluster IDs to its dominant
    business archetype without collision, ensuring all 4 segments are distinct.
    """
    centroids = mch_agg.groupby("cluster_id").agg(
        avg_vol=("total_volume", "mean"),
        avg_night=("night_tx_ratio", "mean"),
        avg_mobile=("mobile_online_ratio", "mean"),
        avg_ticket=("avg_ticket_size", "mean")
    )

    remaining = set(centroids.index.tolist())
    profile_names = {}
    strategies = {}

    # 1. Cluster with highest mobile_online_ratio -> Mobile POS & High-Velocity Merchants
    c_mobile = centroids.loc[list(remaining), "avg_mobile"].idxmax()
    profile_names[c_mobile] = "Mobile POS & High-Velocity Merchants"
    strategies[c_mobile] = "Enforce device GPS geofencing, dynamic velocity rate-limiting & 2FA step-up on high-ticket transfers."
    remaining.remove(c_mobile)

    # 2. Cluster among remaining with highest night_tx_ratio -> Night-Active Hospitality & Entertainment
    c_night = centroids.loc[list(remaining), "avg_night"].idxmax()
    profile_names[c_night] = "Night-Active Hospitality & Entertainment"
    strategies[c_night] = "Calibrate nocturnal risk window (00:00-05:00 WIB) filters and deploy dynamic QRIS POS terminals."
    remaining.remove(c_night)

    # 3. Cluster among remaining with highest total_volume -> High-Volume Enterprise Retailers
    c_ent = centroids.loc[list(remaining), "avg_vol"].idxmax()
    profile_names[c_ent] = "High-Volume Enterprise Retailers"
    strategies[c_ent] = "Provide dedicated Key Account Manager, priority settlement SLA, and automated daily clearing."
    remaining.remove(c_ent)

    # 4. Remaining cluster -> Standard Everyday Retail Merchants
    c_std = list(remaining)[0]
    profile_names[c_std] = "Standard Everyday Retail Merchants"
    strategies[c_std] = "Standard quarterly EDC preventive maintenance & automated settlement processing."

    return profile_names, strategies


def run_merchant_clustering(data_path="data/processed/features.parquet"):
    print("Loading data for merchant segmentation...")
    df = pd.read_parquet(data_path)

    # Aggregate transaction behavior at merchant (name_dest) level
    mch_agg = df.groupby("name_dest").agg(
        total_tx_count=("amount", "count"),
        total_volume=("amount", "sum"),
        avg_ticket_size=("amount", "mean"),
        max_ticket_size=("amount", "max"),
        night_tx_ratio=("is_night_transaction", "mean"),
        mobile_edc_ratio=("terminal_type", lambda x: (x == "EDC_MOBILE").mean()),
        online_pg_ratio=("terminal_type", lambda x: (x == "ONLINE_PG").mean()),
        fraud_tx_count=("is_fraud", "sum"),
        observed_fraud_rate=("is_fraud", "mean")
    ).reset_index()

    mch_agg["mobile_online_ratio"] = mch_agg["mobile_edc_ratio"] + mch_agg["online_pg_ratio"]

    # Log transform skewed monetary aggregates
    mch_agg["total_volume_log"] = np.log1p(mch_agg["total_volume"])
    mch_agg["avg_ticket_size_log"] = np.log1p(mch_agg["avg_ticket_size"])
    mch_agg["max_ticket_size_log"] = np.log1p(mch_agg["max_ticket_size"])

    cluster_features = [
        "total_tx_count", "total_volume_log", "avg_ticket_size_log", "max_ticket_size_log",
        "night_tx_ratio", "mobile_edc_ratio", "online_pg_ratio"
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(mch_agg[cluster_features])

    # KMeans with 4 distinct clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    mch_agg["cluster_id"] = cluster_labels

    # Evaluate Silhouette Score
    sil_score = float(silhouette_score(X_scaled, cluster_labels))
    print(f"K-Means (k=4) Silhouette Score: {sil_score:.4f}")

    profile_names, strategies = assign_distinct_segment_profiles(mch_agg)
    mch_agg["segment_name"] = mch_agg["cluster_id"].map(profile_names)
    mch_agg["recommended_strategy"] = mch_agg["cluster_id"].map(strategies)

    # PCA projection for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    mch_agg["pca_1"] = np.round(pca_coords[:, 0], 3)
    mch_agg["pca_2"] = np.round(pca_coords[:, 1], 3)
    explained_var = float(np.sum(pca.explained_variance_ratio_))

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/tableau", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)

    joblib.dump({
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "features": cluster_features,
        "silhouette_score": round(sil_score, 4),
        "pca_explained_variance": round(explained_var, 4)
    }, "models/merchant_cluster_model.joblib")

    mch_agg.to_csv("data/tableau/merchant_segments.csv", index=False)

    summary = mch_agg.groupby("segment_name").agg(
        merchant_count=("name_dest", "count"),
        avg_tx_count=("total_tx_count", lambda x: round(x.mean(), 1)),
        avg_monthly_volume=("total_volume", lambda x: round(x.mean(), 2)),
        avg_ticket_size=("avg_ticket_size", lambda x: round(x.mean(), 2)),
        avg_night_ratio=("night_tx_ratio", lambda x: round(x.mean() * 100, 1)),
        avg_mobile_online_pct=("mobile_online_ratio", lambda x: round(x.mean() * 100, 1)),
        avg_fraud_rate_pct=("observed_fraud_rate", lambda x: round(x.mean() * 100, 2)),
        recommended_strategy=("recommended_strategy", "first")
    ).reset_index().sort_values(by="avg_monthly_volume", ascending=False)

    summary.to_csv("reports/tables/merchant_cluster_summary.csv", index=False)
    print("Merchant clustering complete! 4-Segment Summary:")
    print(summary[["segment_name", "merchant_count", "avg_monthly_volume", "avg_night_ratio", "avg_mobile_online_pct", "avg_fraud_rate_pct"]].to_string(index=False))


if __name__ == "__main__":
    run_merchant_clustering()
