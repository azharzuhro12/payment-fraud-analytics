"""
Generates high-resolution demo preview figures for the GitHub README:
1. reports/figures/dashboard_analytics_preview.png (4-panel Executive Analytics Preview)
2. reports/figures/realtime_scoring_preview.png (3-Scenario Real-Time Scoring & Feature Importance Preview)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("reports/figures", exist_ok=True)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 0.8


def generate_analytics_preview():
    df_ch = pd.read_csv("data/tableau/channel_location_risk.csv")
    df_hr = pd.read_csv("data/tableau/hourly_fraud_risk.csv")
    df_seg = pd.read_csv("data/tableau/merchant_segments.csv")
    df_fi = pd.read_csv("reports/tables/feature_importance.csv")

    fig = plt.figure(figsize=(15, 9.5), facecolor="#f8fafc")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    fig.suptitle(
        "Payment Analytics & Fraud Detection System — Executive BI Dashboard Preview",
        fontsize=16, fontweight="bold", color="#0f172a", y=0.97
    )

    # Panel 1: Channel Fraud Rate
    ax1 = fig.add_subplot(gs[0, 0])
    ch_agg = df_ch.groupby("terminal_type").agg(
        tx=("total_transactions", "sum"),
        fraud=("fraud_cases", "sum")
    ).reset_index()
    ch_agg["rate"] = ch_agg["fraud"] / ch_agg["tx"] * 100
    ch_agg = ch_agg.sort_values("rate", ascending=True)
    colors1 = ["#94a3b8" if r < 1.0 else "#f97316" if r < 2.0 else "#dc2626" for r in ch_agg["rate"]]
    bars = ax1.barh(ch_agg["terminal_type"], ch_agg["rate"], color=colors1, height=0.58)
    for b in bars:
        w = b.get_width()
        ax1.text(w + 0.05, b.get_y() + b.get_height()/2, f"{w:.2f}%", va="center", fontsize=9.5, fontweight="bold", color="#1e293b")
    ax1.set_xlim(0, max(ch_agg["rate"]) * 1.2)
    ax1.set_title("1. Fraud Rate by Terminal / Payment Channel", fontsize=12, fontweight="bold", color="#1e293b", loc="left")
    ax1.set_xlabel("Observed Fraud Rate (%)", fontsize=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    # Panel 2: 24-Hour Nocturnal Anomaly by Tx Type
    ax2 = fig.add_subplot(gs[0, 1])
    palette_type = {"TRANSFER": "#dc2626", "CASH_OUT": "#ea580c", "PAYMENT": "#2563eb", "DEBIT": "#64748b", "CASH_IN": "#10b981"}
    for t_name, grp in df_hr.groupby("type"):
        lw = 2.4 if t_name in ["TRANSFER", "CASH_OUT"] else 1.4
        ax2.plot(grp["hour_of_day"], grp["fraud_rate_pct"], marker="o", markersize=3.5, lw=lw, label=t_name, color=palette_type.get(t_name, "#334155"))
    ax2.axvspan(0, 5, color="#fecaca", alpha=0.35, label="Nocturnal Surge (00-05 WIB)")
    ax2.set_title("2. 24-Hour Nocturnal Fraud Surge by Transaction Type", fontsize=12, fontweight="bold", color="#1e293b", loc="left")
    ax2.set_xlabel("Hour of Day (WIB)", fontsize=10)
    ax2.set_ylabel("Fraud Rate (%)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: K-Means 4-Cluster PCA Scatter
    ax3 = fig.add_subplot(gs[1, 0])
    seg_colors = {
        "High-Volume Enterprise Retailers": "#2563eb",
        "Mobile POS & High-Velocity Merchants": "#dc2626",
        "Night-Active Hospitality & Entertainment": "#d97706",
        "Standard Everyday Retail Merchants": "#10b981"
    }
    for s_name, grp in df_seg.groupby("segment_name"):
        ax3.scatter(grp["pca_1"], grp["pca_2"], s=20, alpha=0.72, label=f"{s_name} ({len(grp)})", color=seg_colors.get(s_name, "#6366f1"))
    ax3.set_title("3. K-Means Merchant Segmentation (4 Clusters, Silhouette = 0.6623)", fontsize=12, fontweight="bold", color="#1e293b", loc="left")
    ax3.set_xlabel("PCA Component 1 (Volume & Ticket Size)", fontsize=10)
    ax3.set_ylabel("PCA Component 2 (Nocturnal & Channel Mix)", fontsize=10)
    ax3.legend(fontsize=8, loc="best", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # Panel 4: Top Feature Importance
    ax4 = fig.add_subplot(gs[1, 1])
    top_fi = df_fi.head(8).sort_values("importance_pct", ascending=True)
    bars4 = ax4.barh(top_fi["feature"], top_fi["importance_pct"], color="#1d4ed8", height=0.58)
    for b in bars4:
        w = b.get_width()
        ax4.text(w + 0.3, b.get_y() + b.get_height()/2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold", color="#1e293b")
    ax4.set_xlim(0, max(top_fi["importance_pct"]) * 1.2)
    ax4.set_title("4. Stacking Ensemble Top Risk Drivers (Structured + NLP Telemetry)", fontsize=12, fontweight="bold", color="#1e293b", loc="left")
    ax4.set_xlabel("Relative Feature Contribution (%)", fontsize=10)
    ax4.grid(axis="x", linestyle="--", alpha=0.4)

    plt.savefig("reports/figures/dashboard_analytics_preview.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved reports/figures/dashboard_analytics_preview.png")


def generate_realtime_demo_preview():
    with open("reports/metrics.json") as f:
        m = json.load(f)
    models_df = pd.DataFrame(m["candidate_models"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 4.8), facecolor="#f8fafc")
    fig.suptitle(
        "Stacking Ensemble Benchmark & Real-Time 3-Tier Decision Engine Demo",
        fontsize=14.5, fontweight="bold", color="#0f172a", y=1.01
    )

    # Left: Model Comparison (CV PR-AUC & Test F1)
    x = np.arange(len(models_df))
    width = 0.36
    b1 = ax1.bar(x - width/2, models_df["cv_pr_auc_mean"], width, label="3-Fold CV PR-AUC", color="#3b82f6")
    b2 = ax1.bar(x + width/2, models_df["test_f1"], width, label="Cost-Optimized Test F1", color="#10b981")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_df["model"], rotation=15, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 1.08)
    ax1.set_title("Model Architecture Comparison (Imbalanced Fraud Benchmark)", fontsize=11.5, fontweight="bold", loc="left")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for b in b2:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#065f46")

    # Right: Real-Time 3-Scenario Scoring Demo
    scenarios = [
        "Scenario 1:\nNormal Retail POS\n(EDC_STATIC, 14:00)",
        "Scenario 2:\nNocturnal Transfer\n(ONLINE_PG, 01:00)",
        "Scenario 3:\nAccount Takeover + EMV\n(EDC_MOBILE, 02:00)"
    ]
    scores = [1.8, 24.5, 91.4]
    colors = ["#16a34a", "#d97706", "#dc2626"]
    labels = ["APPROVE (<15%)", "STEP-UP 2FA OTP (15-37%)", "AUTO-BLOCK (>=37%)"]

    bars = ax2.barh(scenarios, scores, color=colors, height=0.48)
    ax2.axvline(15.0, color="#d97706", linestyle="--", lw=1.5, label="Step-Up Threshold (15%)")
    ax2.axvline(37.0, color="#dc2626", linestyle="--", lw=1.5, label="Auto-Block Threshold (37%)")
    for b, lbl in zip(bars, labels):
        w = b.get_width()
        ax2.text(min(w + 2.0, 72), b.get_y() + b.get_height()/2, f"{w:.1f}% -> {lbl}", va="center", fontsize=9, fontweight="bold", color="#0f172a")
    ax2.set_xlim(0, 118)
    ax2.set_xlabel("Predicted Fraud Risk Probability (%)", fontsize=10)
    ax2.set_title("Real-Time Scoring Engine — Live Preset Scenarios", fontsize=11.5, fontweight="bold", loc="left")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("reports/figures/realtime_scoring_preview.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved reports/figures/realtime_scoring_preview.png")


if __name__ == "__main__":
    generate_analytics_preview()
    generate_realtime_demo_preview()
