"""
Streamlit Web Application:
- Tab 1: Unified Executive & Merchant Analytics Dashboard
  * Interactive Multi-Dimensional Filter Bar (City, Terminal Channel, Merchant Category)
  * 5 Core Actionable Business Insights (Channel Loss, 24h x Transaction Type Heatmap/Trend,
    Category-City Treemap, Root-Cause Feature Drivers & Financial ROI Threshold Curve,
    K-Means 4-Cluster Segmentation + Priority Audit Watchlist)
- Tab 2: Real-Time Transaction Fraud Prediction Engine
  * One-Click Quick Scenario Presets (Normal POS Approve, Nocturnal Step-Up 2FA, Account Takeover Auto-Block)
  * Interactive Single-Transaction Risk Simulator with Visual Gauge & Triggered Signals
  * Live Batch Transaction Stream / CSV Scoring Simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import json

st.set_page_config(
    page_title="Payment & Fraud Risk Center",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Payment Analytics & Fraud Detection System")

# Load live metrics & artifacts
metrics_file = "reports/metrics.json"
fi_file = "reports/tables/feature_importance.csv"
channel_file = "data/tableau/channel_location_risk.csv"
hourly_file = "data/tableau/hourly_fraud_risk.csv"
seg_file = "data/tableau/merchant_segments.csv"
cluster_summary_file = "reports/tables/merchant_cluster_summary.csv"
cluster_model_file = "models/merchant_cluster_model.joblib"
features_parquet = "data/processed/features.parquet"

metrics_data = {}
if os.path.exists(metrics_file):
    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)

ds = metrics_data.get("dataset_summary", {})
opt_metrics = metrics_data.get("test_metrics_optimal_business_threshold", {})
def_metrics = metrics_data.get("test_metrics_default_threshold", {})
opt_threshold = float(metrics_data.get("optimal_threshold", 0.15))
block_threshold = float(metrics_data.get("block_threshold", 0.37))

tabs = st.tabs([
    "📊 Analytics Dashboard",
    "🎯 Real-Time Fraud Prediction"
])

# ==============================================================================
# TAB 1: UNIFIED ANALYTICS DASHBOARD
# ==============================================================================
with tabs[0]:
    df_ch_raw = pd.read_csv(channel_file) if os.path.exists(channel_file) else pd.DataFrame()
    df_hr_raw = pd.read_csv(hourly_file) if os.path.exists(hourly_file) else pd.DataFrame()

    # --- INTERACTIVE FILTER BAR (SLICERS) ---
    with st.expander("🎛️ Interactive Dashboard Filters (Filter by City, Terminal Channel & Merchant Category)", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        all_cities = sorted(df_ch_raw["location"].unique().tolist()) if not df_ch_raw.empty else []
        all_terminals = sorted(df_ch_raw["terminal_type"].unique().tolist()) if not df_ch_raw.empty else []
        all_cats = sorted(df_ch_raw["merchant_category"].unique().tolist()) if not df_ch_raw.empty else []

        sel_cities = f_col1.multiselect("City / Operational Area", all_cities, default=all_cities)
        sel_terminals = f_col2.multiselect("Terminal / Channel Type", all_terminals, default=all_terminals)
        sel_cats = f_col3.multiselect("Merchant Business Category", all_cats, default=all_cats)

    if not df_ch_raw.empty:
        df_ch = df_ch_raw[
            df_ch_raw["location"].isin(sel_cities or all_cities) &
            df_ch_raw["terminal_type"].isin(sel_terminals or all_terminals) &
            df_ch_raw["merchant_category"].isin(sel_cats or all_cats)
        ].copy()
    else:
        df_ch = pd.DataFrame()

    # --- DYNAMIC KPI HEADER ---
    col1, col2, col3, col4 = st.columns(4)

    if not df_ch.empty:
        total_tx = int(df_ch["total_transactions"].sum())
        total_frauds = int(df_ch["fraud_cases"].sum())
        total_vol = float(df_ch["total_volume_idr"].sum())
        total_fraud_loss = float(df_ch["fraud_loss_idr"].sum())
        fraud_rate = (total_frauds / max(1, total_tx)) * 100.0
        recall_val = float(opt_metrics.get("recall", 0.787))
        prevented_b = (total_fraud_loss * recall_val) / 1e9
    else:
        total_tx = ds.get("total_transactions", 50000)
        total_frauds = ds.get("total_fraud_cases", 680)
        total_vol = ds.get("total_volume_idr", 0)
        fraud_rate = ds.get("observed_fraud_rate_pct", 1.36)
        recall_val = float(opt_metrics.get("recall", 0.787))
        prevented_b = ds.get("estimated_prevented_loss_idr", 0) / 1e9

    vol_str = f"Rp {total_vol/1e12:.2f} Triliun" if total_vol >= 1e12 else f"Rp {total_vol/1e9:.2f} Miliar"

    col1.metric("Total Monitored Transactions", f"{total_tx:,}", f"{total_frauds:,} Fraud Events")
    col2.metric("Total Settlement Volume", vol_str, "Filtered Scope")
    col3.metric("Observed Fraud Rate", f"{fraud_rate:.2f}%", f"Optimal Threshold: {opt_threshold:.2f}", delta_color="off")
    col4.metric("Mitigated Fraud Exposure", f"Rp {prevented_b:.2f} Miliar", f"{recall_val*100:.1f}% Detection Rate")

    st.markdown("---")

    # --- ROW 1: INSIGHT 1 (CHANNEL RISK) & INSIGHT 2 (24H x TRANSACTION TYPE ANOMALY) ---
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("##### 1️⃣ Fraud Rate & Financial Loss by Transaction Channel")
        if not df_ch.empty:
            ch_agg = df_ch.groupby("terminal_type").agg(
                total_tx=("total_transactions", "sum"),
                fraud_cases=("fraud_cases", "sum"),
                total_vol=("total_volume_idr", "sum"),
                fraud_loss=("fraud_loss_idr", "sum")
            ).reset_index()
            ch_agg["Fraud_Rate_Pct"] = np.round(ch_agg["fraud_cases"] / ch_agg["total_tx"].clip(lower=1) * 100, 2)
            ch_agg["Fraud_Loss_Miliar"] = np.round(ch_agg["fraud_loss"] / 1e9, 2)
            ch_agg = ch_agg.sort_values(by="Fraud_Rate_Pct", ascending=False)
        else:
            ch_agg = pd.DataFrame({
                "terminal_type": ["EDC_MOBILE", "ONLINE_PG", "QRIS_DYNAMIC", "EDC_STATIC", "QRIS_STATIC"],
                "Fraud_Rate_Pct": [2.45, 2.12, 0.85, 0.54, 0.38],
                "Fraud_Loss_Miliar": [0.85, 0.62, 0.31, 0.25, 0.16]
            })

        fig_channel = px.bar(
            ch_agg, x="terminal_type", y="Fraud_Rate_Pct", color="Fraud_Rate_Pct",
            hover_data=["Fraud_Loss_Miliar"],
            color_continuous_scale="Reds", text_auto=".2f",
            labels={"terminal_type": "Terminal / Channel", "Fraud_Rate_Pct": "Fraud Rate (%)"}
        )
        fig_channel.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_channel, use_container_width=True)
        st.caption("💡 **Action:** Enforce GPS/IP binding & velocity limits on `EDC_MOBILE` and `ONLINE_PG` terminals (fraud rate ~4×–7× higher than static POS).")

    with c_right:
        hdr_c1, hdr_c2 = st.columns([1.5, 1.0])
        hdr_c1.markdown("##### 2️⃣ 24-Hour Nocturnal Anomaly by Tx Type")
        view_mode = hdr_c2.radio(
            "View Mode", ["By Transaction Type", "Overall Hourly Trend"],
            horizontal=True, label_visibility="collapsed"
        )

        if not df_hr_raw.empty:
            if view_mode == "By Transaction Type":
                hr_type = df_hr_raw.copy()
                fig_hour = px.line(
                    hr_type, x="hour_of_day", y="fraud_rate_pct", color="type", markers=True,
                    labels={"hour_of_day": "Hour of Day (WIB)", "fraud_rate_pct": "Fraud Rate (%)", "type": "Tx Type"}
                )
            else:
                hr_agg = df_hr_raw.groupby("hour_of_day").agg(
                    total_tx=("total_tx", "sum"),
                    fraud_cases=("fraud_cases", "sum")
                ).reset_index()
                hr_agg["Fraud_Rate_Pct"] = np.round(hr_agg["fraud_cases"] / hr_agg["total_tx"] * 100, 2)
                fig_hour = px.line(
                    hr_agg, x="hour_of_day", y="Fraud_Rate_Pct", markers=True,
                    labels={"hour_of_day": "Hour of Day (WIB)", "Fraud_Rate_Pct": "Fraud Rate (%)"}
                )
            fig_hour.add_vrect(x0=0, x1=5, fillcolor="red", opacity=0.12, line_width=0, annotation_text="Nocturnal Surge (00-05 WIB)")
            fig_hour.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=-0.22))
            st.plotly_chart(fig_hour, use_container_width=True)
            st.caption("💡 **Action:** Nocturnal fraud (`00:00-05:00 WIB`) spikes to **10%–12.5%** specifically on `TRANSFER` & `CASH_OUT`, while retail `PAYMENT` stays <1%. Target 2FA OTP specifically on nocturnal transfers/cash-outs.")

    st.markdown("---")

    # --- ROW 2: INSIGHT 3 (FRAUD LOSS TREEMAP) & INSIGHT 4 (ROOT CAUSE DRIVERS + ROI THRESHOLD) ---
    c_m1, c_m2 = st.columns([1.05, 1.0])
    with c_m1:
        st.markdown("##### 3️⃣ Financial Fraud Loss Exposure by Merchant Category & City")
        if not df_ch.empty:
            cat_loss = df_ch.groupby(["merchant_category", "location"]).agg(
                fraud_loss_juta=("fraud_loss_idr", lambda x: round(x.sum() / 1e6, 1)),
                fraud_cases=("fraud_cases", "sum")
            ).reset_index()
            cat_loss = cat_loss[cat_loss["fraud_loss_juta"] > 0]
            if not cat_loss.empty:
                fig_tree = px.treemap(
                    cat_loss, path=["merchant_category", "location"], values="fraud_loss_juta",
                    color="fraud_cases", color_continuous_scale="OrRd",
                    labels={"fraud_loss_juta": "Fraud Loss (Rp Juta)", "fraud_cases": "Fraud Cases"}
                )
                fig_tree.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_tree, use_container_width=True)
            else:
                st.info("No fraud loss events match the current filter selection.")
            st.caption("💡 **Action:** Prioritize merchant KYC & hardware inspections on high-loss categories (`electronics`, `digital_services`, `fashion_apparel`) in top metro areas.")

    with c_m2:
        sub_t1, sub_t2 = st.tabs(["🔍 Root-Cause Fraud Drivers", "💰 Financial ROI & Threshold Optimization"])
        with sub_t1:
            if os.path.exists(fi_file):
                df_fi = pd.read_csv(fi_file).head(9).sort_values(by="importance_pct", ascending=True)
                fig_fi = px.bar(
                    df_fi, x="importance_pct", y="feature", orientation="h",
                    color="importance_pct", color_continuous_scale="Blues", text_auto=".1f",
                    labels={"importance_pct": "Contribution (%)", "feature": "Risk Signal"}
                )
                fig_fi.update_layout(height=285, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig_fi, use_container_width=True)
                st.caption("💡 **Action:** Balance depletion (`is_account_emptied`), unstructured POS telemetry keywords (`device_risk_flag_count`), and settlement discrepancies drive >48% of predictive power.")

        with sub_t2:
            # Compare Unmitigated Loss vs Default 0.50 Threshold vs Optimal Cost-Calibrated 0.15 Threshold on Test Set (scaled to 30-day volume)
            fn_avg_cost = 25_000_000
            fp_avg_cost = 250_000
            scale_factor = 5.0  # 10,000 test set -> 50,000 full 30-day window

            unmitigated_cost = (opt_metrics.get("tp", 107) + opt_metrics.get("fn", 29)) * fn_avg_cost * scale_factor / 1e9
            def_cost = (def_metrics.get("fn", 38) * fn_avg_cost + def_metrics.get("fp", 14) * fp_avg_cost) * scale_factor / 1e9
            opt_cost = (opt_metrics.get("fn", 29) * fn_avg_cost + opt_metrics.get("fp", 35) * fp_avg_cost) * scale_factor / 1e9

            roi_df = pd.DataFrame({
                "Policy": [
                    "1. No ML Model (Rule Baseline)",
                    "2. Default Threshold (0.50)",
                    f"3. Cost-Optimized Threshold ({opt_threshold:.2f})"
                ],
                "Expected_30d_Loss_Miliar": [round(unmitigated_cost, 2), round(def_cost, 2), round(opt_cost, 2)],
                "Detection_Recall_Pct": [0.0, round(def_metrics.get("recall", 0.7206)*100, 1), round(opt_metrics.get("recall", 0.7868)*100, 1)]
            })
            fig_roi = px.bar(
                roi_df, x="Policy", y="Expected_30d_Loss_Miliar", color="Policy",
                text_auto=".2f",
                color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"],
                labels={"Expected_30d_Loss_Miliar": "30-Day Fraud + Friction Cost (Rp Miliar)"}
            )
            fig_roi.update_layout(height=285, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_roi, use_container_width=True)
            st.caption(f"💡 **Why Threshold `{opt_threshold:.2f}`?** Reduces total 30-day financial loss + operational friction from **Rp {unmitigated_cost:.2f} Miliar** down to **Rp {opt_cost:.2f} Miliar** (saving **Rp {unmitigated_cost - opt_cost:.2f} Miliar/month**).")

    st.markdown("---")

    # --- ROW 3: INSIGHT 5 (K-MEANS MERCHANT BEHAVIORAL CLUSTERING & HIGH-RISK WATCHLIST) ---
    sil_text = ""
    if os.path.exists(cluster_model_file):
        c_meta = joblib.load(cluster_model_file)
        sil_val = c_meta.get("silhouette_score")
        pca_var = c_meta.get("pca_explained_variance")
        if sil_val is not None:
            sil_text = f" — Silhouette Score: `{sil_val:.4f}` | 2D PCA Variance: `{pca_var*100:.1f}%`"

    st.markdown(f"##### 5️⃣ Merchant Behavioral Segmentation & Risk Strategy (1,500 Merchants{sil_text})")

    if os.path.exists(seg_file):
        df_seg = pd.read_csv(seg_file)
        c_cl1, c_cl2 = st.columns([1.05, 1.0])

        with c_cl1:
            fig_pca = px.scatter(
                df_seg, x="pca_1", y="pca_2", color="segment_name",
                hover_data=["name_dest", "total_tx_count", "total_volume", "avg_ticket_size", "observed_fraud_rate"],
                labels={"pca_1": "PCA 1 (Volume & Ticket Size)", "pca_2": "PCA 2 (Nocturnal & Channel Mix)"},
                title="PCA 2D Projection of 4 Merchant Behavioral Clusters"
            )
            fig_pca.update_layout(height=380, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_pca, use_container_width=True)

        with c_cl2:
            if os.path.exists(cluster_summary_file):
                df_csum = pd.read_csv(cluster_summary_file)
                st.dataframe(
                    df_csum[[
                        "segment_name", "merchant_count", "avg_monthly_volume",
                        "avg_night_ratio", "avg_mobile_online_pct", "avg_fraud_rate_pct"
                    ]],
                    use_container_width=True,
                    hide_index=True
                )
            for seg in df_seg["segment_name"].unique():
                sub = df_seg[df_seg["segment_name"] == seg]
                strat = sub["recommended_strategy"].iloc[0]
                with st.expander(f"📌 {seg} ({len(sub)} Merchants — Fraud Rate: {sub['observed_fraud_rate'].mean()*100:.2f}%)"):
                    st.write(f"**Strategic Action:** {strat}")
                    st.write(
                        f"**Avg Volume:** Rp {sub['total_volume'].mean():,.0f} | "
                        f"**Avg Ticket:** Rp {sub['avg_ticket_size'].mean():,.0f} | "
                        f"**Nocturnal Ratio:** {sub['night_tx_ratio'].mean()*100:.1f}%"
                    )

        st.markdown("##### 🚨 Priority Merchant Audit Watchlist (Top Anomaly Merchants Requiring Immediate EDC/Account Inspection)")
        watchlist = df_seg[df_seg["fraud_tx_count"] >= 2].sort_values(
            by=["fraud_tx_count", "observed_fraud_rate", "total_volume"], ascending=[False, False, False]
        ).head(8).copy()
        watchlist["observed_fraud_rate_pct"] = np.round(watchlist["observed_fraud_rate"] * 100, 2)
        watchlist["mobile_online_pct"] = np.round(watchlist["mobile_online_ratio"] * 100, 1)
        watchlist["night_tx_pct"] = np.round(watchlist["night_tx_ratio"] * 100, 1)
        watchlist["total_volume_idr"] = watchlist["total_volume"].apply(lambda v: f"Rp {v:,.0f}")
        st.dataframe(
            watchlist[[
                "name_dest", "segment_name", "total_tx_count", "total_volume_idr",
                "fraud_tx_count", "observed_fraud_rate_pct", "mobile_online_pct", "night_tx_pct", "recommended_strategy"
            ]].rename(columns={
                "name_dest": "Merchant ID",
                "segment_name": "Behavioral Cluster",
                "total_tx_count": "Total Tx",
                "total_volume_idr": "Total Volume",
                "fraud_tx_count": "Fraud Cases",
                "observed_fraud_rate_pct": "Fraud Rate (%)",
                "mobile_online_pct": "Mobile/Online (%)",
                "night_tx_pct": "Night Tx (%)",
                "recommended_strategy": "Immediate Action Required"
            }),
            use_container_width=True,
            hide_index=True
        )

# ==============================================================================
# TAB 2: REAL-TIME TRANSACTION FRAUD PREDICTION
# ==============================================================================
with tabs[1]:
    st.subheader("Real-Time Transaction Risk Scoring & Decision Simulator")
    st.caption(f"Powered by **Stacking Ensemble ML** — **Approve:** `< {opt_threshold:.2f}` | **Step-Up 2FA OTP:** `{opt_threshold:.2f} - {block_threshold:.2f}` | **Auto-Block:** `>= {block_threshold:.2f}`")

    # Initialize session state for quick presets
    if "p_type" not in st.session_state:
        st.session_state.update({
            "p_type": "TRANSFER",
            "p_amount": 125_000_000,
            "p_old_bal": 125_000_000,
            "p_new_bal": 0,
            "p_hour": 2,
            "p_terminal": "EDC_MOBILE",
            "p_category": "electronics",
            "p_disc": 25_000_000,
            "p_remarks": "Multiple fallback attempts detected before EMV fallback success"
        })

    st.markdown("##### ⚡ Quick Test Scenarios (One-Click Preset Loader)")
    b_col1, b_col2, b_col3 = st.columns(3)
    if b_col1.button("✅ Scenario 1: Normal Retail POS (Approve)", use_container_width=True):
        st.session_state.update({
            "p_type": "PAYMENT",
            "p_amount": 350_000,
            "p_old_bal": 8_500_000,
            "p_new_bal": 8_150_000,
            "p_hour": 14,
            "p_terminal": "EDC_STATIC",
            "p_category": "retail_grocery",
            "p_disc": 0,
            "p_remarks": "Chip read successful PIN verified"
        })
    if b_col2.button("⚠️ Scenario 2: Suspicious Transfer (Step-Up 2FA OTP)", use_container_width=True):
        st.session_state.update({
            "p_type": "TRANSFER",
            "p_amount": 18_000_000,
            "p_old_bal": 24_000_000,
            "p_new_bal": 6_000_000,
            "p_hour": 1,
            "p_terminal": "ONLINE_PG",
            "p_category": "digital_services",
            "p_disc": 0,
            "p_remarks": "Unusual transaction spike outside merchant business hours"
        })
    if b_col3.button("🚨 Scenario 3: Account Takeover / EMV Fallback (Auto-Block)", use_container_width=True):
        st.session_state.update({
            "p_type": "TRANSFER",
            "p_amount": 145_000_000,
            "p_old_bal": 145_000_000,
            "p_new_bal": 0,
            "p_hour": 2,
            "p_terminal": "EDC_MOBILE",
            "p_category": "electronics",
            "p_disc": 45_000_000,
            "p_remarks": "Multiple fallback attempts detected before EMV fallback success"
        })

    col_input1, col_input2 = st.columns(2)
    tx_types = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
    terminals_list = ["EDC_MOBILE", "ONLINE_PG", "QRIS_DYNAMIC", "EDC_STATIC", "QRIS_STATIC"]
    cats_list = ["electronics", "fashion_apparel", "digital_services", "retail_grocery", "food_beverage", "travel_hospitality"]

    with col_input1:
        tx_type = st.selectbox("Transaction Type", tx_types, index=tx_types.index(st.session_state["p_type"]))
        amount = st.number_input("Transaction Amount (IDR)", value=int(st.session_state["p_amount"]), step=5_000_000)
        old_bal = st.number_input("Customer Current Balance (IDR)", value=int(st.session_state["p_old_bal"]), step=5_000_000)
        new_bal = st.number_input("Customer Balance After Transaction (IDR)", value=int(st.session_state["p_new_bal"]), step=5_000_000)
        hour = st.slider("Hour of Transaction (WIB)", 0, 23, int(st.session_state["p_hour"]))

    with col_input2:
        terminal = st.selectbox("Terminal / Channel", terminals_list, index=terminals_list.index(st.session_state["p_terminal"]))
        category = st.selectbox("Merchant Category", cats_list, index=cats_list.index(st.session_state["p_category"]))
        dest_discrepancy = st.number_input("Merchant Settlement Discrepancy (IDR)", value=int(st.session_state["p_disc"]), step=5_000_000)
        remarks = st.text_input("Device Telemetry / POS Remarks", st.session_state["p_remarks"])

    if st.button("Evaluate Transaction Risk", type="primary", use_container_width=True):
        model_path = "models/fraud_detection_pipeline.joblib"
        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)

            error_orig = round(new_bal + amount - old_bal, 2)
            error_dest = float(dest_discrepancy)
            is_empty = 1 if (new_bal == 0 and old_bal > 0) else 0
            ratio = amount / old_bal if old_bal > 0 else 999.0
            is_night = 1 if (0 <= hour <= 5) else 0
            h_sin = np.sin(2 * np.pi * hour / 24.0)
            h_cos = np.cos(2 * np.pi * hour / 24.0)
            risk_flags = sum(1 for kw in ["fallback", "repeat", "spike", "emptied", "anomaly", "flagged"] if kw in remarks.lower())

            row = pd.DataFrame([{
                "amount": amount, "oldbalance_orig": old_bal, "newbalance_orig": new_bal,
                "oldbalance_dest": 10000000.0, "newbalance_dest": 10000000.0 + amount - error_dest,
                "error_balance_orig": error_orig, "error_balance_dest": error_dest,
                "is_account_emptied": is_empty, "amount_to_oldbalance_ratio": ratio,
                "hour_of_day": hour, "day_of_week": 2, "is_night_transaction": is_night,
                "hour_sin": h_sin, "hour_cos": h_cos, "mch_tx_count_24h": 14,
                "mch_tx_volume_24h": amount * 2.2, "device_risk_flag_count": risk_flags,
                "type": tx_type, "merchant_category": category, "terminal_type": terminal
            }])

            preprocessor = pipeline.named_steps["preprocessor"]
            clf = pipeline.named_steps["stacking_clf"]
            X_t = preprocessor.transform(row)
            prob = float(clf.predict_proba(X_t)[0, 1])

            res_col1, res_col2 = st.columns([1.0, 1.3])
            with res_col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%"},
                    title={"text": "Fraud Probability Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1e293b"},
                        "steps": [
                            {"range": [0, opt_threshold * 100], "color": "#86efac"},
                            {"range": [opt_threshold * 100, block_threshold * 100], "color": "#fde047"},
                            {"range": [block_threshold * 100, 100], "color": "#fca5a5"}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": block_threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                st.markdown("### Risk Evaluation Decision")
                if prob >= block_threshold:
                    st.error(f"🚨 **HIGH RISK TRANSACTION (Risk Score: {prob*100:.1f}%) — DECISION: AUTO-BLOCK**")
                    st.write("**Recommended Protocol:** Immediately decline authorization, temporarily freeze terminal token, and alert SOC/Fraud Ops team.")
                elif prob >= opt_threshold:
                    st.warning(f"⚠️ **MODERATE RISK TRANSACTION (Risk Score: {prob*100:.1f}%) — DECISION: STEP-UP 2FA OTP**")
                    st.write("**Recommended Protocol:** Hold settlement and prompt cardholder with mandatory biometric / SMS OTP challenge.")
                else:
                    st.success(f"✅ **LOW RISK TRANSACTION (Risk Score: {prob*100:.1f}%) — DECISION: APPROVE**")
                    st.write("**Recommended Protocol:** Standard instant settlement authorization.")

                factors = []
                if is_empty or ratio >= 0.90:
                    factors.append(f"High balance depletion ({ratio*100:.1f}% of customer balance)")
                if is_night:
                    factors.append("Nocturnal window transaction (00:00 - 05:00 WIB)")
                if risk_flags > 0:
                    factors.append(f"Unstructured telemetry contains {risk_flags} suspicious keyword(s)")
                if terminal in ["EDC_MOBILE", "ONLINE_PG"]:
                    factors.append(f"Higher-risk channel ({terminal})")
                if abs(error_dest) > 100_000:
                    factors.append(f"Settlement balance discrepancy (Rp {abs(error_dest):,.0f})")
                if factors:
                    st.markdown("**Triggered Risk Signals:**")
                    for f_item in factors:
                        st.markdown(f"- {f_item}")
                else:
                    st.markdown("**Triggered Risk Signals:** None (Clean transaction profile)")
        else:
            st.warning("Model file `models/fraud_detection_pipeline.joblib` not found. Please train models first.")

    st.markdown("---")
    st.markdown("##### 📡 Live Batch Transaction Stream Scorer (Simulate Payment Gateway Queue)")
    st.caption("Evaluate a live batch of incoming transactions simultaneously to see how the model routes traffic into `APPROVE`, `STEP_UP_2FA`, and `BLOCK`.")

    if st.button("🔄 Score Sample Live Stream (15 Incoming Transactions)"):
        if os.path.exists(features_parquet) and os.path.exists("models/fraud_detection_pipeline.joblib"):
            df_sample_pool = pd.read_parquet(features_parquet)
            pipeline = joblib.load("models/fraud_detection_pipeline.joblib")

            # Sample mix of normal and suspicious transactions
            frauds_s = df_sample_pool[df_sample_pool["is_fraud"] == 1].sample(4, random_state=np.random.randint(1, 1000))
            norms_s = df_sample_pool[df_sample_pool["is_fraud"] == 0].sample(11, random_state=np.random.randint(1, 1000))
            batch_df = pd.concat([frauds_s, norms_s]).sample(frac=1.0).reset_index(drop=True)

            feature_cols = [
                "amount", "oldbalance_orig", "newbalance_orig", "oldbalance_dest", "newbalance_dest",
                "error_balance_orig", "error_balance_dest", "is_account_emptied", "amount_to_oldbalance_ratio",
                "hour_of_day", "day_of_week", "is_night_transaction", "hour_sin", "hour_cos",
                "mch_tx_count_24h", "mch_tx_volume_24h", "device_risk_flag_count",
                "type", "merchant_category", "terminal_type"
            ]
            X_b = pipeline.named_steps["preprocessor"].transform(batch_df[feature_cols])
            probs_b = pipeline.named_steps["stacking_clf"].predict_proba(X_b)[:, 1]

            batch_df["risk_score_pct"] = np.round(probs_b * 100, 1)
            batch_df["decision"] = np.where(
                probs_b >= block_threshold, "🚨 BLOCK",
                np.where(probs_b >= opt_threshold, "⚠️ STEP_UP_2FA", "✅ APPROVE")
            )
            batch_df["amount_idr"] = batch_df["amount"].apply(lambda v: f"Rp {v:,.0f}")

            st.dataframe(
                batch_df[[
                    "name_dest", "type", "terminal_type", "merchant_category",
                    "hour_of_day", "amount_idr", "device_remarks", "risk_score_pct", "decision"
                ]].rename(columns={
                    "name_dest": "Merchant ID",
                    "type": "Tx Type",
                    "terminal_type": "Terminal",
                    "merchant_category": "Category",
                    "hour_of_day": "Hour (WIB)",
                    "amount_idr": "Amount",
                    "device_remarks": "Device Telemetry Log",
                    "risk_score_pct": "Risk Score (%)",
                    "decision": "Gateway Decision"
                }),
                use_container_width=True,
                hide_index=True
            )
