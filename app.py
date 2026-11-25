"""
Aplikasi Web Streamlit (Bahasa Indonesia):
- Tab 1: Dashboard Analitik Eksekutif & Risiko Merchant
  * Filter Interaktif Multi-Dimensi (Kota, Kanal Terminal, Kategori Merchant)
  * 5 Insight Bisnis Utama (Distribusi Risiko Kanal, Anomali Transaksi Malam 24 Jam,
    Treemap Kerugian Kategori & Kota, Faktor Pemicu Fraud & Optimasi ROI Threshold,
    Segmentasi 4 Klaster K-Means + Daftar Prioritas Audit Merchant)
- Tab 2: Simulasi & Prediksi Fraud Real-Time
  * 3 Tombol Skenario Cepat (POS Normal Disetujui, Transfer Malam Verifikasi 2FA OTP, Ambil Alih Akun Blokir Otomatis)
  * Simulator Penilaian Risiko Transaksi Tunggal dengan Speedometer Gauge & Sinyal Risiko
  * Simulator Antrean Transaksi Live Batch (15 Transaksi Sekaligus)
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
    page_title="Sistem Analitik Pembayaran & Deteksi Fraud",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Sistem Analitik Pembayaran & Deteksi Fraud")

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

SEG_TRANSLATE = {
    "High-Volume Enterprise Retailers": "Ritel Enterprise Volume Tinggi",
    "Mobile POS & High-Velocity Merchants": "Merchant Mobile POS & Transaksi Cepat",
    "Night-Active Hospitality & Entertainment": "Merchant Aktif Malam (Hospitality & Hiburan)",
    "Standard Everyday Retail Merchants": "Merchant Ritel Harian Standar"
}

STRAT_TRANSLATE = {
    "High-Volume Enterprise Retailers": "Sediakan Key Account Manager khusus, prioritas SLA pencairan harian (daily settlement), dan kliring otomatis.",
    "Mobile POS & High-Velocity Merchants": "Terapkan pembatasan lokasi GPS (geofencing) pada EDC Mobile, batas kecepatan transaksi, & wajib 2FA OTP untuk transfer nominal besar.",
    "Night-Active Hospitality & Entertainment": "Kalibrasi filter risiko jam malam (00:00-05:00 WIB) agar transaksi sah tidak terblokir, serta dorong penggunaan terminal QRIS Dinamis.",
    "Standard Everyday Retail Merchants": "Pencairan settlement otomatis standar & jadwal perawatan preventif mesin EDC setiap kuartal."
}

tabs = st.tabs([
    "📊 Dashboard Analitik",
    "🎯 Prediksi Fraud Real-Time"
])

# ==============================================================================
# TAB 1: DASHBOARD ANALITIK TERPADU
# ==============================================================================
with tabs[0]:
    df_ch_raw = pd.read_csv(channel_file) if os.path.exists(channel_file) else pd.DataFrame()
    df_hr_raw = pd.read_csv(hourly_file) if os.path.exists(hourly_file) else pd.DataFrame()

    # --- FILTER INTERAKTIF (SLICERS) ---
    with st.expander("🎛️ Filter Interaktif Dashboard (Saring berdasarkan Kota, Tipe Terminal & Kategori Merchant)", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        all_cities = sorted(df_ch_raw["location"].unique().tolist()) if not df_ch_raw.empty else []
        all_terminals = sorted(df_ch_raw["terminal_type"].unique().tolist()) if not df_ch_raw.empty else []
        all_cats = sorted(df_ch_raw["merchant_category"].unique().tolist()) if not df_ch_raw.empty else []

        sel_cities = f_col1.multiselect("Kota / Wilayah Operasional", all_cities, default=all_cities)
        sel_terminals = f_col2.multiselect("Tipe Terminal / Kanal Pembayaran", all_terminals, default=all_terminals)
        sel_cats = f_col3.multiselect("Kategori Bisnis Merchant", all_cats, default=all_cats)

    if not df_ch_raw.empty:
        df_ch = df_ch_raw[
            df_ch_raw["location"].isin(sel_cities or all_cities) &
            df_ch_raw["terminal_type"].isin(sel_terminals or all_terminals) &
            df_ch_raw["merchant_category"].isin(sel_cats or all_cats)
        ].copy()
    else:
        df_ch = pd.DataFrame()

    # --- KARTU KPI DINAMIS ---
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

    col1.metric("Total Transaksi Dipantau", f"{total_tx:,}", f"{total_frauds:,} Kasus Fraud")
    col2.metric("Total Volume Settlement", vol_str, "Periode 30 Hari")
    col3.metric("Rasio Fraud Terdeteksi", f"{fraud_rate:.2f}%", f"Ambang Optimal: {opt_threshold:.2f}", delta_color="off")
    col4.metric("Kerugian Fraud Dicegah", f"Rp {prevented_b:.2f} Miliar", f"{recall_val*100:.1f}% Tingkat Deteksi")

    st.markdown("---")

    # --- BARIS 1: INSIGHT 1 (RISIKO KANAL) & INSIGHT 2 (ANOMALI 24 JAM x TIPE TRANSAKSI) ---
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("##### 1️⃣ Tingkat Fraud & Kerugian Finansial per Kanal Terminal")
        if not df_ch.empty:
            ch_agg = df_ch.groupby("terminal_type").agg(
                total_tx=("total_transactions", "sum"),
                fraud_cases=("fraud_cases", "sum"),
                total_vol=("total_volume_idr", "sum"),
                fraud_loss=("fraud_loss_idr", "sum")
            ).reset_index()
            ch_agg["Fraud_Rate_Pct"] = np.round(ch_agg["fraud_cases"] / ch_agg["total_tx"].clip(lower=1) * 100, 2)
            ch_agg["Kerugian_Miliar"] = np.round(ch_agg["fraud_loss"] / 1e9, 2)
            ch_agg = ch_agg.sort_values(by="Fraud_Rate_Pct", ascending=False)
        else:
            ch_agg = pd.DataFrame({
                "terminal_type": ["EDC_MOBILE", "ONLINE_PG", "QRIS_DYNAMIC", "EDC_STATIC", "QRIS_STATIC"],
                "Fraud_Rate_Pct": [2.45, 2.12, 0.85, 0.54, 0.38],
                "Kerugian_Miliar": [0.85, 0.62, 0.31, 0.25, 0.16]
            })

        fig_channel = px.bar(
            ch_agg, x="terminal_type", y="Fraud_Rate_Pct", color="Fraud_Rate_Pct",
            hover_data=["Kerugian_Miliar"],
            color_continuous_scale="Reds", text_auto=".2f",
            labels={"terminal_type": "Tipe Terminal / Kanal", "Fraud_Rate_Pct": "Tingkat Fraud (%)"}
        )
        fig_channel.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_channel, use_container_width=True)
        st.caption("💡 **Rekomendasi Aksi:** Terapkan penguncian GPS/IP & batas kecepatan transaksi pada terminal `EDC_MOBILE` dan `ONLINE_PG` (risiko fraud 4×–7× lebih tinggi dibanding terminal POS statis).")

    with c_right:
        hdr_c1, hdr_c2 = st.columns([1.45, 1.05])
        hdr_c1.markdown("##### 2️⃣ Anomali Fraud 24 Jam per Tipe Transaksi")
        view_mode = hdr_c2.radio(
            "Mode Tampilan", ["Per Tipe Transaksi", "Tren Gabungan Per Jam"],
            horizontal=True, label_visibility="collapsed"
        )

        if not df_hr_raw.empty:
            if view_mode == "Per Tipe Transaksi":
                hr_type = df_hr_raw.copy()
                fig_hour = px.line(
                    hr_type, x="hour_of_day", y="fraud_rate_pct", color="type", markers=True,
                    labels={"hour_of_day": "Jam Transaksi (WIB)", "fraud_rate_pct": "Tingkat Fraud (%)", "type": "Tipe Transaksi"}
                )
            else:
                hr_agg = df_hr_raw.groupby("hour_of_day").agg(
                    total_tx=("total_tx", "sum"),
                    fraud_cases=("fraud_cases", "sum")
                ).reset_index()
                hr_agg["Fraud_Rate_Pct"] = np.round(hr_agg["fraud_cases"] / hr_agg["total_tx"] * 100, 2)
                fig_hour = px.line(
                    hr_agg, x="hour_of_day", y="Fraud_Rate_Pct", markers=True,
                    labels={"hour_of_day": "Jam Transaksi (WIB)", "Fraud_Rate_Pct": "Tingkat Fraud (%)"}
                )
            fig_hour.add_vrect(x0=0, x1=5, fillcolor="red", opacity=0.12, line_width=0, annotation_text="Jam Rawan Malam (00-05 WIB)")
            fig_hour.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=-0.22))
            st.plotly_chart(fig_hour, use_container_width=True)
            st.caption("💡 **Rekomendasi Aksi:** Fraud dini hari (`00:00-05:00 WIB`) melonjak hingga **10%–12,5%** khusus pada `TRANSFER` & `CASH_OUT`, sedangkan `PAYMENT` ritel tetap <1%. Terapkan wajib OTP khusus pada transfer/tarik tunai malam hari.")

    st.markdown("---")

    # --- BARIS 2: INSIGHT 3 (TREEMAP KERUGIAN) & INSIGHT 4 (FAKTOR PEMICU + ROI THRESHOLD) ---
    c_m1, c_m2 = st.columns([1.05, 1.0])
    with c_m1:
        st.markdown("##### 3️⃣ Peta Eksposur Kerugian Fraud per Kategori Bisnis & Kota")
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
                    labels={"fraud_loss_juta": "Kerugian Fraud (Rp Juta)", "fraud_cases": "Jumlah Kasus"}
                )
                fig_tree.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_tree, use_container_width=True)
            else:
                st.info("Tidak ada kejadian kerugian fraud pada filter yang dipilih.")
            st.caption("💡 **Rekomendasi Aksi:** Prioritaskan verifikasi KYC merchant & audit perangkat pada kategori bernilai transaksi tinggi (`electronics`, `digital_services`, `fashion_apparel`) di kota-kota besar.")

    with c_m2:
        sub_t1, sub_t2 = st.tabs(["🔍 Faktor Utama Pemicu Fraud", "💰 Optimasi Biaya & ROI Threshold"])
        with sub_t1:
            if os.path.exists(fi_file):
                df_fi = pd.read_csv(fi_file).head(9).sort_values(by="importance_pct", ascending=True)
                fig_fi = px.bar(
                    df_fi, x="importance_pct", y="feature", orientation="h",
                    color="importance_pct", color_continuous_scale="Blues", text_auto=".1f",
                    labels={"importance_pct": "Kontribusi (%)", "feature": "Sinyal Risiko"}
                )
                fig_fi.update_layout(height=285, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig_fi, use_container_width=True)
                st.caption("💡 **Rekomendasi Aksi:** Pengurasan saldo (`is_account_emptied`), kata kunci anomali log mesin EDC (`device_risk_flag_count`), dan selisih saldo settlement menyumbang >48% kekuatan deteksi model.")

        with sub_t2:
            fn_avg_cost = 25_000_000
            fp_avg_cost = 250_000
            scale_factor = 5.0

            unmitigated_cost = (opt_metrics.get("tp", 107) + opt_metrics.get("fn", 29)) * fn_avg_cost * scale_factor / 1e9
            def_cost = (def_metrics.get("fn", 38) * fn_avg_cost + def_metrics.get("fp", 14) * fp_avg_cost) * scale_factor / 1e9
            opt_cost = (opt_metrics.get("fn", 29) * fn_avg_cost + opt_metrics.get("fp", 35) * fp_avg_cost) * scale_factor / 1e9

            roi_df = pd.DataFrame({
                "Kebijakan": [
                    "1. Tanpa Model ML (Aturan Statis)",
                    "2. Ambang Standar (0.50)",
                    f"3. Ambang Optimasi Biaya ({opt_threshold:.2f})"
                ],
                "Estimasi_Biaya_30Hari_Miliar": [round(unmitigated_cost, 2), round(def_cost, 2), round(opt_cost, 2)]
            })
            fig_roi = px.bar(
                roi_df, x="Kebijakan", y="Estimasi_Biaya_30Hari_Miliar", color="Kebijakan",
                text_auto=".2f",
                color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"],
                labels={"Estimasi_Biaya_30Hari_Miliar": "Biaya Kerugian + Friksi 30 Hari (Rp Miliar)"}
            )
            fig_roi.update_layout(height=285, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_roi, use_container_width=True)
            st.caption(f"💡 **Mengapa Ambang `{opt_threshold:.2f}`?** Menurunkan total kerugian fraud + biaya friksi operasional 30 hari dari **Rp {unmitigated_cost:.2f} Miliar** menjadi **Rp {opt_cost:.2f} Miliar** (hemat **Rp {unmitigated_cost - opt_cost:.2f} Miliar/bulan**).")

    st.markdown("---")

    # --- BARIS 3: INSIGHT 5 (SEGMENTASI K-MEANS 4 KLASTER & DAFTAR AUDIT PRIORITAS) ---
    sil_text = ""
    if os.path.exists(cluster_model_file):
        c_meta = joblib.load(cluster_model_file)
        sil_val = c_meta.get("silhouette_score")
        pca_var = c_meta.get("pca_explained_variance")
        if sil_val is not None:
            sil_text = f" — Silhouette Score: `{sil_val:.4f}` | Variansi 2D PCA: `{pca_var*100:.1f}%`"

    st.markdown(f"##### 5️⃣ Segmentasi Perilaku & Strategi Risiko Merchant (1.500 Merchant{sil_text})")

    if os.path.exists(seg_file):
        df_seg = pd.read_csv(seg_file)
        df_seg["segmen_id"] = df_seg["segment_name"].map(lambda s: SEG_TRANSLATE.get(s, s))
        df_seg["strategi_id"] = df_seg["segment_name"].map(lambda s: STRAT_TRANSLATE.get(s, s))

        c_cl1, c_cl2 = st.columns([1.05, 1.0])

        with c_cl1:
            fig_pca = px.scatter(
                df_seg, x="pca_1", y="pca_2", color="segmen_id",
                hover_data=["name_dest", "total_tx_count", "total_volume", "avg_ticket_size", "observed_fraud_rate"],
                labels={
                    "pca_1": "Komponen Utama 1 (Volume & Ukuran Transaksi)",
                    "pca_2": "Komponen Utama 2 (Rasio Malam & Kanal Mobile)",
                    "segmen_id": "Segmen Merchant"
                },
                title="Proyeksi 2D PCA dari 4 Klaster Perilaku Merchant"
            )
            fig_pca.update_layout(height=380, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_pca, use_container_width=True)

        with c_cl2:
            if os.path.exists(cluster_summary_file):
                df_csum = pd.read_csv(cluster_summary_file)
                df_csum["Segmen Merchant"] = df_csum["segment_name"].map(lambda s: SEG_TRANSLATE.get(s, s))
                st.dataframe(
                    df_csum[[
                        "Segmen Merchant", "merchant_count", "avg_monthly_volume",
                        "avg_night_ratio", "avg_mobile_online_pct", "avg_fraud_rate_pct"
                    ]].rename(columns={
                        "merchant_count": "Jumlah Merchant",
                        "avg_monthly_volume": "Rata-rata Volume (Rp)",
                        "avg_night_ratio": "Transaksi Malam (%)",
                        "avg_mobile_online_pct": "Mobile/Online (%)",
                        "avg_fraud_rate_pct": "Tingkat Fraud (%)"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            for seg_orig in df_seg["segment_name"].unique():
                sub = df_seg[df_seg["segment_name"] == seg_orig]
                seg_indo = SEG_TRANSLATE.get(seg_orig, seg_orig)
                strat_indo = STRAT_TRANSLATE.get(seg_orig, sub["recommended_strategy"].iloc[0])
                with st.expander(f"📌 {seg_indo} ({len(sub)} Merchant — Tingkat Fraud: {sub['observed_fraud_rate'].mean()*100:.2f}%)"):
                    st.write(f"**Rekomendasi Strategi Bisnis:** {strat_indo}")
                    st.write(
                        f"**Rata-rata Volume Bulanan:** Rp {sub['total_volume'].mean():,.0f} | "
                        f"**Rata-rata Nilai Transaksi:** Rp {sub['avg_ticket_size'].mean():,.0f} | "
                        f"**Rasio Malam:** {sub['night_tx_ratio'].mean()*100:.1f}%"
                    )

        st.markdown("##### 🚨 Daftar Prioritas Audit Merchant (Top Merchant Anomali yang Membutuhkan Inspeksi EDC/Akun Segera)")
        watchlist = df_seg[df_seg["fraud_tx_count"] >= 2].sort_values(
            by=["fraud_tx_count", "observed_fraud_rate", "total_volume"], ascending=[False, False, False]
        ).head(8).copy()
        watchlist["observed_fraud_rate_pct"] = np.round(watchlist["observed_fraud_rate"] * 100, 2)
        watchlist["mobile_online_pct"] = np.round(watchlist["mobile_online_ratio"] * 100, 1)
        watchlist["night_tx_pct"] = np.round(watchlist["night_tx_ratio"] * 100, 1)
        watchlist["total_volume_idr"] = watchlist["total_volume"].apply(lambda v: f"Rp {v:,.0f}")
        st.dataframe(
            watchlist[[
                "name_dest", "segmen_id", "total_tx_count", "total_volume_idr",
                "fraud_tx_count", "observed_fraud_rate_pct", "mobile_online_pct", "night_tx_pct", "strategi_id"
            ]].rename(columns={
                "name_dest": "ID Merchant",
                "segmen_id": "Klaster Perilaku",
                "total_tx_count": "Total Transaksi",
                "total_volume_idr": "Total Volume",
                "fraud_tx_count": "Kasus Fraud",
                "observed_fraud_rate_pct": "Rasio Fraud (%)",
                "mobile_online_pct": "Mobile/Online (%)",
                "night_tx_pct": "Rasio Malam (%)",
                "strategi_id": "Tindakan Investigasi yang Disarankan"
            }),
            use_container_width=True,
            hide_index=True
        )

# ==============================================================================
# TAB 2: PREDIKSI FRAUD TRANSAKSI REAL-TIME
# ==============================================================================
with tabs[1]:
    st.subheader("Simulator Penilaian Risiko & Keputusan Transaksi Real-Time")
    st.caption(f"Ditenagai **Stacking Ensemble ML** — **Setujui (Approve):** `< {opt_threshold:.2f}` | **Verifikasi 2FA OTP:** `{opt_threshold:.2f} - {block_threshold:.2f}` | **Blokir Otomatis (Auto-Block):** `>= {block_threshold:.2f}`")

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

    st.markdown("##### ⚡ Uji Skenario Cepat (Klik untuk Memuat Parameter Otomatis)")
    b_col1, b_col2, b_col3 = st.columns(3)
    if b_col1.button("✅ Skenario 1: Belanja Ritel POS Normal (Disetujui)", use_container_width=True):
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
    if b_col2.button("⚠️ Skenario 2: Transfer Malam Mencurigakan (Wajib 2FA OTP)", use_container_width=True):
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
    if b_col3.button("🚨 Skenario 3: Pengurasan Akun / EMV Fallback (Blokir Otomatis)", use_container_width=True):
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
        tx_type = st.selectbox("Tipe Transaksi", tx_types, index=tx_types.index(st.session_state["p_type"]))
        amount = st.number_input("Nominal Transaksi (IDR)", value=int(st.session_state["p_amount"]), step=5_000_000)
        old_bal = st.number_input("Saldo Awal Pelanggan (IDR)", value=int(st.session_state["p_old_bal"]), step=5_000_000)
        new_bal = st.number_input("Saldo Akhir Pelanggan (IDR)", value=int(st.session_state["p_new_bal"]), step=5_000_000)
        hour = st.slider("Jam Transaksi (WIB)", 0, 23, int(st.session_state["p_hour"]))

    with col_input2:
        terminal = st.selectbox("Tipe Terminal / Kanal Pembayaran", terminals_list, index=terminals_list.index(st.session_state["p_terminal"]))
        category = st.selectbox("Kategori Bisnis Merchant", cats_list, index=cats_list.index(st.session_state["p_category"]))
        dest_discrepancy = st.number_input("Selisih Settlement Merchant (IDR)", value=int(st.session_state["p_disc"]), step=5_000_000)
        remarks = st.text_input("Log Telemetry Perangkat / Catatan POS", st.session_state["p_remarks"])

    if st.button("Evaluasi Risiko Transaksi", type="primary", use_container_width=True):
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
                    title={"text": "Skor Probabilitas Fraud"},
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
                st.markdown("### Keputusan Evaluasi Risiko")
                if prob >= block_threshold:
                    st.error(f"🚨 **RISIKO TINGGI (Skor Risiko: {prob*100:.1f}%) — KEPUTUSAN: BLOKIR OTOMATIS (AUTO-BLOCK)**")
                    st.write("**Protokol Tindakan:** Tolak otorisasi transaksi seketika, bekukan sementara token terminal EDC, dan kirim peringatan ke tim SOC/Fraud Ops.")
                elif prob >= opt_threshold:
                    st.warning(f"⚠️ **RISIKO MENENGAH (Skor Risiko: {prob*100:.1f}%) — KEPUTUSAN: VERIFIKASI 2FA OTP (STEP-UP)**")
                    st.write("**Protokol Tindakan:** Tahan penyelesaian transaksi dan minta pemegang kartu melakukan konfirmasi biometrik / kode OTP SMS.")
                else:
                    st.success(f"✅ **RISIKO RENDAH (Skor Risiko: {prob*100:.1f}%) — KEPUTUSAN: DISETUJUI (APPROVE)**")
                    st.write("**Protokol Tindakan:** Otorisasi penyelesaian transaksi (*instant settlement*) disetujui.")

                factors = []
                if is_empty or ratio >= 0.90:
                    factors.append(f"Pengurasan saldo tinggi ({ratio*100:.1f}% dari saldo pelanggan)")
                if is_night:
                    factors.append("Transaksi pada jam rawan dini hari (00:00 - 05:00 WIB)")
                if risk_flags > 0:
                    factors.append(f"Log telemetry perangkat mengandung {risk_flags} kata kunci keamanan mencurigakan")
                if terminal in ["EDC_MOBILE", "ONLINE_PG"]:
                    factors.append(f"Kanal pembayaran dengan profil risiko lebih tinggi ({terminal})")
                if abs(error_dest) > 100_000:
                    factors.append(f"Terdeteksi selisih saldo settlement merchant (Rp {abs(error_dest):,.0f})")
                if factors:
                    st.markdown("**Sinyal Risiko yang Terpicu:**")
                    for f_item in factors:
                        st.markdown(f"- {f_item}")
                else:
                    st.markdown("**Sinyal Risiko yang Terpicu:** Tidak ada (Profil transaksi normal & aman)")
        else:
            st.warning("File model `models/fraud_detection_pipeline.joblib` belum ditemukan. Silakan jalankan training model terlebih dahulu.")

    st.markdown("---")
    st.markdown("##### 📡 Simulator Antrean Transaksi Live Batch (Simulasi Antrean Payment Gateway)")
    st.caption("Evaluasi 15 transaksi yang masuk secara bersamaan untuk melihat bagaimana model mengarahkan transaksi ke `DISETUJUI`, `WAJIB 2FA OTP`, dan `BLOKIR`.")

    if st.button("🔄 Nilai Sampel Antrean Live (15 Transaksi Masuk)"):
        if os.path.exists(features_parquet) and os.path.exists("models/fraud_detection_pipeline.joblib"):
            df_sample_pool = pd.read_parquet(features_parquet)
            pipeline = joblib.load("models/fraud_detection_pipeline.joblib")

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
                probs_b >= block_threshold, "🚨 BLOKIR",
                np.where(probs_b >= opt_threshold, "⚠️ WAJIB 2FA OTP", "✅ DISETUJUI")
            )
            batch_df["amount_idr"] = batch_df["amount"].apply(lambda v: f"Rp {v:,.0f}")

            st.dataframe(
                batch_df[[
                    "name_dest", "type", "terminal_type", "merchant_category",
                    "hour_of_day", "amount_idr", "device_remarks", "risk_score_pct", "decision"
                ]].rename(columns={
                    "name_dest": "ID Merchant",
                    "type": "Tipe Transaksi",
                    "terminal_type": "Terminal",
                    "merchant_category": "Kategori",
                    "hour_of_day": "Jam (WIB)",
                    "amount_idr": "Nominal",
                    "device_remarks": "Log Telemetry Perangkat",
                    "risk_score_pct": "Skor Risiko (%)",
                    "decision": "Keputusan Gateway"
                }),
                use_container_width=True,
                hide_index=True
            )
