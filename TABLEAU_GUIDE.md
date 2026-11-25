# Panduan Menghubungkan & Membuat Dashboard Tableau Public

File dataset yang telah diekstrak dan siap di-drag-and-drop ke Tableau sudah tersedia di direktori:
📂 `data/tableau/`

Berikut adalah 3 dataset utama yang telah diagregasi & distrukturkan khusus untuk visualisasi BI:
1. `channel_location_risk.csv` (Risiko fraud berdasarkan tipe terminal EDC/QRIS, kategori merchant, dan kota)
2. `hourly_fraud_risk.csv` (Pola siklus risiko fraud per jam 00:00 - 23:00 WIB dan tipe transaksi)
3. `merchant_segments.csv` (Hasil clustering K-Means & koordinat PCA untuk 1.500 profil merchant)

---

## 🛠️ Langkah-Langkah Membuat Dashboard di Tableau Public (Gratis)

### Langkah 1: Buka Tableau & Hubungkan Data
1. Buka software **Tableau Desktop** / **Tableau Public** (atau via web [Tableau Public Web Authoring](https://public.tableau.com/)).
2. Pilih menu **Connect** -> **Text file**.
3. Pilih salah satu file dari folder `data/tableau/channel_location_risk.csv`.

---

### Langkah 2: Buat 4 Sheet Visualisasi

#### Sheet 1: Heatmap Risiko Terminal Device vs Kategori Merchant
- **Columns**: `terminal_type` (EDC_MOBILE, EDC_STATIC, QRIS, dsb.)
- **Rows**: `merchant_category`
- **Color**: `fraud_rate_pct` (Ubah palette ke *Red-Gold diverging* atau *Orange-Blue*)
- **Label / Text**: `fraud_rate_pct`
- *Insight yang ditampilkan*: Terlihat jelas bahwa EDC_MOBILE dan ONLINE_PG memiliki intensitas fraud tertinggi.

#### Sheet 2: Analisis Siklus Jam Transaksi (Hourly Line Chart)
- Hubungkan data `hourly_fraud_risk.csv`.
- **Columns**: `hour_of_day` (Continuous, 0 sampai 23).
- **Rows**: `fraud_rate_pct` (Measure: Average) dan `total_volume_idr` (Dual Axis).
- **Color**: `type` (TRANSFER, CASH_OUT, PAYMENT).
- *Insight yang ditampilkan*: Lonjakan tajam fraud pada jam 00:00 - 05:00 WIB (*Nocturnal Risk Spike*).

#### Sheet 3: Sebaran Geografis Risiko Fraud per Kota di Indonesia
- Pakai data `channel_location_risk.csv`.
- Klik kanan kolom `location` -> pilih **Geographic Role** -> **City** (atau buat simbol peta berdasarkan kota: Jakarta, Surabaya, Bandung, Medan, Bali/Denpasar, Makassar).
- **Size**: `total_volume_idr`
- **Color**: `fraud_loss_idr`
- *Insight yang ditampilkan*: Menampilkan kota dengan volume settlement terbesar dan total potensi fraud yang berhasil dicegah.

#### Sheet 4: Scatter Plot Segmentasi Merchant (K-Means & PCA)
- Hubungkan data `merchant_segments.csv`.
- **Columns**: `pca_1`
- **Rows**: `pca_2`
- **Color**: `segment_name` (*High-Volume Enterprise Retailers* vs *Standard Retail Merchants*)
- **Tooltip**: `name_dest`, `total_volume`, `observed_fraud_rate`, `recommended_strategy`
- *Insight yang ditampilkan*: Membuktikan pemahaman Machine Learning Unsupervised untuk strategi akun merchant di Prima Vista.

---

### Langkah 3: Gabungkan Menjadi 1 Dashboard Interaktif
1. Klik **New Dashboard** di bagian bawah Tableau.
2. Atur ukuran ke **Automatic** atau **Generic Desktop (1366 x 768)**.
3. Tarik Sheet 1, Sheet 2, Sheet 3, dan Sheet 4 ke dalam layout dashboard.
4. Tambahkan **Text Box** di bagian atas untuk Header:
   > **PT Prima Vista Solusi — Executive Payment Risk & Terminal Fraud Center**
5. Aktifkan tombol **"Use as Filter"** pada Sheet Heatmap Terminal agar saat user mengklik "EDC_MOBILE", seluruh grafik lainnya ikut tersaring secara otomatis.

---

### Langkah 4: Publikasikan ke Tableau Public
1. Klik menu **Server** -> **Tableau Public** -> **Save to Tableau Public As...**.
2. Beri judul: `PT Prima Vista Solusi - Payment Fraud & Merchant Risk Analytics`.
3. Salin URL publiknya (contoh: `https://public.tableau.com/app/profile/azhar/viz/...`) dan cantumkan di CV serta `README.md` GitHub kamu.

