# Modul 11 (BI) - Business Dashboard & Deployment: Visualisasi Insight dengan Streamlit

Tujuan Pembelajaran: Mahasiswa mampu membuat dashboard interaktif untuk Business Intelligence menggunakan Streamlit yang menampilkan grafik performa bisnis secara real-time.

## Materi

### 1) Mengapa Dashboard Penting bagi Bisnis?

Dashboard adalah representasi visual dari informasi terpenting yang diperlukan untuk mencapai satu atau lebih tujuan bisnis. 
- Efisiensi: Manajer tidak perlu membaca laporan tabel yang panjang.
- Real-time: Keputusan dapat diambil berdasarkan data terbaru.
- Fokus: Menampilkan Key Performance Indicators (KPI).

### 2) Komponen Dashboard BI

- **Metrik Utama (KPI)**: Angka tunggal yang sangat krusial (misal: Total Laba, Jumlah Pelanggan Baru).
- **Grafik Tren (Line Chart)**: Melihat pertumbuhan dari waktu ke waktu.
- **Grafik Perbandingan (Bar/Pie Chart)**: Perbandingan antar kategori produk atau wilayah.
- **Filter/Slicer**: Memungkinkan pengguna untuk mengganti rentang waktu atau kategori tertentu.

---

## Praktikum: Membangun Dashboard Penjualan di Google Colab

Kita akan membangun dashboard bisnis sederhana menggunakan library `Streamlit`.

### 1. Menulis Kode Dashboard (`dashboard.py`)

Gunakan magic command di cell Google Colab:

```python
%%writefile dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Data Penjualan (Simulasi DB)
data = {
    'Produk': ['Laptop', 'HP', 'Televisi', 'Laptop', 'HP', 'Televisi', 'Laptop', 'HP'],
    'Wilayah': ['Jawa', 'Sumatra', 'Bali', 'Jawa', 'Sumatra', 'Jawa', 'Bali', 'Bali'],
    'Penjualan': [15, 12, 10, 18, 14, 11, 25, 9],
    'Bulan': ['Jan', 'Jan', 'Jan', 'Feb', 'Feb', 'Mar', 'Mar', 'Mar']
}
df = pd.DataFrame(data)

# Judul Dashboard
st.set_page_config(layout="wide")
st.title("Business Intelligence Dashboard 📊")
st.markdown("Dashboard ini menampilkan performa penjualan unit per wilayah.")

# Bagian KPI (Metrik Utama)
col1, col2, col3 = st.columns(3)
col1.metric("Total Produk Terjual", df['Penjualan'].sum())
col2.metric("Rata-rata Penjualan", f"{df['Penjualan'].mean():.2f}")
col3.metric("Jumlah Wilayah", df['Wilayah'].nunique())

# Bagian Visualisasi Utama
col4, col5 = st.columns(2)

with col4:
    st.subheader("Tren Penjualan per Produk")
    fig_bar = px.bar(df, x='Bulan', y='Penjualan', color='Produk', barmode='group')
    st.plotly_chart(fig_bar, use_container_width=True)

with col5:
    st.subheader("Distribusi Penjualan per Wilayah")
    fig_pie = px.pie(df, values='Penjualan', names='Wilayah')
    st.plotly_chart(fig_pie, use_container_width=True)

# Fitur Filter
wilayah_pilih = st.selectbox("Pilih Wilayah untuk Detail Data:", df['Wilayah'].unique())
st.write(f"Menampilkan data untuk wilayah: {wilayah_pilih}")
st.table(df[df['Wilayah'] == wilayah_pilih])
```

### 2. Deployment Sementara via Google Colab

Gunakan perintah terminal di cell baru:

```bash
!pip install streamlit plotly
!npm install -g localtunnel
!streamlit run dashboard.py & npx localtunnel --port 8501
```

---

## Latihan/Tugas

1.  Ubah `st.title` Dashboard Anda menjadi nama perusahaan fiktif buatan Anda sendiri. 
2.  Tambahkan satu kolom baru "Profit" pada dataset di atas dan buatlah satu diagram batang (`px.bar`) baru yang menampilkan profit per produk.
3.  Berikan pendapat Anda: Apa perbedaan utama antara dashboard operasional (untuk staf harian) dengan dashboard strategis (untuk pemilik perusahaan/CEO) dalam konteks BI?
