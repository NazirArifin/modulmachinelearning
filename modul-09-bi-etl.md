# Modul 09 (BI) - Arsitektur Business Intelligence & ETL Dasar

Tujuan Pembelajaran: Mahasiswa memahami arsitektur Business Intelligence (BI) dan mampu melakukan proses ETL (Extract, Transform, Load) sederhana untuk menyiapkan data siap analisis.

## Materi

### 1) Apa itu Business Intelligence (BI)?

Business Intelligence adalah sekumpulan teknik dan alat untuk mentransformasi data mentah menjadi informasi yang bermakna bagi analisis bisnis. Fokus utama BI adalah memberikan wawasan (*insight*) untuk pengambilan keputusan.

### 2) Arsitektur BI

Arsitektur BI biasanya melibatkan:
- **Data Sources**: ERP, CRM, file Excel, database operasional.
- **ETL (Extract, Transform, Load)**: Proses mengambil data, membersihkan/mengubahnya, dan memasukkannya ke tempat penyimpanan baru.
- **Data Warehouse**: Penyimpanan terpusat untuk data yang sudah dibersihkan.
- **Data Visualization/Reporting**: Dashboard untuk pengguna akhir (PowerBI, Tableau, Streamlit).

### 3) Konsep ETL

- **Extract**: Mengambil data dari berbagai format (CSV, SQL, API).
- **Transform**: Membersihkan data (handling nulls, formatting dates, aggregating).
- **Load**: Menyimpan data hasil transformasi ke sistem target.

---

## Praktikum: Pipeline ETL Sederhana di Google Colab

Kita akan mensimulasikan proses ETL dari data transaksi penjualan mentah.

### 1. Extract: Memuat Data Mentah

```python
import pandas as pd
import numpy as np

# Simulasi data transaksi mentah dari 2 cabang (Extract)
data_cabang_a = {
    'transaksi_id': [101, 102, 103],
    'tgl': ['2026-03-01', '01/03/2026', '2026-03-02'],
    'nilai': [50000, 75000, np.nan]
}
data_cabang_b = {
    'transaksi_id': [201, 202],
    'tgl': ['2026-03-01', '2026-03-03'],
    'nilai': [120000, 45000]
}

df_a = pd.DataFrame(data_cabang_a)
df_b = pd.DataFrame(data_cabang_b)

# Menggabungkan data (Join/Union)
df_raw = pd.concat([df_a, df_b], ignore_index=True)
print("Data Mentah Gabungan:")
print(df_raw)
```

### 2. Transform: Pembersihan & Standardisasi

```python
# 1. Standardisasi format tanggal
df_raw['tgl'] = pd.to_datetime(df_raw['tgl'], dayfirst=True, errors='coerce')

# 2. Menghapus atau mengisi data kosong (Cleaning)
df_raw['nilai'] = df_raw['nilai'].fillna(df_raw['nilai'].mean())

# 3. Menambahkan kolom baru (Aggregation/Calculation)
df_raw['pajak'] = df_raw['nilai'] * 0.11
df_raw['total'] = df_raw['nilai'] + df_raw['pajak']

print("\nData Setelah Transformasi:")
print(df_raw)
```

### 3. Load: Menyimpan ke "Data Warehouse" (CSV)

```python
# Load ke file target
df_raw.to_csv('data_penjualan_bi_clean.csv', index=False)
print("\nData berhasil disimpan ke 'data_penjualan_bi_clean.csv'")
```

---

## Latihan/Tugas

1.  Jelaskan perbedaan utama antara **Data Warehouse** dengan **Database Operasional** (seperti database kasir di minimarket)!
2.  Implementasikan tahap **Transform** baru: Tambahkan kolom "Kategori_Nilai" yang berisi "Tinggi" jika total > 100.000 dan "Rendah" jika sebaliknya.
3.  Cari dataset "Sales Data" di Kaggle, lalu lakukan proses ETL sederhana untuk membersihkan kolom tanggal dan nama kota.
