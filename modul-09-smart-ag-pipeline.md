# Modul 09 - Arsitektur Smart Agriculture & Data Pipeline

Tujuan Pembelajaran: Mahasiswa memahami arsitektur sistem pertanian cerdas (*Smart Agriculture*) dan mampu mengolah data sensor IoT mentah menjadi dataset yang siap digunakan untuk Machine Learning.

## Materi

### 1) Arsitektur Smart Agriculture

Sistem Smart Agriculture modern melibatkan beberapa layer:
- **Perangkat IoT (Edge)**: Sensor kelembapan tanah, suhu, kelembapan udara, sensor NPK, pH tanah.
- **Data Pipeline**: Alur pengiriman data dari sensor ke cloud (biasanya menggunakan protokol MQTT atau HTTP).
- **Prosesing (Cloud)**: Tempat data disimpan (database) dan diproses (Google Colab, AWS, Azure).
- **Aksi/Insight**: Dashboard pemantauan, sistem irigasi otomatis, prediksi penyakit tanaman.

### 2) Data Ingestion & Cleaning

Data sensor IoT seringkali memiliki masalah:
- **Data Kosong (Missing Values)**: Karena sensor mati atau kehilangan sinyal.
- **Outliers**: Karena gangguan sensor (misal: suhu terbaca 200°C tiba-tiba).
- **Duplikasi**: Karena pengiriman ulang paket data yang gagal sebelumnya.

---

## Praktikum: Mengolah Data IoT di Google Colab

Kita akan mensimulasikan dataset sensor dari sebuah lahan pertanian.

### 1. Simulasi Load Data & Cleaning

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset dummy sensor tanah
# Kolom: Timestamp, Soil_Moisture (%), Temperature (°C), Humidity (%)
data = {
    'Timestamp': pd.date_range(start='2026-03-01', periods=10, freq='H'),
    'Soil_Moisture': [30, 32, np.nan, 35, 34, 150, 31, 30, 28, 29], # Ada outlier 150 dan NaN
    'Temperature': [25, 26, 26, 27, 26, 28, 27, 26, 25, 25],
    'Humidity': [70, 71, 72, 70, 69, 70, 71, 72, 73, 72]
}
df = pd.DataFrame(data)

# 1. Menangani Missing Value (NaN) dengan data sebelumnya (Forward Fill)
df['Soil_Moisture'] = df['Soil_Moisture'].ffill()

# 2. Menangani Outlier (Kelembapan tanah tidak mungkin di atas 100%)
# Kita ganti outlier dengan nilai rata-rata atau batas atas masuk akal
df.loc[df['Soil_Moisture'] > 100, 'Soil_Moisture'] = df['Soil_Moisture'].median()

# Tampilkan data bersih
print(df)
```

### 2. Monitoring & Thresholding (Praktis)

Dalam Smart Agriculture, kita sering memerlukan sistem peringatan dini.

```python
def check_irrigation_need(moisture_value):
    # Ambang batas siram: kelembapan di bawah 30%
    if moisture_value < 30:
        return "Nyalakan Irigasi!"
    else:
        return "Kondisi Aman"

# Tambah kolom status irigasi
df['Action'] = df['Soil_Moisture'].apply(check_irrigation_need)
print(df[['Timestamp', 'Soil_Moisture', 'Action']])
```

---

## Latihan/Tugas

1.  Berdasarkan materi arsitektur di atas, jelaskan bila kita punya 1000 sensor yang mengirim data setiap 1 detik, apa dampaknya terhadap Google Colab jika kita melatih model secara langsung tanpa *Data Pipeline* yang baik?
2.  Cari dataset "Soil Moisture and Temperature" di Kaggle, lalu lakukan pembersihan data (cleaning) terhadap nilai yang dianggap tidak masuk akal (misal: kelembapan negatif).
3.  Buatlah fungsi `check_pest_risk` yang mengembalikan pesan "Waspada Penyakit!" jika kelembapan udara (Humidity) di atas 80% dan suhu tanah di atas 30°C.
