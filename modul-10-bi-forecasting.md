# Modul 10 (BI) - Analisis Prediktif & Forecasting: Sales Prediction

Tujuan Pembelajaran: Mahasiswa mampu menggunakan teknik regresi untuk memprediksi tren bisnis (penjualan) di masa depan berdasarkan data historis.

## Materi

### 1) Apa itu Analisis Prediktif?

Analisis prediktif adalah penggunaan data, algoritma statistik, dan teknik machine learning untuk mengidentifikasi probabilitas hasil di masa depan berdasarkan data historis. Tujuannya adalah untuk meminimalkan risiko dan memaksimalkan peluang bisnis.

### 2) Peramalan (Forecasting) vs Prediksi (Prediction)

| Istilah | Fokus | Contoh |
| --- | --- | --- |
| **Forecasting** | Urutan waktu (Time Series) | "Berapa penjualan roti di hari Senin depan?" |
| **Prediction** | Fitur yang menyertai suatu kejadian | "Apakah pelanggan ini akan membeli mobil mewah?" |

### 3) Konsep ROI dalam Machine Learning

Di dunia bisnis, akurasi model ML harus berujung pada penghematan biaya atau peningkatan pendapatan. Contoh: Stok yang terlalu banyak (stok mati) vs Stok yang kurang (kehilangan peluang jual).

---

## Praktikum: Prediksi Penjualan Berdasarkan Iklan

Kita akan memprediksi total penjualan berdasarkan anggaran iklan di berbagai media (TV, Radio, Newspaper).

### 1. Load Dataset

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Dataset Iklan & Penjualan (Simulasi sederhana)
data = {
    'TV': [230, 44, 17, 151, 180, 8, 57, 120, 199, 100],
    'Radio': [37, 39, 45, 41, 10, 2, 48, 32, 2, 12],
    'Newspaper': [69, 45, 69, 58, 58, 1, 23, 11, 2, 6],
    'Sales': [22, 10, 9, 16, 12, 4, 11, 15, 12, 10]
}
df = pd.DataFrame(data)

# Korelasi antar iklan dan penjualan
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
plt.title('Korelasi Anggaran Media vs Penjualan')
plt.show()

# Split Data (80% Training, 20% Testing)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Training & Peramalan Sederhana

```python
# Model Linear Regression
model_sales = LinearRegression()
model_sales.fit(X_train, y_train)

# Prediksi Penjualan Masa Depan
pred_sales = model_sales.predict(X_test)

# Evaluasi Prediksi
mae = mean_absolute_error(y_test, pred_sales)
r2 = r2_score(y_test, pred_sales)

print(f"Rata-rata Kesalahan Prediksi: ± {mae:.2f} Unit")
print(f"Skor Akurasi Prediksi (R-Squared): {r2 * 100:.2f}%")
```

---

## Latihan/Tugas

1.  Berdasarkan heatmap di atas, media cetak manakah (TV/Radio/Newspaper) yang memiliki hubungan terkuat dengan jumlah penjualan?
2.  Gunakan model di atas untuk memprediksi berapa penjualan yang didapat jika perusahaan menganggarkan TV=200, Radio=50, dan Newspaper=10.
3.  Cari dataset "Advertising" di Kaggle (yang memiliki ~200 baris data). Terapkan langkah di atas dan simpulkan media mana yang paling efektif.
