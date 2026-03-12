# Modul 10 - Analisis Spasial & Time-Series di Pertanian

Tujuan Pembelajaran: Mahasiswa memahami dataset bertipe urutan waktu (*Time-Series*) dan mampu menggunakannya untuk memprediksi hasil panen atau kebutuhan air pada tanaman.

## Materi

### 1) Apa itu Time-Series Data?

Data Time-Series adalah urutan observasi yang diambil pada waktu tertentu (misal: suhu setiap jam selama setahun). Di pertanian, data ini sangat penting untuk memahami musim, periode tanam, dan pola iklim.

### 2) Karakteristik Data Pertanian

- **Tren**: Peningkatan/penurunan dalam jangka panjang (misal: suhu bumi meningkat).
- **Musiman (Seasonality)**: Pola yang berulang pada periode tertentu (misal: musim hujan setiap akhir tahun).
- **Stasioneritas**: Sifat data yang statistikanya (rata-rata/variansi) konstan dari waktu ke waktu.

### 3) Analisis Spasial

Analisis spasial melibatkan data koordinat (Latitude/Longitude) untuk memetakan kondisi lahan. Contoh: pembuatan peta kesuburan tanah atau peta deteksi dini hama.

---

## Praktikum: Prediksi Hasil Panen Sederhana (Regression)

Kita akan menggunakan teknik regresi untuk memprediksi hasil panen berdasarkan curah hujan dan pupuk.

### 1. Load & Explore Data

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Simulasi data panen padi
# Area: ID Sawah, Rainfall: Curah Hujan (mm), Fertilizer: Kg Pupuk, Yield: Hasil Panen (Ton)
data = {
    'Rainfall': [1200, 1300, 1250, 1100, 1400, 1500, 1350, 1150, 1200, 1450],
    'Fertilizer': [50, 55, 52, 45, 60, 65, 58, 48, 51, 62],
    'Yield': [4.5, 5.0, 4.8, 4.1, 5.4, 5.9, 5.2, 4.3, 4.6, 5.6]
}
df = pd.DataFrame(data)

# Korelasi antar fitur
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
plt.title('Korelasi Fitur Pertanian')
plt.show()

# Split data
X = df[['Rainfall', 'Fertilizer']]
y = df['Yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Modeling & Prediction

```python
# Training Model Regresi Linear
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)

# Prediksi contoh data baru
new_data = [[1300, 55]] # Curah hujan 1300, pupuk 55
pred_yield = model_reg.predict(new_data)
print(f"Prediksi Hasil Panen: {pred_yield[0]:.2f} Ton")
```

---

## Latihan/Tugas

1.  Mengapa **Linear Regression** cocok digunakan untuk prediksi numerik (seperti tonase panen) sedangkan **Logistic Regression** tidak?
2.  Cari data pertumbuhan tanaman (misal: tinggi jagung vs hari setelah tanam) dan gunakan **Moving Average** (rata-rata bergerak) untuk menghaluskan fluktuasi data harian di Pandas (`df['kolom'].rolling(window=3).mean()`).
3.  Berikan pendapat Anda: Manakah yang lebih berpengaruh terhadap hasil panen di dataset di atas: curah hujan atau penggunaan pupuk? (Petunjuk: Cek nilai Koefisien model di `model_reg.coef_`).
