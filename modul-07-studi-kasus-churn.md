# Modul 07 - Studi Kasus Business Case: Customer Churn Prediction

Tujuan Pembelajaran: Mahasiswa mampu menerapkan seluruh alur kerja Machine Learning (dari EDA hingga Evaluasi) pada sebuah problem bisnis nyata: Churn Prediction.

## Materi

### 1) Apa itu Customer Churn?

Customer Churn adalah kejadian di mana pelanggan berhenti menggunakan produk atau layanan perusahaan (misal: berhenti berlangganan ISP, bank, atau aplikasi). Memprediksi siapa yang akan *churn* memungkinkan perusahaan untuk memberikan promo agar mereka tidak pergi.

### 2) Alur Kerja Proyek

1.  **Definisi Masalah**: Memprediksi kolom `Churn` (Target: Ya/Tidak).
2.  **Eksplorasi Data (EDA)**: Mencari korelasi antar fitur (misal: apakah pelanggan dengan biaya bulanan tinggi lebih cenderung pergi?).
3.  **Preprocessing**: Handling missing values, encoding data kategorikal, scaling.
4.  **Modeling & Evaluation**: Memilih model terbaik berdasarkan F1-Score (karena data churn biasanya imbalanced).

---

## Praktikum: Membangun Churn Prediction Model

### 1. Persiapan Dataset

Gunakan dataset populer `Telco Customer Churn` yang sudah dibersihkan.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dummy data (simulasi Telco Churn)
data = {
    'gender': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1], # 1: Male, 0: Female
    'SeniorCitizen': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 29.85, 99.85, 89.10, 24.50, 67.50, 45.40],
    'TotalCharges': [29.85, 1889.50, 108.15, 1840.75, 151.65, 820.50, 1400.10, 48.00, 1500.20, 120.40],
    'Churn': [0, 0, 1, 0, 0, 1, 1, 0, 1, 0] # Target
}
df = pd.DataFrame(data)

# EDA Sederhana
sns.countplot(data=df, x='Churn')
plt.title('Distribusi Churn')
plt.show()

# Pisahkan Fitur - Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Modeling & Evaluation

```python
# Training Model Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

# Evaluasi Lengkap
y_pred = model_rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 3. Feature Importance

```python
# Mengetahui fitur mana yang paling berpengaruh terhadap keputusan Churn
importances = pd.Series(model_rf.feature_importances_, index=X.columns)
importances.nlargest(5).plot(kind='barh')
plt.title('Fitur Terpenting dalam Memprediksi Churn')
plt.show()
```

---

## Latihan/Tugas

1.  Berdasarkan grafik **Feature Importance**, manakah faktor terkuat yang menyebabkan pelanggan berpindah?
2.  Carilah dataset "Telco Churn" asli di Kaggle (yang memiliki ~7000 baris data). Terapkan model XGBoost yang dipelajari di Modul 4 dan bandingkan hasilnya dengan Random Forest!
3.  Berikan satu rekomendasi bisnis untuk perusahaan tersebut berdasarkan hasil model Anda (Contoh: "Beri diskon untuk pelanggan dengan MonthlyCharges di atas X rupiah").
