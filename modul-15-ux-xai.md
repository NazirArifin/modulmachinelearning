# Modul 15 - UX untuk Sistem AI: Explainable AI (XAI)

Tujuan Pembelajaran: Mahasiswa memahami pentingnya transparansi dalam model Machine Learning (agar tidak sekadar "Black Box") dan mampu memvisualisasikan faktor pendukung prediksi model (XAI).

## Materi

### 1) Apa itu Explainable AI (XAI)?

Explainable AI (XAI) adalah sekumpulan metode dan teknik yang memungkinkan pengguna manusia memahami dan mempercayai hasil serta *output* yang dibuat oleh algoritma *Machine Learning*. 

Contoh: Tidak cukup bagi dokter untuk tahu "Pasien ini sakit X", dokter perlu tahu "Mengapa model menebak sakit X? (karena fitur A sangap tinggi)".

### 2) Black Box vs White Box Model

- **White Box**: Model yang secara alami mudah dijelaskan (misal: Decision Tree, Linear Regression).
- **Black Box**: Model yang sangat akurat tapi sulit dipahami cara kerjanya secara internal (misal: Deep Learning, XGBoost, Random Forest).

### 3) Teknik XAI: SHAP & LIME

Dua library paling populer untuk memberikan penjelasan pada model Black Box:
- **SHAP (SHapley Additive exPlanations)**: Berdasarkan teori permainan untuk mengalokasikan pengaruh ke setiap fitur.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Menjelaskan prediksi individu dengan melatih model lokal yang dapat diinterpretasi.

---

## Praktikum: Menjelaskan Hasil Prediksi Model

Kita akan menggunakan library `SHAP` untuk melihat fitur apa yang paling berpengaruh pada model kita secara visual.

### 1. Training Model

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Data Breast Cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Model Random Forest (Black Box)
model = RandomForestClassifier()
model.fit(X, y)
```

### 2. Implementasi SHAP

*Catatan: Anda mungkin butuh `!pip install shap`*

```python
import shap

# Membuat explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualisasi ringkasan fitur (Summary Plot)
shap.summary_plot(shap_values[1], X)
```

**Interpretasi:** Fitur di bagian paling atas adalah fitur yang memiliki dampak paling besar terhadap keputusan model. Warna merah menunjukkan nilai fitur yang tinggi, warna biru menunjukkan nilai yang rendah.

---

## Latihan/Tugas

1.  Berdasarkan hasil `summary_plot` di atas, sebutkan 3 fitur utama yang paling memengaruhi model untuk menebak "Malignant" (Ganas) pada penyakit tersebut.
2.  Mengapa **Explainable AI** sangat kritis dalam bidang-bidang seperti Hukum (Legal), Kesehatan (Medical), dan Keuangan (Finance)?
3.  Implementasikan SHAP pada model Churn Prediction yang Anda buat di Modul 07. Apakah hasil SHAP sama dengan hasil `feature_importances_` bawaan Random Forest/XGBoost?
