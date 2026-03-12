# Modul 03 - Algoritma Klasik: Logistic Regression & Decision Tree

Tujuan Pembelajaran: Mahasiswa mampu mengimplementasikan algoritma klasifikasi klasik (Logistic Regression dan Decision Tree) menggunakan Scikit-Learn dan memahami cara kerjanya.

## Materi

### 1) Logistic Regression

Meskipun namanya "regresi", algoritma ini digunakan untuk **klasifikasi** (biasanya biner: Ya/Tidak, 0/1). Algoritma ini memprediksi probabilitas sebuah observasi masuk ke dalam kelas tertentu menggunakan fungsi *Sigmoid*.

- Output: Nilai antara 0 dan 1.
- Kelebihan: Sederhana, cepat, dan mudah diinterpretasi.

### 2) Decision Tree

Decision Tree adalah model prediksi yang menggunakan struktur pohon. Data dibagi secara bercabang berdasarkan nilai fitur tertentu hingga mencapai keputusan akhir (daun/leaf).

- Cocok untuk data numerik maupun kategorikal.
- Memiliki sifat *Non-linear*.
- Kelemahan: Mudah mengalami *Overfitting* jika pohon terlalu dalam.

---

## Praktikum: Implementasi Logistic Regression & Decision Tree

Buka [Google Colab](https://colab.research.google.com/) dan jalankan kode berikut.

### 1. Persiapan Dataset

Kita gunakan dataset `Social_Network_Ads` yang menggambarkan apakah seseorang membeli produk setelah melihat iklan.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset buatan sederhana (untuk ilustrasi praktis)
data = {
    'Age': [19, 35, 26, 27, 32, 25, 34, 45, 28, 48, 43, 49, 21, 52],
    'EstimatedSalary': [19000, 20000, 43000, 58000, 76000, 33000, 44000, 26000, 150000, 33000, 112000, 36000, 42000, 150000],
    'Purchased': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Pisahkan Fitur dan Label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split Data (80% Latih, 20% Uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Wajib untuk Logistic Regression)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### 2. Implementasi Logistic Regression

```python
# Training Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Prediksi
y_pred_log = log_reg.predict(X_test)

# Evaluasi
print(f"Akurasi Logistic Regression: {accuracy_score(y_test, y_pred_log) * 100:.2f}%")
```

### 3. Implementasi Decision Tree

```python
# Training Model
tree_clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
tree_clf.fit(X_train, y_train)

# Prediksi
y_pred_tree = tree_clf.predict(X_test)

# Evaluasi
print(f"Akurasi Decision Tree: {accuracy_score(y_test, y_pred_tree) * 100:.2f}%")
```

---

## Latihan/Tugas

1.  Mengapa **Feature Scaling** (`StandardScaler`) sangat penting untuk algoritma *Logistic Regression* tetapi kurang kritis untuk *Decision Tree*?
2.  Ganti parameter `criterion='entropy'` pada `DecisionTreeClassifier` menjadi `criterion='gini'`. Amati apakah ada perbedaan pada akurasi.
3.  Carilah dataset "Titanic" di Kaggle atau lewat library Seaborn (`sns.load_dataset('titanic')`), lalu coba prediksi apakah penumpang selamat menggunakan salah satu algoritma di atas!
