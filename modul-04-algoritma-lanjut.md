# Modul 04 - Algoritma Lanjut: SVM, Random Forest, & XGBoost

Tujuan Pembelajaran: Mahasiswa mampu mengimplementasikan algoritma Machine Learning yang lebih kompleks (Support Vector Machine dan Ensemble Methods) untuk mendapatkan performa prediksi yang lebih baik.

## Materi

### 1) Support Vector Machine (SVM)

SVM bekerja dengan cara mencari *hyperplane* (bidang pemisah) terbaik yang memisahkan dua kelas data dengan margin maksimal. Bagus untuk data dimensi tinggi.

### 2) Random Forest

Random Forest adalah kumpulan dari banyak *Decision Tree* (Ensemble). Setiap pohon memberikan suara (*vote*), dan hasil terbanyak diambil sebagai prediksi akhir. Ini sangat efektif mengurangi *Overfitting*.

### 3) XGBoost (Extreme Gradient Boosting)

XGBoost adalah implementasi *Gradient Boosted Decision Trees* yang sangat cepat dan akurat. Algoritma ini sering memenangkan kompetisi ML di Kaggle karena efisiensinya.

---

## Praktikum: Perbandingan SVM, Random Forest, & XGBoost

Gunakan dataset `Breast Cancer` secara praktis untuk membandingkan ketiga algoritma ini.

### 1. Persiapan Data

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### 2. Implementasi SVM

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train, y_train)
acc_svm = accuracy_score(y_test, model_svm.predict(X_test))
print(f"Akurasi SVM: {acc_svm * 100:.2f}%")
```

### 3. Implementasi Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, model_rf.predict(X_test))
print(f"Akurasi Random Forest: {acc_rf * 100:.2f}%")
```

### 4. Implementasi XGBoost

Jika belum terinstall, gunakan `!pip install xgboost`.

```python
from xgboost import XGBClassifier

model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
acc_xgb = accuracy_score(y_test, model_xgb.predict(X_test))
print(f"Akurasi XGBoost: {acc_xgb * 100:.2f}%")
```

---

## Latihan/Tugas

1.  Bandingkan akurasi ketiga model di atas. Model mana yang memberikan hasil terbaik untuk dataset ini?
2.  Ubah parameter `kernel` pada SVM menjadi `rbf`. Apakah ada perubahan akurasi?
3.  Ubah jumlah `n_estimators` pada Random Forest menjadi `10` dan `500`. Bagaimana pengaruhnya terhadap kecepatan *training* dan akurasi?
