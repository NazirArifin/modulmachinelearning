# Modul 06 - Evaluasi Model II: ROC-AUC, Cross-Validation, & Imbalanced Data

Tujuan Pembelajaran: Mahasiswa mampu melakukan validasi model yang lebih mendalam serta menangani data yang tidak seimbang (*imbalanced*).

## Materi

### 1) K-Fold Cross Validation

Metode validasi di mana data dibagi menjadi $k$ bagian (*folds*). Model dilatih $k$ kali, masing-masing menggunakan $k-1$ bagian untuk training dan 1 bagian untuk testing. Hasil akhirnya adalah rata-rata dari semua pengujian tersebut. 

Tujuannya: Menghindari keberuntungan/kebetulan semesta saat membagi data (*split*).

### 2) ROC-AUC Curve

ROC (*Receiver Operating Characteristic*) adalah grafik yang menggambarkan performa model klasifikasi pada berbagai tingkat ambang batas (*threshold*). AUC (*Area Under the Curve*) mengukur seluruh area dua dimensi di bawah kurva ROC. 

- Semakin mendekati 1, model semakin baik dalam membedakan kelas.

### 3) Menangani Imbalanced Data (SMOTE)

Data yang sangat tidak seimbang bisa "menipu" model. Salah satu cara menanganinya adalah dengan **Oversampling** kelas yang jumlahnya sedikit. 
**SMOTE** (*Synthetic Minority Over-sampling Technique*) menciptakan data sintetis baru, bukan sekadar menduplikasi data yang ada.

---

## Praktikum: Validasi dan Penanganan Imbalanced Data

Jalankan eksperimen berikut di Google Colab.

### 1. K-Fold Cross Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

# Load Data
digits = load_digits()
X, y = digits.data, digits.target

# Model RF
model_rf = RandomForestClassifier(n_estimators=100)

# K-Fold CV (5 Folds)
skor_cv = cross_val_score(model_rf, X, y, cv=5)
print(f"Skor tiap Fold: {skor_cv}")
print(f"Rata-rata Akurasi: {skor_cv.mean():.2f}")
```

### 2. ROC-AUC (Biner)

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Data Dummy
X_bin, y_bin = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Model
log_reg = LogisticRegression()
log_reg.fit(X_bin, y_bin)

# Hitung Probabilitas
y_probs = log_reg.predict_proba(X_bin)[:, 1]

# Plot ROC
fpr, tpr, thresholds = roc_curve(y_bin, y_probs)
auc = roc_auc_score(y_bin, y_probs)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f"Skor AUC: {auc:.2f}")
```

### 3. Implementasi SMOTE

*Catatan: Anda mungkin butuh `!pip install imbalanced-learn`*

```python
from imblearn.over_sampling import SMOTE
from collections import Counter

# Sebelum SMOTE
print("Sebelum SMOTE:", Counter(y_bin))

# Melakukan SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_bin, y_bin)

# Setelah SMOTE
print("Setelah SMOTE:", Counter(y_resampled))
```

---

## Latihan/Tugas

1.  Mengapa **K-Fold Cross Validation** lebih baik digunakan daripada `train_test_split` tunggal saat datanya sedikit?
2.  Apa yang dimaksud dengan nilai **AUC = 0.5**? Apakah model tersebut berguna? Jelaskan!
3.  Implementasikan SMOTE pada proyek klasifikasi nyata (misalnya dataset "Credit Card Fraud Detection" dari Kaggle) dan bandingkan akurasinya sebelum vs sesudah SMOTE.
