# Modul 05 - Evaluasi Model I: Confusion Matrix, Precision, Recall, & F1-Score

Tujuan Pembelajaran: Mahasiswa memahami pentingnya metrik evaluasi selain akurasi dan mampu menghitungnya melalui praktek langsung.

## Materi

### 1) Mengapa Akurasi Saja Tidak Cukup?

Pada data yang tidak seimbang (*Imbalanced Data*), misalnya 95% data adalah kelas A dan 5% kelas B, model yang menebak "semuanya kelas A" akan memiliki akurasi 95%, namun ia gagal total mengenali kelas B. Di sinilah metrik lain diperlukan.

### 2) Confusion Matrix

Confusion matrix adalah tabel yang membandingkan prediksi model dengan label asli.
- **TP (True Positive)**: Positif diprediksi Positif.
- **TN (True Negative)**: Negatif diprediksi Negatif.
- **FP (False Positive)**: Negatif diprediksi Positif (salah sangka).
- **FN (False Negative)**: Positif diprediksi Negatif (salah lepas).

### 3) Metrik Utama

- **Precision**: Dari semua yang diprediksi positif, berapa yang benar-benar positif?
- **Recall**: Dari semua positif yang ada, berapa banyak yang berhasil dideteksi?
- **F1-Score**: Rata-rata harmonik antara Precision dan Recall. Bagus untuk data imbalanced.

---

## Praktikum: Menghitung Metrik Evaluasi

Gunakan dataset buatan untuk memahami perbedaan metrik ini.

### 1. Persiapan Data Latih & Model Dasar

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Data asli (0: Tidak Beli, 1: Beli)
y_asli = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# Prediksi model (Anggap saja model kita menebak)
y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1]

# Confusion Matrix
cm = confusion_matrix(y_asli, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Prediksi')
plt.ylabel('Asli')
plt.title('Confusion Matrix Visualization')
plt.show()
```

### 2. Mencetak Classification Report

```python
# Report lengkap
report = classification_report(y_asli, y_pred, target_names=['Tidak Beli', 'Beli'])
print(report)
```

---

## Latihan/Tugas

1.  Berdasarkan hasil `classification_report` di atas, berikan penjelasan sederhana perbedaan antara **Precision** dan **Recall** untuk kelas "Beli".
2.  Ganti manual nilai `y_pred` agar model melakukan kesalahan lebih banyak (FP dan FN meningkat). Amati perubahan nilai F1-Score.
3.  Kapan kita lebih mementingkan **Recall** daripada **Precision**? Berikan satu contoh kasus nyata di bidang medis!
