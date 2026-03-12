# Modul 11 - Cloud ML & Model Deployment Basics: Save & Load Model

Tujuan Pembelajaran: Mahasiswa mampu melatih model di lingkungan cloud (Google Colab) dan menyimpan model tersebut dalam format file (`.pkl` atau `.joblib`) agar dapat digunakan kembali tanpa harus melatih ulang (Model Persistence).

## Materi

### 1) Apa itu Cloud ML?

Cloud ML merujuk pada pemanfaatan infrastruktur cloud untuk tugas-tugas Machine Learning. Layanan populer meliputi:
- **Google Colab**: Lingkungan Jupyter Notebook gratis berbasis cloud dengan dukungan GPU/TPU.
- **AWS SageMaker / Azure ML**: Platform profesional untuk siklus hidup ML lengkap.
- **Google Cloud AI Platform**: Layanan enterprise untuk deployment model.

### 2) Model Persistence (Menyimpan Model)

Dalam pengembangan aplikasi, kita tidak ingin melatih ulang model setiap kali aplikasi dijalankan (memakan waktu dan sumber daya). Kita menyimpan pola/bobot yang sudah dipelajari model ke dalam sebuah file.

Ada dua library utama dalam Python untuk ini:
1.  **Pickle**: Library bawaan Python.
2.  **Joblib**: Lebih efisien untuk model dengan dataset besar yang mengandung array NumPy.

---

## Praktikum: Menyimpan dan Memuat Model di Colab

Buka Google Colab dan ikuti langkah berikut.

### 1. Melatih dan Menyimpan Model

```python
import joblib
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# 3. Simpan model ke file format joblib
nama_file = 'model_iris_v1.joblib'
joblib.dump(model, nama_file)

print(f"Model berhasil disimpan sebagai {nama_file}")
```

### 2. Memuat dan Menggunakan Model Kembali

```python
# Hapus variabel model untuk membuktikan kita memuat dari file
del model

# 4. Load model dari file
model_termuat = joblib.load('model_iris_v1.joblib')

# 5. Prediksi menggunakan model yang dimuat
data_baru = [[5.1, 3.5, 1.4, 0.2]] # Fitur bunga Iris Setosa
hasil = model_termuat.predict(data_baru)

print(f"Hasil Prediksi dari Model Termuat: {iris.target_names[hasil[0]]}")
```

### 3. Menyambungkan ke Google Drive (Praktis)

Di Google Colab, Anda bisa menyimpan file langsung ke Google Drive Anda:

```python
from google.colab import drive
drive.mount('/content/drive')
# joblib.dump(model, '/content/drive/MyDrive/model_iris.joblib')
```

---

## Latihan/Tugas

1.  Coba simpan model Random Forest dari Modul 04 menggunakan library `pickle` (bukan `joblib`). Cek dokumentasi Python untuk caranya!
2.  Apa risiko keamanan utama jika kita memuat model (`.pkl` atau `.joblib`) dari sumber yang tidak terpercaya (misal: download dari internet tanpa verifikasi)?
3.  Simpan model terbaik Anda dari proyek Churn Prediction (Modul 07) ke Google Drive Anda, lalu coba muat model tersebut di notebook baru yang berbeda.
