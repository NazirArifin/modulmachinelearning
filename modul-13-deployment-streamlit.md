# Modul 13 - Deployment Frameworks: Membangun Aplikasi Web ML via Streamlit

Tujuan Pembelajaran: Mahasiswa mampu membuat prototipe aplikasi web untuk prediksi Machine Learning menggunakan framework Streamlit langsung dari Google Colab.

## Materi

### 1) Apa itu Streamlit?

Streamlit adalah library Python sumber terbuka yang memungkinkan kita membuat aplikasi web interaktif untuk proyek data dan machine learning dalam waktu singkat, tanpa perlu keahlian Web Development (HTML/CSS/JS) yang mendalam.

### 2) Mengapa Deployment Penting?

Tanpa deployment, model ML hanya berupa file code yang jalan di komputer kita sendiri. Deployment memungkinkan orang lain (pengguna) berinteraksi dengan model kita melalui antarmuka visual (UI).

### 3) Menjalankan Streamlit di Google Colab

Biasanya Streamlit dijalankan di komputer lokal. Namun, kita bisa menggunakan layanan tunnel seperti `localtunnel` agar aplikasi yang berjalan di Google Colab dapat diakses melalui browser.

---

## Praktikum: Membangun Aplikasi Prediksi Bunga Iris

Jalankan perintah berikut di Google Colab Anda.

### 1. Persiapan File Aplikasi (`app.py`)

Kita buat file `app.py` yang berisi kode Streamlit kita:

```python
# Tulis ke file app.py menggunakan magic command Colab
%%writefile app.py
import streamlit as st
import joblib
import numpy as np

# Load model yang sudah dilatih (pastikan sudah ada file .joblib-nya!)
# Jika belum ada, jalankan dulu kode training model di Modul 11
model = joblib.load('model_iris_v1.joblib')

st.title("Aplikasi Klasifikasi Bunga Iris 🌸")
st.write("Masukkan ukuran kelopak dan mahkota bunga untuk melihat jenisnya.")

# Input Form
sepal_l = st.number_input("Panjang Sepal", min_value=0.0, max_value=10.0, value=5.0)
sepal_w = st.number_input("Lebar Sepal", min_value=0.0, max_value=10.0, value=3.0)
petal_l = st.number_input("Panjang Petal", min_value=0.0, max_value=10.0, value=1.0)
petal_w = st.number_input("Lebar Petal", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Prediksi"):
    prediksi = model.predict([[sepal_l, sepal_w, petal_l, petal_w]])
    nama_bunga = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Hasil Prediksi: {nama_bunga[prediksi[0]]}")
```

### 2. Menjalankan Streamlit di Background

Gunakan perintah terminal berikut di cell baru Colab:

```bash
# 1. Install localtunnel
!npm install -g localtunnel

# 2. Jalankan Streamlit di background
!streamlit run app.py & npx localtunnel --port 8501
```

*Klik link yang muncul (biasanya berakhiran .loca.lt) untuk melihat aplikasi Anda.*

---

## Latihan/Tugas

1.  Buatlah aplikasi Streamlit sederhana untuk menghitung **"Prediksi Hasil Panen"** (menggunakan model regresi dari Modul 10). Ganti input numerik menjadi input berkategori menggunakan `st.selectbox` jika datanya kategorikal.
2.  Pelajari dokumentasi Streamlit: Gantilah `st.write` di judul aplikasi Anda menjadi `st.header` atau tambahkan gambar menggunakan `st.image`.
3.  Berikan pendapat Anda: Apa keuntungan terbesar menggunakan Streamlit daripada membangun web menggunakan Flask atau Django untuk profil seorang Data Scientist?
