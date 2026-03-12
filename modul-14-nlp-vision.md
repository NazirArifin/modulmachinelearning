# Modul 14 - Data Multimodal: Dasar NLP & Computer Vision

Tujuan Pembelajaran: Mahasiswa mengenali pengolahan data non-tabular (teks dan gambar) menggunakan library Machine Learning modern.

## Materi

### 1) Apa itu Data Multimodal?

Multimodal merujuk pada integrasi data dari berbagai sumber (teks, gambar, audio, dan data numerik). 
- **Natural Language Processing (NLP)**: Memproses bahasa manusia (teks/suara).
- **Computer Vision (CV)**: Memproses citra visual (gambar/video).

### 2) Dasar NLP: Sentiment Analysis

Proses klasifikasi teks untuk menentukan apakah sebuah kalimat bernada Positif, Negatif, atau Netral. Contoh: Review produk pertanian atau opini tentang kebijakan pupuk.

### 3) Dasar Computer Vision: Image Classification

Mempelajari cara komputer mengenali objek dalam gambar. Contoh: Identifikasi jenis penyakit daun pada tanaman lewat foto.

---

## Praktikum: Implementasi Sederhana NLP & CV

Gunakan Google Colab untuk menjalankan eksperimen berikut.

### 1. NLP: Sentiment Analysis (VADER)

```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download library pendukung
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Contoh Review Lahan Pertanian
review = "Gagal panen tahun ini membuat petani sangat sedih dan kecewa."
hasil = sia.polarity_scores(review)

print(f"Hasil Analisis Sentimen: {hasil}")
# Jika compound score > 0.05 (Positif), < -0.05 (Negatif)
```

### 2. Computer Vision: Deteksi Objek Sederhana (Pretrained Model)

Kita akan menggunakan model yang sudah dilatih oleh Google (InceptionV3) untuk mengenali isi gambar secara instan.

```python
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pretrained model
model = InceptionV3(weights='imagenet')

# Simulasi load gambar (perlu input URL gambar real atau upload ke Colab)
# img_path = 'daun_sakit.jpg'
# img = image.load_img(img_path, target_size=(299, 299))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# preds = model.predict(x)
# print('Hasil Prediksi:', decode_predictions(preds, top=3)[0])
```

---

## Latihan/Tugas

1.  Carilah dataset teks review produk di Shopee/Tokopedia, lalu buatlah diagram batang (`sns.countplot`) untuk kategori rating (1-5).
2.  Apa perbedaan utama antara **Preprocessing** pada data teks (Tokenizing, Stemming) dengan data gambar (Resizing, Normalizing)?
3.  Berikan satu ide penggunaan **Multimodal Data** (NLP + CV) dalam satu aplikasi Smart Agriculture! (Contoh: "Kamera deteksi hama disertai input suara petani").
