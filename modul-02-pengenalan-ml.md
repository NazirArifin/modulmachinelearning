# Modul 02 - Dasar Konsep Machine Learning: Supervised vs Unsupervised

Tujuan Pembelajaran: Mahasiswa memahami perbedaan antara Supervised dan Unsupervised Learning melalui implementasi praktis menggunakan Google Colab dan dataset sederhana.

## Materi

### 1) Apa itu Machine Learning?

Machine Learning (ML) adalah cabang AI yang memungkinkan komputer untuk belajar dari data tanpa diprogram secara eksplisit. Fokus utama ML adalah membuat model yang dapat melakukan prediksi atau menemukan pola dalam data.

### 2) Supervised vs Unsupervised Learning

| Karakteristik | Supervised Learning | Unsupervised Learning |
| --- | --- | --- |
| **Data** | Memiliki Label (Target) | Tidak Memiliki Label |
| **Tujuan** | Memprediksi Nilai/Kategori | Menemukan Pola/Struktur Tersembunyi |
| **Contoh Kasus** | Prediksi Harga Rumah, Deteksi Spam | Segmentasi Pelanggan, Deteksi Anomali |
| **Output** | Label/Nilai Kontinu | Klaster (Kelompok) |

### 3) Alur Kerja Praktis ML di Google Colab

Biasanya melibatkan langkah-langkah berikut:
1.  **Import Library**: Mengambil alat yang dibutuhkan (Pandas, Scikit-Learn).
2.  **Load Data**: Memasukkan data ke lingkungan Colab.
3.  **Exploratory Data Analysis (EDA)**: Memahami data (statistik deskriptif, visualisasi).
4.  **Split Data**: Membagi data menjadi data latih (*Train*) dan data uji (*Test*).
5.  **Model Training**: Melatih model menggunakan data latih.
6.  **Evaluation**: Mengevaluasi performa model menggunakan data uji.

---

## Praktikum: Implementasi Sederhana

Buka [Google Colab](https://colab.research.google.com/) dan jalankan kode berikut untuk membandingkan Klasifikasi (Supervised) dan Clustering (Unsupervised).

### 1. Import Library dan Dataset

Kita akan menggunakan dataset `Iris` yang sangat populer untuk latihan ML.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.head()
```

### 2. Supervised Learning: Klasifikasi Bunga Iris

Kita akan melatih model untuk memprediksi jenis bunga berdasarkan fiturnya.

```python
# Persiapan data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model K-Nearest Neighbors
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)

# Prediksi dan Skor Akurasi
accuracy = model_knn.score(X_test, y_test)
print(f"Akurasi Model Supervised (KNN): {accuracy * 100:.2f}%")
```

### 3. Unsupervised Learning: Clustering Bunga Iris

Di sini kita berpura-pura tidak tahu jenis bunganya dan meminta model mengelompokkan data sendiri.

```python
# Kita hanya ambil fiturnya saja tanpa label target
X_unsupervised = df.drop('target', axis=1)

# Training model K-Means untuk mencari 3 kelompok (cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_unsupervised)

# Visualisasi hasil clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='cluster', palette='viridis')
plt.title('Hasil Unsupervised Learning: K-Means Clustering')
plt.show()
```

---

## Latihan/Tugas

1.  Jelaskan mengapa pada praktikum Unsupervised Learning di atas kita menggunakan `n_clusters=3`?
2.  Cobalah ganti nilai `n_neighbors` pada bagian Supervised Learning menjadi `1` atau `10`. Apa dampaknya terhadap akurasi?
3.  Carilah satu contoh dataset di [Kaggle](https://www.kaggle.com/datasets) yang termasuk kategori Supervised Learning dan sebutkan fitur serta targetnya!
