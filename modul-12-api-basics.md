# Modul 12 - Arsitektur Aplikasi Cerdas: REST API Basics untuk Machine Learning

Tujuan Pembelajaran: Mahasiswa memahami konsep dasar integrasi model Machine Learning ke dalam aplikasi melalui arsitektur REST API.

## Materi

### 1) Apa itu Aplikasi Cerdas?

Aplikasi cerdas adalah perangkat lunak (web, mobile, desktop) yang memiliki fitur berbasis Machine Learning (misal: rekomendasi belanja di e-commerce, asisten suara, pengenalan wajah).

### 2) Konsep Client-Server & REST API

Agar aplikasi (Client) bisa menggunakan model ML (Server), dibutuhkan jembatan komunikasi yang disebut **API (Application Programming Interface)**.

- **Client**: Aplikasi yang meminta prediksi (misal: Aplikasi Android).
- **Server**: Tempat model ML berada (misal: Server Flask/FastAPI).
- **REST (Representational State Transfer)**: Arsitektur komunikasi standar menggunakan protokol HTTP (GET, POST, dll).
- **JSON (JavaScript Object Notation)**: Format data standar untuk kirim/terima informasi antar Client-Server.

### 3) Alur Prediksi Lewat API

1.  Aplikasi Client mengirim data (misal: sensor tanah) dalam format JSON ke URL API.
2.  Server API menerima data, mengubahnya menjadi array NumPy, dan memasukkannya ke `model.predict()`.
3.  Server API mengirim balik hasil prediksi dalam format JSON ke Client.

---

## Praktikum: Simulasi Pengiriman Data Berformat JSON

Meskipun kita belum membangun server penuh (akan dipelajari di Modul 13), kita harus paham format datanya.

### 1. Python Dictionary ke JSON

```python
import json

# Data sensor tanah dari hardware IoT (Python Dictionary)
data_sensor = {
    'deviceId': 'SUKABUMI_001',
    'temperature': 28.5,
    'moisture': 35.0
}

# Mengubah data kearah format JSON (String) untuk dikirim lewat internet
json_payload = json.dumps(data_sensor)
print(f"Format JSON untuk dikirim: {json_payload}")

# Menerima balik JSON (di sisi Server)
data_diterima = json.loads(json_payload)
print(f"Suhu yang diterima server: {data_diterima['temperature']} °C")
```

### 2. Memanggil API Publik (Optional)

Untuk melihat cara kerja API sungguhan, kita bisa mengambil data cuaca (API Gratis):

```python
import requests

# Link API publik (contoh untuk info harga crypto)
url = "https://api.coindesk.com/v1/bpi/currentprice.json"
response = requests.get(url)
data = response.json()

print(f"Data Harga Bitcoin: {data['bpi']['USD']['rate']}")
```

---

## Latihan/Tugas

1.  Buatlah struktur JSON untuk mengirim data pasien rumah sakit yang terdiri dari: Nama, Umur, Tekanan Darah (Sistolik), dan apakah sedang Merokok (True/False).
2.  Mengapa format **JSON** lebih populer digunakan di API daripada format **XML** atau file **Excel**?
3.  Jelaskan perbedaan antara metode **GET** (mengambil data) dan **POST** (mengirim data) dalam konteks pengiriman data sensor ke server Machine Learning!
