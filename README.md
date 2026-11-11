# 🍎 FruitScan – Deteksi Kesegaran Buah

FruitScan merupakan aplikasi berbasis web yang digunakan untuk mendeteksi kondisi kesegaran buah menggunakan YOLOv8n.

Aplikasi ini dapat mendeteksi 4 jenis buah, yaitu apel, pisang, jeruk, dan delima. Hasil deteksi akan menunjukkan apakah buah termasuk dalam kondisi segar atau busuk.

# Fitur Aplikasi

- Deteksi buah melalui gambar.
- Mendeteksi apel, pisang, jeruk, dan delima.
- Menampilkan kondisi buah (segar atau busuk).
- Menampilkan kotak deteksi pada gambar.
- Menampilkan nilai confidence.
- Pengaturan batas confidence.
- Menampilkan riwayat hasil deteksi.
- Download riwayat deteksi dalam format CSV.

# Teknologi yang Digunakan

- Python
- Streamlit
- YOLOv8n
- Pillow
- Pandas
- NumPy

# Cara Menjalankan Aplikasi

1. Download project melalui tombol **Code → Download ZIP** di GitHub.
2. Ekstrak file ZIP ke komputer.
3. Buka folder project menggunakan Visual Studio Code.
4. Buka Terminal di Visual Studio Code.
5. Install library yang dibutuhkan dengan mengetik `pip install -r requirements.txt`.
6. Setelah instalasi selesai, jalankan aplikasi dengan mengetik `streamlit run app_deteksi_buah.py`.
7. Buka browser dan akses `http://localhost:8501`.

**Catatan:** Pastikan file `best.pt` berada dalam satu folder dengan `app_deteksi_buah.py`.

# Jenis Buah yang Dideteksi

| 1 | Apel | Segar / Busuk 
| 2 | Pisang | Segar / Busuk 
| 3 | Jeruk | Segar / Busuk 
| 4 | Delima | Segar / Busuk 

# Kekurangan Program

Beberapa kekurangan yang masih terdapat pada aplikasi ini:

- Buah di luar kelas yang tersedia masih berpotensi salah terdeteksi.
- Hasil deteksi terkadang kurang tepat jika gambar buram atau buah saling bertumpuk.
- Hasil deteksi dapat dipengaruhi oleh pencahayaan dan kualitas gambar.

# Pengembangan Selanjutnya

Aplikasi ini masih bisa dikembangkan, terutama dalam hal peningkatan akurasi deteksi, penambahan jenis buah, dan penyimpanan riwayat agar dapat tersimpan secara permanen.

# Repository

Repository project:  
https://github.com/Lodfa666/deteksi_buah
