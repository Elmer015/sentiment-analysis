# Sentiment Analysis Bahasa Indonesia (Streamlit)

Aplikasi ini melakukan analisis sentimen teks Bahasa Indonesia menggunakan model:

- `indobenchmark/indobert-base-p1`

Karena model tersebut adalah base model (belum fine-tuned khusus klasifikasi sentimen), aplikasi ini memakai pendekatan:

1. Ubah teks input menjadi embedding menggunakan IndoBERT.
2. Bandingkan embedding input dengan embedding prototipe sentimen (`positif`, `netral`, `negatif`) menggunakan cosine similarity.
3. Pilih label dengan skor tertinggi.

## Cara Menjalankan

1. Buat virtual environment (opsional tapi direkomendasikan).
2. Install dependency:

```bash
pip install -r requirements.txt
```

3. Jalankan Streamlit:

```bash
streamlit run app.py
```

4. Buka URL yang tampil di terminal (biasanya `http://localhost:8501`).

## Catatan

- Saat pertama kali dijalankan, model akan diunduh dari Hugging Face sehingga butuh koneksi internet.
- Untuk akurasi lebih tinggi, sebaiknya lanjutkan dengan fine-tuning model pada dataset sentimen Bahasa Indonesia.
