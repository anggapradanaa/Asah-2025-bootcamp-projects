# Analisis Sentimen Ulasan Aplikasi Gojek

Proyek ini melakukan analisis sentimen terhadap ulasan pengguna aplikasi Gojek di Google Play Store menggunakan dua skema pemodelan machine learning.

## Struktur Proyek

```
├── scraping_data.ipynb     # Pengambilan data ulasan dari Google Play
├── modelling.ipynb         # Preprocessing, pelabelan, dan pelatihan model
├── inference.ipynb         # Inferensi / prediksi sentimen teks baru
├── requirements.txt
└── saved_models/
    ├── tfidf_vectorizer.pkl
    ├── random_forest_model.pkl
    ├── word2vec_model.bin
    ├── lstm_model.h5
    └── label_encoder.pkl
```

## Dataset

- **Sumber:** Google Play Store (ID, Bahasa Indonesia)
- **Jumlah:** 12.000 ulasan aplikasi `com.gojek.app`
- **Kolom:** `userName`, `score`, `content`, `thumbsUpCount`, `reviewCreatedVersion`, `at`, `replyContent`, `repliedAt`
- **Distribusi rating:** 1★ (2.838) · 2★ (404) · 3★ (451) · 4★ (585) · 5★ (7.722)

## Alur Pipeline

1. **Scraping** — Ambil ulasan via `google-play-scraper` (sort: terbaru, 200/batch)
2. **Preprocessing** — Case folding → cleaning (emoji, angka, tanda baca) → stopword removal (Sastrawi) → stemming (Sastrawi)
3. **Pelabelan** — Lexicon-based (kamus kata positif/negatif Bahasa Indonesia) → label: `positif`, `negatif`, `netral`
4. **Split Data** — 80% train / 20% test, stratified
5. **Modelling** — Dua skema (lihat di bawah)
6. **Inferensi** — Prediksi teks baru menggunakan model yang sudah disimpan

## Skema Pemodelan

| | Skema 1 | Skema 2 |
|---|---|---|
| **Representasi teks** | TF-IDF (max 5.000 fitur, bigram) | Word2Vec embedding |
| **Balancing** | SMOTE | SMOTE |
| **Model** | Random Forest (500 trees) + XGBoost | LSTM |
| **Output** | `saved_models/random_forest_model.pkl` | `saved_models/lstm_model.h5` |

## Hasil Evaluasi Model

Label sentimen: `positif`, `negatif`, `netral`

### Random Forest (TF-IDF + SMOTE)

| Split | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Train | 0.9181 | 0.9198 | 0.9181 | 0.9186 |
| **Test** | **0.8767** | **0.8999** | **0.8767** | **0.8835** |

Per kelas (test):

| Kelas | Precision | Recall | F1 |
|---|---|---|---|
| Negatif | 0.60 | 0.86 | 0.70 |
| Netral | 0.93 | 0.87 | 0.90 |
| Positif | 0.99 | 0.89 | 0.93 |

### XGBoost (TF-IDF + SMOTE)

| Split | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Train | 0.9895 | 0.9896 | 0.9895 | 0.9895 |
| **Test** | **0.9788** | **0.9793** | **0.9788** | **0.9789** |

Per kelas (test):

| Kelas | Precision | Recall | F1 |
|---|---|---|---|
| Negatif | 0.93 | 0.98 | 0.95 |
| Netral | 0.99 | 0.98 | 0.98 |
| Positif | 0.98 | 0.98 | 0.98 |

### LSTM (Word2Vec + SMOTE)

| Split | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Train | 0.8973 | 0.8991 | 0.8973 | 0.8967 |
| **Test** | **0.9367** | **0.9362** | **0.9367** | **0.9364** |

Per kelas (test):

| Kelas | Precision | Recall | F1 |
|---|---|---|---|
| Negatif | 0.83 | 0.81 | 0.82 |
| Netral | 0.95 | 0.95 | 0.95 |
| Positif | 0.97 | 0.97 | 0.97 |

### Ringkasan Perbandingan (Data Test)

| Model | Accuracy | F1-Score |
|---|---|---|
| Random Forest | 0.8767 | 0.8835 |
| LSTM | 0.9367 | 0.9364 |
| **XGBoost** | **0.9788** | **0.9789** |

> XGBoost mencapai performa terbaik di semua metrik pada data test.

## Instalasi

```bash
pip install -r requirements.txt
```

> Pastikan Python 3.8+ dan TensorFlow/Keras sudah kompatibel dengan lingkungan Anda.

## Cara Penggunaan

Jalankan notebook secara berurutan:

```
1. scraping_data.ipynb   → menghasilkan gojek_reviews_v2.csv
2. modelling.ipynb       → menghasilkan model di folder saved_models/
3. inference.ipynb       → prediksi sentimen teks baru
```

Contoh inferensi:

```python
predict_rf("Aplikasi gojek sangat bagus dan mudah digunakan!")
# → ('positif', [prob_negatif, prob_netral, prob_positif])

predict_lstm("Jelek banget sering error")
# → ('negatif', confidence_score)
```

## Dependensi Utama

`pandas` · `numpy` · `Sastrawi` · `scikit-learn` · `imbalanced-learn` · `gensim` · `tensorflow` · `keras` · `xgboost` · `google-play-scraper` · `joblib`
