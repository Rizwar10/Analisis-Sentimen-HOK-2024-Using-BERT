# 🎮 Sentiment Analysis untuk Review Game Honor of Kings

Analisis sentimen review game Honor of Kings menggunakan **BERT (Bidirectional Encoder Representations from Transformers)**. Model ini mengklasifikasikan review menjadi tiga kategori: **Positive**, **Neutral**, dan **Negative**.

## Dataset
Data dikumpulkan dari scraping review game Honor of Kings periode Juni - Agustus 2024.
Dataset berisi 73,741 review dari Google Play Store dengan kolom:
- `content`: Teks review
- `score`: Rating 1-5

Review dibersihkan dan dikonversi menjadi label sentimen berdasarkan score:
- Score > 3 → Positive
- Score < 3 → Negative  
- Score = 3 → Neutral

## Requirements

Script ini berjalan di Google Colab dengan GPU (T4). Pastikan runtime sudah diset ke GPU sebelum menjalankan:

```
Runtime → Change runtime type → T4 GPU
```

Dependencies utama:
- transformers 4.30.2
- torch
- pandas, numpy, sklearn
- seaborn, matplotlib, wordcloud

## Struktur Project

```
1-8: Setup & Data Preparation
   - Mount Google Drive
   - Load & clean dataset
   - Tokenization dengan BERT

9-10: Training
   - 3 epochs
   - Learning rate: 2e-5
   - Batch size: 16
   - Train/test split: 80/20

11-13: Evaluation
   - Classification report
   - Confusion matrix
   - Error analysis

14: Prediction Function
   - Fungsi untuk prediksi teks baru
   - Mengembalikan sentiment + confidence score

15-16: Save & Summary
   - Model disimpan ke Google Drive
   - Download visualisasi
```

## Hasil

**Final Metrics:**
- Accuracy: 84.15%
- F1-Score (macro): 0.5913
- F1-Score (weighted): 0.8225

**Per-Class Performance:**
- Negative: Precision 0.76 | Recall 0.82 | F1 0.79
- Neutral: Precision 0.27 | Recall 0.05 | F1 0.08
- Positive: Precision 0.89 | Recall 0.93 | F1 0.90

Model sangat baik untuk klasifikasi positive dan negative, tapi struggle dengan neutral karena jumlah data yang tidak seimbang (neutral hanya 879 dari 73,741 reviews).

## Usage

### Training

Upload dataset ke Google Drive di path:
```
/content/drive/MyDrive/Project Data Analytics/Sentimen Analysis/Data/scrapped_data_105077_honor of kings.csv
```

Run semua cell secara berurutan. Training займет sekitar 20-30 menit dengan T4 GPU.

### Prediction

Setelah training selesai, gunakan fungsi `predict_sentiment()`:

```python
result = predict_sentiment(
    "Game ini sangat bagus dan menyenangkan!",
    model, 
    tokenizer, 
    device
)

print(result['sentiment'])  # positive
print(result['confidence']) # 0.994
```

### Load Model yang Sudah Disimpan

```python
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    '/content/drive/MyDrive/bert_sentiment_model'
)
tokenizer = BertTokenizer.from_pretrained(
    '/content/drive/MyDrive/bert_sentiment_model'
)
```

## Visualisasi

Script menghasilkan beberapa visualisasi:
- Distribusi sentimen (bar & donut chart)
- Distribusi score dan panjang teks
- Word clouds per sentimen
- Training history (loss & accuracy)
- Confusion matrix
- Error analysis

Semua plot otomatis tersimpan sebagai PNG dan bisa didownload.

## Catatan

- Model ini di-fine-tune dari `bert-base-uncased`
- Text preprocessing menghapus URL, mentions, dan stopwords bahasa Indonesia
- Untuk hasil terbaik, review harus minimal 5 karakter setelah cleaning
- Model disimpan ke Google Drive agar tidak hilang saat session berakhir

## Limitasi

- Class imbalance: Neutral sangat sedikit (1.2% dari total data)
- Model base adalah BERT English, tapi data review mixed (Indonesia + English)
- Optimal untuk binary classification (positive/negative) daripada 3-class