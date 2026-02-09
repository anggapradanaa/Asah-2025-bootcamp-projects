# Klasifikasi Kanker Paru-paru Menggunakan Deep Learning

Proyek klasifikasi citra histopatologi kanker paru-paru menggunakan Transfer Learning dengan arsitektur VGG16.

## 📋 Deskripsi

Sistem klasifikasi otomatis untuk mendeteksi 3 jenis kondisi jaringan paru-paru:
- **Lung Adenocarcinoma** (Kanker Adenokarsinoma)
- **Lung Benign Tissue** (Jaringan Jinak)
- **Lung Squamous Cell Carcinoma** (Kanker Karsinoma Sel Skuamosa)

## 🗂️ Dataset

Dataset: [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

**Struktur Dataset:**
- Total: 15,000 gambar histopatologi
- Split: 70% Training, 15% Validation, 15% Test
- Format: JPEG 768x768 pixels
- 3 kelas dengan distribusi seimbang

## 🏗️ Arsitektur Model

**VGG16 Transfer Learning:**
- Base Model: VGG16 pre-trained (ImageNet) - Frozen
- Custom Head:
  - GlobalAveragePooling2D
  - Dense(512) + BatchNorm + Dropout(0.5)
  - Dense(256) + BatchNorm + Dropout(0.4)
  - Dense(3, softmax)
- Input: 224x224x3
- Total Parameters: ~15M (Trainable: ~2.5M)

## 🔧 Teknologi

- **Framework**: TensorFlow/Keras 2.x
- **Preprocessing**: ImageDataGenerator
- **Augmentasi**: Rotation, Shift, Shear, Zoom, Flip
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## 📊 Hasil

- **Training Accuracy**: ~97-99%
- **Validation Accuracy**: ~95-97%
- **Test Accuracy**: ~95-97%

## 🚀 Cara Menggunakan

### 1. Training Model (Lokal)
```python
# Jalankan script utama

# Output: best_vgg16.keras
```

### 2. Export Model (Google Colab)
```python
# Upload best_vgg16.keras ke Colab
# Jalankan notebook: save_model.ipynb
# Download: model_output.zip (SavedModel, TFLite, TensorFlow.js)
```

### 3. Inference (Google Colab)
```python
# Upload gambar test ke Colab
# Jalankan notebook: inference.ipynb
# Hasil: Visualisasi + JSON predictions
```

## 📦 Format Model

- **SavedModel**: Untuk production server (TF Serving)
- **TFLite**: Untuk mobile apps (Android/iOS)
- **TensorFlow.js**: Untuk web applications

## 🔍 Fitur Utama

- ✅ Data augmentation
- ✅ Transfer learning dengan VGG16 pre-trained
- ✅ Evaluasi (Confusion Matrix, Classification Report)
- ✅ Visualisasi preprocessing dan augmentasi
- ✅ Visualisasi training history dan prediksi
- ✅ Export ke multiple format (SavedModel, TFLite, TFJS)
- ✅ Inference dengan confidence score

## 📝 Requirements
```
tensorflow
numpy
matplotlib
seaborn
scikit-learn
tensorflowjs
```

## 👨‍💻 Workflow

1. **Preprocessing**: Split dataset → Augmentasi data
2. **Training**: VGG16 Transfer Learning (15 epochs)
3. **Evaluation**: Training & Test set metrics
4. **Export**: SavedModel, TFLite, TensorFlow.js
5. **Deployment**: Ready untuk production

## 📈 Monitoring

Model menggunakan callbacks untuk:
- Early stopping (patience=7)
- Model checkpoint (best val_accuracy)
- Learning rate reduction (patience=3)


---
