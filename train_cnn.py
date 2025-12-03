import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ===========================
# 1️⃣ Path ke dataset
# ===========================
dataset_path = 'dataset'  # Ganti dengan folder dataset kamu
if not os.path.exists(dataset_path):
    raise FileNotFoundError("❌ Folder 'dataset' tidak ditemukan! Pastikan struktur folder sudah benar.")

# ===========================
# 2️⃣ Data augmentation & pembagian train/val
# ===========================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ===========================
# 3️⃣ Bangun model CNN
# ===========================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(7, activation='softmax')  # 7 kelas bentuk wajah
])

# ===========================
# 4️⃣ Compile model
# ===========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===========================
# 5️⃣ Training model
# ===========================
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)

# ===========================
# 6️⃣ Simpan model
# ===========================
model.save('face_shape_cnn_model.h5')
print("✅ Model selesai dilatih dan disimpan sebagai 'face_shape_cnn_model.h5'")
