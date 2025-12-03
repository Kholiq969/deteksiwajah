import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ===========================
# 1️⃣ Load model CNN
# ===========================
model_path = 'face_shape_cnn_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model tidak ditemukan! Jalankan train_cnn.py terlebih dahulu.")

model = tf.keras.models.load_model(model_path)
print("✅ Model CNN berhasil dimuat!")

# ===========================
# 2️⃣ Label kelas & rekomendasi gaya rambut pria
# ===========================
class_labels = ['oval', 'round', 'square', 'rectangle', 'heart', 'diamond', 'triangle']

hair_recommendations_men = {
    'oval': "Cocok hampir semua gaya — coba pompadour, slick back, atau quiff.",
    'round': "Gunakan potongan dengan volume di atas seperti pompadour atau high fade untuk menambah kesan panjang.",
    'square': "Coba gaya undercut atau textured crop agar rahang terlihat tegas tapi proporsional.",
    'rectangle': "Hindari rambut terlalu panjang di atas, pilih gaya side part atau comb over agar wajah tidak tampak terlalu panjang.",
    'heart': "Gunakan fringe atau gaya dengan poni untuk menyeimbangkan dahi yang lebar.",
    'diamond': "Coba gaya messy fringe atau top volume untuk menonjolkan tulang pipi tanpa mempertegas dagu.",
    'triangle': "Gunakan gaya dengan volume di atas dan sisi tipis (high fade atau pompadour)."
}

# ===========================
# 3️⃣ Load detektor wajah OpenCV
# ===========================
face_cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(face_cascade_path):
    raise FileNotFoundError("❌ File haarcascade_frontalface_default.xml tidak ditemukan! "
                            "Unduh dari: https://github.com/opencv/opencv/tree/master/data/haarcascades")

face_cascade = cv2.CascadeClassifier(face_cascade_path)

# ===========================
# 4️⃣ Jalankan kamera (Mirror Mode)
# ===========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Tidak dapat mengakses kamera.")
    exit()

print("🎥 Kamera aktif (mode mirror) — Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Gagal membaca frame dari kamera.")
        break

    # 🔁 Mirror (seperti kaca)
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Preprocessing untuk model
        face_resized = cv2.resize(face_img, (128, 128))
        face_array = image.img_to_array(face_resized) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        # Prediksi bentuk wajah
        prediction = model.predict(face_array, verbose=0)
        label_index = np.argmax(prediction)
        label = class_labels[label_index]
        recommendation = hair_recommendations_men[label]

        # 🖥️ Tampilkan hasil di TERMINAL
        print(f"\n[INFO] Bentuk wajah terdeteksi: {label.capitalize()}")
        print(f"[REKOMENDASI] {recommendation}")

        # 🎯 Gambar kotak di wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 🧾 Teks hasil di layar
        cv2.putText(frame, f"Bentuk: {label.capitalize()}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset = y + h + 30
        for line in recommendation.split('\n'):
            cv2.putText(frame, line, (x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)
            y_offset += 25

    cv2.imshow("💇‍♂️ Face Shape & Hair Recommendation (Men) - Mirror Mode", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Program dihentikan.")
