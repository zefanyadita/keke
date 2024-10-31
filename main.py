import cv2
import numpy as np

# Load Haar Cascade untuk deteksi helm
helmet_cascade = cv2.CascadeClassifier('C:\\Users\\ACER\\Downloads\\Haarcascade helmet\\haarcascade_helmet (1).xml')

# Fungsi untuk mendeteksi helm dan memperpanjang hanya tepi bawah dari bounding box
def detect_helmet_bottom_extended(img_path, vertical_padding=50, horizontal_padding=100, extra_padding=50):
    # Membaca gambar
    img = cv2.imread(img_path)
    
    # Cek apakah gambar berhasil dimuat
    if img is None:
        print("Error: Gambar tidak dapat dimuat. Pastikan jalur file benar.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Menerapkan Gaussian Blur untuk mengurangi noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Mendeteksi helm
    helmets = helmet_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

    print(f"Jumlah helm yang terdeteksi: {len(helmets)}")  # Debugging

    if len(helmets) > 0:
        # Menemukan kotak terbesar berdasarkan luas
        largest_helmet = max(helmets, key=lambda box: box[2] * box[3])

        # Mendapatkan koordinat dan ukuran helm yang terdeteksi
        x, y, w, h = largest_helmet

        # Menghitung koordinat bounding box dengan padding ekstra di bawah
        x1 = max(0, x - horizontal_padding)
        y1 = max(0, y - vertical_padding)
        x2 = min(img.shape[1], x + w + 2 * horizontal_padding)
        y2 = min(img.shape[0], y + h + 3 * vertical_padding + extra_padding)

        # Menggambar kotak di sekitar helm
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, 'Helmet Detected', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Jika tidak ada helm terdeteksi, tambahkan teks "Helmet Not Detected"
        cv2.putText(img, 'Helmet Not Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Menampilkan hasil
    cv2.imshow('Grayscale Image', gray)  # Menampilkan gambar grayscale
    cv2.imshow('Original Image', img)  # Menampilkan gambar asli
    cv2.imshow('Helmet Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ganti jalur gambar di sini
img_path = 'C:\\Users\\ACER\\Downloads\\Haarcascade helmet\\poto helm yang kece-20241031T034456Z-001\\poto helm yang kece\\Pakai Helm0.jpg'  # Ganti dengan jalur gambar Anda

# Menjalankan deteksi helm
detect_helmet_bottom_extended(img_path, vertical_padding=20, horizontal_padding=80, extra_padding=30)
