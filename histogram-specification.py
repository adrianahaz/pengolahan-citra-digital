import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar sumber dan target
img_source = cv2.imread('parrot.jpeg', 0)  # Baca sebagai grayscale
img_target = cv2.imread('test-img.jpeg', 0)  # Ganti dengan gambar target Anda

# Hitung histogram kumulatif untuk gambar sumber dan target
hist_source, bins = np.histogram(img_source.flatten(), 256, [0, 256])
hist_target, bins = np.histogram(img_target.flatten(), 256, [0, 256])
cdf_source = hist_source.cumsum()
cdf_target = hist_target.cumsum()
cdf_source = cdf_source / cdf_source.max()
cdf_target = cdf_target / cdf_target.max()

# Buat lookup table
lut = np.zeros(256, dtype='uint8')
j = 0
for i in range(256):
    while j < 256 and cdf_target[j] <= cdf_source[i]:
        j += 1
    lut[i] = j - 1

# Terapkan lookup table ke gambar sumber
img_result = cv2.LUT(img_source, lut)  # Tampilkan hasil
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img_source, cmap='gray')
plt.title('Gambar Sumber')
plt.subplot(132)
plt.imshow(img_target, cmap='gray')
plt.title('Gambar Target')
plt.subplot(133)
plt.imshow(img_result, cmap='gray')
plt.title('Hasil Histogram Specification')
plt.tight_layout()
plt.show()

# Tampilkan histogram
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.hist(img_source.ravel(), 256, [0, 256])
plt.title('Histogram Sumber')
plt.subplot(132)
plt.hist(img_target.ravel(), 256, [0, 256])
plt.title('Histogram Target')
plt.subplot(133)
plt.hist(img_result.ravel(), 256, [0, 256])
plt.title('Histogram Hasil')
plt.tight_layout()
plt.show()
