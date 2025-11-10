import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca citra
img = cv2.imread('img/wanita.jpg', 0)

# Terapkan Fourier Transform
F = np.fft.fft2(img)
F1 = np.fft.fftshift(F)

# Pusatkan FFT
# Tampilkan magnitude spektrum Fourier
F2 = np.abs(F1)  # Get the magnitude
F2_log = np.log(F2 + 1)  # Use log

for i in range(96, 99):
    for j in range(263, 266):
        F1[i, j] = 0

for i in range(116, 119):
    for j in range(283, 286):
        F1[i, j] = 0

for i in range(261, 264):
    for j in range(263, 266):
        F1[i, j] = 0

for i in range(282, 285):
    for j in range(283, 286):
        F1[i, j] = 0

for i in range(428, 431):
    for j in range(263, 266):
        F1[i, j] = 0

for i in range(407, 410):
    for j in range(283, 286):
        F1[i, j] = 0

# Kembalikan ke ranah spasial
J = np.fft.ifftshift(F1)
J = np.fft.ifft2(J)
J = np.abs(J)

# Ambil bagian real (magnitude)
J = np.uint8(J)

# Tampilkan magnitude spektrum setelah filtering
F2_filtered = np.abs(F1)
F2_filtered_log = np.log(F2_filtered + 1)
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(
    100*F2_log, cmap='gray'), plt.title('Magnitude Spectrum')
plt.subplot(133), plt.imshow(100*F2_filtered_log,
                             cmap='gray'), plt.title('Filtered Magnitude Spectrum')
plt.tight_layout()
plt.show()

# Tampilkan hasil rekonstruksi
plt.figure(figsize=(10, 5))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(J, cmap='gray'), plt.title(
    'Reconstructed Image (Filtered)')
plt.subplot(133)
comparison = np.hstack((img[:256, :256], J[:256, :256]))
plt.imshow(comparison, cmap='gray')
plt.title('Comparison')
plt.tight_layout()
plt.show()
