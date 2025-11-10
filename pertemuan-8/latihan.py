import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca citra
img = cv2.imread('img/india.jpg', 0)

# Terapkan Fourier Transform
F = np.fft.fft2(img)
F1 = np.fft.fftshift(F)  

# Pusatkan FFT
# Tampilkan magnitude spektrum Fourier
F2 = np.abs(F1)  # Get the magnitude
F2_log = np.log(F2 + 1)  # Use log

for i in range(37, 40):      
  for j in range(225, 228):          
    F1[i, j] = 0

for i in range(74, 77):      
  for j in range(225, 228):          
    F1[i, j] = 0

for i in range(234, 237):      
  for j in range(225, 228):          
    F1[i, j] = 0

for i in range(348, 351):      
  for j in range(225, 228):          
    F1[i, j] = 0

for i in range(507, 510):      
  for j in range(225, 228):          
    F1[i, j] = 0

for i in range(545, 548):      
  for j in range(225, 228):          
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
plt.subplot(132), plt.imshow(100*F2_log, cmap='gray'), plt.title('Magnitude Spectrum')
plt.subplot(133), plt.imshow(100*F2_filtered_log, cmap='gray'), plt.title('Filtered Magnitude Spectrum')
plt.tight_layout()
plt.show()

# Tampilkan hasil rekonstruksi
plt.figure(figsize=(10, 5))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(J, cmap='gray'), plt.title('Reconstructed Image (Filtered)')
plt.subplot(133)
comparison = np.hstack((img[:256, :256], J[:256, :256]))
plt.imshow(comparison, cmap='gray')
plt.title('Comparison')
plt.tight_layout()
plt.show()
