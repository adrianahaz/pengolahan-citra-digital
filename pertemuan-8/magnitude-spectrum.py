import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img/wanita.jpg', 0)
plt.imshow(img, cmap='gray')
F = np.fft.fft2(img)
F2 = np.fft.fftshift(F)
F2 = np.abs(F2)
F2 = np.log(F2 + 1)
plt.figure(figsize=(15, 5))
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(100*F2, cmap='gray'), plt.title('Magnitude Spectrum')
plt.tight_layout()
plt.show()
