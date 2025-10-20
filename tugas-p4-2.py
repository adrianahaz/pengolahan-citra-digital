import cv2
import numpy as np
import matplotlib.pyplot as plt

# BGR Image
image = cv2.imread('test-img.jpeg')
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Separating color channels
R = img_rgb[:, :, 0]  # red layer
G = img_rgb[:, :, 1]  # green layer
B = img_rgb[:, :, 2]  # blue layer

# Equalize each channel separately
r_equi = cv2.equalizeHist(R)
g_equi = cv2.equalizeHist(G)
b_equi = cv2.equalizeHist(B)

# Merge the channels and create new image
equi_im = cv2.merge([r_equi, g_equi, b_equi])

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
histRed = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
plt.plot(histRed, color='red', label='Red Channel')
histGreen = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
plt.plot(histGreen, color='green', label='Green Channel')
histBlue = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
plt.plot(histBlue, color='blue', label='Blue Channel')
plt.title("Histogram Asli")

plt.subplot(2, 2, 3)
plt.imshow(equi_im)
plt.title("Equalized Image")
plt.axis('off')

plt.subplot(2, 2, 4)
histRed_eq = cv2.calcHist([equi_im], [0], None, [256], [0, 256])
plt.plot(histRed_eq, color='red', label='Red Channel')
histGreen_eq = cv2.calcHist([equi_im], [1], None, [256], [0, 256])
plt.plot(histGreen_eq, color='green', label='Green Channel')
histBlue_eq = cv2.calcHist([equi_im], [2], None, [256], [0, 256])
plt.plot(histBlue_eq, color='blue', label='Blue Channel')
plt.title("Histogram Setelah Equalization")

plt.show()
