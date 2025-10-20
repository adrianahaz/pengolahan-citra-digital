import cv2
import numpy as np
import matplotlib.pyplot as plt
# using opencv to read an image
# BGR Image
image = cv2.imread('parrot.jpeg')
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# seperating colour channels
R = img_rgb[:, :, 0]  # red layer
G = img_rgb[:, :, 1]  # green layer
B = img_rgb[:, :, 2]  # blue layer
# equilize each channel seperately
r_equi = cv2.equalizeHist(R)
g_equi = cv2.equalizeHist(G)
b_equi = cv2.equalizeHist(B)
# calculate histograms for each channel seperately
R_histo = cv2.calcHist([r_equi], [0], None, [256], [0, 256])
G_histo = cv2.calcHist([g_equi], [0], None, [256], [0, 256])
B_histo = cv2.calcHist([b_equi], [0], None, [256], [0, 256])
# merge the channels and create new image
equi_im = cv2.merge([r_equi, g_equi, b_equi])
# Buat figure dengan 2x3 subplot
plt.figure(figsize=(15, 10))
# visualize the channel histograms seperately
plt.subplot(2, 3, 1)
plt.plot(R_histo, 'r', label="Red")
plt.title("Red Channel Histogram")
plt.xlabel("Nilai Pixel")
plt.ylabel("Frekuensi")
plt.legend()
plt.xlim([0, 256])
plt.subplot(2, 3, 2)
plt.plot(G_histo, 'g', label="Green")
plt.title("Green Channel Histogram")
plt.xlabel("Nilai Pixel")
plt.ylabel("Frekuensi")
plt.legend()
plt.xlim([0, 256])
plt.subplot(2, 3, 3)
plt.plot(B_histo, 'b', label="Blue")
plt.title("Blue Channel Histogram")
plt.xlabel("Nilai Pixel")
plt.ylabel("Frekuensi")
plt.legend()
plt.xlim([0, 256])
# visualize the equalized channels
plt.subplot(2, 3, 4)
plt.imshow(r_equi, cmap='Reds')
plt.title("Red Channel Equalized")
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(g_equi, cmap='Greens')
plt.title("Green Channel Equalized")
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(b_equi, cmap='Blues')
plt.title("Blue Channel Equalized")
plt.axis('off')
plt.tight_layout()
plt.show()

# Display original and equalized images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(equi_im)
plt.title("Equalized Image")
plt.axis('off')
plt.tight_layout()
plt.show()
