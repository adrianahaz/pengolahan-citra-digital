import matplotlib.pyplot as plt
import cv2

# Baca gambar menggunakan cv2 (dalam format BGR)
img = cv2.imread('../parrot.jpeg')

# Grayscale + Histogram
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(g, cmap="gray")
plt.axis("off")
plt.title("Grayscale")

plt.subplot(132)
plt.hist(g.ravel(), 256, [0, 256], color="k")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(133)
histg = cv2.calcHist([g], [0], None, [256], [0, 256])
plt.plot(histg)
plt.title("Gray Histogram (cv2.calcHist)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
