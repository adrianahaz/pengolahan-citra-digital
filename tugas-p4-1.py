import matplotlib.pyplot as plt
import cv2

# Baca gambar menggunakan cv2 (dalam format BGR)
img = cv2.imread('test-img.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Buat figure utama
fig = plt.figure(figsize=(12, 8))

# Subplot kiri (Gambar)
ax1 = plt.subplot(121)
ax1.imshow(img_rgb)
ax1.axis("off")
ax1.set_title("RGB Image")

# Subplot kanan (Histogram)
ax2 = plt.subplot(122)
ax2.axis("off")

# Subplot kecil di dalam subplot kanan
inner_r = fig.add_axes([0.56, 0.60, 0.18, 0.28])  # Red
inner_g = fig.add_axes([0.56, 0.25, 0.18, 0.28])  # Green
inner_b = fig.add_axes([0.78, 0.43, 0.18, 0.28])  # Blue

# Histogram merah
hist_red = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
inner_r.plot(hist_red, color="red")
inner_r.set_title("Red Channel")
inner_r.set_xlim([0, 256])

# Histogram hijau
hist_green = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
inner_g.plot(hist_green, color="green")
inner_g.set_title("Green Channel")
inner_g.set_xlim([0, 256])

# Histogram biru
hist_blue = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
inner_b.plot(hist_blue, color="blue")
inner_b.set_title("Blue Channel")
inner_b.set_xlim([0, 256])

plt.show()
