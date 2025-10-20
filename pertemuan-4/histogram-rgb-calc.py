import matplotlib.pyplot as plt
import cv2

img = cv2.imread('../parrot.jpeg')

# RGB Image + Histogram
plt.figure(figsize=(12, 5))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create two subplots
plt.subplot(121), plt.imshow(img_rgb), plt.axis("off"), plt.title("RGB")

plt.subplot(122)
# hist = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
# plt.plot(hist, color="red", label="Red")

hist = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
plt.plot(hist, color="green", label="Green")

# hist = cv2.calcHist([img_rgb], [3], None, [256], [0, 256])
# plt.plot(hist, color="blue", label="Blue")

# Set the title and labels for the histogram
plt.title("RGB Histogram"), plt.xlabel("Intensity"), plt.ylabel("Frequency")
plt.legend()

plt.xlim([0, 256])
plt.show()
