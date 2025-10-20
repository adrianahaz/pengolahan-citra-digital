import matplotlib.pyplot as plt
import cv2

img = cv2.imread('../parrot.jpeg')

# RGB Image + Histogram
plt.figure(figsize=(12, 5))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create two subplots
plt.subplot(121), plt.imshow(img_rgb), plt.axis("off"), plt.title("RGB")

plt.subplot(122)
# Separate color channels
r, g, b = cv2.split(img_rgb)

# Plot histogram for each channel
plt.hist(r.ravel(), bins=256, range=[0, 256],
         color="red", alpha=0.5, label="red")
plt.hist(g.ravel(), bins=256, range=[0, 256],
         color="green", alpha=0.5, label="green")
plt.hist(b.ravel(), bins=256, range=[0, 256],
         color="blue", alpha=0.5, label="blue")

# Set the title labels for the histogram
plt.title("RGB Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.legend()

# Set the x-axis limits for the histogram
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
