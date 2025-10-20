import matplotlib.pyplot as plt
import cv2

# Baca gambar menggunakan cv2 (dalam format BGR)
img = cv2.imread('parrot.jpeg')

# RGB Image + Histogram
# Create a new figure with a width of 12 inches and height of 5 inches# Convert BGR to RGB for display
plt.figure(figsize=(12, 5))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Create two subplots: one for the RGB image and one for the histogram
plt.subplot(121), plt.imshow(img_rgb), plt.axis("off"), plt.title("RGB")
plt.subplot(122)  # Switch to the second subplot for the histogram
hist = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
# Plot the histogram using the corresponding color
plt.plot(hist, color="red", label="Red")
# hist = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
# plt.plot(hist, color="green", label="Green")
# hist = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
# plt.plot(hist, color="blue", label="Blue")
# Set the title and labels for the histogram
plt.title("RGB Histogram"), plt.xlabel("Intensity"), plt.ylabel("Frequency")
plt.legend()  # Add legend to distinguish color channels
# Set the x-axis limits for the histogram
plt.xlim([0, 256])
plt.show()
