import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV version:", cv2.__version__)

img = np.zeros((300, 300, 3), dtype="uint8")
cv2.circle(img, (150, 150), 100, (0, 0, 255), -1)

# OpenCV pakai BGR, matplotlib pakai RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Test Image")
plt.axis("off")
plt.show()
