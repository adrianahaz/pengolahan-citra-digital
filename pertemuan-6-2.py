import cv2
import numpy as np
import matplotlib.pyplot as plt

nilai1 = np.array([255, 100, 0], dtype=np.uint8)
nilai2 = np.array([100, 255, 0], dtype=np.uint8)
nilai3 = np.array([0, 100, 255], dtype=np.uint8)

RGB = np.zeros((1, 3, 3), dtype=np.uint8)
RGB[0, 0] = nilai1
RGB[0, 1] = nilai2
RGB[0, 2] = nilai3

plt.figure(figsize=(15, 10))
plt.subplot(311), plt.imshow(RGB), plt.title("RGB")
plt.tight_layout()
plt.axis('off')
plt.show()
