import cv2
import numpy as np
import matplotlib.pyplot as plt

A1 = np.array([6, 8, 2], dtype=np.uint8)
# A2 = np.array([2, 20, 3], dtype=np.uint8)
# A3 = A1 - A2
array_2d = np.array([
    [0, 64, 128],
    [255, 192, 128]
], dtype=np.uint8)

array_3d = np.zeros((1, 3, 3), dtype=np.uint8)
array_3d[:, :, 0] = A1
array_3d[:, :, 1] = A1 * 10
array_3d[:, :, 2] = A1 + 200

plt.figure(figsize=(15, 10))

plt.subplot(311), plt.imshow([A1], cmap='gray'), plt.title('A1D')
plt.subplot(312), plt.imshow(array_2d, cmap='gray'), plt.title('A2D')
plt.subplot(313), plt.imshow(array_3d, cmap='gray'), plt.title('A3D')
# plt.subplot(313), plt.imshow([A3], cmap='gray'), plt.title('A3')
plt.tight_layout()
plt.axis('off')
plt.show()
