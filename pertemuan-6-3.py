import cv2
import numpy as np
import matplotlib.pyplot as plt

# img1 = cv2.imread('img/cat-l.png')
img1 = cv2.imread('img/t-1.jpg')
gambar1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# img2 = cv2.imread('img/cat-r.png')
img2 = cv2.imread('img/t-2.jpg')
gambar2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# added = gambar1 + gambar2
diff = cv2.subtract(gambar2, gambar1)

plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(gambar1), plt.title('Cat Left')
plt.subplot(233), plt.imshow(gambar2), plt.title('Cat Right')
# plt.subplot(235), plt.imshow(added), plt.title('Added')
plt.subplot(235), plt.imshow(diff), plt.title('Subtracted')
plt.tight_layout()
plt.show()
