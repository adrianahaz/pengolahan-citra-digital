import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('parrot.jpeg')
parrot = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('parrot-invert.jpeg')
parrotI = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

added = parrot + parrotI

plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(parrot), plt.title('Parrot')
plt.subplot(233), plt.imshow(parrotI), plt.title('Parrot Invert')
plt.subplot(235), plt.imshow(added), plt.title('Parrot + Invert')
plt.tight_layout()
plt.show()
