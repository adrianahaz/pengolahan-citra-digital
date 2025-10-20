import cv2
import numpy as np
import matplotlib.pyplot as plt


def transformation_operations_demo():
    # Baca gambar
    img = cv2.imread('parrot.jpeg')
    # 1. Translasi
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, 50], [0, 1, 50]])
    translated = cv2.warpAffine(img, M, (cols, rows))
    # 2. Rotasi
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    # 3. Scaling
    scaled = cv2.resize(img, None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_LINEAR)
    # 4. Affine transformation
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    affine = cv2.warpAffine(img, M, (cols, rows))
    # Tampilkan hasil
    plt.figure(figsize=(12, 8))
    plt.subplot(251), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
    plt.title('Original')
    plt.subplot(252), plt.imshow(cv2.cvtColor(translated,
                                              cv2.COLOR_BGR2RGB)), plt.title('Translated')
    plt.subplot(253), plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)),
    plt.title('Rotated')
    plt.subplot(254), plt.imshow(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)),
    plt.title('Scaling')
    plt.subplot(255), plt.imshow(cv2.cvtColor(affine, cv2.COLOR_BGR2RGB)),
    plt.title('Affine')
    plt.subplot(256), plt.hist(img.ravel(), 128, [
        0, 256], color='black', alpha=0.7),
    plt.title('Histogram Original')
    plt.subplot(257), plt.hist(translated.ravel(), 128,
                               [0, 256], color='black', alpha=0.7),
    plt.title('Histogram Translasi')
    plt.subplot(258), plt.hist(rotated.ravel(), 128,
                               [0, 256], color='black', alpha=0.7),
    plt.title('Histogram Rotate')
    plt.subplot(259), plt.hist(scaled.ravel(), 128,
                               [0, 256], color='black', alpha=0.7),
    plt.title('Histogram Scalling')
    plt.subplot(2, 5, 10), plt.hist(affine.ravel(), 128,
                                    [0, 256], color='black', alpha=0.7),
    plt.title('Histogram Affine                                           ')
    plt.tight_layout()
    plt.show()


transformation_operations_demo()
