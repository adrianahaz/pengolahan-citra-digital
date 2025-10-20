import cv2
import numpy as np
import matplotlib.pyplot as plt


def spatial_operations_demo():
    # Baca gambar
    img = cv2.imread('parrot.jpeg', 0)
    # Baca sebagai grayscale
    # 1. Low-pass filtering (Gaussian blur)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # 2. High-pass filtering
    high_pass = cv2.subtract(img, blurred)
    # 3. Edge detection (Sobel)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX,
                          dtype=cv2.CV_8U)
    # 4. Morfologi (dilasi)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    # Tampilkan hasil
    plt.figure(figsize=(15, 10))
    plt.subplot(251), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(252), plt.imshow(
        blurred, cmap='gray'), plt.title('Low-pass(Blurred)')
    plt.subplot(253), plt.imshow(
        high_pass, cmap='gray'), plt.title('High-pass')
    plt.subplot(254), plt.imshow(edges, cmap='gray'), plt.title('Edges')
    plt.subplot(255), plt.imshow(dilated, cmap='gray'), plt.title('Dilated')
    plt.subplot(256), plt.hist(img.ravel(), 128, [
        0, 256], color='black', alpha=0.7),
    plt.title('Histogram Original')
    plt.subplot(257), plt.hist(blurred.ravel(), 128,
                               [0, 256], color='black', alpha=0.7),
    plt.title('Histogram Low-pass filtering')
    plt.subplot(258), plt.hist(high_pass.ravel(), 128,
                               [0, 256], color='black', alpha=0.7),
    plt.title('Histogram High-pass filtering')
    plt.subplot(259), plt.hist(edges.ravel(), 128, [
        0, 256], color='black', alpha=0.7),
    plt.title('Histogram Edges')
    plt.subplot(2, 5, 10), plt.hist(dilated.ravel(),
                                    128, [0, 256], color='black', alpha=0.7),
    plt.title('Histogram Dilated')
    plt.tight_layout()
    plt.show()


spatial_operations_demo()
