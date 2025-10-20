import cv2
import numpy as np
import matplotlib.pyplot as plt


def point_operations_demo():
    # Baca gambar
    img = cv2.imread('test-img.jpeg', 0)  # Baca sebagai grayscale

    # 1. Pengaturan kecerahan
    brightness = 50
    brightened = cv2.add(img, brightness)

    # 2. Thresholding
    _, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 3. Negasi citra
    negated = 255 - img

    # 4. Operasi aritmatika (contoh: menambahkan dua citra)
    added = cv2.add(img, img)

    # Tampilkan hasil
    plt.figure(figsize=(12, 8))
    plt.subplot(251), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(252), plt.imshow(brightened, cmap='gray'),
    plt.title('Brightened')
    plt.subplot(253), plt.imshow(thresholded, cmap='gray'),
    plt.title('Thresholded')
    plt.subplot(254), plt.imshow(negated, cmap='gray'), plt.title('Negated')
    plt.subplot(255), plt.imshow(added, cmap='gray'), plt.title('Added')
    plt.subplot(256), plt.hist(img.ravel(), 128, [
        0, 256], color='black', alpha=0.7),
    plt.title('Histogram Original')
    plt.subplot(257), plt.hist(brightened.ravel(), 128,
                               [0, 256], color='black', alpha=0.7),
    plt.title('Histogram Brightened')
    plt.subplot(258), plt.hist(thresholded.ravel(), 128,
                               [0, 256], color='black', alpha=0.7),
    plt.title('Histogram Thresholded')
    plt.subplot(259), plt.hist(negated.ravel(), 128,
                               [0, 256], color='black', alpha=0.7),
    plt.title('Histogram Negated')
    plt.subplot(2, 5, 10), plt.hist(added.ravel(), 128,
                                    [0, 256], color='black', alpha=0.7),
    plt.title('Histogram Added')
    plt.tight_layout()
    plt.show()


point_operations_demo()
