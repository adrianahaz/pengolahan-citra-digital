import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Memuat gambar
    image = cv2.imread('parrot.jpeg')

    # convert the color from BGR to RGB for displaying using matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_red = cv2.inRange(image_hsv, lower_red, upper_red)
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(image_hsv, lower_blue, upper_blue)
    result_red = cv2.bitwise_and(image, image, mask=mask_red)
    result_green = cv2.bitwise_and(image, image, mask=mask_green)
    result_blue = cv2.bitwise_and(image, image, mask=mask_blue)
    image_red = cv2.cvtColor(result_red, cv2.COLOR_BGR2RGB)
    image_green = cv2.cvtColor(result_green, cv2.COLOR_BGR2RGB)
    image_blue = cv2.cvtColor(result_blue, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 10))
    plt.subplot(231), plt.imshow(image_rgb), plt.title('RGB Image')
    plt.subplot(232), plt.imshow(image_hsv), plt.title('HSV Image')
    plt.subplot(233), plt.imshow(image_red), plt.title('Red Color Image')
    plt.subplot(234), plt.imshow(image_green), plt.title('Green Color Image')
    plt.subplot(235), plt.imshow(image_blue), plt.title('Blue Color Image')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
