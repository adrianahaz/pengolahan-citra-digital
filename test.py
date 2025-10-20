import cv2
import matplotlib.pyplot as plt


def main():
    img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    window_image = 'Image'
    (rows, cols) = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 70, 1)
    res = cv2.warpAffine(img, M, (cols, rows))

    filename = 'AdrianAhmadAlZidan.jpg'
    cv2.imwrite(filename, res)
    img = cv2.imread(filename)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.title(window_image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
