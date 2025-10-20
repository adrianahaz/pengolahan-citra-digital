import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2


def main():
    # Baca gambar menggunakan cv2 (format BGR)
    img = cv2.imread('./test-img.jpeg')

    # Ubah dari BGR ke RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Buat figure dan layout grid
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1])  # 3 baris, 2 kolom

    # Kolom kiri (gambar), merge semua baris
    ax_img = plt.subplot(gs[:, 0])
    ax_img.imshow(img_rgb)
    ax_img.axis("off")
    ax_img.set_title("RGB Image")

    # Kolom kanan (3 histogram vertikal)
    colors = ("red", "green", "blue")
    for i, color in enumerate(colors):
        ax = plt.subplot(gs['12'+str(i), 1])
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
        ax.set_xlim([0, 256])
        ax.set_title(f"{color.capitalize()} Channel")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
