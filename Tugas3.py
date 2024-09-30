import numpy as np
import cv2
import matplotlib.pyplot as plt

# Fungsi untuk perhitungan konvolusi manual dengan debug output
def manual_convolution(image, kernel):
    img_height, img_width = image.shape
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    convolved_image = np.zeros((img_height, img_width))

    # Melakukan konvolusi manual
    for i in range(1, img_height + 1):
        for j in range(1, img_width + 1):
            region = padded_image[i - 1:i + 2, j - 1:j + 2]
            convolved_value = np.sum(kernel * region)
            convolved_image[i - 1, j - 1] = convolved_value

            # Debug output
            print(f"Processing pixel ({i-1}, {j-1}):")
            print("Region:")
            print(region)
            print("Kernel:")
            print(kernel)
            print("Region * Kernel:")
            print(kernel * region)
            print(f"Sum of elements: {convolved_value}")
            print("=" * 30)

    return convolved_image

# Fungsi untuk menampilkan histogram dari citra
def display_histogram(image, title):
    plt.figure()
    plt.title(f"Histogram {title}")
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.show()

# Memuat gambar dan resize untuk mempercepat proses
image_rgb = cv2.imread('manusia.jpg')  # Memuat gambar berwarna
image_rgb = cv2.resize(image_rgb, (100, 100))  # Resize gambar menjadi 100x100
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)  # Mengonversi ke grayscale

# Membuat kernel/mask/filter untuk konvolusi
mask1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # Kernel untuk edge detection
mask2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Kernel Sobel

# Menerapkan konvolusi secara manual
print("Manual Convolution with Kernel 1:")
convolved_edge1 = manual_convolution(image_gray, mask1)

print("\nManual Convolution with Kernel 2:")
convolved_edge2 = manual_convolution(image_gray, mask2)

# Menampilkan gambar, hasil konvolusi, dan histogram
plt.figure(figsize=(14, 14))

# Gambar RGB
plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
plt.title('Citra RGB')
plt.axis('off')

# Histogram RGB
plt.subplot(3, 4, 2)
for i, col in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.title('Histogram RGB')
plt.xlim([0, 256])

# Gambar Grayscale
plt.subplot(3, 4, 3)
plt.imshow(image_gray, cmap='gray')
plt.title('Citra Grayscale')
plt.axis('off')

# Histogram Grayscale
plt.subplot(3, 4, 4)
plt.hist(image_gray.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('Histogram Grayscale')

# Hasil konvolusi dengan kernel 1
plt.subplot(3, 4, 5)
plt.imshow(convolved_edge1, cmap='gray')
plt.title('Konvolusi Kernel 1')
plt.axis('off')

# Histogram Hasil Konvolusi Kernel 1
plt.subplot(3, 4, 6)
plt.hist(convolved_edge1.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('Histogram Kernel 1')

# Hasil konvolusi dengan kernel 2
plt.subplot(3, 4, 7)
plt.imshow(convolved_edge2, cmap='gray')
plt.title('Konvolusi Kernel 2')
plt.axis('off')

# Histogram Hasil Konvolusi Kernel 2
plt.subplot(3, 4, 8)
plt.hist(convolved_edge2.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('Histogram Kernel 2')

plt.tight_layout()
plt.show()
