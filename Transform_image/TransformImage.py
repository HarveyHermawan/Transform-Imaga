from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

input_image = imread("test.png")

if input_image.max() <= 1.0:
    input_image = (input_image * 255).astype(np.uint8)

gamma = 1.04
r_const, g_const, b_const = 0.2126, 0.7152, 0.0722

r, g, b = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]
grayscale = r_const * r ** gamma + g_const * g ** gamma + b_const * b ** gamma

kernel_size = 11
height, width, channels = input_image.shape

padded_image = np.pad(input_image, [(kernel_size // 2, kernel_size // 2),
                                    (kernel_size // 2, kernel_size // 2),
                                    (0, 0)], mode='edge')

blurred_image = np.zeros_like(input_image)
for i in range(height):
    for j in range(width):
        for c in range(channels):
            kernel = padded_image[i:i + kernel_size, j:j + kernel_size, c]
            blurred_image[i, j, c] = np.mean(kernel)

blurred_image = blurred_image.astype(np.uint8)

fig = plt.figure(1)

ax1 = fig.add_subplot(131)
ax1.imshow(input_image)
ax1.set_title("Original Image")
ax1.axis('off')

ax2 = fig.add_subplot(132)
ax2.imshow(blurred_image)
ax2.set_title("Blurred Image")
ax2.axis('off')

ax3 = fig.add_subplot(133)
ax3.imshow(grayscale, cmap=plt.cm.get_cmap('gray'))
ax3.set_title("Grayscale Image")
ax3.axis('off')

plt.tight_layout()
plt.show()
