import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('R:\Programs\Deep Learning\WhatsApp Image 2022-03-18 at 6.54.12 AM.jpeg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# Display the original and segmented images
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(binary_image, cmap='gray'), plt.title('Segmented Image')
plt.show()
