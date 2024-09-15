import cv2
import numpy as np

# Load the image
image = cv2.imread('right_eye.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply different low-pass filters
blurred_average = cv2.blur(gray_image, (5, 5))
blurred_gaussian = cv2.GaussianBlur(gray_image, (5, 5), 0)
blurred_median = cv2.medianBlur(gray_image, 5)
blurred_bilateral = cv2.bilateralFilter(gray_image, 5, 50, 50)

# ... apply FFT low-pass filter ...
# Perform FFT
fft = np.fft.fft2(gray_image)

# Shift FFT to center
fft_shift = np.fft.fftshift(fft)

# Create a low-pass filter mask
rows, cols = gray_image.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols))
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# Apply the mask
fft_shift_masked = fft_shift * mask

# Shift back and perform inverse FFT
fft_unshift = np.fft.ifftshift(fft_shift_masked)
blurred_image = np.real(np.fft.ifft2(fft_unshift))
blurred_image = np.uint8(np.clip(blurred_image, 0, 255))

# Save the blurred images
cv2.imwrite('right_eye_blurred_average.jpg', blurred_average)
cv2.imwrite('right_eye_blurred_gaussian.jpg', blurred_gaussian)
cv2.imwrite('right_eye_blurred_median.jpg', blurred_median)
cv2.imwrite('right_eye_blurred_bilateral.jpg', blurred_bilateral)
cv2.imwrite('right_eye_blurred_fft.jpg', blurred_image)