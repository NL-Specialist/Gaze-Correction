import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_image_intensity(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        return "Image not found or invalid file format."
    
    # Calculate the average pixel intensity
    average_intensity = np.mean(image)
    print("average_intensity: ", average_intensity)
    
    # Plot the histogram of pixel intensities
    plt.figure(figsize=(10, 6))
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title('Pixel Intensity Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    return average_intensity

analyze_image_intensity('datasets/Auto6/at_camera/image_1/full_frame.jpg')