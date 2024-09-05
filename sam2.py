import cv2
from ultralytics import SAM

print("Starting the SAM model script.")

# Load the SAM model
print("Loading the SAM model.")
model = SAM("sam2_b.pt")

# Display model information (optional)
print("Displaying model information.")
model.info()

# Define the path to the input image
input_image_path = r"datasets\Test3_Dataset\at_camera\image_1\full_frame.jpg"
print(f"Input image path: {input_image_path}")

# Load the image using OpenCV
print("Loading the image.")
image = cv2.imread(input_image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()
else:
    print(f"Image loaded successfully. Image shape: {image.shape}")

# Rotate the image 90 degrees counterclockwise
print("Rotating the image 90 degrees counterclockwise.")
image_rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Print shape of the rotated image
print(f"Rotated image shape: {image_rotated.shape}")

# Segment the entire rotated image using the SAM model
print("Running segmentation on the rotated image.")
results = model(image_rotated)

# Check if the results contain masks
if hasattr(results, 'masks') and results.masks is not None:
    print(f"Number of masks found: {len(results.masks)}")
    # Draw the segmentation results on the image
    for i, mask in enumerate(results.masks):
        print(f"Processing mask {i+1}/{len(results.masks)}.")
        image_rotated[mask] = (0, 255, 0)  # Example color for the mask

    # Display the segmented and rotated image
    print("Displaying the segmented and rotated image.")
    cv2.imshow("Segmented Image", image_rotated)
else:
    print("Error: No masks found in the results.")

# Wait until a key is pressed to close the window
print("Waiting for key press to close the window.")
cv2.waitKey(0)

# Close all OpenCV windows
print("Closing all OpenCV windows.")
cv2.destroyAllWindows()

# Optionally, save the segmented and rotated image to a file
output_image_path = r"datasets\Test3_Dataset\at_camera\image_1\full_frame_segmented.jpg"
print(f"Saving the segmented and rotated image to: {output_image_path}")
cv2.imwrite(output_image_path, image_rotated)

print("Script finished.")
