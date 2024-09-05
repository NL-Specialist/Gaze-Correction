import cv2
from ultralytics import SAM
import numpy as np

# Initialize variables to store eye points
eye_points = []

def click_event(event, x, y, flags, params):
    """Mouse click event to capture eye points."""
    global eye_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the clicked point (x, y) to the eye_points list
        print(f"Point selected: ({x}, {y})")
        eye_points.append([x, y])
        
        # Once two points are selected, close the image window
        if len(eye_points) == 2:
            cv2.destroyAllWindows()

# Load the image
image_path = "sam_test_image.jpg"
image = cv2.imread(image_path)

# Display the image and wait for clicks
print("Please click on your left eye, then on your right eye.")
cv2.imshow("Select Eyes", image)
cv2.setMouseCallback("Select Eyes", click_event)

cv2.waitKey(0)  # Wait for the two points to be selected

# Ensure two points are selected
if len(eye_points) != 2:
    print("Error: Two points were not selected. Please try again.")
    exit()

# Load the SAM model
print("Loading the SAM model.")
model = SAM("sam2_b.pt")

# Segment with the selected points (eyes)
eye_labels = [1, 1]  # Label as foreground for both eyes
eye_results = model(image_path, points=eye_points, labels=eye_labels)

# Access the mask data from the results
eye_mask = eye_results[0].masks.data[0].cpu().numpy()

# Convert to uint8 format for saving (0-255 scale)
eye_image = (eye_mask * 255).astype('uint8')

# Define the path to the output image
output_eye_image_path = "sam_test_result_eyes.jpg"

# Save the segmented eye mask image
print(f"Saving the eye segmentation result at: {output_eye_image_path}")
cv2.imwrite(output_eye_image_path, eye_image)

print("Eye segmentation image saved successfully.")
