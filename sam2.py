import cv2
from ultralytics import SAM
import numpy as np

# Define the remove_eyes function
def remove_eyes(frame):
    """Removes eyes from the input frame using two points selected by the user."""
    
    # Initialize variables to store eye points
    eye_points = []

    def click_event(event, x, y, flags, params):
        """Mouse click event to capture eye points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Append the clicked point (x, y) to the eye_points list
            print(f"Point selected: ({x}, {y})")
            eye_points.append([x, y])

            # Once two points are selected, close the image window
            if len(eye_points) == 2:
                cv2.destroyAllWindows()

    # Display the frame and wait for clicks to select eye points
    print("Please click on your left eye, then on your right eye.")
    cv2.imshow("Select Eyes", frame)
    cv2.setMouseCallback("Select Eyes", click_event)

    cv2.waitKey(0)  # Wait for the two points to be selected

    # Ensure two points are selected
    if len(eye_points) != 2:
        print("Error: Two points were not selected. Please try again.")
        return frame  # Return the original frame if points were not selected

    # Load the SAM model
    print("Loading the SAM model.")
    model = SAM("sam2_b.pt")

    # Segment with the selected points (eyes)
    eye_labels = [1, 1]  # Label as foreground for both eyes
    eye_results = model(frame, points=eye_points, labels=eye_labels)

    # Access the mask data from the results
    eye_mask = eye_results[0].masks.data.sum(axis=0).cpu().numpy()  # Combine the two eye masks

    # Convert the mask to uint8 format and create an inverted mask
    eye_mask = (eye_mask > 0).astype('uint8')  # Binary mask where eyes are segmented
    inverse_mask = 1 - eye_mask  # Invert the mask to remove eyes from the image

    # Apply the inverse mask to the original frame to remove the eye segments
    result_image = frame * inverse_mask[:, :, np.newaxis]

    return result_image

