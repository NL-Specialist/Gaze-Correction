import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat

# Open the camera input (camera 1 in this case)
camera = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get camera properties
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv2.CAP_PROP_FPS)

# Open the virtual camera
with pyvirtualcam.Camera(width=height, height=width, fps=fps, fmt=PixelFormat.BGR) as vcam:
    print(f'Virtual camera ready with resolution {height}x{width} at {fps} FPS (rotated 90 degrees)')

    while True:
        # Capture frame-by-frame from the real camera
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Manipulate the frame (example: convert to grayscale)
        manipulated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        manipulated_frame = cv2.cvtColor(manipulated_frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for virtual cam

        # Rotate the frame 90 degrees anti-clockwise
        rotated_frame = cv2.rotate(manipulated_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Output the rotated frame to the virtual camera
        vcam.send(rotated_frame)

        # Wait for the next frame
        vcam.sleep_until_next_frame()

        # Show the manipulated and rotated frame for debugging purposes (optional)
        cv2.imshow('Manipulated & Rotated Frame', rotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the real camera and close any OpenCV windows
camera.release()
cv2.destroyAllWindows()
