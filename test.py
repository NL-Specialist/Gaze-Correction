import cv2
import numpy as np
import mediapipe as mp

# Function to extract eye region and return an image with a transparent background
def extract_eye_region(frame, eye_landmarks):
    h, w, _ = frame.shape
    # Convert landmarks to 2D pixel coordinates
    eye_coords = np.array([(int(landmark.x * w), int(landmark.y * h)) for landmark in eye_landmarks], np.int32)

    # Create a mask with the same size as the frame, filled with zeros (black)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Fill the eye region on the mask with white
    cv2.fillPoly(mask, [eye_coords], 255)

    # Create a transparent background (4 channels, the 4th being the alpha channel)
    transparent_image = np.zeros((h, w, 4), dtype=np.uint8)

    # Copy the original frame (BGR) where the mask is white
    eye_region = cv2.bitwise_and(frame, frame, mask=mask)

    # Add the eye region to the transparent image (first 3 channels for RGB, the last for Alpha)
    transparent_image[:, :, :3] = eye_region
    transparent_image[:, :, 3] = mask  # Use the mask as the alpha channel

    # Make all fully black (0, 0, 0) pixels in the eye region transparent by setting alpha to 0
    black_pixels = np.all(eye_region == [0, 0, 0], axis=-1)  # Detect black pixels in RGB channels
    transparent_image[black_pixels, 3] = 0  # Set alpha channel to 0 for these pixels

    # Crop the transparent image to the bounding box of the eye region
    x, y, w, h = cv2.boundingRect(eye_coords)
    cropped_eye = transparent_image[y:y+h, x:x+w]

    return cropped_eye

# Function to draw contours around the eyes
def draw_eye_contours(image, results, img_h, img_w, thickness):
    # Initialize lists to store eye points
    left_eye_points = []
    right_eye_points = []

    # Iterate over face landmarks
    for face_landmarks in results.multi_face_landmarks:
        for idx, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)

            # left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 145 ]]
            # right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380]]
            
            # Check if landmark is part of the left or right eye
            if idx in [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173]:
                left_eye_points.append((x, y))
            elif idx in [362, 398, 384, 385, 386, 387, 388, 390, 373, 374, 380, 466]:
                right_eye_points.append((x, y))

    # Draw contours around the eyes
    draw_contours(image, left_eye_points,right_eye_points, (0, 255, 0), thickness)
    # draw_contours(image, right_eye_points, (0, 255, 0))

    return image

# Function to draw contours between points
def draw_contours(image, left_eye_points, right_eye_points, color, thickness):
    # Draw the green lines around the left eye in the required order
    if len(left_eye_points) > 0:
        cv2.line(image, left_eye_points[0], left_eye_points[12], color, thickness)
        cv2.line(image, left_eye_points[0], left_eye_points[11], color, thickness)
        cv2.line(image, left_eye_points[11], left_eye_points[10], color,thickness)
        cv2.line(image, left_eye_points[10], left_eye_points[9], color, thickness)
        cv2.line(image, left_eye_points[9], left_eye_points[8], color,  thickness)
        cv2.line(image, left_eye_points[8], left_eye_points[7], color,  thickness)
        cv2.line(image, left_eye_points[7], left_eye_points[6], color,  thickness)
        cv2.line(image, left_eye_points[6], left_eye_points[5], color,  thickness)
        cv2.line(image, left_eye_points[5], left_eye_points[4], color,  thickness)
        cv2.line(image, left_eye_points[4], left_eye_points[3], color,  thickness)
        cv2.line(image, left_eye_points[3], left_eye_points[2], color,  thickness)
        cv2.line(image, left_eye_points[2], left_eye_points[12], color, thickness)
    else:
        print('FAILED DRAWING LEFT CONTOURS')

    # Draw the green lines around the right eye in the required order
    if len(right_eye_points) > 0:
        cv2.line(image, right_eye_points[0], right_eye_points[10], color, thickness)
        cv2.line(image, right_eye_points[10], right_eye_points[4], color, thickness)
        cv2.line(image, right_eye_points[4], right_eye_points[5], color,  thickness)
        cv2.line(image, right_eye_points[5], right_eye_points[6], color,  thickness)
        cv2.line(image, right_eye_points[6], right_eye_points[7], color,  thickness)
        cv2.line(image, right_eye_points[7], right_eye_points[8], color,  thickness)
        cv2.line(image, right_eye_points[8], right_eye_points[11], color, thickness)
        cv2.line(image, right_eye_points[11], right_eye_points[9], color, thickness)
        cv2.line(image, right_eye_points[9], right_eye_points[1], color,  thickness)
        cv2.line(image, right_eye_points[1], right_eye_points[2], color,  thickness)
        cv2.line(image, right_eye_points[2], right_eye_points[3], color,  thickness)
        cv2.line(image, right_eye_points[3], right_eye_points[0], color,  thickness)
    else:    
        print('FAILED DRAWING RIGHT CONTOURS')

# Function to detect pupils, draw contours, and save the eye images with transparency
def detect_pupils_and_save_eyes(frame, frame_with_contours, frame_with_dots, face_mesh, mp_drawing, mp_face_mesh, drawing_spec, show_face_mesh=True, show_pupils=True):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find face meshes
    results = face_mesh.process(image_rgb)
    img_h, img_w, _ = frame.shape

    # Draw eye contours for verification
    frame_with_contours = draw_eye_contours(frame_with_contours, results, img_h, img_w, 1)

    # Check if frame is valid before saving
    if frame_with_contours is not None and not np.all(frame_with_contours == 0):
        cv2.imwrite("image_with_eye_contours.jpg", frame_with_contours)
    else:
        print("Error: Frame with contours is invalid.")
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks for left and right eyes in correct order        
            # left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173]] # [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 145 ]
            # right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 398, 384, 385, 386, 387, 388, 390, 373, 374, 380, 466]] # [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380]

            # Left and right eye landmark indices as per MediaPipe
            left_eye_indices = [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173]
            right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 390, 373, 374, 380, 466]

            # Draw dots for all left eye landmarks
            for idx in left_eye_indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * frame_with_dots.shape[1])
                y = int(landmark.y * frame_with_dots.shape[0])
                cv2.circle(frame_with_dots, (x, y), 1, (0, 0, 255), -1)  # Red dots for left eye
                
            # Draw dots for all right eye landmarks
            for idx in right_eye_indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * frame_with_dots.shape[1])
                y = int(landmark.y * frame_with_dots.shape[0])
                cv2.circle(frame_with_dots, (x, y), 1, (255, 0, 0), -1)  # Blue dots for right eye

            # # Get coordinates for dots
            # left_eye_left_dot = (int(left_eye_landmarks[0].x * img_w), int(left_eye_landmarks[0].y * img_h))  # Point 33 (red dot)
            # left_eye_right_dot = (int(left_eye_landmarks[8].x * img_w), int(left_eye_landmarks[8].y * img_h))  # Point 144 (blue dot)
            # left_eye_top_dot = (int(left_eye_landmarks[4].x * img_w), int(left_eye_landmarks[4].y * img_h))  # Point 33 (red dot)
            # left_eye_bottom_dot = (int(left_eye_landmarks[12].x * img_w), int(left_eye_landmarks[12].y * img_h))  # Point 144 (blue dot)

            # right_eye_left_dot = (int(right_eye_landmarks[0].x * img_w), int(right_eye_landmarks[0].y * img_h))  # Point 362 (red dot)
            # right_eye_right_dot = (int(right_eye_landmarks[8].x * img_w), int(right_eye_landmarks[8].y * img_h))  # Point 380 (blue dot
            # right_eye_top_dot = (int(right_eye_landmarks[4].x * img_w), int(right_eye_landmarks[4].y * img_h))  # Point 362 (red dot)
            # right_eye_bottom_dot = (int(right_eye_landmarks[12].x * img_w), int(right_eye_landmarks[12].y * img_h))  # Point 380 (blue dot)

            # # Draw dots
            # cv2.circle(frame_with_dots, left_eye_left_dot, 1, (0, 0, 255), -1)  # Red dot on left eye (point 33)
            # cv2.circle(frame_with_dots, left_eye_right_dot, 1, (0, 0, 255), -1)  # Blue dot on left eye (point 144)
            # cv2.circle(frame_with_dots, left_eye_top_dot, 1, (0, 0, 255), -1)  # Red dot on left eye (point 33)
            # cv2.circle(frame_with_dots, left_eye_bottom_dot, 1, (0, 0, 255), -1)  # Blue dot on left eye (point 144)

            # cv2.circle(frame_with_dots, right_eye_left_dot, 1, (255, 0, 0), -1)  # Red dot on right eye (point 362)
            # cv2.circle(frame_with_dots, right_eye_right_dot, 1, (255, 0, 0), -1)  # Blue dot on right eye (point 380)
            # cv2.circle(frame_with_dots, right_eye_top_dot, 1, (255, 0, 0), -1)  # Red dot on right eye (point 362)
            # cv2.circle(frame_with_dots, right_eye_bottom_dot, 1, (255, 0, 0), -1)  # Blue dot on right eye (point 380)

            # Extract and save the left eye region with transparency
            left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144]] # 
            
            left_eye_image = extract_eye_region(frame, left_eye_landmarks)
            cv2.imwrite("left_eye_transparent.png", left_eye_image)

            # Extract and save the right eye region with transparency
            right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374]] # 
            right_eye_image = extract_eye_region(frame, right_eye_landmarks)
            cv2.imwrite("right_eye_transparent.png", right_eye_image)

        # Save the image with drawn contours for verification
        cv2.imwrite("image_with_eye_dots.jpg", frame_with_dots)


# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Load your image
# image = cv2.imread("destination_image.jpg")
# frame_with_contours = cv2.imread("destination_image.jpg")

image = cv2.imread("at_image.png")
frame_with_contours = cv2.imread("at_image.png")
frame_with_dots = cv2.imread("at_image.png")

# Detect pupils, draw contours, and save eyes
detect_pupils_and_save_eyes(image, frame_with_contours, frame_with_dots, face_mesh, mp_drawing, mp_face_mesh, drawing_spec, show_face_mesh=False, show_pupils=True)