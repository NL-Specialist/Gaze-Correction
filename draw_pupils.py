import cv2
import numpy as np

# Define the function to draw dots on the pupils
def draw_pupil_dots(frame, left_eye_landmarks, right_eye_landmarks, threshold=50):
    def get_pupil_center(eye_landmarks, frame, threshold):
        min_x = np.min(eye_landmarks[:, 0]) + 5
        max_x = np.max(eye_landmarks[:, 0]) - 10
        min_y = np.min(eye_landmarks[:, 1]) + 1
        max_y = np.max(eye_landmarks[:, 1]) - 1
        eye_region = frame[min_y:max_y, min_x:max_x]

        # Convert to grayscale before applying median blur and thresholding
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        blurred_eye = cv2.medianBlur(gray_eye, 5)
        _, thresholded_eye = cv2.threshold(blurred_eye, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        _, _, stats, centroids = cv2.connectedComponentsWithStats(thresholded_eye, 4)
        
        if len(centroids) > 1:
            largest_component_index = np.argmax(stats[1:, -1]) + 1
            pupil_center = centroids[largest_component_index]
            pupil_center_x = int(pupil_center[0] + min_x)
            pupil_center_y = int(pupil_center[1] + min_y)
            return (pupil_center_x, pupil_center_y)
        else:
            return None

    # Detect pupil centers for both eyes
    left_pupil_center = get_pupil_center(left_eye_landmarks, frame, threshold)
    right_pupil_center = get_pupil_center(right_eye_landmarks, frame, threshold)

    # Draw green dots on the pupils if detected
    if left_pupil_center:
        cv2.circle(frame, left_pupil_center, 3, (0, 255, 0), -1)
    if right_pupil_center:
        cv2.circle(frame, right_pupil_center, 3, (0, 255, 0), -1)

    return frame

# Process frame to add pupil tracking dots using face_mesh landmarks
def process_frame_with_face_mesh(face_mesh, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Define landmark indices for each eye
    eye_landmarks_indices = {
        "left_eye": [7, 163, 144, 153, 154, 155, 173, 157, 158, 159, 160, 161],
        "right_eye": [384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 398]
    }

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract coordinates for left and right eyes
            left_eye_landmarks = np.array(
                [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                  int(face_landmarks.landmark[i].y * frame.shape[0]))
                 for i in eye_landmarks_indices["left_eye"]]
            )
            right_eye_landmarks = np.array(
                [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                  int(face_landmarks.landmark[i].y * frame.shape[0]))
                 for i in eye_landmarks_indices["right_eye"]]
            )

            # Draw pupil dots on frame
            frame = draw_pupil_dots(frame, left_eye_landmarks, right_eye_landmarks, threshold=50)

    return frame

# Example usage:
# Load the input image
frame = cv2.imread("image.png")

# Assuming `face_mesh` is already initialized (e.g., from MediaPipe)
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    processed_frame = process_frame_with_face_mesh(face_mesh, frame)

# Save the result
cv2.imwrite("image_with_pupils.png", processed_frame)
