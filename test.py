import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Start capturing video input from the specified camera (camera 1)
cap = cv2.VideoCapture(1)

# Threshold values
PITCH_THRESHOLD = 150  # degrees
# YAW_THRESHOLD = 15  # degrees
ROLL_THRESHOLD = 5  # degrees

def calculate_angle(vector1, vector2):
    """Calculate the angle between two vectors."""
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Rotate the frame 90 degrees counterclockwise
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find face meshes
    results = face_mesh.process(image_rgb)

    # Draw the face mesh annotations on the image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # Extract landmarks
            landmarks = face_landmarks.landmark

            # Define key landmarks
            nose_tip = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
            chin = np.array([landmarks[152].x, landmarks[152].y, landmarks[152].z])
            left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
            right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])

            # Calculate vectors
            nose_to_chin = chin - nose_tip
            nose_to_left_eye = left_eye - nose_tip
            nose_to_right_eye = right_eye - nose_tip
            eye_to_eye = right_eye - left_eye

            # Reference vertical vector (upward direction)
            vertical_vector = np.array([0, -1, 0])

            # Calculate pitch (up/down tilt)
            pitch_angle = calculate_angle(nose_to_chin, vertical_vector)

            # Calculate yaw (left/right turn)
            yaw_angle = calculate_angle(nose_to_left_eye, nose_to_right_eye)

            # Calculate roll (in-plane rotation)
            horizontal_vector = np.array([1, 0, 0])
            roll_angle = calculate_angle(eye_to_eye, horizontal_vector)

            # Determine if looking at the camera or away
            if pitch_angle > PITCH_THRESHOLD and roll_angle < ROLL_THRESHOLD: # or yaw_angle < YAW_THRESHOLD
                head_pose = "Looking at Camera"
            else:
                head_pose = "Looking Away"

            # Display values and head pose on the image
            cv2.putText(image, f"Pitch: {pitch_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"Yaw: {yaw_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"Roll: {roll_angle:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, head_pose, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('MediaPipe Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
