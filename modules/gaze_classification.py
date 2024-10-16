import cv2
import mediapipe as mp
import numpy as np

# ==========================================================
# GazeClassifier Module
# ==========================================================
# This module provides a class `GazeClassifier` for classifying gaze direction
# using MediaPipe's Face Mesh. The classification is based on the pitch and
# roll angles of the head.

# How Gaze is Calculated:
# 1. Initialize MediaPipe Face Mesh and Drawing utilities.
# 2. Capture a frame from the video input and convert it to RGB.
# 3. Use MediaPipe Face Mesh to detect facial landmarks in the frame.
# 4. Extract specific landmarks (nose tip, chin, left eye, right eye) to
#    define key vectors.
# 5. Calculate pitch and roll angles using these vectors.
# 6. Classify gaze direction based on the calculated pitch and roll angles.
# ==========================================================

class GazeClassifier:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        
        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Threshold values
        self.PITCH_THRESHOLD = 147  # degrees
        self.YAW_THRESHOLD = 5  # degrees

    def calculate_angle(self, vector1, vector2):
        """Calculate the angle between two vectors."""
        unit_vector1 = vector1 / np.linalg.norm(vector1)
        unit_vector2 = vector2 / np.linalg.norm(vector2)
        dot_product = np.dot(unit_vector1, unit_vector2)
        angle = np.arccos(dot_product)
        return np.degrees(angle)

    def classify_gaze(self, frame, show_face_mesh=True):
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find face meshes
        results = self.face_mesh.process(image_rgb)

        # Draw the face mesh annotations on the image if show_face_mesh is True
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if show_face_mesh:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec)

                # Extract landmarks
                landmarks = face_landmarks.landmark

                # Define key landmarks
                nose_tip = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
                chin = np.array([landmarks[152].x, landmarks[152].y, landmarks[152].z])
                left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
                right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])

                # Calculate vectors
                nose_to_chin = chin - nose_tip
                eye_to_eye = right_eye - left_eye

                # Reference vertical vector (upward direction)
                vertical_vector = np.array([0, -1, 0])

                # Calculate pitch (up/down tilt)
                pitch_angle = self.calculate_angle(nose_to_chin, vertical_vector)

                # Calculate roll (in-plane rotation)
                horizontal_vector = np.array([1, 0, 0])
                yaw_angle = self.calculate_angle(eye_to_eye, horizontal_vector)

                # Determine if looking at the camera or away
                if pitch_angle > self.PITCH_THRESHOLD and yaw_angle < self.YAW_THRESHOLD:
                    head_pose = "Gaze Direction: Camera"
                else:
                    head_pose = "Gaze Direction: Away"

                return head_pose

        return "Gaze: Unknown"

