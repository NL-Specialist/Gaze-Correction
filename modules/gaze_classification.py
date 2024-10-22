import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

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

        self.EYE_POSITION_THRESHOLD = 3

    def calculate_angle(self, vector1, vector2):
        """Calculate the angle between two vectors."""
        unit_vector1 = vector1 / np.linalg.norm(vector1)
        unit_vector2 = vector2 / np.linalg.norm(vector2)
        dot_product = np.dot(unit_vector1, unit_vector2)
        angle = np.arccos(dot_product)
        return np.degrees(angle)

   # Function to calculate EAR (Eye Aspect Ratio)
    def calculate_ear(self, eye):
        # Compute the Euclidean distances between the two sets of vertical eye landmarks (p2, p6) and (p3, p5)
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # Compute the Euclidean distance between the horizontal eye landmarks (p1, p4)
        C = dist.euclidean(eye[0], eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def classify_gaze(self, frame, show_face_mesh, show_pupils):
        old_frame = frame 
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)

        # Process the image and find face meshes
        results = self.face_mesh.process(image_rgb)

        # EAR threshold for detecting closed eyes
        EAR_THRESHOLD = 0.22  # Adjusted for more sensitivity
        GAZE_OFFSET_THRESHOLD = 2  # Reduced from 3 for more sensitivity

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if show_face_mesh:
                    self.mp_drawing.draw_landmarks(
                        image=old_frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec)

                # Extract landmarks for left and right eyes using the specified indices
                left_eye_landmarks = [face_landmarks.landmark[i] for i in [7, 163, 144, 153, 154, 155, 173, 157, 158, 159, 160, 161]]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in [384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 398]]

                # Convert landmarks to pixel positions
                left_eye_pts = np.array([[int(landmark.x * old_frame.shape[1]), int(landmark.y * old_frame.shape[0])] for landmark in left_eye_landmarks])
                right_eye_pts = np.array([[int(landmark.x * old_frame.shape[1]), int(landmark.y * old_frame.shape[0])] for landmark in right_eye_landmarks])

                # Calculate EAR for both eyes
                left_ear = self.calculate_ear(left_eye_pts)
                right_ear = self.calculate_ear(right_eye_pts)

                # Check if eyes are closed
                if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                    print("Eyes are closed")
                    gaze_direction = "Gaze Direction: Eyes Closed"
                    return gaze_direction

                # Approximate the pupil position by averaging the eye region's landmarks
                left_eye_center = np.mean(left_eye_pts, axis=0).astype(int)
                right_eye_center = np.mean(right_eye_pts, axis=0).astype(int)

                # Calculate horizontal and vertical offsets
                left_eye_position = left_eye_center[0] - np.mean(left_eye_pts[:, 0])  # Horizontal offset
                right_eye_position = right_eye_center[0] - np.mean(right_eye_pts[:, 0])  # Horizontal offset
                left_eye_vertical_offset = left_eye_center[1] - np.mean(left_eye_pts[:, 1])  # Vertical offset
                right_eye_vertical_offset = right_eye_center[1] - np.mean(right_eye_pts[:, 1])  # Vertical offset

                # Print debug information
                print(f"Left Eye Position Offset: {left_eye_position}, Vertical Offset: {left_eye_vertical_offset}")
                print(f"Right Eye Position Offset: {right_eye_position}, Vertical Offset: {right_eye_vertical_offset}")
                print(f"Horizontal Threshold: {GAZE_OFFSET_THRESHOLD}")

                # Improved threshold to determine if the user is looking at the camera
                if (abs(left_eye_position) < GAZE_OFFSET_THRESHOLD and abs(right_eye_position) < GAZE_OFFSET_THRESHOLD and 
                    abs(left_eye_vertical_offset) < GAZE_OFFSET_THRESHOLD and abs(right_eye_vertical_offset) < GAZE_OFFSET_THRESHOLD):
                    print("Gaze classified as: Camera")
                    gaze_direction = "Gaze Direction: Camera"
                else:
                    print("Gaze classified as: Away")
                    gaze_direction = "Gaze Direction: Away"

                if show_pupils:
                    # Optionally visualize the pupil centers
                    cv2.circle(old_frame, tuple(left_eye_center), 3, (0, 255, 0), -1)
                    cv2.circle(old_frame, tuple(right_eye_center), 3, (0, 255, 0), -1)

                return gaze_direction

        return "Gaze: Unknown"



