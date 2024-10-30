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
    



