import cv2
import dlib
import numpy as np
import os
import glob

class GazeClassifier:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def classify_gaze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)

        def get_gaze_ratio(eye_points, facial_landmarks):
            eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points], np.int32)
            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            cv2.polylines(mask, [eye_region], True, 255, 2)
            cv2.fillPoly(mask, [eye_region], 255)
            eye = cv2.bitwise_and(gray, gray, mask=mask)
            
            min_x = np.min(eye_region[:, 0])
            max_x = np.max(eye_region[:, 0])
            min_y = np.min(eye_region[:, 1])
            max_y = np.max(eye_region[:, 1])
            
            gray_eye = eye[min_y: max_y, min_x: max_x]
            _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
            
            height, width = threshold_eye.shape
            left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
            left_side_white = cv2.countNonZero(left_side_threshold)
            
            right_side_threshold = threshold_eye[0: height, int(width / 2): width]
            right_side_white = cv2.countNonZero(right_side_threshold)
            
            if left_side_white == 0:
                gaze_ratio = 1
            elif right_side_white == 0:
                gaze_ratio = 5
            else:
                gaze_ratio = left_side_white / right_side_white
            return gaze_ratio

        for face in faces:
            landmarks = self.predictor(gray, face)
            
            left_eye_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            
            gaze_ratio = (left_eye_ratio + right_eye_ratio) / 2
            
            if gaze_ratio <= 1.5:
                return "Gaze: Looking Forward"
            else:
                return "Gaze: Looking Away"
        
        return "Gaze: Unknown"

# Path to the dlib predictor
# predictor_path = r"D:\OneDrive - North-West University\Rocketbook\2024\EERI474\FYP\Gaze Correction FYP\classifiers\shape_predictor_68_face_landmarks.dat"

# # Directory containing the images
# image_directory = r"D:\OneDrive - North-West University\Rocketbook\2024\EERI474\FYP\Gaze Correction FYP\output_images\xe"

# # Initialize the classifier
# gaze_classifier = GazeClassifier(predictor_path)

# # Process each image
# for folder in os.listdir(image_directory):
#     folder_path = os.path.join(image_directory, folder)
#     if os.path.isdir(folder_path):
#         image_path = os.path.join(folder_path, "full_frame.jpg")
#         if os.path.exists(image_path):
#             image = cv2.imread(image_path)
#             if image is not None:
#                 gaze_result = gaze_classifier.classify_gaze(image)
#                 print(f"Image: {image_path}, Gaze Classification: {gaze_result}")
#             else:
#                 print(f"Failed to load image: {image_path}")
