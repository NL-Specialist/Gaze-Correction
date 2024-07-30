import cv2
import mediapipe as mp
from modules.gaze_classification import GazeClassifier  # Correct import statement
import numpy as np
import dlib

class Eyes:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                         max_num_faces=1,
                                                         min_detection_confidence=0.5,
                                                         min_tracking_confidence=0.5)
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        self.fixed_box_size = 50  # Fixed bounding box size: 50x50 pixels
        self.offset_y = 20  # Offset to move the bounding box up
        self.left_eye_offset_x = 0  # Offset to move the left eye bounding box left/right
        self.left_eye_offset_y = 5  # Offset to move the left eye bounding box up/down
        self.right_eye_offset_x = 5  # Offset to move the right eye bounding box left/right
        self.right_eye_offset_y = 10  # Offset to move the right eye bounding box up/down
        self.left_eye_bbox = None
        self.right_eye_bbox = None

        # Load the overlay images
        self.left_eye_img = cv2.imread('D:/OneDrive - North-West University/Rocketbook/2024/EERI474/FYP/Gaze Correction FYP/datasets/at_camera2/image_1/left_eye.jpg')
        self.right_eye_img = cv2.imread('D:/OneDrive - North-West University/Rocketbook/2024/EERI474/FYP/Gaze Correction FYP/datasets/at_camera2/image_1/right_eye.jpg')


        # Initialize GazeClassifier with the path to the shape predictor
        predictor_path = 'D:/OneDrive - North-West University/Rocketbook/2024/EERI474/FYP/Gaze Correction FYP/classifiers/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('D:/OneDrive - North-West University/Rocketbook/2024/EERI474/FYP/Gaze Correction FYP/classifiers/shape_predictor_68_face_landmarks.dat')
        self.left_eye_indices = list(range(36, 42))
        self.right_eye_indices = list(range(42, 48))

        self.gaze_classifier = GazeClassifier(predictor_path)

    def process_frame(self, frame, draw_rectangles=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._draw_eye_boxes(frame, face_landmarks, draw_rectangles)
        return frame

    def _draw_eye_boxes(self, frame, face_landmarks, draw_rectangles):
        h, w, _ = frame.shape
        left_eye = [face_landmarks.landmark[i] for i in [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]]
        right_eye = [face_landmarks.landmark[i] for i in [362, 398, 384, 385, 386, 387, 388, 390, 391, 392, 393, 373, 374, 380, 466]]

        def _get_bounding_box(landmarks):
            x_min = min([pt.x * w for pt in landmarks])
            y_min = min([pt.y * h for pt in landmarks])
            x_max = max([pt.x * w for pt in landmarks])
            y_max = max([pt.y * h for pt in landmarks])
            # Center the bounding box around the detected landmarks
            center_x = int((x_min + x_max) // 2)
            center_y = int((y_min + y_max) // 2)
            half_size = self.fixed_box_size // 2
            x_min = max(0, center_x - half_size)
            y_min = max(0, center_y - half_size)
            x_max = min(w, center_x + half_size)
            y_max = min(h, center_y + half_size)
            return (x_min, y_min), (x_max, y_max), center_y

        left_eye_bbox, left_eye_center_y = _get_bounding_box(left_eye)[:2], _get_bounding_box(left_eye)[2]
        right_eye_bbox, right_eye_center_y = _get_bounding_box(right_eye)[:2], _get_bounding_box(right_eye)[2]

        # Calculate the average y-coordinate to align both boxes
        avg_center_y = (left_eye_center_y + right_eye_center_y) // 2 - self.offset_y

        # Apply offsets to the left eye bounding box
        self.left_eye_bbox = ((left_eye_bbox[0][0] + self.left_eye_offset_x, avg_center_y - self.fixed_box_size // 2 + self.left_eye_offset_y),
                              (left_eye_bbox[1][0] + self.left_eye_offset_x, avg_center_y + self.fixed_box_size // 2 + self.left_eye_offset_y))

        # Apply offsets to the right eye bounding box
        self.right_eye_bbox = ((right_eye_bbox[0][0] + self.right_eye_offset_x, avg_center_y - self.fixed_box_size // 2 + self.right_eye_offset_y),
                               (right_eye_bbox[1][0] + self.right_eye_offset_x, avg_center_y + self.fixed_box_size // 2 + self.right_eye_offset_y))

        if draw_rectangles:
            cv2.rectangle(frame, self.left_eye_bbox[0], self.left_eye_bbox[1], (0, 255, 0), 2)
            cv2.rectangle(frame, self.right_eye_bbox[0], self.right_eye_bbox[1], (0, 255, 0), 2)

            # Determine gaze direction
            gaze_direction = self.gaze_classifier.classify_gaze(frame)
            cv2.putText(frame, gaze_direction, (w // 2 - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def get_left_eye_region(self, frame):
        if self.left_eye_bbox:
            (x_min, y_min), (x_max, y_max) = self.left_eye_bbox
            frame = self.draw_selected_landmarks(frame, show_eyes=True, show_mouth=False, show_face_outline=False, show_text=False)
            left_eye_frame = frame[y_min:y_max, x_min:x_max]
            return left_eye_frame # self.mark_eye_region(left_eye_frame)
        return None

    def get_right_eye_region(self, frame):
        if self.right_eye_bbox:
            (x_min, y_min), (x_max, y_max) = self.right_eye_bbox
            frame = self.draw_selected_landmarks(frame, show_eyes=True, show_mouth=False, show_face_outline=False, show_text=False)
            right_eye_frame = frame[y_min:y_max, x_min:x_max]
            return right_eye_frame # self.mark_eye_region(right_eye_frame)
        return None      

    def draw_selected_landmarks(self, frame, show_eyes=False, show_mouth=False, show_face_outline=False, show_text=False):
        img_h, img_w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * img_w)
                    y = int(landmark.y * img_h)

                    if show_eyes and idx in [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
                                             362, 398, 384, 385, 386, 387, 388, 390, 373, 374, 380, 466]:
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Draw small circle at each landmark point for eyes
                        if show_text : cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    elif show_mouth and idx in [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13]:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)  # Draw small circle at each landmark point for mouth
                        if show_text : cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    elif show_face_outline and idx in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365]:
                        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)  # Draw small circle at each landmark point for face outline
                        if show_text : cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        return frame






    def correct_gaze(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._calculate_eye_boxes(face_landmarks, frame)
                frame = self._overlay_eye_images(frame)
        return frame


    def _calculate_eye_boxes(self, face_landmarks, frame):
        h, w, _ = frame.shape
        left_eye = [face_landmarks.landmark[i] for i in [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]]
        right_eye = [face_landmarks.landmark[i] for i in [362, 398, 384, 385, 386, 387, 388, 390, 391, 392, 393, 373, 374, 380, 466]]

        def _get_bounding_box(landmarks):
            x_min = min([pt.x * w for pt in landmarks])
            y_min = min([pt.y * h for pt in landmarks])
            x_max = max([pt.x * w for pt in landmarks])
            y_max = max([pt.y * h for pt in landmarks])
            center_x = int((x_min + x_max) // 2)
            center_y = int((y_min + y_max) // 2)
            half_size = self.fixed_box_size // 2
            x_min = max(0, center_x - half_size)
            y_min = max(0, center_y - half_size)
            x_max = min(w, center_x + half_size)
            y_max = min(h, center_y + half_size)
            return (x_min, y_min), (x_max, y_max), center_y

        left_eye_bbox, left_eye_center_y = _get_bounding_box(left_eye)[:2], _get_bounding_box(left_eye)[2]
        right_eye_bbox, right_eye_center_y = _get_bounding_box(right_eye)[:2], _get_bounding_box(right_eye)[2]

        avg_center_y = (left_eye_center_y + right_eye_center_y) // 2 - self.offset_y

        self.left_eye_bbox = ((left_eye_bbox[0][0] + self.left_eye_offset_x, avg_center_y - self.fixed_box_size // 2 + self.left_eye_offset_y),
                              (left_eye_bbox[1][0] + self.left_eye_offset_x, avg_center_y + self.fixed_box_size // 2 + self.left_eye_offset_y))

        self.right_eye_bbox = ((right_eye_bbox[0][0] + self.right_eye_offset_x, avg_center_y - self.fixed_box_size // 2 + self.right_eye_offset_y),
                               (right_eye_bbox[1][0] + self.right_eye_offset_x, avg_center_y + self.fixed_box_size // 2 + self.right_eye_offset_y))
    
    def _overlay_eye_images(self, frame):
        def overlay_image(eye_bbox, eye_img):
            (x_min, y_min), (x_max, y_max) = eye_bbox
            eye_region = frame[y_min:y_max, x_min:x_max]
            eye_img_resized = cv2.resize(eye_img, (x_max - x_min, y_max - y_min))

            if eye_img_resized.shape[2] == 4:  # Check if the image has an alpha channel
                alpha_s = eye_img_resized[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    eye_region[:, :, c] = (alpha_s * eye_img_resized[:, :, c] +
                                           alpha_l * eye_region[:, :, c])
            else:
                frame[y_min:y_max, x_min:x_max] = eye_img_resized

        overlay_image(self.left_eye_bbox, self.left_eye_img)
        overlay_image(self.right_eye_bbox, self.right_eye_img)
        return frame

