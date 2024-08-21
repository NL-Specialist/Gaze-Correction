import cv2
import mediapipe as mp
from modules.gaze_classification import GazeClassifier
import numpy as np
import dlib
import logging
import os
import requests

class Eyes:
    def __init__(self):
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                             max_num_faces=1,
                                                             min_detection_confidence=0.5,
                                                             min_tracking_confidence=0.5)
            self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
            self.fixed_box_height = 24  # Fixed bounding box height
            self.fixed_box_width = 50   # Fixed bounding box width
            self.fixed_box_dimensions = (self.fixed_box_width, self.fixed_box_height)
            self.offset_y = 20  # Offset to move the bounding box up
            self.left_eye_offset_x = 0  # Offset to move the left eye bounding box left/right
            self.left_eye_offset_y = 5  # Offset to move the left eye bounding box up/down
            self.right_eye_offset_x = 5  # Offset to move the right eye bounding box left/right
            self.right_eye_offset_y = 5  # Offset to move the right eye bounding box up/down
            self.left_eye_bbox = None
            self.right_eye_bbox = None

            self.should_correct_gaze = False
            self.gaze_direction = "Gaze Direction: Away"

            # Initialize overlay images
            self.left_eye_img = None
            self.right_eye_img = None

            self.gaze_classifier = GazeClassifier()

            # try:
            #     response = requests.post("http://192.168.0.58:8021/load_model/", json={"model_name":"Test1"})
            #     if response.status_code == 200:
            #         print(response.status())
            # except Exception as e:
            #     print("Error loading GAN: ", e)
            # self.set_default_overlay_eyes()
        except Exception as e:
            logging.error(f"Error initializing Eyes: {e}")
            raise

    def set_default_overlay_eyes(self):
        try:
            # Get the current working directory
            current_dir = os.getcwd()

            # Define the path to the datasets folder
            datasets_path = os.path.join(current_dir, 'datasets')

            # Get the first folder inside the datasets folder
            first_folder = os.listdir(datasets_path)[0]

            # Define the path to the at_camera folder inside the first folder
            at_camera_path = os.path.join(datasets_path, first_folder, 'at_camera')

            # Define the path to the image_1 folder inside the at_camera folder
            image_1_path = os.path.join(at_camera_path, 'image_1')

            # Define the paths to the left_eye and right_eye images
            left_eye_img_path = os.path.join(image_1_path, 'left_eye.jpg')
            right_eye_img_path = os.path.join(image_1_path, 'right_eye.jpg')

            # Read the images using cv2
            self.left_eye_img = cv2.imread(left_eye_img_path)
            self.right_eye_img = cv2.imread(right_eye_img_path)

            if self.left_eye_img is None or self.right_eye_img is None:
                raise FileNotFoundError("Left or right eye image not found.")
        except Exception as e:
            logging.error(f"Error in get_new_eyes: {e}")
            raise

    def process_frame(self, frame, show_face_mesh=True, classify_gaze=True, draw_rectangles=True):
        try:
            h, w, _ = frame.shape

            if classify_gaze:
                # Determine gaze direction
                self.gaze_direction = self.gaze_classifier.classify_gaze(frame, show_face_mesh)
                self.should_correct_gaze = self.gaze_direction == "Gaze Direction: Away"

                cv2.putText(frame, self.gaze_direction, (w // 2 - 100, 70 * h // 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if draw_rectangles and self.left_eye_bbox and self.right_eye_bbox:
                cv2.rectangle(frame, self.left_eye_bbox[0], self.left_eye_bbox[1], (0, 255, 0), 2)
                cv2.rectangle(frame, self.right_eye_bbox[0], self.right_eye_bbox[1], (0, 255, 0), 2)

            return frame
        except Exception as e:
            logging.error(f"Error in process_frame: {e}")
            raise

    def capture_left_eye_region(self, frame):
        try:
            left_eye_frame, _ = self.extract_eye_region(frame, eye="left")
            return self.place_on_colored_square(left_eye_frame, bg_color=(255, 0, 0))
        except Exception as e:
            logging.error(f"Error in capture_left_eye_region: {e}")
            raise

    def capture_right_eye_region(self, frame, show_eyes=False):
        try:
            _, right_eye_frame = self.extract_eye_region(frame, eye="right")
            return self.place_on_colored_square(right_eye_frame, bg_color=(255, 0, 0))
        except Exception as e:
            logging.error(f"Error in capture_right_eye_region: {e}")
            raise

    def place_on_colored_square(self, eye_image, bg_color=(255, 0, 0)):
        if bg_color is None:
            return eye_image
    
        # Create a colored square with the specified background color
        colored_square = np.full((self.fixed_box_dimensions[1], self.fixed_box_dimensions[0], 3), bg_color, dtype=np.uint8)
        eye_h, eye_w, _ = eye_image.shape
    
        # Calculate the offsets to center the eye image on the square
        x_offset = max((self.fixed_box_dimensions[0] - eye_w) // 2, 0)
        y_offset = max((self.fixed_box_dimensions[1] - eye_h) // 2, 0)
    
        # Place the eye image on the colored square
        colored_square[y_offset:y_offset + eye_h, x_offset:x_offset + eye_w] = eye_image
    
        return colored_square



    def _extract_eye_region(self, frame, eye_points, bg_color=(255, 0, 0)):
        if bg_color is None:
            return frame

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(eye_points)], (255))
        eye_region = cv2.bitwise_and(frame, frame, mask=mask)

        x, y, w, h = cv2.boundingRect(np.array(eye_points))
        eye_crop = eye_region[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]

        colored_background = np.full_like(eye_crop, bg_color, dtype=np.uint8)
        colored_background[mask_crop == 255] = eye_crop[mask_crop == 255]

        return colored_background



    def extract_eye_region(self, frame, eye):
        try:
            img_h, img_w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    eye_landmarks = {
                        "left_eye": [7, 163, 144, 153, 154, 155, 173, 157, 158, 159, 160, 161],
                        "right_eye": [384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 398]
                    }

                    left_eye_points = []
                    right_eye_points = []
                    
                    for idx in eye_landmarks["left_eye"]:
                        x = int(face_landmarks.landmark[idx].x * img_w)
                        y = int(face_landmarks.landmark[idx].y * img_h)
                        left_eye_points.append((x, y))

                    for idx in eye_landmarks["right_eye"]:
                        x = int(face_landmarks.landmark[idx].x * img_w)
                        y = int(face_landmarks.landmark[idx].y * img_h)
                        right_eye_points.append((x, y))

                    left_eye_frame = self._extract_eye_region(frame, left_eye_points,  (255, 0, 0))
                    right_eye_frame = self._extract_eye_region(frame, right_eye_points, (255, 0, 0))

                    return left_eye_frame , right_eye_frame

            return frame, None
        except Exception as e:
            logging.error(f"Error in extract_eye_region: {e}")
            raise
    

    def get_left_eye_region(self, frame, show_eyes=False):
        try:
            frame = self.draw_selected_landmarks(frame, show_eyes=show_eyes, show_mouth=False, show_face_outline=False, show_text=False)
            if self.left_eye_bbox:
                (x_min, y_min), (x_max, y_max) = self.left_eye_bbox
                left_eye_frame = frame[y_min:y_max, x_min:x_max]
                return left_eye_frame
            return None
        except Exception as e:
            logging.error(f"Error in get_left_eye_region: {e}")
            raise

    def get_right_eye_region(self, frame, show_eyes=False):
        try:
            frame = self.draw_selected_landmarks(frame, show_eyes=show_eyes, show_mouth=False, show_face_outline=False, show_text=False)
            if self.right_eye_bbox:
                (x_min, y_min), (x_max, y_max) = self.right_eye_bbox
                right_eye_frame = frame[y_min:y_max, x_min:x_max]
                return right_eye_frame
            return None
        except Exception as e:
            logging.error(f"Error in get_right_eye_region: {e}")
            raise      

    def draw_selected_landmarks(self, frame, show_eyes=False, show_mouth=False, show_face_outline=False, show_text=False):
        try:
            if show_eyes==False and show_mouth==False and show_face_outline==False and show_text==False:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        self._calculate_eye_boxes(face_landmarks, frame)
                return frame
            
            img_h, img_w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = int(landmark.x * img_w)
                        y = int(landmark.y * img_h)

                        # "left_eye": [7, 163, 144, 153, 154, 155, 173, 157, 158, 159, 160, 161],
                        # "right_eye": [384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 398]

                        if show_eyes and idx in [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
                                                 362, 398, 384, 385, 386, 387, 388, 390, 373, 374, 380, 466]:
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Draw small circle at each landmark point for eyes
                            if show_text:
                                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                        elif show_mouth and idx in [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13]:
                            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)  # Draw small circle at each landmark point for mouth
                            if show_text:
                                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                        elif show_face_outline and idx in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365]:
                            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)  # Draw small circle at each landmark point for face outline
                            if show_text:
                                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            return frame
        except Exception as e:
            logging.error(f"Error in draw_selected_landmarks: {e}")
            raise

    def correct_gaze(self, frame, corrected_left_eye_path, corrected_right_eye_path):
        try:
            if self.should_correct_gaze:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        self._calculate_eye_boxes(face_landmarks, frame)

                        # Load the corrected eye images from the provided paths
                        if not corrected_left_eye_path == None:
                            self.left_eye_img = cv2.imread(corrected_left_eye_path)
                            self.overlay_image(frame, self.left_eye_bbox, self.left_eye_img)
                        if not corrected_right_eye_path == None:
                            self.right_eye_img = cv2.imread(corrected_right_eye_path)
                            self.overlay_image(frame, self.right_eye_bbox, self.right_eye_img)

            return frame
        except Exception as e:
            logging.error(f"Error in correct_gaze: {e}")
            raise


    def _calculate_eye_boxes(self, face_landmarks, frame):
        try:
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
                half_width = self.fixed_box_width // 2
                half_height = self.fixed_box_height // 2
                x_min = max(0, center_x - half_width)
                y_min = max(0, center_y - half_height)
                x_max = min(w, center_x + half_width)
                y_max = min(h, center_y + half_height)
                return (x_min, y_min), (x_max, y_max), center_y

            left_eye_bbox, left_eye_center_y = _get_bounding_box(left_eye)[:2], _get_bounding_box(left_eye)[2]
            right_eye_bbox, right_eye_center_y = _get_bounding_box(right_eye)[:2], _get_bounding_box(right_eye)[2]

            avg_center_y = (left_eye_center_y + right_eye_center_y) // 2 - self.offset_y

            self.left_eye_bbox = ((left_eye_bbox[0][0] + self.left_eye_offset_x, avg_center_y - self.fixed_box_height // 2 + self.left_eye_offset_y),
                                  (left_eye_bbox[1][0] + self.left_eye_offset_x, avg_center_y + self.fixed_box_height // 2 + self.left_eye_offset_y))

            self.right_eye_bbox = ((right_eye_bbox[0][0] + self.right_eye_offset_x, avg_center_y - self.fixed_box_height // 2 + self.right_eye_offset_y),
                                   (right_eye_bbox[1][0] + self.right_eye_offset_x, avg_center_y + self.fixed_box_height // 2 + self.right_eye_offset_y))
        except Exception as e:
            logging.error(f"Error in _calculate_eye_boxes: {e}")
            raise

    

    def overlay_image(self, frame, eye_bbox, eye_img):
        try:
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
        
        except Exception as e:
            logging.error(f"Error in overlay_image: {e}")
            raise

