# Core libraries
import os
import time
import logging

# Third-party image processing libraries
import cv2
import numpy as np
from skimage.exposure import match_histograms

# Face and eye detection libraries
import dlib
import mediapipe as mp

# Gaze classification module
from modules.gaze_classification import GazeClassifier

# SAM (Segment Anything Model) library for advanced image segmentation
from ultralytics import SAM

# HTTP requests library
import requests


class Eyes:
    def __init__(self):
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                             max_num_faces=1,
                                                             min_detection_confidence=0.4,
                                                             min_tracking_confidence=0.4)
            self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
            self.fixed_box_height = 24  # Fixed bounding box height
            self.fixed_box_width = 50   # Fixed bounding box width
            self.fixed_box_dimensions = (self.fixed_box_width, self.fixed_box_height)


            self.offset_y = 20 # 18  For Laptop ---> Offset to move the bounding box up

            self.left_eye_offset_x = -5  # Offset to move the left eye bounding box left/right
            self.left_eye_offset_y = 0  # Offset to move the left eye bounding box up/down

            self.right_eye_offset_x = 10  # Offset to move the right eye bounding box left/right
            self.right_eye_offset_y = 0  # Offset to move the right eye bounding box up/down
            
            self.left_eye_bbox = None
            self.right_eye_bbox = None

            self.should_correct_gaze = True
            self.gaze_direction = "Gaze Direction: Away"
            self.classify_count = 0
            self.classify_count_total = 0

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

    def draw_pupil_dots(self, frame, left_eye_landmarks, right_eye_landmarks, threshold=50):
        def get_pupil_center(eye_landmarks, frame, threshold):
            min_x = np.min(eye_landmarks[:, 0]) + 5
            max_x = np.max(eye_landmarks[:, 0]) - 10
            min_y = np.min(eye_landmarks[:, 1]) - 10
            max_y = np.max(eye_landmarks[:, 1]) + 10
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

        looking_at_camera = False
        print("Before looking_at_camera: ", looking_at_camera)
        print("Before =============================================")
        print("Before RIGHT PUPIL CENTER: ", right_pupil_center)
        print("Before Right -> right_eye_landmarks[0]: ", right_eye_landmarks[0])
        print("Before Right -> right_eye_landmarks[8]: ", right_eye_landmarks[8])
        print("Before Bottom -> right_eye_landmarks[11]: ", right_eye_landmarks[11])
        print("Before Top -> right_eye_landmarks[4]: ", right_eye_landmarks[4])

        

        # Right Vertical Thresholds
        y_diff = right_pupil_center[1] - right_eye_landmarks[4][1]

        print('Y-DIFF: ', y_diff)
        # Is Looking UP ?
        if y_diff <= 0:
            x_diff = right_pupil_center[0] - right_eye_landmarks[4][0]
            print('X-DIFF: ', x_diff)

            # # Is Looking UP ?
            # if abs(x_diff) < 17:
            looking_at_camera = True

        print("After looking_at_camera: ", looking_at_camera)
        print("After =============================================")
        print("After RIGHT PUPIL CENTER: ", right_pupil_center)
        print("After Right -> right_eye_landmarks[0]: ", right_eye_landmarks[0])
        print("After Right -> right_eye_landmarks[8]: ", right_eye_landmarks[8])
        print("After Bottom -> right_eye_landmarks[11]: ", right_eye_landmarks[11])
        print("After Top -> right_eye_landmarks[4]: ", right_eye_landmarks[4])

        return looking_at_camera, frame

    def process_frame_with_face_mesh(self, face_landmarks, frame):
        # Define landmark indices for each eye
        eye_landmarks_indices = {
            "left_eye": [7, 163, 144, 153, 154, 155, 173, 157, 158, 159, 160, 161],
            "right_eye": [384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 398]
        }

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
        looking_at_camera, frame = self.draw_pupil_dots(frame, left_eye_landmarks, right_eye_landmarks, threshold=50)

        return looking_at_camera, frame

    def classify_gaze(self, frame, show_face_mesh, show_pupils):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if show_face_mesh:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec
                    )


                if show_pupils:
                    looking_at_camera, frame = self.process_frame_with_face_mesh(face_landmarks, frame)

                if looking_at_camera == False:
                    gaze_direction = "Gaze Direction: Away"
                else: gaze_direction = "Gaze Direction: At Camera"

                return gaze_direction

        return "Gaze: Unknown"

    def process_frame(self, frame, show_face_mesh, classify_gaze, draw_rectangles):
        try:
            h, w, _ = frame.shape
            
            # Determine gaze direction
            if classify_gaze: 
                if self.classify_count >= self.classify_count_total:
                    self.classify_count = 0
                    self.gaze_direction = self.classify_gaze(frame, show_face_mesh, True)
                else: self.classify_count +=1

                self.should_correct_gaze = self.gaze_direction == "Gaze Direction: Away"
                cv2.putText(frame, self.gaze_direction, (w // 2 - 100, h // 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
            temp_frame = frame
            temp_frame = self.draw_selected_landmarks(temp_frame, show_eyes=show_eyes, show_mouth=False, show_face_outline=False, show_text=False)
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
                        elif show_mouth and idx in [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 
                                                    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 
                                                    308, 415, 310, 311, 312, 13, 82, 81, 42, 80, 
                                                    191, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]:
                            # Draw small circle at each landmark point for the entire mouth
                            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
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

    def correct_gaze(self, frame, extract_eyes):
        try:
            print("Starting gaze correction")

            # Set default paths dynamically
            corrected_left_eye_path = os.path.abspath('generated_images/left_eye.png')
            corrected_right_eye_path = os.path.abspath('generated_images/right_eye.png')

            # Save the original frame for away gaze
            dest_image_path = 'destination_image.png'  # Use .png to preserve alpha channel
            dest_image = frame


            if extract_eyes:
                # Add a fully opaque alpha channel (255 for all pixels)
                alpha_channel = np.ones((dest_image.shape[0], dest_image.shape[1]), dtype=dest_image.dtype) * 255
                dest_image = np.dstack([dest_image, alpha_channel])

                cv2.imwrite(dest_image_path, dest_image)
                print(f"Saved frame to {dest_image_path}")

                corrected_image = cv2.imread(dest_image_path, cv2.IMREAD_UNCHANGED)
                print(f"Reading WITH alpha channel, corrected_image.shape: {corrected_image.shape}")
            else:
                cv2.imwrite(dest_image_path, dest_image)
                print(f"Saved frame to {dest_image_path}")

                corrected_image = cv2.imread(dest_image_path, cv2.IMREAD_COLOR)
                print(f"Reading WITHOUT alpha channel, corrected_image.shape: {corrected_image.shape}")

            if self.should_correct_gaze:
                print("Gaze correction is enabled")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print("Converted frame to RGB for face mesh processing")

                results = self.face_mesh.process(rgb_frame)
                print("Processed face landmarks")

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        print("Face landmarks detected, calculating eye boxes")
                        
                        max_retries = 50
                        retry_count = 0
                        while retry_count < max_retries:
                            if os.path.exists(corrected_left_eye_path) and os.path.getsize(corrected_left_eye_path) > 0:
                                self.left_eye_img = cv2.imread(corrected_left_eye_path, cv2.IMREAD_UNCHANGED)
                                if self.left_eye_img is not None:
                                    break
                            else:
                                time.sleep(1)
                                retry_count += 1

                        retry_count = 0
                        while retry_count < max_retries:
                            if os.path.exists(corrected_right_eye_path) and os.path.getsize(corrected_right_eye_path) > 0:
                                self.right_eye_img = cv2.imread(corrected_right_eye_path, cv2.IMREAD_UNCHANGED)
                                if self.right_eye_img is not None:
                                    break
                            else:
                                time.sleep(1)
                                retry_count += 1

                        if self.left_eye_img is not None and self.right_eye_img is not None:
                            left_eye_landmarks = [7, 163, 144, 153, 154, 155, 173, 157, 158, 159, 160, 161]
                            right_eye_landmarks = [384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 398]
                            

                            def _extract_eye_region_with_alpha(frame, eye_points):
                                height, width = frame.shape[:2]
                                mask = np.zeros((height, width), dtype=np.uint8)
                                cv2.fillPoly(mask, [np.array(eye_points)], 255)

                                # Extract eye region with mask
                                eye_region = cv2.bitwise_and(frame, frame, mask=mask)
                                
                                # Crop the eye region based on mask
                                x, y, w, h = cv2.boundingRect(np.array(eye_points))
                                eye_crop = eye_region[y:y+h, x:x+w]
                                mask_crop = mask[y:y+h, x:x+w]

                                # Add an alpha channel
                                eye_with_alpha = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2BGRA)
                                eye_with_alpha[:, :, 3] = mask_crop  # Set alpha channel based on mask
                                
                                return eye_with_alpha, (x, y, w, h)

                            def extract_eye_region(frame, face_mesh_results):
                                img_h, img_w, _ = frame.shape

                                if face_mesh_results.multi_face_landmarks:
                                    for face_landmarks in face_mesh_results.multi_face_landmarks:
                                        left_eye_points = [(int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h)) for idx in left_eye_landmarks]
                                        right_eye_points = [(int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h)) for idx in right_eye_landmarks]

                                        left_eye_frame, left_bbox = _extract_eye_region_with_alpha(frame, left_eye_points)
                                        right_eye_frame, right_bbox = _extract_eye_region_with_alpha(frame, right_eye_points)

                                        return left_eye_frame, right_eye_frame, left_bbox, right_bbox
                                return None, None, None, None
                            
                            results2 = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            

                            if extract_eyes:
                                left_eye2, right_eye2, left_bbox, right_bbox = extract_eye_region(frame, results2)
                            else:
                                self._calculate_eye_boxes(face_landmarks, frame)
                                left_bbox  = self.left_eye_bbox
                                right_bbox = self.right_eye_bbox

                            self.overlay_image(corrected_image, self.left_eye_img, extract_eyes, left_bbox)
                            self.overlay_image(corrected_image, self.right_eye_img, extract_eyes, right_bbox)

                            corrected_image_path = 'corrected_image.png'
                            cv2.imwrite(corrected_image_path, corrected_image)
                            print(f"Corrected frame saved to {corrected_image_path}")

                            # To much processing, offloaded to GPU server - IMPORTANT KEEP THIS COMMENTED
                            # if extract_eyes:
                            #     corrected_image = extract(corrected_image_path=corrected_image_path, dest_image_path=dest_image_path)
                            # else:
                            #     corrected_image = frame
                        else:
                            logging.error("Failed to load one or both corrected eye images.")
                            return frame

            return corrected_image

        except cv2.error as e:
            logging.error(f"OpenCV error in correct_gaze: {e}")
            return frame  # Return original frame on OpenCV error
        except Exception as e:
            logging.error(f"Error in correct_gaze: {e}")
            return frame  # Return original frame on any other error





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
        
    def overlay_boxes(self, frame, eye_bbox, eye_img):
        try:
            (x_min, y_min), (x_max, y_max) = eye_bbox
            eye_region = frame[y_min:y_max, x_min:x_max]
            eye_img_resized = cv2.resize(eye_img, (x_max - x_min, y_max - y_min))

            # Histogram matching to adjust contrast and color
            eye_img_matched = match_histograms(eye_img_resized, eye_region)

            # Ensure both images have the same data type (uint8)
            if eye_img_matched.dtype != eye_region.dtype:
                eye_img_matched = eye_img_matched.astype(eye_region.dtype)

            # Increase the weight of the eye image to make it more prominent
            blended_eye = cv2.addWeighted(eye_region, 0.1, eye_img_matched, 0.9, 0)

            # If the eye image has an alpha channel, blend only the RGB channels
            if eye_img_resized.shape[2] == 4:  # Check if the image has an alpha channel
                eye_img_rgb = eye_img_resized[:, :, :3]  # Use only RGB channels
                mask = eye_img_resized[:, :, 3]  # Alpha channel
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0  # Normalize alpha mask

                # Perform alpha blending based on the alpha channel
                frame[y_min:y_max, x_min:x_max] = (1 - mask) * eye_region + mask * eye_img_rgb
            else:
                # No alpha channel, use the blended image
                frame[y_min:y_max, x_min:x_max] = blended_eye

        except Exception as e:
            logging.error(f"Error in overlay_image: {e}")
            raise

    def overlay_image(self, frame, eye_img, extract_eyes, eye_bbox):
        try:
            
            if extract_eyes:
                def blend_eye_region(base_image, eye_region, position):
                    x, y, w, h = position
                    overlay = base_image[y:y+h, x:x+w]

                    # Resize eye region to match overlay dimensions if needed
                    eye_region_resized = cv2.resize(eye_region, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_AREA)

                    # Blend the eye region using the alpha channel
                    alpha_eye = eye_region_resized[:, :, 3] / 255.0
                    for c in range(3):  # Blend each color channel
                        overlay[:, :, c] = overlay[:, :, c] * (1 - alpha_eye) + eye_region_resized[:, :, c] * alpha_eye

                    base_image[y:y+h, x:x+w] = overlay
                    return base_image
                
                
                
                frame = blend_eye_region(frame, eye_img, eye_bbox)

            else:         
                (x_min, y_min), (x_max, y_max) = eye_bbox
                eye_region = frame[y_min:y_max, x_min:x_max]
                eye_img_resized = cv2.resize(eye_img, (x_max - x_min, y_max - y_min))

                # Directly overlay the resized eye image onto the frame
                frame[y_min:y_max, x_min:x_max] = eye_img


        except Exception as e:
            logging.error(f"Error in overlay_image: {e}")
            raise


# Function to load an image from a file
def load_image(path):
    return cv2.imread(path)

# Function to process face landmarks in an image
def process_face_landmarks(image, eyes_processor):
    # Convert image to RGB and process face landmarks
    img_h, img_w, _ = image.shape
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = eyes_processor.face_mesh.process(frame_rgb)
    return results, img_h, img_w

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

            # Check if landmark is part of the left or right eye
            if idx in [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173]:
                left_eye_points.append((x, y))
            elif idx in [362, 398, 384, 385, 386, 387, 388, 390, 373, 374, 380, 466]:
                right_eye_points.append((x, y))

    # Draw contours around the eyes
    draw_contours(image, left_eye_points,right_eye_points, (0, 255, 0), thickness)
    # draw_contours(image, right_eye_points, (0, 255, 0))

# Function to draw contours between points
def draw_contours(image, left_eye_points, right_eye_points, color, thickness):
    # Draw the green lines around the left eye in the required order
    if len(left_eye_points) > 0:
        cv2.line(image, left_eye_points[0], left_eye_points[12], color, thickness)
        cv2.line(image, left_eye_points[0], left_eye_points[11], color, thickness)
        cv2.line(image, left_eye_points[11], left_eye_points[10], color,thickness)
        cv2.line(image, left_eye_points[10], left_eye_points[9], color, thickness)
        cv2.line(image, left_eye_points[9], left_eye_points[6], color,  thickness)
        # cv2.line(image, left_eye_points[8], left_eye_points[7], color,  thickness)
        # cv2.line(image, left_eye_points[7], left_eye_points[6], color,  thickness)
        cv2.line(image, left_eye_points[6], left_eye_points[5], color,  thickness)
        cv2.line(image, left_eye_points[5], left_eye_points[4], color,  thickness)
        cv2.line(image, left_eye_points[4], left_eye_points[3], color,  thickness)
        cv2.line(image, left_eye_points[3], left_eye_points[2], color,  thickness)
        cv2.line(image, left_eye_points[2], left_eye_points[12], color, thickness)

    # Draw the green lines around the right eye in the required order
    if len(right_eye_points) > 0:
        cv2.line(image, right_eye_points[0], right_eye_points[10], color, thickness)
        cv2.line(image, right_eye_points[10], right_eye_points[4], color, thickness)
        cv2.line(image, right_eye_points[4], right_eye_points[5], color,  thickness)
        cv2.line(image, right_eye_points[5], right_eye_points[6], color,  thickness)
        # cv2.line(image, right_eye_points[6], right_eye_points[7], color,  thickness)
        # cv2.line(image, right_eye_points[7], right_eye_points[8], color,  thickness)
        cv2.line(image, right_eye_points[6], right_eye_points[11], color, thickness)
        cv2.line(image, right_eye_points[11], right_eye_points[9], color, thickness)
        cv2.line(image, right_eye_points[9], right_eye_points[1], color,  thickness)
        cv2.line(image, right_eye_points[1], right_eye_points[2], color,  thickness)
        cv2.line(image, right_eye_points[2], right_eye_points[3], color,  thickness)
        cv2.line(image, right_eye_points[3], right_eye_points[0], color,  thickness)

# Function to detect green contours with positive or negative padding
def detect_green_contours(image, padding=1):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a very strict HSV range for the color green (0, 255, 0)
    lower_green = np.array([35, 150, 150])  # Slightly broader range to capture close colors
    upper_green = np.array([85, 255, 255])  # Slightly broader range to capture close colors
    
    # Create a mask to detect green
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours from the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for padding
    padding_mask = np.zeros_like(green_mask)
    
    # Draw contours on the padding mask
    for contour in contours:
        cv2.drawContours(padding_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Choose dilation or erosion based on the padding value
    kernel_size = max(abs(padding) * 2 + 1, 3)  # Ensure kernel is at least 3x3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if padding > 0:
        # Apply dilation for positive padding
        padded_mask = cv2.dilate(padding_mask, kernel, iterations=1)
    else:
        # Apply erosion for negative padding (shrink contours)
        padded_mask = cv2.erode(padding_mask, kernel, iterations=1)
    
    # Find new contours from the padded mask
    padded_contours, _ = cv2.findContours(padded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return padded_contours, green_mask

import cupy as cp

kernel = np.ones((3, 3), np.uint8)

def clear_warped_images():
    global left_eye_region
    left_eye_region = None
    return True

clear_warped_images()

def process_input_image(corrected_image_path):
    try:
        global kernel
        
        start_time = time.time()
        total_start_time = start_time
        image = load_image(corrected_image_path)
        image_cpy = load_image(corrected_image_path)
        print(f"Time to copy input image: {time.time() - start_time:.4f} seconds")
        
        # Initialize Eyes class and process face landmarks
        eyes_processor = Eyes()
        start_time = time.time()
        results, img_h, img_w = process_face_landmarks(image, eyes_processor)
        draw_eye_contours(image, results, img_h, img_w, 1)
        print(f"Time to process input face landmarks: {time.time() - start_time:.4f} seconds")
            
        # Detect green contours and mask
        start_time = time.time()
        contours, green_mask = detect_green_contours(image, 1)
        print(f"Time to detect green contours: {time.time() - start_time:.4f} seconds")
            
        # Extract the eye regions from the input image
        left_eye_contour = contours[0]
        right_eye_contour = contours[1]
            
        # Create masks for left and right eye regions
        start_time = time.time()
        left_eye_mask = np.zeros_like(image_cpy)  # Create an all-black mask
        cv2.drawContours(left_eye_mask, [left_eye_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        right_eye_mask = np.zeros_like(image_cpy)  # Create an all-black mask
        cv2.drawContours(right_eye_mask, [right_eye_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        print(f"Time to create eye masks: {time.time() - start_time:.4f} seconds")
            
        # Enlarge masks
        start_time = time.time()
        left_eye_mask = cv2.dilate(left_eye_mask, kernel, iterations=2)
        right_eye_mask = cv2.dilate(right_eye_mask, kernel, iterations=2)
        print(f"Time to enlarge masks: {time.time() - start_time:.4f} seconds")
            
        # Extract eye regions using enlarged masks
        start_time = time.time()
        left_eye_region = cv2.bitwise_and(image_cpy, left_eye_mask)
        right_eye_region = cv2.bitwise_and(image_cpy, right_eye_mask)
        print(f"Time to extract eye regions: {time.time() - start_time:.4f} seconds")
            
        return left_eye_region, right_eye_region, left_eye_contour, right_eye_contour, image

    except cv2.error as e:
        print(f"OpenCV error: {e}")
    except Exception as e:
        print(f"Unexpected error in process_input_image: {e}")

def process_destination_image(dest_image_path):
    try:
        
        start_time = time.time()
        dest_image = load_image(dest_image_path)
        copy_dest_image = load_image(dest_image_path)
        print(f"Time to load destination image: {time.time() - start_time:.4f} seconds")
        
        # Initialize Eyes class and process face landmarks
        dest_eyes_processor = Eyes()
        start_time = time.time()
        dest_results, dest_img_h, dest_img_w = process_face_landmarks(dest_image, dest_eyes_processor)
        draw_eye_contours(dest_image, dest_results, dest_img_h, dest_img_w, 1)
        print(f"Time to process destination face landmarks: {time.time() - start_time:.4f} seconds")
            
        # Detect green contours and mask
        start_time = time.time()
        dest_contours, dest_green_mask = detect_green_contours(dest_image, 1)
        print(f"Time to detect green contours: {time.time() - start_time:.4f} seconds")
            
        # Get the contours for the eyes in the destination image
        dest_left_eye_contour = dest_contours[0]
        dest_right_eye_contour = dest_contours[1]
            
        return dest_image, dest_left_eye_contour, dest_right_eye_contour, copy_dest_image

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except cv2.error as e:
        print(f"OpenCV error: {e}")
    except Exception as e:
        print(f"Unexpected error in process_destination_image: {e}")

def combine_images(frame, left_eye_region, right_eye_region, left_eye_contour, right_eye_contour, image, dest_image, dest_left_eye_contour, dest_right_eye_contour, copy_dest_image):
    try:
        # Define the points for the perspective transform
        start_time = time.time()
        left_eye_points = left_eye_contour.reshape(-1, 2).astype(np.float32)
        left_eye_points = np.float32([left_eye_points.min(axis=0), [left_eye_points[:, 0].max(), left_eye_points[:, 1].min()], 
                                      left_eye_points.max(axis=0), [left_eye_points[:, 0].min(), left_eye_points[:, 1].max()]])
        dest_left_eye_points = dest_left_eye_contour.reshape(-1, 2).astype(np.float32)
        dest_left_eye_points = np.float32([dest_left_eye_points.min(axis=0), [dest_left_eye_points[:, 0].max(), dest_left_eye_points[:, 1].min()], 
                                           dest_left_eye_points.max(axis=0), [dest_left_eye_points[:, 0].min(), dest_left_eye_points[:, 1].max()]])
        right_eye_points = right_eye_contour.reshape(-1, 2).astype(np.float32)
        right_eye_points = np.float32([right_eye_points.min(axis=0), [right_eye_points[:, 0].max(), right_eye_points[:, 1].min()], 
                                       right_eye_points.max(axis=0), [right_eye_points[:, 0].min(), right_eye_points[:, 1].max()]])
        dest_right_eye_points = dest_right_eye_contour.reshape(-1, 2).astype(np.float32)
        dest_right_eye_points = np.float32([dest_right_eye_points.min(axis=0), [dest_right_eye_points[:, 0].max(), dest_right_eye_points[:, 1].min()], 
                                            dest_right_eye_points.max(axis=0), [dest_right_eye_points[:, 0].min(), dest_right_eye_points[:, 1].max()]])
        print(f"Time to define points for perspective transform: {time.time() - start_time:.4f} seconds")
        
        # Perform perspective transform
        # Check if any regions are None before proceeding
        if left_eye_region is None or right_eye_region is None:
            left_eye_region, right_eye_region, left_eye_contour, right_eye_contour, image = process_input_image(frame)
        
        start_time = time.time()
        H_left, _ = cv2.findHomography(left_eye_points, dest_left_eye_points)
        warped_left_eye = cv2.warpPerspective(left_eye_region, H_left, (copy_dest_image.shape[1], copy_dest_image.shape[0]))
        H_right, _ = cv2.findHomography(right_eye_points, dest_right_eye_points)
        warped_right_eye = cv2.warpPerspective(right_eye_region, H_right, (copy_dest_image.shape[1], copy_dest_image.shape[0]))
        print(f"Time to perform perspective transform: {time.time() - start_time:.4f} seconds")
        
        # Create masks for warped eye regions
        start_time = time.time()
        left_eye_warped_mask = np.zeros_like(copy_dest_image, dtype=np.uint8)
        cv2.drawContours(left_eye_warped_mask, [dest_left_eye_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        right_eye_warped_mask = np.zeros_like(copy_dest_image, dtype=np.uint8)
        cv2.drawContours(right_eye_warped_mask, [dest_right_eye_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        print(f"Time to create masks for warped regions: {time.time() - start_time:.4f} seconds")
        
        # Combine images using masking
        start_time = time.time()
        combined_image = np.copy(copy_dest_image)
        cv2.copyTo(warped_left_eye, left_eye_warped_mask, combined_image)
        cv2.copyTo(warped_right_eye, right_eye_warped_mask, combined_image)
        print(f"Time to combine images: {time.time() - start_time:.4f} seconds")
        
        return combined_image

    except cv2.error as e:
        print(f"OpenCV error during combine_images: {e}")
    except Exception as e:
        print(f"Unexpected error in combine_images: {e}")

def extract(corrected_image_path, dest_image_path):
    try:
        global left_eye_region, right_eye_region, left_eye_contour, right_eye_contour, image 
        start_time = time.time()
        
        if left_eye_region is None:
            left_eye_region, right_eye_region, left_eye_contour, right_eye_contour, image = process_input_image(corrected_image_path)

        dest_image, dest_left_eye_contour, dest_right_eye_contour, copy_dest_image = process_destination_image(dest_image_path)
        
        # Ensure combined_image is converted back to CPU memory if needed for further steps
        combined_image = combine_images(corrected_image_path, left_eye_region, right_eye_region, left_eye_contour, right_eye_contour, image, dest_image, dest_left_eye_contour, dest_right_eye_contour, copy_dest_image)
        
        print(f"FINAL EXTRACTION TIME: {time.time() - start_time:.4f} seconds")
        
        return combined_image  # Convert back to numpy array if needed by other parts of the program

    except Exception as e:
        print(f"No eyes found for extraction, returning original image: {e}")
        return load_image(dest_image_path)


