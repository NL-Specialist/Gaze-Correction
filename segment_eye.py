import os
import time
import cv2
import logging
import mediapipe as mp
import numpy as np

logging.basicConfig(level=logging.ERROR)

WHITE_SQUARE_DIMENSIONS = (50, 24)
DELAY_BETWEEN_IMAGES = 0.5  # Delay in seconds

class EyeSegmenter:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    def draw_selected_landmarks(self, frame):
        try:
            if frame is None:
                raise ValueError("Input frame is None. Check if the image path is correct.")
            
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

                    return self._extract_eye_region(frame, left_eye_points), self._extract_eye_region(frame, right_eye_points)
            
            return None, None
        except Exception as e:
            logging.error(f"Error in draw_selected_landmarks: {e}")
            raise

    def _extract_eye_region(self, frame, eye_points):
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(eye_points)], (255, 255, 255))
        eye_region = cv2.bitwise_and(frame, mask)
        
        x, y, w, h = cv2.boundingRect(np.array(eye_points))
        eye_crop = eye_region[y:y+h, x:x+w]

        white_background = np.full_like(eye_crop, 255)
        white_background[mask[y:y+h, x:x+w] == 255] = eye_crop[mask[y:y+h, x:x+w] == 255]

        return white_background

def place_on_white_square(eye_image):
    white_square = np.ones((WHITE_SQUARE_DIMENSIONS[1], WHITE_SQUARE_DIMENSIONS[0], 3), dtype=np.uint8) * 255
    eye_h, eye_w, _ = eye_image.shape
    x_offset = (WHITE_SQUARE_DIMENSIONS[0] - eye_w) // 2
    y_offset = (WHITE_SQUARE_DIMENSIONS[1] - eye_h) // 2
    white_square[y_offset:y_offset+eye_h, x_offset:x_offset+eye_w] = eye_image

    return white_square

def process_image(input_image_path, left_eye_output_path, right_eye_output_path):
    try:
        segmenter = EyeSegmenter()
        frame = cv2.imread(input_image_path)

        if frame is None:
            raise ValueError(f"Could not read image from path: {input_image_path}")

        left_eye, right_eye = segmenter.draw_selected_landmarks(frame)

        if left_eye is not None:
            left_eye_square = place_on_white_square(left_eye)
            cv2.imwrite(left_eye_output_path, left_eye_square)
            print(f"Left eye image saved at {left_eye_output_path}")

        if right_eye is not None:
            right_eye_square = place_on_white_square(right_eye)
            cv2.imwrite(right_eye_output_path, right_eye_square)
            print(f"Right eye image saved at {right_eye_output_path}")
    except Exception as e:
        logging.error(f"Error in process_image: {e}")
        raise

def process_all_images(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root).startswith("image_"):
            full_frame_path = os.path.join(root, "full_frame.jpg")
            left_eye_output_path = os.path.join(root, "left_eye.jpg")
            right_eye_output_path = os.path.join(root, "right_eye.jpg")
            process_image(full_frame_path, left_eye_output_path, right_eye_output_path)
            time.sleep(DELAY_BETWEEN_IMAGES)

if __name__ == "__main__":
    base_dir = 'datasets/My_Test1_Dataset'
    subdirs = ['at_camera', 'away']
    sets = ['train'] # , 'train', 'validate'

    for subdir in subdirs:
        for set_name in sets:
            process_all_images(os.path.join(base_dir, subdir, set_name))
