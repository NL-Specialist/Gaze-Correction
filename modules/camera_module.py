import cv2
import logging
import threading
import queue
from modules.eyes import Eyes
import os
import numpy as np
import time
import requests

class CameraModule:
    def __init__(self):
        self.camera_on = False
        self.camera = None
        self.device_nr = 1
        self.eyes_processor = Eyes()
        self.frame_queue = queue.Queue(maxsize=10)
        self.camera_thread = None
        self.stop_event = threading.Event()
        self.left_eye_frame = None
        self.right_eye_frame = None
        # Attributes to store the latest frames
        self.latest_frame = None
        self.latest_left_and_right_eye_frames = None
        self.active_model = 'disabled'

    def set_active_model(self, active_model):
        self.active_model = active_model
        return True

    def toggle_camera(self):
        try:
            self.camera_on = not self.camera_on

            if self.camera_on:
                self._start_camera()
            else:
                self._stop_camera()

            return self.camera_on
        except Exception as e:
            logging.error(f"Error in toggle_camera: {e}")
            raise

    def _start_camera(self):
        try:
            def try_camera(device):
                camera = cv2.VideoCapture(device)
                start_time = time.time()
                while not camera.isOpened() and (time.time() - start_time < 60):
                    time.sleep(1)
                return camera

            self.camera = try_camera(1)
            if not self.camera.isOpened():
                logging.error("Failed to open camera 1, trying camera 0")
                self.device_nr = 0
                self.camera = try_camera(self.device_nr)

            if not self.camera.isOpened():
                logging.error("Failed to open camera 0")
                self.camera_on = False
            else:
                if not self.camera_thread or not self.camera_thread.is_alive():
                    self.stop_event.clear()
                    self.camera_thread = threading.Thread(target=self._generate_frames)
                    self.camera_thread.start()
        except Exception as e:
            logging.error(f"Error in _start_camera: {e}")
            raise

    def _stop_camera(self):
        try:
            self.stop_event.set()
            if self.camera:
                self.camera.release()
                self.camera = None
        except Exception as e:
            logging.error(f"Error in _stop_camera: {e}")
            raise

    def _generate_frames(self):
        try:
            retry_attempts = 3
            retry_delay = 2  # seconds

            while not self.stop_event.is_set():
                if not self.camera_on or self.camera is None:
                    threading.Event().wait(1)
                    continue

                success, frame = self.camera.read()
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                attempts = 0
                while not success and attempts < retry_attempts:
                    logging.warning(f"Failed to read frame from camera, retrying ({attempts + 1}/{retry_attempts})...")
                    threading.Event().wait(retry_delay)
                    success, frame = self.camera.read()
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    attempts += 1

                if not success:
                    logging.error("Failed to read frame from camera after multiple attempts")
                    break

                # Place the raw frame in the queue
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logging.error("Failed to encode frame to JPEG")
                    continue

                frame_bytes = buffer.tobytes()
                self.frame_queue.put(frame_bytes)

                # Update latest frame
                self.latest_frame = frame
        except Exception as e:
            logging.error(f"Error in _generate_frames: {e}")
            raise

    def generate_frames(self, stream):
        try:
            while self.camera_on:
                frame_bytes = self.frame_queue.get()
                if stream == "live-video-left-and-right-eye":
                    self.latest_left_and_right_eye_frames = self.get_frame("live-video-left-and-right-eye")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
            raise

    def capture_frame_from_queue(self, image_dir):
        try:
            logging.debug("Attempting to capture frame from queue")
            if self.latest_frame is None:
                logging.error("No frame available to capture")
                raise RuntimeError("Failed to capture frame from queue")
    
            frame = self.latest_frame
    
            # Save full frame
            cv2.imwrite(os.path.join(image_dir, 'full_frame.jpg'), frame)
    
            # Process and save left eye
            self.left_eye_frame = self.eyes_processor.get_left_eye_region(frame)
            if self.left_eye_frame is not None and self.left_eye_frame.size != 0:
                cv2.imwrite(os.path.join(image_dir, 'left_eye.jpg'), self.left_eye_frame)
            else:
                logging.warning("No left eye capture_region found")
    
            # Process and save right eye
            self.right_eye_frame = self.eyes_processor.get_right_eye_region(frame)
            if self.right_eye_frame is not None and self.right_eye_frame.size != 0:
                cv2.imwrite(os.path.join(image_dir, 'right_eye.jpg'), self.right_eye_frame)
            else:
                logging.warning("No right eye capture_region found")
    
            logging.debug("Frame capture from queue completed")
        except Exception as e:
            logging.error(f"Error in capture_frame_from_queue: {e}")
            raise

    def get_frame(self, stream):
        try:
            frame_bytes = self.frame_queue.get(timeout=1)
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                logging.error("Failed to decode frame bytes into an image")
                return None

            

            if stream == "live-video-left-and-right-eye":
                frames = {}

                self.left_eye_frame = self.eyes_processor.get_left_eye_region(frame, False)
                if self.left_eye_frame is not None and self.left_eye_frame.size != 0:
                    ret, buffer = cv2.imencode('.jpg', self.left_eye_frame)
                    if ret:
                        frames['left_eye'] = buffer.tobytes()
                else:
                    logging.warning("No left eye region found")

                self.right_eye_frame = self.eyes_processor.get_right_eye_region(frame, False)
                if self.right_eye_frame is not None and self.right_eye_frame.size != 0:
                    ret, buffer = cv2.imencode('.jpg', self.right_eye_frame)
                    if ret:
                        frames['right_eye'] = buffer.tobytes()
                else:
                    logging.warning("No right eye region found")
                
                return frames

            elif stream == "live-video-1":
                processed_frame = self.eyes_processor.draw_selected_landmarks(frame, show_eyes=True, show_mouth=False, show_face_outline=False, show_text=False)
                ret, buffer = cv2.imencode('.jpg', processed_frame)

                if ret:
                    return buffer.tobytes()
                logging.error("Failed to encode processed frame to JPEG")
                return None

            elif stream == "live-video-left":
                processed_frame = self.eyes_processor.process_frame(frame, show_face_mesh=False, classify_gaze=True, draw_rectangles=False)

                # segmented_images_path = os.path.join('SEGMENT_EYES', 'input_image.jpg')
                # cv2.imwrite(segmented_images_path, processed_frame)

                # ret, buffer = cv2.imencode('.jpg', processed_frame)
                # if ret:
                #     with open(segmented_images_path, "rb") as segmented_img_file:
                #         files = {"file": segmented_img_file}
                #         remove_eyes_response = requests.post("http://192.168.0.58:8021/remove_eyes/", files=files)

                #     if remove_eyes_response.status_code == 200:
                #         remove_eyes = remove_eyes_response.content
                
                #         ret, buffer = cv2.imencode('.jpg', remove_eyes)
                #         if ret:
                #             return buffer.tobytes()
                        
                
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if ret:
                    return buffer.tobytes()
                else: 
                    logging.error("Failed to remove eyes from frame")
                    return None

            elif stream == "live-video-right":
                if not self.active_model == 'disabled' and self.eyes_processor.should_correct_gaze == True:
                    # Generate left eye
                    self.left_eye_frame = self.eyes_processor.get_left_eye_region(frame, False)
                    if self.left_eye_frame is not None and self.left_eye_frame.size != 0:
                        left_eye_image_path = os.path.join('INPUT_EYES', 'left_eye.jpg')
                        cv2.imwrite(left_eye_image_path, self.left_eye_frame)

                        ret, buffer = cv2.imencode('.jpg', self.left_eye_frame)
                        if ret:
                            with open(left_eye_image_path, "rb") as left_eye_img_file:
                                files = {"file": left_eye_img_file}
                                corrected_left_eye_response = requests.post("http://192.168.0.58:8021/generate_image/", files=files)

                            if corrected_left_eye_response.status_code == 200:
                                corrected_left_eye = corrected_left_eye_response.content
                                left_eye_image_output_path = os.path.join('OUTPUT_EYES', 'corrected_left_eye.jpg')
                                with open(left_eye_image_output_path, "wb") as f:
                                    f.write(corrected_left_eye)
                            else:
                                logging.error("Error: Could not correct Left eye")
                                ret, buffer = cv2.imencode('.jpg', frame)
                                if ret:
                                    return buffer.tobytes()
                    else:
                        logging.warning("No left eye region found")
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            return buffer.tobytes()
                    
                    # Generate right eye
                    # right_eye_image_output_path = None
                    self.right_eye_frame = self.eyes_processor.get_right_eye_region(frame, False)
                    if self.right_eye_frame is not None and self.right_eye_frame.size != 0:
                        right_eye_image_path = os.path.join('INPUT_EYES', 'right_eye.jpg')
                        cv2.imwrite(right_eye_image_path, self.right_eye_frame)

                        ret, buffer = cv2.imencode('.jpg', self.right_eye_frame)
                        if ret:
                            with open(right_eye_image_path, "rb") as right_eye_img_file:
                                files = {"file": right_eye_img_file}
                                corrected_right_eye_response = requests.post("http://192.168.0.58:8021/generate_image/", files=files)

                            if corrected_right_eye_response.status_code == 200:
                                corrected_right_eye = corrected_right_eye_response.content
                                right_eye_image_output_path = os.path.join('OUTPUT_EYES', 'corrected_right_eye.jpg')
                                with open(right_eye_image_output_path, "wb") as f:
                                    f.write(corrected_right_eye)
                            else:
                                logging.error("Error: Could not correct Right eye")
                                ret, buffer = cv2.imencode('.jpg', frame)
                                if ret:
                                    return buffer.tobytes()
                    else:
                        logging.warning("No right eye region found")
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            return buffer.tobytes()

                    frame = self.eyes_processor.correct_gaze(frame, left_eye_image_output_path, right_eye_image_output_path)
                    cv2.imwrite('my_frame.jpg', frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    return buffer.tobytes()
                logging.error("Failed to encode processed frame to JPEG")
                return None

            else:
                logging.error(f"ERROR: Stream name not recognized: {stream}")
                return None

        except queue.Empty:
            logging.error("Error in get_frame: Frame queue is empty")
            return None
        except Exception as e:
            logging.error(f"Error in get_frame: {e}")
            raise