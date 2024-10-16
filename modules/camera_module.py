import time
import cv2
import logging
import threading
import queue
from modules.eyes import Eyes
import os
import numpy as np
import time
import requests
import pyvirtualcam
from pyvirtualcam import PixelFormat
import concurrent.futures

# Define global variable
vcam_on = True




class CameraModule:
    def __init__(self):
        self.camera_on = False
        self.camera = None
        self.device_nr = 2 # 0 or 1 or 2
        self.eyes_processor = Eyes()
        self.frame_queue = queue.Queue(maxsize=15)
        self.camera_thread = None
        self.stop_event = threading.Event()
        self.left_eye_frame = None
        self.right_eye_frame = None
        # Attributes to store the latest frames
        self.latest_frame = None
        self.latest_left_and_right_eye_frames = None
        self.active_model = 'disabled'
        
        self.future_left_eye = None  # Initialize to None
        self.future_right_eye = None  # Initialize to None
        self.executor = None
        self.initialize_executor()

        self.lock = threading.Lock()  # Optional lock if you need more control
        self.thread_running = False
        
        # self.vcam = self.init_vcam(640, 480)  # For webcam
        # self.vcam = self.init_vcam(960, 540)    # For iphone

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

            self.camera = try_camera(self.device_nr)
            if not self.camera.isOpened():
                print(f"Failed to open camera {self.device_nr}, trying camera 0")
                self.camera = try_camera(0)
                self.device_nr = 0

            if not self.camera.isOpened():
                print("Failed to open camera 0")
                self.camera_on = False
            else:
                if not self.camera_thread or not self.camera_thread.is_alive():
                    self.stop_event.clear()
                    self.camera_thread = threading.Thread(target=self._generate_frames)
                    self.camera_thread.start()
        except Exception as e:
            print(f"Error in _start_camera: {e}")
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
                if not success:
                    logging.error("Failed to read frame from camera")
                    continue

                # Add logging for frame timestamp
                timestamp = time.time()
                logging.debug(f"Captured frame at timestamp: {timestamp}")

                if self.device_nr == 2:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    
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
            frame = self._get_decoded_frame()
            if frame is None:
                return None

            if stream == "live-video-left-and-right-eye":
                return self._process_left_and_right_eye_frame(frame)

            elif stream == "live-video-1":
                return self._process_live_video_1_frame(frame)

            elif stream == "live-video-left":
                return self._process_live_video_left_frame(frame)

            elif stream == "live-video-right":
                return self._process_live_video_right_frame_async(self.latest_frame)

            else:
                logging.error(f"ERROR: Stream name not recognized: {stream}")
                return None

        except queue.Empty:
            logging.error("Error in get_frame: Frame queue is empty")
            return None
        except Exception as e:
            logging.error(f"Error in get_frame: {e}")
            raise

    def _get_decoded_frame(self):
        try:
            frame_bytes = self.frame_queue.get(timeout=1)
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                logging.error("Failed to decode frame bytes into an image")
            return frame
        except Exception as e:
            logging.error(f"Error in decoding frame: {e}")
            return None

    def _process_left_and_right_eye_frame(self, frame):
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

    def _process_live_video_1_frame(self, frame):
        processed_frame = self.eyes_processor.draw_selected_landmarks(
            frame, show_eyes=True, show_mouth=False, show_face_outline=False, show_text=False)
        processed_frame = self.eyes_processor.process_frame(
            processed_frame, show_face_mesh=False, classify_gaze=True, draw_rectangles=True)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            return buffer.tobytes()
        logging.error("Failed to encode processed frame to JPEG")
        return None

    def _process_live_video_left_frame(self, frame):
        processed_frame = self.eyes_processor.process_frame(frame, show_face_mesh=False, classify_gaze=True, draw_rectangles=False)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            return buffer.tobytes()
        logging.error("Failed to encode processed frame to JPEG")
        return None

    def _process_live_video_right_frame_async(self, frame):
        # Check if the active model is enabled and gaze correction is required
        if not self.active_model == 'disabled' and self.eyes_processor.should_correct_gaze:
            print("Active model is enabled and gaze correction is required")

            # Only generate images if no other thread is running
            if not self.thread_running:
                with self.lock:  # Acquire lock to prevent race conditions
                    if not self.thread_running:  # Double check within the lock
                        self._generate_eye_images_async(frame)
                        self.thread_running = True
            else:
                print("Thread already running, skipping image generation")

            # Always use the latest corrected eye images from the folder
            frame = self.eyes_processor.correct_gaze(frame)

        else:
            self.stop_generation()
            self.eyes_processor.should_correct_gaze = False

        # Send the frame to the virtual camera if enabled
        if vcam_on:
            self.vcam.send(frame)
            self.vcam.sleep_until_next_frame()

        # Encode frame and return as bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            return buffer.tobytes()
        else:
            logging.error("Failed to encode frame")

    def _generate_eye_images_async(self, frame):
        # Start a new thread to generate the eye images without blocking
        threading.Thread(target=self._generate_eye_images, args=(frame,), daemon=True).start()

    def _generate_eye_images(self, frame): 
        try:
            time.sleep(10)
            start_time = time.time()
            print("[INFO] Starting image generation for both eyes...")

            # Process left eye
            left_eye_frame = self.eyes_processor.get_left_eye_region(frame, False)
            left_eye_image_path = os.path.join('INPUT_EYES', 'left_eye.jpg')
            if left_eye_frame is not None and left_eye_frame.size != 0:
                cv2.imwrite(left_eye_image_path, left_eye_frame)
            else:
                logging.warning("No left eye region found")

            # Process right eye
            right_eye_frame = self.eyes_processor.get_right_eye_region(frame, False)
            right_eye_image_path = os.path.join('INPUT_EYES', 'right_eye.jpg')
            if right_eye_frame is not None and right_eye_frame.size != 0:
                cv2.imwrite(right_eye_image_path, right_eye_frame)
            else:
                logging.warning("No right eye region found")

            # Check if both eyes were found
            if os.path.exists(left_eye_image_path) and os.path.exists(right_eye_image_path):
                print("[INFO] Sending both eye images to http://192.168.0.58:8021/generate_image/")
                
                # Opening the files outside the with block to avoid early closure
                left_eye_img_file = open(left_eye_image_path, "rb")
                right_eye_img_file = open(right_eye_image_path, "rb")
                
                files = {
                    "left_eye": left_eye_img_file,
                    "right_eye": right_eye_img_file
                }
                response_start_time = time.time()
                corrected_eye_response = requests.post("http://192.168.0.58:8021/generate_image/", files=files)
                response_time = time.time() - response_start_time
                print(f"[INFO] Response received for both eyes in {response_time:.2f} seconds.")

                # Close the files after the request
                left_eye_img_file.close()
                right_eye_img_file.close()

                if corrected_eye_response.status_code == 200:
                    # Parse the JSON response to get the file paths
                    corrected_images = corrected_eye_response.json()

                    output_left_eye_image_path = corrected_images["left_eye"]
                    output_right_eye_image_path = corrected_images["right_eye"]

                    print(f"[SUCCESS] Corrected eye images available at {output_left_eye_image_path} and {output_right_eye_image_path}.")
                    total_time = time.time() - start_time
                    print(f"[INFO] Total time taken for both eyes image processing: {total_time:.2f} seconds.")

                    # Mark the thread as no longer running
                    with self.lock:
                        self.thread_running = False

                    return output_left_eye_image_path, output_right_eye_image_path
                else:
                    logging.error(f"[ERROR] Failed to correct eye images. HTTP Status: {corrected_eye_response.status_code}")
            else:
                logging.warning("Either left or right eye image was not found.")
            return None, None

        except Exception as e:
            logging.error(f"[ERROR] Error generating eye images: {e}")
            # Mark the thread as no longer running in case of error
            with self.lock:
                self.thread_running = False
            return None, None





    def initialize_executor(self):
        # Re-initialize executor if it has been shut down
        if self.executor is None or self.executor._shutdown:
            print("[INFO] Initializing a new executor...")
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            print("[INFO] Executor initialized and ready for new tasks.")

    def stop_generation(self):        
        # Cancel the left eye image generation if it is still running
        if hasattr(self, 'future_left_eye') and self.future_left_eye is not None and not self.future_left_eye.done():
            cancelled = self.future_left_eye.cancel()
            if cancelled:
                print("[INFO] Left eye generation task was successfully canceled.")
            else:
                print("[WARNING] Left eye generation task could not be canceled (it may have already completed).")

        # Cancel the right eye image generation if it is still running
        if hasattr(self, 'future_right_eye') and self.future_right_eye is not None and not self.future_right_eye.done():
            cancelled = self.future_right_eye.cancel()
            if cancelled:
                print("[INFO] Right eye generation task was successfully canceled.")
            else:
                print("[WARNING] Right eye generation task could not be canceled (it may have already completed).")

        # Shutdown the executor if it exists and has not been shut down
        if self.executor is not None:
            if not self.executor._shutdown:
                self.executor.shutdown(wait=False)
                self.executor = None  # Set to None to allow re-initialization
                print("[INFO] Executor has been shut down. No more tasks will be processed.")
            else:
                print("[INFO] Executor was already shut down.")



    def init_vcam(self, width, height):
        # Initialize the virtual camera (this can be done once outside the function)
        return pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=PixelFormat.BGR)