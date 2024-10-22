# Core libraries
import os
import time
import logging
import base64
import threading
import queue
import concurrent.futures

# Third-party image and video processing libraries
import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat

# Async I/O and HTTP request libraries
import aiofiles
import httpx
import requests

# Custom module for eye processing
from modules.eyes import Eyes, clear_warped_images


# Define global variable
vcam_on = True


class CameraModule:
    def __init__(self):
        self.camera_on = False
        self.settings = {}
        self.camera = None
        self.device_nr = 2 # 0 or 1 or 2
        self.eyes_processor = Eyes()
        self.frame_queue = queue.Queue(maxsize=50)
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
        
        self.vcam = self.init_vcam(640, 480)  # For webcam
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
        """
        Continuously capture frames from the camera and process them for encoding
        and adding to the frame queue. Ensures that valid frames with monotonically
        increasing timestamps are passed into the MediaPipe graph.
        """
        try:
            previous_timestamp = None  # Track previous frame's timestamp
            while not self.stop_event.is_set():
                if not self.camera_on or self.camera is None:
                    time.sleep(1)
                    continue

                success, frame = self.camera.read()

                # Ensure the frame is valid
                if not success or frame is None or frame.size == 0:
                    logging.warning("Invalid or empty frame captured. Skipping this frame.")
                    continue

                current_timestamp = time.time()

                frame_rate = 20  # Set your camera's frame rate here
                timestamp_increment = 1 / frame_rate

                # Ensure monotonically increasing timestamps
                if previous_timestamp is not None and current_timestamp <= previous_timestamp:
                    current_timestamp = previous_timestamp + timestamp_increment
                    logging.warning(f"Adjusted timestamp for monotonicity: {current_timestamp}")

                previous_timestamp = current_timestamp

                # Rotate the frame if required (specific to your camera)
                if self.device_nr == 2:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                # Encode the frame into JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logging.error("Failed to encode frame to JPEG. Skipping this frame.")
                    continue

                frame_bytes = buffer.tobytes()

                # Check if the frame queue is full and discard the oldest frame if necessary
                if self.frame_queue.full():
                    logging.warning("Frame queue is full. Clearing queue to prevent overflow.")
                    self.frame_queue.queue.clear()  # Clear the queue to handle the backlog

                # Add the valid frame to the queue
                self.frame_queue.put(frame_bytes)

                # Update the latest valid frame
                self.latest_frame = frame

        except Exception as e:
            logging.error(f"Error in _generate_frames: {e}")



    def capture_frame_from_queue(self, dataset_name, camera_direction, total_images):
        print(f"capture_frame_from_queue called with: dataset_name={dataset_name}, camera_direction={camera_direction}")
        
        try:
            # Set up dataset directory once
            dataset_dir = os.path.join("datasets", dataset_name)
            out_dir = os.path.join(dataset_dir, "at_camera" if camera_direction == "lookingAtCamera" else "away")
            os.makedirs(out_dir, exist_ok=True)

            # Use globbing or regex to avoid parsing filenames manually
            existing_image_numbers = sorted(
                [int(f.split('_')[1]) for f in os.listdir(out_dir) if f.startswith("image_") and f.split('_')[1].isdigit()],
                reverse=True
            )
            
            image_number = existing_image_numbers[0] + 1 if existing_image_numbers else 1

            # Buffer to store frames
            frame_buffer = []

            # Capture the frames
            for i in range(total_images):
                print(f"Attempting to capture frame {i+1} from queue")
                frame_bytes = self.frame_queue.get()
                frame =  cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

                if frame is None:
                    print("No frame available to capture")
                    raise RuntimeError("Failed to capture frame from queue")
                else:
                    print(f"Frame {i+1} captured successfully, size: {frame.shape}")
                
                # Append the captured frame to the buffer
                frame_buffer.append((frame, image_number))
                image_number += 1
                time.sleep(0.035)

            # After capturing all frames, process the buffer and save the images
            for i, (frame, img_num) in enumerate(frame_buffer):
                image_dir = os.path.join(out_dir, f"image_{img_num}")
                os.makedirs(image_dir, exist_ok=True)

                # Save full frame
                cv2.imwrite(os.path.join(image_dir, 'full_frame.jpg'), frame)

                # Process and save left eye
                self.left_eye_frame = self.eyes_processor.get_left_eye_region(frame, False)
                if self.left_eye_frame is not None and self.left_eye_frame.size != 0:
                    cv2.imwrite(os.path.join(image_dir, 'left_eye.jpg'), self.left_eye_frame)

                # Process and save right eye
                self.right_eye_frame = self.eyes_processor.get_right_eye_region(frame, False)
                if self.right_eye_frame is not None and self.right_eye_frame.size != 0:
                    cv2.imwrite(os.path.join(image_dir, 'right_eye.jpg'), self.right_eye_frame)

            logging.debug("All frames captured and saved successfully")

        except Exception as e:
            logging.error(f"Error in capture_frame_from_queue: {e}")
            raise

    def set_frame_settings(self, stream, show_face_mesh, classify_gaze, draw_rectangles, show_eyes, show_mouth, show_face_outline, show_text, extract_eyes):
        try:
            # Ensure the stream name is one of the expected ones
            if stream not in ['live-video-left', 'live-video-right', 'live-video-left-and-right-eye', 'live-video-1']:
                raise ValueError(f"Stream name '{stream}' is not recognized")

            self.settings["live-video-left"] = {
                'show_face_mesh': show_face_mesh,
                'classify_gaze': classify_gaze,
                'draw_rectangles': draw_rectangles,
                'show_eyes': show_eyes,
                'show_mouth': show_mouth,
                'show_face_outline': show_face_outline,
                'show_text': show_text,
                'extract_eyes': extract_eyes
            }

            self.settings['live-video-right'] = {
                'show_face_mesh': False,
                'classify_gaze': False,
                'draw_rectangles': False,
                'show_eyes': False,
                'show_mouth': False,
                'show_face_outline': False,
                'show_text': False,
                'extract_eyes': extract_eyes
            }

            self.settings['live-video-left-and-right-eye'] = {
                'show_face_mesh': False,
                'classify_gaze': False,
                'draw_rectangles': False,
                'show_eyes': False,
                'show_mouth': False,
                'show_face_outline': False,
                'show_text': False,
                'extract_eyes': False
            }

            self.settings['live-video-1'] = {
                'show_face_mesh': False,
                'classify_gaze': False,
                'draw_rectangles': False,
                'show_eyes': False,
                'show_mouth': False,
                'show_face_outline': False,
                'show_text': False,
                'extract_eyes': False
            }
            print(f'Set frame settings for stream {stream} to: {self.settings[stream]}')
            return True
        except Exception as e:
            print(f'[ERROR] error in set_frame_settings: {e}')
            return False


    def get_frame(self, stream):
        try:
            frame = self._get_decoded_frame_from_queue()
            if frame is None:
                return None

            if stream == "live-video-left-and-right-eye":
                return self._process_left_and_right_eye_frame(frame)

            elif stream == "live-video-1":
                return self._process_live_video_1_frame(frame)

            elif stream == "live-video-left":
                return self._process_live_video_left_frame(frame)

            elif stream == "live-video-right":
                return self._process_live_video_right_frame_async(frame)

            else:
                logging.error(f"ERROR: Stream name not recognized: {stream}")
                return None

        except queue.Empty:
            logging.error("Error in get_frame: Frame queue is empty")
            return None
        except Exception as e:
            logging.error(f"Error in get_frame: {e}")
            raise

    def _get_decoded_frame_from_queue(self):
        try:
            frame_bytes = self.frame_queue.get()
            frame =  self.latest_frame #cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                logging.error("Failed to decode frame bytes into an image")
            return frame
        except Exception as e:
            logging.error(f"Error in decoding frame: {e}")
            return None

    def _process_left_and_right_eye_frame(self, frame):
        #self.latest_left_and_right_eye_frames = self.get_frame(stream="live-video-left-and-right-eye", show_face_mesh=False, classify_gaze=False, draw_rectangles=False, show_eyes=False, show_mouth=False, show_face_outline=False, show_text=False, extract_eyes=False)
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
            frame, show_eyes=False, show_mouth=False, show_face_outline=False, show_text=False)
        processed_frame = self.eyes_processor.process_frame(
            processed_frame, show_face_mesh=False, classify_gaze=False, draw_rectangles=True)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            return buffer.tobytes()
        logging.error("Failed to encode processed frame to JPEG")
        return None

    def _process_live_video_left_frame(self, frame):
        processed_frame = self.eyes_processor.draw_selected_landmarks(frame, show_eyes=self.settings['live-video-left']['show_eyes'], show_mouth=self.settings['live-video-left']['show_mouth'], show_face_outline=self.settings['live-video-left']['show_face_outline'], show_text=self.settings['live-video-left']['show_text'])
        
        processed_frame = self.eyes_processor.process_frame(processed_frame, 
                                                            show_face_mesh=self.settings['live-video-left']['show_face_mesh'], 
                                                            classify_gaze=self.settings['live-video-left']['classify_gaze'], 
                                                            draw_rectangles=self.settings['live-video-left']['draw_rectangles'])
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            return buffer.tobytes()
        logging.error("Failed to encode processed frame to JPEG")
        return None
        
    def _process_live_video_right_frame_async(self, frame):
        # Check if the active model is enabled and gaze correction is required
        print("RUNNING STREAM FOR: live-video-right 1")
        print("self.active_model: ", self.active_model)
        print("self.eyes_processor.should_correct_gaze: ", self.eyes_processor.should_correct_gaze)
        if not self.active_model == 'disabled' and self.eyes_processor.should_correct_gaze:
            print("RUNNING STREAM FOR: live-video-right 2")
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
            frame = self.eyes_processor.correct_gaze(frame, self.settings['live-video-right']['extract_eyes'])

        else:
            self.stop_generation()
            # self.eyes_processor.should_correct_gaze = False

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
            start_time = time.time()
            print("[INFO] Starting image generation for both eyes...")

            # Process left eye
            print("[INFO] Extracting left eye region...")
            left_eye_frame = self.eyes_processor.get_left_eye_region(frame, False)
            left_eye_image_data = None
            if left_eye_frame is not None and left_eye_frame.size != 0:
                _, left_eye_image_data = cv2.imencode('.jpg', left_eye_frame)
            else:
                logging.warning("No left eye region found")
            
            # Process right eye
            print("[INFO] Extracting right eye region...")
            right_eye_frame = self.eyes_processor.get_right_eye_region(frame, False)
            right_eye_image_data = None
            if right_eye_frame is not None and right_eye_frame.size != 0:
                _, right_eye_image_data = cv2.imencode('.jpg', right_eye_frame)
            else:
                logging.warning("No right eye region found")
            
            # Check if both eyes were found
            if left_eye_image_data is not None and right_eye_image_data is not None:
                print("[INFO] Sending both eye images to /generate_image/")

                # Convert image data to bytes and prepare for sending
                files = {
                    "left_eye": ("left_eye.jpg", left_eye_image_data.tobytes(), "image/jpeg"),
                    "right_eye": ("right_eye.jpg", right_eye_image_data.tobytes(), "image/jpeg")
                }

                try:
                    # Make the POST request synchronously using requests
                    response_start_time = time.time()
                    corrected_eye_response = requests.post("http://192.168.0.58:8021/generate_image/", files=files)
                    response_time = time.time() - response_start_time
                    print(f"[INFO] Response received for both eyes in {response_time:.2f} seconds.")

                    # Check if the response is successful
                    if corrected_eye_response.status_code == 200:
                        corrected_images = corrected_eye_response.json()

                        # Ensure directory for saving images exists
                        if not os.path.exists('generated_images'):
                            os.makedirs('generated_images')

                        # Save the corrected images
                        with open('generated_images/left_eye.jpg', 'wb') as f:
                            f.write(base64.b64decode(corrected_images["left_eye"]))
                        with open('generated_images/right_eye.jpg', 'wb') as f:
                            f.write(base64.b64decode(corrected_images["right_eye"]))

                        print("[SUCCESS] Corrected eye images saved locally.")
                        self.thread_running = False
                        # clear_warped_images()
                    else:
                        logging.error(f"[ERROR] Failed to correct eye images. HTTP Status: {corrected_eye_response.status_code}")
                except requests.RequestException as e:
                    logging.error(f"[ERROR] HTTP request failed: {str(e)}")
            else:
                logging.warning("Either left or right eye image was not found.")

            # Print time taken for entire operation
            total_time = time.time() - start_time
            print(f"[INFO] Total time taken for eye image generation: {total_time:.2f} seconds.")

            return None, None
        except Exception as e:
            logging.error(f"[ERROR] Exception occurred during image processing: {str(e)}")
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