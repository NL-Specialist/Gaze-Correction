import os
import logging
import threading
from flask import Flask, render_template, jsonify, request, Response, send_file
from flask_socketio import SocketIO, emit
from flask import send_from_directory
from modules.camera_module import CameraModule
from dotenv import load_dotenv
import zipfile
import io
import time
import json
import random
import requests
import base64
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)

socketio = SocketIO(app)

camera_module = CameraModule()
camera_state = {'camera_on': False}  # Global variable to store camera state

max_workers = 300  # Default number of images to send simultaneously
total_epochs = 1
epoch_count = 0
total_files = 0
files_sent = 0

DEBUG = True

DATASETS_PATH = os.getenv("DATASETS_PATH")
if DEBUG:
    print("[DEBUG] DATASETS_PATH: ", DATASETS_PATH)

ACTIVE_DATASET_PATH = ''
progress = 0  # Global variable to store training progress
generator_losses = []  # Global list to store generator loss
discriminator_losses = []  # Global list to store discriminator loss
dataset_loading_progress = 0

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/backend/get_datasets', methods=['GET'])
def get_datasets():
    # List all entries in the directory
    entries = os.listdir(DATASETS_PATH)
    
    # Filter out files, keep only directories
    datasets = [entry for entry in entries if os.path.isdir(os.path.join(DATASETS_PATH, entry))]
    
    # Create a list of dictionaries for the datasets
    datasets_list = [{'value': dataset, 'text': dataset} for dataset in datasets]
    
    # Return the list as JSON
    return jsonify(datasets_list)

@app.route('/backend/download_dataset/<dataset_name>', methods=['GET'])
def download_dataset(dataset_name):
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)
    
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        return jsonify({'error': 'Dataset not found'}), 404

    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dataset_path)
                zip_file.write(file_path, arcname)
    
    zip_buffer.seek(0)

    return send_file(zip_buffer, as_attachment=True, download_name=f'{dataset_name}.zip', mimetype='application/zip')

@app.route('/backend/create_dataset', methods=['POST'])
def create_dataset():
    global ACTIVE_DATASET_PATH

    data = request.json
    dataset_name = data.get('datasetName')
    
    if not dataset_name:
        return jsonify({'error': 'Dataset name is required'}), 400

    dataset_path = os.path.join(DATASETS_PATH, dataset_name)
    
    if os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset already exists'}), 400
    
    try:
        os.makedirs(dataset_path)
        ACTIVE_DATASET_PATH = dataset_path
        return jsonify({'success': 'Dataset created successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/backend/append_dataset', methods=['POST'])
def append_dataset():
    global ACTIVE_DATASET_PATH

    data = request.json
    dataset_name = data.get('datasetName')
    
    if not dataset_name:
        return jsonify({'error': 'Dataset name is required'}), 400

    dataset_path = os.path.join(DATASETS_PATH, dataset_name)
    ACTIVE_DATASET_PATH = dataset_path

    if not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset does not exist!'}), 400
    
    try:
        return jsonify({'success': 'Dataset appended successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/backend/capture-images', methods=['POST'])
def capture_images():
    data = request.get_json()
    logging.debug(f"Received data: {data}")

    dataset_mode = data.get("datasetMode")
    dataset_name = data.get("datasetName")
    cameraDirection = data.get("cameraDirection")
    print("cameraDirection: ", cameraDirection)

    if not dataset_mode:
        return jsonify({"error": "Dataset mode is required"}), 400

    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    dataset_dir = os.path.join("datasets", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    out_dir = os.path.join(dataset_dir, "at_camera" if cameraDirection == "lookingAtCamera" else "away")
    os.makedirs(out_dir, exist_ok=True)

    image_number = len(os.listdir(out_dir)) + 1
    image_dir = os.path.join(out_dir, f"image_{image_number}")
    os.makedirs(image_dir, exist_ok=True)

    logging.debug(f"Output directory: {dataset_dir}")
    logging.debug(f"Image directory: {image_dir}")

    try:
        camera_module.capture_frame_from_queue(image_dir)
        logging.debug("Frame capture completed successfully")
        return jsonify({"message": "Capture initiated"})
    except Exception as e:
        logging.error(f"Error capturing frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/backend/toggle-camera', methods=['POST'])
def toggle_camera():
    global camera_state
    logging.debug("Received request to toggle camera")
    print("Processing toggle camera request...")

    camera_status = camera_module.toggle_camera()
    camera_state['camera_on'] = camera_status

    socketio.emit('camera_status', {'camera_on': camera_status})

    status_message = f"{'On' if camera_status else 'Off'}"
    print(status_message)
    return jsonify({"message": status_message, "status": status_message})

@app.route('/backend/camera-status', methods=['GET'])
def camera_status():
    global camera_state
    return jsonify({"camera_on": camera_state['camera_on']})

@socketio.on('start_video')
def handle_start_video(data):
    try:
        stream = data.get('stream')
        print("Stream: ", stream)
        
        while camera_state['camera_on']:
            frame = camera_module.get_frame(stream)
            
            if frame:
                if stream == "live-video-left-and-right-eye":
                    if 'left_eye' in frame and 'right_eye' in frame:
                        socketio.emit(stream, {
                            'type': 'left_and_right_eye',
                            'left_eye': frame['left_eye'],
                            'right_eye': frame['right_eye']
                        })
                    else:
                        logging.warning("Left or right eye frame missing in 'live-video-left-and-right-eye' stream")
                else:
                    socketio.emit(stream, {
                        'type': stream,
                        'frame': frame
                    })
            else:
                logging.warning("No frame received from get_frame() method")
    
    except Exception as e:
        logging.error(f"Error in start_video: {e}")

def get_checkpoint_images(timeout=300):
    global generator_losses, discriminator_losses, epoch_count
    try:
        print("Sending get checkpoint request, waiting for response...")
        response = requests.post("http://192.168.0.58:8021/get_checkpoint_image/", timeout=timeout)
        response.raise_for_status()

        checkpoint_data = response.json()
        print(f"Response received: {checkpoint_data}")
        checkpoint_images_data = checkpoint_data.get("images")
        epoch_count = checkpoint_data.get("epoch_count")

        if checkpoint_images_data:
            checkpoint_images_path = "image_checkpoints"
            if not os.path.exists(checkpoint_images_path):
                os.mkdir(checkpoint_images_path)
                print(f"Created directory: {checkpoint_images_path}")

            progress = checkpoint_data.get("progress")
            print(f"Processing checkpoint images for progress: {progress}%")

            saved_images = []

            for filename, content in checkpoint_images_data.items():
                print(f"Saving image: {filename}")
                image_path = os.path.join(checkpoint_images_path, filename)
                image_content = base64.b64decode(content)

                with open(image_path, "wb") as f:
                    f.write(image_content)
                    print(f"Saved image to: {image_path}")
                    saved_images.append(filename)

            generator_loss = checkpoint_data.get("generator_loss", 0)
            discriminator_loss = checkpoint_data.get("discriminator_loss", 0)

            print(f"Updated Checkpoint Progress: {progress}%")
            print(f"Updated Checkpoint Generator loss: {generator_loss}")
            print(f"Updated Checkpoint Discriminator loss: {discriminator_loss}")

            generator_losses.append(generator_loss)
            discriminator_losses.append(discriminator_loss)

            return saved_images, progress, epoch_count

        else:
            print("Checkpoint images data is missing")
            return None, None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None
    except ValueError:
        print("Invalid JSON response")
        return None, None
    except KeyError as e:
        print(f"Missing expected key in response: {e}")
        return None, None


@app.route('/datasets/<path:filename>')
def serve_dataset_files(filename):
    return send_from_directory('datasets', filename)

@app.route('/image_checkpoints/<filename>')
def send_image(filename):
    return send_from_directory('image_checkpoints', filename)

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    global dataset_loading_progress, files_sent, total_files    
    dataset_loading_progress = 0
    data = request.json
    print("[INFO] Content received:", data)

    dataset_name = data.get('dataset', 'Duncan')
    print(f"[INFO] Dataset name set to: {dataset_name}")

    folder_path = data.get('datasets', f'datasets/{dataset_name}')
    print(f"[INFO] Folder path set to: {folder_path}")

    total_files = sum([len(files) for r, d, files in os.walk(folder_path)])
    print(f"[INFO] Total files to send: {total_files}")
    files_sent = 0

    def send_file(file_path, rel_path):
        global dataset_loading_progress, files_sent, total_files
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            data = {'path': rel_path}
            response = requests.post("http://192.168.0.58:8021/save_dataset_image", files=files, data=data)
            if response.status_code == 200:
                files_sent += 1
                dataset_loading_progress = int((files_sent / total_files) * 100)
                # print(f"[INFO] Progress: {dataset_loading_progress}% ; Files Sent:{files_sent} ; Total Files: {total_files}")
            else:
                print(f"[ERROR] Sending {file_path} failed with status code {response.status_code}")
                print("[ERROR] Response text: ", response.text)

    def send_files_concurrently(folder_path):
        global dataset_loading_progress
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other file extensions if needed
                        file_path = os.path.join(root, filename)
                        file_path_normalized = file_path.replace("\\", "/")
                        rel_path = os.path.relpath(file_path_normalized, folder_path).replace("\\", "/")
                        executor.submit(send_file, file_path, rel_path)
                    else:
                        print(f"[DEBUG] Skipped file: {filename} (not an image)")
        time.sleep(5)
        print("[INFO] All files sent successfully.")
        dataset_loading_progress = 100  # Set progress to 100% after all files are sent

    # Start the file sending process in a background thread
    threading.Thread(target=send_files_concurrently, args=(folder_path,)).start()

    return jsonify({"message": "Dataset loading started"})

@app.route('/dataset_loading_progress_stream')
def dataset_loading_progress_stream():
    def event_generator():
        global files_sent, total_files, dataset_loading_progress
        while dataset_loading_progress < 100:
            yield f"data: {json.dumps({'file_count': files_sent, 'total_files': total_files, 'progress': dataset_loading_progress})}\n\n".encode('utf-8')
            time.sleep(1)
        yield f"data: {json.dumps({'progress': 100})}\n\n".encode('utf-8')
    
    return Response(event_generator(), content_type='text/event-stream')

@app.route('/start_training', methods=['POST'])
def start_training():
    global total_epochs, epoch_count
    data = request.json
    folder_path = data.get('dataset_path')
    total_epochs = data.get('epochs')
    learning_rate = data.get('learning_rate')
    epoch_count = 0

    # Notify the server to start training after all files are sent
    response = requests.post("http://192.168.0.58:8021/train", json={'dataset_path': folder_path, 'epochs': total_epochs, 'learning_rate': learning_rate})
    if response.status_code == 200:
        print(f"[INFO] Training started successfully: {response.status_code}")
    else:
        print(f"[ERROR] Starting training failed with status code {response.status_code}")
        print("[ERROR] Response text: ", response.text)

    return jsonify({"message": "Training started successfully"})


@app.route('/training_progress', methods=['GET'])
def training_progress():
    def event_generator():
        global progress, epoch_count, total_epochs, generator_losses, discriminator_losses

        while progress < 100:
            time.sleep(1)
            saved_images, progress, epoch_count = get_checkpoint_images()
            if saved_images:
                print("Sending images: ", saved_images)
                before_image_path = next((img for img in saved_images if "input" in img), None)
                before_image_path = os.path.join('image_checkpoints', before_image_path)
                after_image_path = next((img for img in saved_images if "predicted" in img), None)
                after_image_path = os.path.join('image_checkpoints', after_image_path)
                if before_image_path and after_image_path:
                    yield f"data: {json.dumps({'progress': progress, 'epoch_count': epoch_count, 'total_epochs': total_epochs, 'before_image': before_image_path, 'after_image': after_image_path, 'generator_loss': generator_losses[-1], 'discriminator_loss': discriminator_losses[-1]})}\n\n".encode('utf-8')
                else:
                    yield f"data: {json.dumps({'error': 'Failed to get required checkpoint images'})}\n\n".encode('utf-8')
            else:
                yield f"data: {json.dumps({'error': 'Failed to get checkpoint images'})}\n\n".encode('utf-8')
        yield f"data: {json.dumps({'progress': progress})}\n\n".encode('utf-8')

    return Response(event_generator(), content_type='text/event-stream')


if __name__ == "__main__":
    socketio.run(app, host='127.0.0.1', port=8000, debug=False)
