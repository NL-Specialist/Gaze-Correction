import os
import logging
import threading
from flask import Flask, render_template, jsonify, request, Response, send_file
from flask_socketio import SocketIO, emit
from modules.camera_module import CameraModule
from dotenv import load_dotenv
import zipfile
import io

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

camera_module = CameraModule()
camera_state = {'camera_on': False}  # Global variable to store camera state

DEBUG = True

DATASETS_PATH = os.getenv("DATASETS_PATH")
if DEBUG == True:
    print("[DEBUG] DATASETS_PATH: ", DATASETS_PATH)

ACTIVE_DATASET_PATH = ''

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/backend/get_datasets', methods=['GET'])
def get_datasets():
    # List all entries in the directory
    entries         =       os.listdir(DATASETS_PATH)
    
    # Filter out files, keep only directories
    datasets        =       [entry for entry in entries if os.path.isdir(os.path.join(DATASETS_PATH, entry))]
    
    # Create a list of dictionaries for the datasets
    datasets_list   =       [{'value': dataset, 'text': dataset} for dataset in datasets]
    
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

    data                    =      request.json
    dataset_name            =      data.get('datasetName')
    
    if not dataset_name:
        return jsonify({'error': 'Dataset name is required'}), 400

    dataset_path            =      os.path.join(DATASETS_PATH, dataset_name)
    
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

    data                    =       request.json
    dataset_name            =       data.get('datasetName')
    
    if not dataset_name:
        return jsonify({'error': 'Dataset name is required'}), 400

    dataset_path            =       os.path.join(DATASETS_PATH, dataset_name)
    ACTIVE_DATASET_PATH     =       dataset_path

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

    dataset_mode        =       data.get("datasetMode")
    dataset_name        =       data.get("datasetName")

    if not dataset_mode:
        return jsonify({"error": "Dataset mode is required"}), 400

    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    output_dir = os.path.join("datasets", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    image_number = len(os.listdir(output_dir)) + 1
    image_dir = os.path.join(output_dir, f"image_{image_number}")
    os.makedirs(image_dir, exist_ok=True)

    logging.debug(f"Output directory: {output_dir}")
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
    print("Processing toggle camera request...")  # Indicate processing started

    # Toggle the camera state
    camera_status = camera_module.toggle_camera()
    camera_state['camera_on'] = camera_status  # Update the global state

    # Emit camera status to all connected clients
    socketio.emit('camera_status', {'camera_on': camera_status})

    status_message = f"{'On' if camera_status else 'Off'}"
    print(status_message)  # Indicate processing done
    return jsonify({"message": status_message, "status": status_message})


@app.route('/backend/camera-status', methods=['GET'])
def camera_status():
    global camera_state
    return jsonify({"camera_on": camera_state['camera_on']})

@socketio.on('start_video')
def handle_start_video(data):
    try:
        stream = data.get('stream')
        print("stream: ", stream)
        while camera_state['camera_on']:
            frame = camera_module.get_frame(stream)
            if frame:
                if stream == "live-video-left-and-right-eye":
                    if ('left_eye' in frame) and ('right_eye' in frame):
                        socketio.emit(stream, {'type': 'left_and_right_eye', 'frame': frame})
                else:
                    socketio.emit(stream, {'type': stream, 'frame': frame})
    except Exception as e:
        logging.error(f"Error in start_video: {e}")



if __name__ == "__main__":
    socketio.run(app, host='127.0.0.1', port=8000, debug=False)
