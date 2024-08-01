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

# Define a route for capturing images, accepting POST requests at '/backend/capture-images'
@app.route('/backend/capture-images', methods=['POST'])
def capture_images():
    # Get the JSON data sent in the request
    data = request.get_json()
    # Log the received data for debugging
    logging.debug(f"Received data: {data}")

    # Extract specific data fields from the JSON
    dataset_mode        =       data.get("datasetMode")
    dataset_name        =       data.get("datasetName")
    cameraDirection     =       data.get("cameraDirection")
    # Print the camera direction to the console (for debugging)
    print("cameraDirection: ", cameraDirection)

    # Check if dataset mode is provided, if not, return an error response
    if not dataset_mode:
        return jsonify({"error": "Dataset mode is required"}), 400

    # Check if dataset name is provided, if not, return an error response
    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    # Define the output directory path using the dataset name
    dataset_dir = os.path.join("datasets", dataset_name)
    # Create the output directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)

    # Determine save directory as 'at_camera' or 'away'
    out_dir = os.path.join(dataset_dir, "at_camera" if cameraDirection == "lookingAtCamera" else "away"); os.makedirs(out_dir, exist_ok=True)

    # Determine the next image number based on the existing files in the output directory
    image_number = len(os.listdir(out_dir)) + 1
    # Define the directory for the new image using the image number
    image_dir = os.path.join(out_dir, f"image_{image_number}")
    # Create the image directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)

    # Log the output directory path for debugging
    logging.debug(f"Output directory: {dataset_dir}")
    # Log the image directory path for debugging
    logging.debug(f"Image directory: {image_dir}")

    try:
        # Attempt to capture a frame using the camera module and save it to the image directory
        camera_module.capture_frame_from_queue(image_dir)
        # Log a successful frame capture for debugging
        logging.debug("Frame capture completed successfully")
        # Return a success message as JSON
        return jsonify({"message": "Capture initiated"})
    except Exception as e:
        # Log any errors that occur during frame capture
        logging.error(f"Error capturing frame: {str(e)}")
        # Return an error message as JSON
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
        print("Stream: ", stream)
        
        while camera_state['camera_on']:
            frame = camera_module.get_frame(stream)
            
            if frame:
                if stream == "live-video-left-and-right-eye":
                    if 'left_eye' in frame and 'right_eye' in frame:
                        # Emit both left and right eye frames
                        socketio.emit(stream, {
                            'type': 'left_and_right_eye',
                            'left_eye': frame['left_eye'],
                            'right_eye': frame['right_eye']
                        })
                    else:
                        logging.warning("Left or right eye frame missing in 'live-video-left-and-right-eye' stream")
                else:
                    # Emit the frame for other streams
                    socketio.emit(stream, {
                        'type': stream,
                        'frame': frame
                    })
            else:
                logging.warning("No frame received from get_frame() method")
    
    except Exception as e:
        logging.error(f"Error in start_video: {e}")



if __name__ == "__main__":
    socketio.run(app, host='127.0.0.1', port=8000, debug=False)
