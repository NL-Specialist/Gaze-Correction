# FastAPI imports
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi import BackgroundTasks

# Pydantic for data validation
from pydantic import BaseModel

# Custom module imports
from modules.eyes_gan_dataset import EYES_GAN_DATASET
from modules.eyes_gan_generator import EYES_GAN_GENERATOR
from modules.eyes_gan_discriminator import EYES_GAN_DISCRIMINATOR
from modules.eyes_gan import EYES_GAN
from modules.split_datasets import split_folders
from modules.eyes import Eyes 
# Standard library imports
import os
import base64
import zipfile
import shutil
import logging
import asyncio
import threading
import glob

# Machine learning and deep learning libraries
import tensorflow as tf
import torch
from ultralytics import SAM

# Numerical and image processing libraries
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp

# Asynchronous and multi-threading utilities
from concurrent.futures import ThreadPoolExecutor

# Uvicorn for running FastAPI server
import uvicorn


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


app = FastAPI()

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

executor = ThreadPoolExecutor()

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Global variables for GAN components
eyes_gan_left = None
eyes_gan_right = None

generator_right = None
discriminator_right = None

generator_left = None
discriminator_left = None

epoch_count = 0
prev_progress = -1
dataset_exists = False


prev_checkpoint_eye_type = '_left'

# Directories and settings
DATASETS_DIRECTORY = 'datasets'
EYE_TYPE = 'both'  # or 'left' or 'right'
EPOCHS = 1

async def load_GAN(model_name, left_eye_device_type, right_eye_device_type):
    global eyes_gan_left, eyes_gan_right, generator_left, discriminator_left, generator_right, discriminator_right
    
    if model_name == 'disabled':
        # await offload_model()
        return {"status": "Model offloaded successfully.", "checkpoint_list":[]}
    
    # Initialize the generator
    generator_left = EYES_GAN_GENERATOR(input_shape=(24, 50, 3), gpu_index=left_eye_device_type)
    generator_right = EYES_GAN_GENERATOR(input_shape=(24, 50, 3), gpu_index=right_eye_device_type)

    # Initialize the discriminator
    discriminator_left = EYES_GAN_DISCRIMINATOR(input_shape=(24, 50, 3), gpu_index=left_eye_device_type)
    discriminator_right = EYES_GAN_DISCRIMINATOR(input_shape=(24, 50, 3), gpu_index=right_eye_device_type)

    # Initialize EYES_GAN
    eyes_gan_left = EYES_GAN(model_name + '_left', generator_left, discriminator_left, left_eye_device_type)
    eyes_gan_right = EYES_GAN(model_name +'_right', generator_right, discriminator_right, right_eye_device_type)
    
    return True


class LoadModelRequest(BaseModel):
    load_model_name: str
    
@app.post("/load_model/")
async def load_model(request: LoadModelRequest):
    """
    Initialize and load the EYES_GAN model from the latest checkpoint.
    
    This endpoint initializes the generator and discriminator models, 
    combines them into the EYES_GAN, and restores the model from the latest checkpoint.
    
    Returns:
        A JSON response indicating the success of the model loading operation.
        A List of checkpoints corresponding to the loaded model
    """
    global model_name
    
    model_name = request.load_model_name
    
    if model_name == 'disabled':
        # await offload_model()
        return {"status": "Model disabled.", "checkpoint_list":[]}
    
    await load_GAN(model_name, "0", "0")
    
    # Get list of trained checkpoints
    checkpoint_list_left = await get_model_checkpoints(model_name=model_name+'_left')
    checkpoint_list_right = await get_model_checkpoints(model_name=model_name+'_right')

    return {"status": "Model loaded successfully.", "checkpoint_list":checkpoint_list_left}

async def get_model_checkpoints(model_name):
    if model_name == 'disabled':
        return []
    
    checkpoints_folder = os.path.join('models', model_name, 'training_checkpoints')
    
    # Get the list of files in the folder
    files = os.listdir(checkpoints_folder)

    # Extract unique checkpoint names (without extensions) and filter out 'checkpoint'
    checkpoint_list = sorted(set(f.split('.')[0] for f in files if f.split('.')[0] != 'checkpoint'))

    # Sort based on the numeric part of the checkpoint names
    checkpoint_list = sorted(checkpoint_list, key=lambda x: int(x.split('-')[1]), reverse=True)

    # Print the result for debugging purposes
    print("checkpoint_list: ", checkpoint_list)
    
    return checkpoint_list


# @app.post("/offload_model/")
async def offload_model():
    """
    Offload the EYES_GAN model and release GPU resources.
    
    This endpoint sets the global variables to None, deletes the model,
    and clears the GPU memory.
    
    Returns:
        A JSON response indicating the success of the model offloading operation.
    """
    global eyes_gan_left, eyes_gan_right, generator_right, discriminator_right, generator_left, discriminator_left, model_name
    
    if eyes_gan_left is not None or eyes_gan_right is not None:
        del eyes_gan_left
        del eyes_gan_right
    if generator_left is not None or generator_right is not None:
        del generator_left
        del generator_right
    if discriminator_left is not None or discriminator_right is not None:
        del discriminator_left
        del discriminator_right
    
    # Clear the GPU cache
    torch.cuda.empty_cache()
    
    eyes_gan_left = None
    eyes_gan_right = None
    
    generator_right = None
    discriminator_right = None
    
    generator_left = None
    discriminator_left = None
    
    model_name = None
    
    return {"status": "Model offloaded successfully."}

class RestoreCheckpointRequest(BaseModel):
    checkpoint_nr: int
    
@app.post('/restore_checkpoint/')
async def restore_checkpoint(request: RestoreCheckpointRequest):
    global eyes_gan_left, eyes_gan_right 
    # Restore the model from the specified checkpoint
    if eyes_gan_left == None or eyes_gan_right == None:
        print("ERROR: Restoring checkpoint failed, eye gans not loaded")
        return {"status": "ERROR: Restoring checkpoint failed, eye gans not loaded."}
    
    await eyes_gan_left.restore(request.checkpoint_nr)
    await eyes_gan_right.restore(request.checkpoint_nr)
    
    return {"status": "Checkpoint restored successfully."}

@app.post("/generate_image/")
async def generate_image(frame: UploadFile = File(...), extract_eyes: bool = Form(True)):
    global eyes_gan_left, eyes_gan_right

    if eyes_gan_left is None :  # or eyes_gan_right is None
        logging.error("Model not loaded. Call /load_model/ first.")
        return {"error": "Model not loaded. Call /load_model/ first."}

    try:
        # Read the uploaded image
        print("[INFO] Reading uploaded frame...")
        img = read_image_sync(frame)

        if img is None:
            logging.error("[ERROR] Failed to decode the image.")
            return {"error": "Invalid image format"}

        img = tf.expand_dims(img, axis=0)
        full_frame_path = os.path.join("INPUT_EYES", "full_frame.jpg")
        full_frame_encoded = tf.io.encode_jpeg(tf.squeeze(img, axis=0))
        
        print(f'Saving Full Frame to : {full_frame_path}')
        with open(full_frame_path, "wb") as f:
            f.write(full_frame_encoded.numpy())
        
        image = cv2.imread(full_frame_path)
        
        # Process left and right eyes
        eyes_processor = Eyes()
        left_img = eyes_processor.get_left_eye_region(image, False)
        right_img = eyes_processor.get_right_eye_region(image, False)
        
        input_left_eye_path = os.path.join("INPUT_EYES", "left_eye.jpg")
        input_right_eye_path = os.path.join("INPUT_EYES", "right_eye.jpg")
        
        print(f"[INFO] Saving left eye to {input_left_eye_path} and right eye to {input_right_eye_path}...")
        cv2.imwrite(input_left_eye_path, left_img)
        cv2.imwrite(input_right_eye_path, right_img)
        
        # Expand dims before passing to GAN model (to match expected input shape)
        left_img = np.expand_dims(left_img, axis=0)  # Shape becomes (1, 24, 50, 3)
        right_img = np.expand_dims(right_img, axis=0)  # Shape becomes (1, 24, 50, 3)

        # GAN model prediction and returning corrected images (remains the same)
        output_left_filepath = os.path.join("generated_images", "left_eye.jpg")
        output_right_filepath = os.path.join("generated_images", "right_eye.jpg")

             
        await eyes_gan_left.predict(left_img, save_path=output_left_filepath)
        await eyes_gan_right.predict(right_img, save_path=output_right_filepath)   
        
        
        left_eye_img = cv2.imread(output_left_filepath)
        right_eye_img = cv2.imread(output_right_filepath)

        if extract_eyes:
            # Overlay new eyes onto frame
            eyes_processor.overlay_boxes(image, eyes_processor.left_eye_bbox, left_eye_img)
            eyes_processor.overlay_boxes(image, eyes_processor.right_eye_bbox, right_eye_img)
            
            cv2.imwrite('image_test.png', image)
            
            # Process the image and find face meshes
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract landmarks for left and right eyes in correct order
                    left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144,]]
                    right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,]]
            
                    # Extract and save the left eye region with No Background
                    left_eye_image = extract_eye_region(image, left_eye_landmarks)
                    output_left_no_background_filepath = os.path.join("OUTPUT_EYES", "left_eye_transparent.png")
                    cv2.imwrite(output_left_no_background_filepath, left_eye_image)
                    print(f"[SUCCESS] Saved corrected left eye with transparent background to: {output_left_no_background_filepath}")

                    # Extract and save the right eye region with No Background
                    right_eye_image = extract_eye_region(image, right_eye_landmarks)
                    output_right_no_background_filepath = os.path.join("OUTPUT_EYES", "right_eye_transparent.png")
                    cv2.imwrite(output_right_no_background_filepath, right_eye_image)
                    print(f"[SUCCESS] Saved corrected right eye with transparent background to: {output_right_no_background_filepath}")
            
            with open(output_left_no_background_filepath, "rb") as left_eye_no_background_file, \
                    open(output_right_no_background_filepath, "rb") as right_eye_no_background_file:
                left_eye_bytes = left_eye_no_background_file.read()
                right_eye_bytes = right_eye_no_background_file.read()
        else:
            with open(output_left_filepath, "rb") as left_eye_file, \
                    open(output_right_filepath, "rb") as right_eye_file:
                left_eye_bytes = left_eye_file.read()
                right_eye_bytes = right_eye_file.read()

        return {
            "left_eye": base64.b64encode(left_eye_bytes).decode(),
            "right_eye": base64.b64encode(right_eye_bytes).decode()
        }

    except Exception as e:
        logging.error(f"[ERROR] An error occurred during image generation: {str(e)}")
        return {"error": str(e)}

def extract_eye_region(frame, eye_landmarks):
    h, w, _ = frame.shape

    # Convert landmarks to 2D pixel coordinates
    eye_coords = np.array([(int(landmark.x * w), int(landmark.y * h)) for landmark in eye_landmarks], np.int32)

    # Create a mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [eye_coords], 255)

    # Create a transparent background
    transparent_image = np.zeros((h, w, 4), dtype=np.uint8)

    # Copy the original frame where the mask is white
    eye_region = cv2.bitwise_and(frame, frame, mask=mask)

    # Add the eye region to the transparent image
    transparent_image[:, :, :3] = eye_region
    transparent_image[:, :, 3] = mask

    # Make the region outside the eye partially transparent (20% transparency, 80% opacity)
    # First copy the full original frame to the transparent image
    transparent_image[:, :, :3] = frame
    transparent_image[:, :, 3] = 200  # 80% opacity for the entire frame

    # Set the eye region to fully opaque (alpha = 255)
    transparent_image[mask == 255, 3] = 255

    # Crop the transparent image to the eye region
    x, y, w, h = cv2.boundingRect(eye_coords)
    cropped_eye = transparent_image[y:y+h, x:x+w]

    return cropped_eye



def read_image_sync(file: UploadFile):
    """
    Read and decode the uploaded image file synchronously.
    
    This helper function reads the contents of an uploaded image file and 
    decodes it into a tensor.
    
    Args:
        file: An uploaded image file.
    
    Returns:
        A decoded image tensor.
    """
    # Read the file contents synchronously
    contents = file.file.read()  # This reads the file as bytes
    
    # Decode the image using TensorFlow's decode_image function
    image_tensor = tf.image.decode_image(contents, channels=3)
    
    return image_tensor



@app.post("/save_dataset_image")
async def save_dataset_image(file: UploadFile = File(...), path: str = Form(...), dataset_name: str = Form(...)):
    save_path = os.path.join(DATASETS_DIRECTORY, dataset_name, path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    logging.info(f"Saved file to: {save_path}")
    return JSONResponse(content={"message": "File saved successfully"})

import asyncio
import logging
from fastapi import HTTPException

class TrainRequest(BaseModel):
    train_model_name: str
    dataset_path: str
    epochs: int
    learning_rate: float

@app.post("/train")
async def train(request: TrainRequest):
    global eyes_gan_left, eyes_gan_right, model_name, epoch_count, ACTIVE_DATASET_DIRECTORY, EPOCHS, dataset_exists, prev_checkpoint_eye_type

    model_name = request.train_model_name
    ACTIVE_DATASET_DIRECTORY = request.dataset_path

    if not dataset_exists:
        # Define the path to the 'away' directory
        away_directory = os.path.join(ACTIVE_DATASET_DIRECTORY, 'away')

        # Count the number of folders in the 'away' directory
        folder_count = sum([1 for entry in os.scandir(away_directory) if entry.is_dir()])

        # If there are more than 3 folders, perform the split
        if folder_count > 3:
            logging.debug(f"Splitting dataset at: {ACTIVE_DATASET_DIRECTORY}")
            
            # Run the blocking function in a separate thread using asyncio.to_thread
            await split_folders(ACTIVE_DATASET_DIRECTORY)
            
            logging.debug(f"Dataset split successfully!")
        else:
            logging.debug(f"Skipping dataset split as there are {folder_count} folders in the 'away' directory.")

        logging.debug(f"Loading images...")
        # await asyncio.sleep(10)
        

    eyes_gan_dataset = EYES_GAN_DATASET(dataset_path=ACTIVE_DATASET_DIRECTORY, debug=True)

    train_dataset_left, test_dataset_left, val_dataset_left = eyes_gan_dataset.prepare_datasets(eye_type='left')
    train_dataset_right, test_dataset_right, val_dataset_right = eyes_gan_dataset.prepare_datasets(eye_type='right')

    EPOCHS = request.epochs

    # Wrapping training in separate asyncio threads
    async def train_model_left():
        await asyncio.to_thread(
            eyes_gan_left.fit,
            model_name + '_left',
            train_dataset_left,
            test_dataset_left,
            EPOCHS,
            request.learning_rate
        )

    async def train_model_right():
        await asyncio.to_thread(
            eyes_gan_right.fit,
            model_name + '_right',
            train_dataset_right,
            test_dataset_right,
            EPOCHS,
            request.learning_rate
        )

    await load_GAN(model_name, "0", "1")

    if eyes_gan_left is None or eyes_gan_right is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load_model/ first.")

    # Schedule both training tasks in the background
    asyncio.create_task(train_model_left())
    asyncio.create_task(train_model_right())

    epoch_count = 0
    prev_checkpoint_eye_type = '_left'
    return {"status": True}



def generate_initial_test_images(test_dataset):
    """
    Generate and save initial images from the test dataset.
    
    This helper function generates initial images from the test dataset and saves them to disk.
    
    Args:
        test_dataset: The test dataset from which to generate initial images.
    """
    def save_image(image, filepath):
        if len(image.shape) == 4:
            image = tf.squeeze(image, axis=0)
        image = (image + 1.0) * 127.5
        image = tf.cast(image, tf.uint8)
        encoded_image = tf.image.encode_jpeg(image)
        tf.io.write_file(filepath, encoded_image)

    logging.info("Generating initial images from test set...")
    for i, (inp, tar) in enumerate(test_dataset.take(5)):
        save_image(inp[0], f'initial_images/input/image_{i}.jpg')
        save_image(tar[0], f'initial_images/target/image_{i}.jpg')
        eyes_gan_left.generate_images(inp[0], tar[0], f'initial_images/generated/', step=i+1)

class DoesDatasetExistRequest(BaseModel):
    dataset_path: str

@app.post("/does_dataset_exist/")
async def does_dataset_exist(request: DoesDatasetExistRequest):
    global dataset_exists
    dataset_name = os.path.basename(request.dataset_path)  # Extract the dataset name from the path
    left_folder = os.path.join("models", dataset_name + "_left")
    right_folder = os.path.join("models", dataset_name + "_right")

    if os.path.exists(request.dataset_path):
        print(f'[INFO] Dataset Exists! Deleting it: {request.dataset_path}')
        
        # Delete the dataset file or directory
        if os.path.isdir(request.dataset_path):
            shutil.rmtree(request.dataset_path)  # Remove directory and contents
        else:
            os.remove(request.dataset_path)  # Remove file

        # Delete the corresponding left and right folders
        if os.path.exists(left_folder):
            print(f'[INFO] Deleting left folder: {left_folder}')
            shutil.rmtree(left_folder)

        if os.path.exists(right_folder):
            print(f'[INFO] Deleting right folder: {right_folder}')
            shutil.rmtree(right_folder)

        dataset_exists = False
        return {'exists': dataset_exists, 'message': 'Dataset and related folders have been deleted'}
    else:
        print(f'[INFO] Dataset does not exist: {request.dataset_path}')
        dataset_exists = False
        return {'exists': dataset_exists, 'message': 'Dataset does not exist'}

class GetCheckpointImageRequest(BaseModel):
    checkpoint_image_model_name: str

@app.post("/get_checkpoint_image/")
async def get_checkpoint_image(request: GetCheckpointImageRequest):
    global epoch_count, prev_progress, eyes_gan_left, eyes_gan_right, prev_checkpoint_eye_type
    
    if prev_checkpoint_eye_type == '_left':
        prev_checkpoint_eye_type = '_right'
        gan = eyes_gan_right
    else: 
        prev_checkpoint_eye_type = '_left'
        gan = eyes_gan_left
        
    base_path = os.path.join("models", request.checkpoint_image_model_name + prev_checkpoint_eye_type, "image_checkpoints")
    print(f"INFO: Starting get_checkpoint_image function")
    print(f"My epoch count: {epoch_count}, base path: {base_path}")
    
    patterns = [
        f"image_at_epoch_{epoch_count}_step_*.png",
        f"image_input_at_epoch_{epoch_count}_step_*.png",
        f"image_target_at_epoch_{epoch_count}_step_*.png",
        f"image_predicted_at_epoch_{epoch_count}_step_*.png"
    ]

    progress = 0
    print('DEBUG: Checking for checkpoints')
    
    try:
        matched_files = []
        for pattern in patterns:
            print(f"DEBUG: Looking for files with pattern: {pattern}")
            matched_files.extend(glob.glob(os.path.join(base_path, pattern)))

        if matched_files:
            print(f"DEBUG: Found matched files: {matched_files}")
            
            images_content = {}
            
            # Ensure epoch_count is within the valid range for the lists
            if epoch_count < len(gan.progress) and \
               epoch_count < len(gan.gen_loss) and \
               epoch_count < len(gan.disc_loss):
               
                eyes_gan_left_progress = eyes_gan_left.progress[epoch_count]
                eyes_gan_left_gen_loss = eyes_gan_left.gen_loss[epoch_count]
                eyes_gan_left_disc_loss = eyes_gan_left.disc_loss[epoch_count]
                
                eyes_gan_right_progress = eyes_gan_right.progress[epoch_count]
                eyes_gan_right_gen_loss = eyes_gan_right.gen_loss[epoch_count]
                eyes_gan_right_disc_loss = eyes_gan_right.disc_loss[epoch_count]
                
                progress = ( (eyes_gan_left_progress +eyes_gan_right_progress)/2)
                gen_loss = ( (eyes_gan_left_gen_loss+ eyes_gan_right_gen_loss)/2)
                disc_loss = ( (eyes_gan_left_disc_loss+ eyes_gan_right_disc_loss)/2)
                
                print(f"DEBUG: Current progress: {progress}, Previous progress: {prev_progress}")
                
                if prev_progress == progress:
                    print(f"DEBUG: Progress has not changed.")
                    return {"error": "Progress has not advanced"}
                
                prev_progress = progress  # Update the previous progress

                for file_path in matched_files:
                    print(f"DEBUG: Processing file: {file_path}")
                    with open(file_path, "rb") as image_file:
                        image_content = base64.b64encode(image_file.read()).decode('utf-8')
                    images_content[os.path.basename(file_path)] = image_content

                epoch_count += 1
                print(f'DEBUG: Sent Checkpoint Image for epoch {epoch_count}, progress {progress}')
                
                return {
                    "images": images_content,
                    "progress": int(progress),
                    "epoch_count": epoch_count,
                    "generator_loss": str(gen_loss.numpy()),
                    "discriminator_loss": str(disc_loss.numpy())
                }

            else:
                print(f"ERROR: epoch_count {epoch_count} is out of bounds for progress/gen_loss/disc_loss")
                return {"error": "Epoch count out of range"}

    except Exception as e:
        print(f"ERROR: Exception occurred: {e}")
        return {"error": str(e)}

    # Return if no files are found or if progress has completed
    return {
        "images": '',
        "progress": int(100),
        "epoch_count": epoch_count,
        "generator_loss": 0,
        "discriminator_loss": 0
    }



    

async def read_image(file: UploadFile):
    """Reads an uploaded image file and converts it to a numpy array."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)  # Using frombuffer to avoid deprecation warning
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


# GAZE CLASSIFICATION
@app.post("/classify_gaze/")
async def classify_gaze(left_eye: UploadFile = File(...), right_eye: UploadFile = File(...)):
    """
    Generate images using the loaded EYES_GAN model for both eyes.
    
    Args:
        left_eye: Uploaded left eye image file.
        right_eye: Uploaded right eye image file.
    
    Returns:
        JSON containing the gaze direction.
    """
    global discriminator_left, discriminator_right

    # Ensure models are loaded
    if discriminator_left is None or discriminator_right is None:
        await load_GAN('Today')

    try:
        # Read the uploaded left and right eye images
        left_img = await read_image(left_eye)
        right_img = await read_image(right_eye)
        
        # Check if images are read properly
        if left_img is None or right_img is None:
            logging.error("Failed to read one or both images.")
            return {"error": "Invalid images. Please upload valid eye images."}

        # Preprocess the images
        left_img = tf.expand_dims(left_img, axis=0)
        right_img = tf.expand_dims(right_img, axis=0)

        # Perform inference
        real_or_fake_left = is_real_or_fake_inference(discriminator_left, left_img)
        real_or_fake_right = is_real_or_fake_inference(discriminator_right, right_img)
        
        # Check for valid inference results
        if real_or_fake_left is None or real_or_fake_right is None:
            logging.error("Inference returned None for one or both images.")
            return {"error": "Failed to classify gaze during inference."}

        # Average the results to classify gaze
        avg = (real_or_fake_left + real_or_fake_right) / 2

        gaze_direction = 'Gaze Direction: Camera' if avg > 0.5 else 'Gaze Direction: Away'
        
        return {"gaze_direction": gaze_direction}

    except Exception as e:
        logging.error(f"Error in classify_gaze: {e}")
        return {"error": "Failed to classify gaze. Please try again."}



def is_real_or_fake_inference(discriminator, input_image):
    # Call the new discriminator that accepts only one input image
    logits = discriminator([input_image], training=False)
    
    # Apply sigmoid to convert logits to probabilities
    probabilities = tf.sigmoid(logits)
    
    # Average the output probabilities
    avg_probability = tf.reduce_mean(probabilities).numpy()
    
    # Print the result
    if avg_probability > 0.5:
        print("The image is classified as REAL: ", avg_probability)
    else:
        print("The image is classified as FAKE: ", avg_probability)
    
    return avg_probability


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8021)