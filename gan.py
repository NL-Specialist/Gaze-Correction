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

import optuna

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

def objective(eye_type, trial, train_ds, test_ds, val_ds, debug=True):
    if debug:
        print("[DEBUG] Starting hyperparameter search with Optuna trial...")

    # Define the search space for hyperparameters
    gen_learning_rate = trial.suggest_loguniform('gen_learning_rate', 1e-5, 1e-3)
    disc_learning_rate = trial.suggest_loguniform('disc_learning_rate', 1e-5, 1e-3)
    beta_1 = trial.suggest_float('beta_1', 0.5, 0.9)
    lambda_value = trial.suggest_int('lambda_value', 50, 200, step=50)

    if debug:
        print(f"[DEBUG] Suggested hyperparameters - Generator Learning Rate: {gen_learning_rate}, "
              f"Discriminator Learning Rate: {disc_learning_rate}, Beta_1: {beta_1}, Lambda: {lambda_value}")

    # Instantiate generator and discriminator with custom hyperparameters
    generator = EYES_GAN_GENERATOR(input_shape=(24, 50, 3), output_channels=3)
    discriminator = EYES_GAN_DISCRIMINATOR()
    gan_model = EYES_GAN('opt_eyes_gan'+'_'+ eye_type, generator, discriminator)

    if debug:
        print("[DEBUG] GAN model instantiated with generator and discriminator.")

    # Set the optimizers with trial hyperparameters
    gan_model.generator_optimizer = tf.keras.optimizers.Adam(gen_learning_rate, beta_1=beta_1)
    gan_model.discriminator_optimizer = tf.keras.optimizers.Adam(disc_learning_rate, beta_1=beta_1)
    gan_model.generator.LAMBDA = lambda_value  # Update generator L1 weight

    if debug:
        print("[DEBUG] Optimizers configured for generator and discriminator with suggested hyperparameters.")

    # Run training for a small number of epochs to evaluate performance
    if debug:
        print("[DEBUG] Starting GAN model training...")
    gan_model.fit(model_name='opt_eyes_gan'+'_'+ eye_type, train_ds=train_ds, test_ds=test_ds, epochs=5)
    if debug:
        print("[DEBUG] GAN model training complete.")

    # Evaluate the model on validation set and return the loss for Optuna to minimize
    save_dir = "validation_images"
    validation_dataset_percentage = 100  # for example, validate on 10% of the dataset
    if debug:
        print("[DEBUG] Starting model evaluation on validation set...")
    val_loss = gan_model.validate(val_dataset=val_ds, save_dir=save_dir, validation_dataset_percentage=validation_dataset_percentage)
    if debug:
        print(f"[DEBUG] Validation loss calculated: {val_loss}")

    return val_loss  # Optuna aims to minimize this metric



async def load_GAN(model_name, eye_type, device_type, train_ds=None, test_ds=None, val_ds=None):
    global eyes_gan_left, eyes_gan_right, generator_left, discriminator_left, generator_right, discriminator_right
    
    if model_name == 'disabled':
        # await offload_model()
        return {"status": "Model offloaded successfully.", "checkpoint_list":[]}
    
    # Run hyperparameter optimization only if datasets are provided
    if train_ds is not None and test_ds is not None and val_ds is not None:
        print("[INFO] Running Optuna hyperparameter optimization...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(eye_type, trial, train_ds, test_ds, val_ds), n_trials=5)  # Specify number of trials
    
        
        # Retrieve and print the best hyperparameters
        best_params = study.best_params
        print("Best hyperparameters:", best_params)
        # Define the path for the text file where best_params will be saved
        best_params_path = f'best_hyperparameters_{eye_type}.txt'

        # Write best_params to the file
        with open(best_params_path, 'w') as file:
            file.write(f"Best hyperparameters: {best_params}")

        print(f"[INFO] Best hyperparameters saved to {best_params_path}")


    # Initialize the generator and discriminator with optimized or default parameters
    generator = EYES_GAN_GENERATOR(input_shape=(24, 50, 3), gpu_index=device_type)
    discriminator = EYES_GAN_DISCRIMINATOR(input_shape=(24, 50, 3), gpu_index=device_type)

    # Initialize EYES_GAN with the best parameters
    eye_gan = EYES_GAN(model_name + '_' + eye_type, generator, discriminator, device_type)
    
    return eye_gan




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
    global model_name, eyes_gan_left, eyes_gan_right
    
    model_name = request.load_model_name
    
    if model_name == 'disabled':
        # await offload_model()
        return {"status": "Model disabled.", "checkpoint_list":[]}
    
    eyes_gan_right = await load_GAN(model_name, 'right', "0")
    eyes_gan_left = await load_GAN(model_name, 'left', "0")
    

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
    
    print('RESTORED CHECKPOINT NUMBER: ', request.checkpoint_nr)
    
    return {"status": "Checkpoint restored successfully."}

def load_image_as_tensor(image_path):
    image = tf.io.read_file(image_path)  # Read the image file
    image = tf.image.decode_image(image, channels=3)  # Decode the image into a tensor
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to [0, 1] float
    image = (image * 2.0) - 1.0  # Normalize to [-1, 1]
    image = tf.expand_dims(image, axis=0)  # Add a batch dimension
    return image

@app.post("/generate_image/")
async def generate_image(frame: UploadFile = File(...), extract_eyes: bool = Form(True)):
    global eyes_gan_left, eyes_gan_right

    if eyes_gan_left is None or eyes_gan_right is None:
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
        
        #### LEFT EYE ####
        left_img = eyes_processor.get_left_eye_region(image, False)
        input_left_eye_path = os.path.join("INPUT_EYES", "left_eye.png")
        cv2.imwrite(input_left_eye_path, left_img)
        print(f"[INFO] Saved LEFT input eye to {input_left_eye_path}")
        
        left_img = load_image_as_tensor(input_left_eye_path)
        output_left_filepath = os.path.join("generated_images", "left_eye.png")
        await eyes_gan_left.predict(left_img, save_path=output_left_filepath)
        print(f"[INFO] Predicted LEFT generated eye to {output_left_filepath}")
        
        #### RIGHT EYE ####
        right_img = eyes_processor.get_right_eye_region(image, False)
        input_right_eye_path = os.path.join("INPUT_EYES", "right_eye.png")
        cv2.imwrite(input_right_eye_path, right_img)
        print(f"[INFO] Saved right input eye to {input_right_eye_path}")
        
        right_img = load_image_as_tensor(input_right_eye_path)
        output_right_filepath = os.path.join("generated_images", "right_eye.png")
        await eyes_gan_right.predict(right_img, save_path=output_right_filepath) 
        print(f"[INFO] Predicted RIGHT generated eye to {output_right_filepath}")
        
        
        if extract_eyes:
            # Load eyes
            left_eye_img = cv2.imread(output_left_filepath)
            right_eye_img = cv2.imread(output_right_filepath)
        
            # Overlay new eyes onto frame
            eyes_processor.overlay_boxes(image, eyes_processor.left_eye_bbox, left_eye_img)
            eyes_processor.overlay_boxes(image, eyes_processor.right_eye_bbox, right_eye_img)

            # Process the image and find face meshes
            results = face_mesh.process(image)
            
            left_eye1, right_eye1, left_bbox1, right_bbox1 = extract_eye_region(image, results)

            output_left_no_background_filepath = os.path.join("OUTPUT_EYES", "left_eye_transparent.png")
            cv2.imwrite(output_left_no_background_filepath, left_eye1)
            print(f"[SUCCESS] Saved corrected left eye with transparent background to: {output_left_no_background_filepath}")

            output_right_no_background_filepath = os.path.join("OUTPUT_EYES", "right_eye_transparent.png")
            cv2.imwrite(output_right_no_background_filepath, right_eye1)
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
            "right_eye": base64.b64encode(right_eye_bytes).decode(),
        }

    except Exception as e:
        logging.error(f"[ERROR] An error occurred during image generation: {str(e)}")
        return {"error": str(e)}


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Updated landmarks for left and right eyes
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
    await asyncio.sleep(0.01)
    return JSONResponse(content={"message": "File saved successfully"})


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
        )

    async def train_model_right():
        await asyncio.to_thread(
            eyes_gan_right.fit,
            model_name + '_right',
            train_dataset_right,
            test_dataset_right,
            EPOCHS,
        )

    eyes_gan_left = await load_GAN(model_name, 'left', "0", train_dataset_left, test_dataset_left, val_dataset_left)
    eyes_gan_right = await load_GAN(model_name, 'right', "1", train_dataset_right, test_dataset_right, val_dataset_right)

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