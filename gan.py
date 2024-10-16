from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi import BackgroundTasks
from pydantic import BaseModel
from modules.eyes_gan_dataset import EYES_GAN_DATASET
from modules.eyes_gan_generator import EYES_GAN_GENERATOR
from modules.eyes_gan_discriminator import EYES_GAN_DISCRIMINATOR
from modules.eyes_gan import EYES_GAN
from modules.split_datasets import split_folders
import os
import base64
import tensorflow as tf
import uvicorn
import zipfile
import shutil
import logging
import asyncio
import threading
import glob
import torch
from ultralytics import SAM
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import threading
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


app = FastAPI()

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

executor = ThreadPoolExecutor()

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

device_left = "0"
device_right = "1"


# Directories and settings
DATASETS_DIRECTORY = 'datasets'
EYE_TYPE = 'both'  # or 'left' or 'right'
EPOCHS = 1

async def load_GAN(model_name):
    global eyes_gan_left, eyes_gan_right, generator_left, discriminator_left, generator_right, discriminator_right
    
    if model_name == 'disabled':
        # await offload_model()
        return {"status": "Model offloaded successfully.", "checkpoint_list":[]}
    
    # Initialize the generator
    generator_left = EYES_GAN_GENERATOR(input_shape=(24, 50, 3), gpu_index=device_left)
    generator_right = EYES_GAN_GENERATOR(input_shape=(24, 50, 3), gpu_index=device_right)

    # Initialize the discriminator
    discriminator_left = EYES_GAN_DISCRIMINATOR(input_shape=(24, 50, 3), gpu_index=device_left)
    discriminator_right = EYES_GAN_DISCRIMINATOR(input_shape=(24, 50, 3), gpu_index=device_right)

    # Initialize EYES_GAN
    eyes_gan_left = EYES_GAN(model_name + '_left', generator_left, discriminator_left, device_left)
    eyes_gan_right = EYES_GAN(model_name +'_right', generator_right, discriminator_right, device_right)
    
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
    
    await load_GAN(model_name)
    
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
    await eyes_gan_left.restore(request.checkpoint_nr)
    await eyes_gan_right.restore(request.checkpoint_nr)
    
    return {"status": "Checkpoint restored successfully."}

@app.post("/generate_image/")
async def generate_image(left_eye: UploadFile = File(...), right_eye: UploadFile = File(...)):
    """
    Generate images using the loaded EYES_GAN model for both eyes.
    
    Args:
        left_eye: Uploaded left eye image file.
        right_eye: Uploaded right eye image file.
    
    Returns:
        JSON containing the paths of the corrected images for both eyes.
    """
    global eyes_gan_left, eyes_gan_right

    if eyes_gan_left is None or eyes_gan_right is None:
        return {"error": "Model not loaded. Call /load_model/ first."}

    # Read the uploaded left and right eye images
    left_img = await read_image(left_eye)
    right_img = await read_image(right_eye)

    left_img = tf.expand_dims(left_img, axis=0)
    right_img = tf.expand_dims(right_img, axis=0)

    # Generate the images for both eyes
    output_left_filepath = os.path.join("generated_images", left_eye.filename)
    output_right_filepath = os.path.join("generated_images", right_eye.filename)

    await eyes_gan_left.predict(left_img, save_path=output_left_filepath)
    await eyes_gan_right.predict(right_img, save_path=output_right_filepath)

    # Return the paths of the generated images in a JSON response
    return {
        "left_eye": output_left_filepath,
        "right_eye": output_right_filepath
    }



async def read_image(file: UploadFile):
    """
    Read and decode the uploaded image file.
    
    This helper function reads the contents of an uploaded image file and 
    decodes it into a tensor.
    
    Args:
        file: An uploaded image file.
    
    Returns:
        A decoded image tensor.
    """
    contents = await file.read()
    return tf.image.decode_image(contents, channels=3)


@app.post("/save_dataset_image")
async def save_dataset_image(file: UploadFile = File(...), path: str = Form(...), dataset_name: str = Form(...)):
    save_path = os.path.join(DATASETS_DIRECTORY, dataset_name, path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    logging.info(f"Saved file to: {save_path}")
    return JSONResponse(content={"message": "File saved successfully"})

class TrainRequest(BaseModel):
    train_model_name: str
    dataset_path: str
    epochs: int
    learning_rate: float

@app.post("/train/")
async def train(request: TrainRequest):
    global eyes_gan_left, eyes_gan_right, model_name, epoch_count, ACTIVE_DATASET_DIRECTORY, EPOCHS, dataset_exists

    model_name = request.train_model_name
    await load_GAN(model_name)
    
    if eyes_gan_left is None or eyes_gan_right is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load_model/ first.")

    ACTIVE_DATASET_DIRECTORY = request.dataset_path

    if not dataset_exists:
        logging.debug(f"Splitting dataset at: {ACTIVE_DATASET_DIRECTORY}")
        split_folders(ACTIVE_DATASET_DIRECTORY)
        logging.debug(f"Loading images...")
        await asyncio.sleep(20)
        logging.debug(f"Dataset split successfully!")

    eyes_gan_dataset = EYES_GAN_DATASET(dataset_path=ACTIVE_DATASET_DIRECTORY, debug=True)

    train_dataset_left, test_dataset_left, val_dataset_left = eyes_gan_dataset.prepare_datasets(eye_type='left')
    train_dataset_right, test_dataset_right, val_dataset_right = eyes_gan_dataset.prepare_datasets(eye_type='right')

    EPOCHS = request.epochs

    def train_model_left():
        eyes_gan_left.fit(
            model_name + '_left', 
            train_dataset_left, 
            test_dataset_left, 
            epochs=EPOCHS, 
            learning_rate=request.learning_rate
        )

    def train_model_right():
        eyes_gan_right.fit(
            model_name + '_right', 
            train_dataset_right, 
            test_dataset_right, 
            epochs=EPOCHS, 
            learning_rate=request.learning_rate
        )

    # Run the training tasks concurrently
    threading.Thread(target=train_model_left).start()
    threading.Thread(target=train_model_right).start()
    epoch_count = 0
    return {"status": "TRAINING_STARTED"}


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
    global epoch_count, prev_progress, eyes_gan_left
    
    base_path = os.path.join("models", request.checkpoint_image_model_name +'_left', "image_checkpoints")
    print(f"INFO: Starting get_checkpoint_image function")
    print(f"My epoch count: {epoch_count}, base path: {base_path}")
    
    patterns = [
        f"image_at_epoch_{epoch_count}_step_*.png",
        f"image_input_at_epoch_{epoch_count}_step_*.png",
        f"image_target_at_epoch_{epoch_count}_step_*.png",
        f"image_predicted_at_epoch_{epoch_count}_step_*.png"
    ]

    max_retries = 200
    retries = max_retries
    progress = 0
    print('DEBUG: Trying checkpoints with max retries:', max_retries)
    
    while retries > 0 and progress < 100 and epoch_count+1<EPOCHS:
        try:
            matched_files = []
            for pattern in patterns:
                print(f"DEBUG: Looking for files with pattern: {pattern}")
                matched_files.extend(glob.glob(os.path.join(base_path, pattern)))

            if matched_files:
                print(f"DEBUG: Found matched files: {matched_files}")
                
                images_content = {}
                progress = eyes_gan_left.progress[epoch_count]
                print(f"DEBUG: Current progress: {progress}, Previous progress: {prev_progress}")
                
                if prev_progress == progress:
                    print(f"DEBUG: Progress has not changed, retrying... (retries left: {retries})")
                    retries = max_retries  # Reset the retries counter
                    continue  # Start the loop again

                prev_progress = progress  # Update the previous progress
                gen_loss = eyes_gan_left.gen_loss[epoch_count]
                disc_loss = eyes_gan_left.disc_loss[epoch_count]
                print(f"DEBUG: Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}")

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
                
        except Exception as e:
            print(f"ERROR: Exception occurred: {e}")
            await asyncio.sleep(1)
            continue

        print(f'WARN: Retrying in 1 second... (retries left: {retries})')
        await asyncio.sleep(1)
        retries -= 1

        return {
            "images": '',
            "progress": int(100),
            "epoch_count": EPOCHS,
            "generator_loss": 0,
            "discriminator_loss": 0
            }
    return {"error": "File not found or error occurred after max retries"}

    

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