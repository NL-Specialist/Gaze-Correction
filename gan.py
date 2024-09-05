from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
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

app = FastAPI()

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

# Global variables for GAN components
eyes_gan = None
generator = None
discriminator = None
epoch_count = 0
prev_progress = -1
dataset_exists = False

# Directories and settings
DATASETS_DIRECTORY = 'datasets'
EYE_TYPE = 'both'  # or 'left' or 'right'
EPOCHS = 1

async def load_GAN(model_name):
    global eyes_gan, generator, discriminator
    
    if model_name == 'disabled':
        offload_model()
        return {"status": "Model offloaded successfully.", "checkpoint_list":[]}
    
    # Initialize the generator
    generator = EYES_GAN_GENERATOR(input_shape=(24, 50, 3))

    # Initialize the discriminator
    discriminator = EYES_GAN_DISCRIMINATOR(input_shape=(24, 50, 3))

    # Initialize EYES_GAN
    eyes_gan = EYES_GAN(model_name, generator, discriminator)
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
    
    await load_GAN(model_name)
    
    # Get list of trained checkpoints
    checkpoint_list = await get_model_checkpoints(model_name=model_name)

    return {"status": "Model loaded successfully.", "checkpoint_list":checkpoint_list}

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
    global eyes_gan, generator, discriminator, model_name
    
    if eyes_gan is not None:
        del eyes_gan
    if generator is not None:
        del generator
    if discriminator is not None:
        del discriminator
    
    # Clear the GPU cache
    torch.cuda.empty_cache()
    
    eyes_gan = None
    generator = None
    discriminator = None
    model_name = None
    
    return {"status": "Model offloaded successfully."}

class RestoreCheckpointRequest(BaseModel):
    checkpoint_nr: int
    
@app.post('/restore_checkpoint/')
def retore_checkpoint(request: RestoreCheckpointRequest):
    global eyes_gan
    # Restore the model from the latest checkpoint
    eyes_gan.restore(request.checkpoint_nr)
    return {"status": "Checkpoint restored successfully."}

@app.post("/generate_image/")
async def generate_image(file: UploadFile = File(...)):
    """
    Generate an image using the loaded EYES_GAN model.
    
    This endpoint receives an image file, decodes it, processes it using the 
    generator model, and saves the generated image to a specified path.
    
    Args:
        file: An uploaded image file to be processed by the model.
    
    Returns:
        The generated image file.
    """
    global eyes_gan
    if eyes_gan is None:
        return {"error": "Model not loaded. Call /load_model/ first."}

    # Read the uploaded file
    img = await read_image(file)
    img = tf.expand_dims(img, axis=0)

    # Generate image
    output_filepath = os.path.join("generated_images", file.filename)
    generated_image = eyes_gan.predict(img, save_path=output_filepath)

    return FileResponse(output_filepath)


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

# Define a Pydantic model for the input
class TrainRequest(BaseModel):
    train_model_name: str
    dataset_path: str
    epochs: int
    learning_rate: float
    
@app.post("/train/")
async def train(request: TrainRequest):
    global eyes_gan, model_name, epoch_count, ACTIVE_DATASET_DIRECTORY, dataset_exists
    
    model_name = request.train_model_name
    
    await load_GAN(model_name)
    
    if eyes_gan is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load_model/ first.")

    ACTIVE_DATASET_DIRECTORY = request.dataset_path
    logging.debug(f"Set active dataset: {ACTIVE_DATASET_DIRECTORY}")
    
    if not dataset_exists:
        print(f'[INFO] Dataset does not exist: ')
        logging.debug(f"Splitting dataset...")
        split_folders(ACTIVE_DATASET_DIRECTORY)
        logging.debug(f"Dataset split successfully!")
    
    logging.info("Loading dataset...")
    eyes_gan_dataset = EYES_GAN_DATASET(dataset_path=ACTIVE_DATASET_DIRECTORY, debug=True)
    train_dataset, test_dataset, val_dataset = eyes_gan_dataset.prepare_datasets(eye_type=EYE_TYPE)
    logging.info("Dataset loaded successfully.")

    logging.info(f"Starting training loop... Total Epochs: {request.epochs}")

    def train_model():
        global model_name
        eyes_gan.fit(model_name, train_dataset, test_dataset, epochs=request.epochs, learning_rate=request.learning_rate)

    # Create and start the thread

    training_thread = threading.Thread(target=train_model)
    training_thread.start()

    logging.info("Training started.")
    epoch_count = 0

    return {"status": "TRAINING SUCCESSFUL"}

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
        eyes_gan.generate_images(inp[0], tar[0], f'initial_images/generated/', step=i+1)

class DoesDatasetExistRequest(BaseModel):
    dataset_path: str

@app.post("/does_dataset_exist/")
async def does_dataset_exist(request: DoesDatasetExistRequest):
    global dataset_exists
    if os.path.exists(request.dataset_path):
        print(f'[INFO] Dataset Exists! : {request.dataset_path}')
        dataset_exists = True
        return {'exists':dataset_exists}
    else:
        print(f'[INFO] Dataset does not exists: {request.dataset_path}')
        dataset_exists = False
        return {'exists':dataset_exists}

class GetCheckpointImageRequest(BaseModel):
    checkpoint_image_model_name: str

@app.post("/get_checkpoint_image/")
async def get_checkpoint_image(request: GetCheckpointImageRequest):
    global epoch_count, prev_progress, eyes_gan
    
    base_path = os.path.join("models", request.checkpoint_image_model_name, "image_checkpoints")
    print("My epoch count: ", epoch_count)
    patterns = [
        f"image_at_epoch_{epoch_count}_step_*.png",
        f"image_input_at_epoch_{epoch_count}_step_*.png",
        f"image_target_at_epoch_{epoch_count}_step_*.png",
        f"image_predicted_at_epoch_{epoch_count}_step_*.png"
    ]

    max_retries = 200
    retries = max_retries
    progress = 0
    print('DEBUG: Trying checkpoints...')
    while retries > 0 and progress < 100:
        try:
            matched_files = []
            for pattern in patterns:
                matched_files.extend(glob.glob(os.path.join(base_path, pattern)))

            if matched_files:
                print('DEBUG: Sent Checkpoint Image.')
                images_content = {}
                progress = eyes_gan.progress[epoch_count]
                if prev_progress == progress:
                    retries = max_retries  # Reset the retries counter
                    continue  # Start the loop again

                prev_progress = progress  # Update the previous progress
                gen_loss = eyes_gan.gen_loss[epoch_count]
                disc_loss = eyes_gan.disc_loss[epoch_count]

                for file_path in matched_files:
                    with open(file_path, "rb") as image_file:
                        image_content = base64.b64encode(image_file.read()).decode('utf-8')
                    images_content[os.path.basename(file_path)] = image_content

                epoch_count += 1
                return {
                    "images": images_content,
                    "progress": int(progress),
                    "epoch_count": epoch_count,
                    "generator_loss": str(gen_loss.numpy()),
                    "discriminator_loss": str(disc_loss.numpy())
                }
        except Exception as e:
            # print('ERROR: Get checkpoints :', e)
            await asyncio.sleep(1)
            continue

        # print('ERROR: Get checkpoints retrying in 5 seconds...')
        await asyncio.sleep(1)
        retries -= 1

    return {"error": "File not found or error occurred after max retries"}
    


async def read_image(file: UploadFile):
    """Reads an uploaded image file and converts it to a numpy array."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)  # Using frombuffer to avoid deprecation warning
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

@app.post("/remove_eyes/")
async def remove_eyes(file: UploadFile = File(...)):
    """Removes eyes from the input frame using two points provided by the user.

    Args:
        file: The input image frame (uploaded file).
        eye_points: A list of two points representing the eyes in the form [[x1, y1], [x2, y2]].

    Returns:
        The modified frame with the eyes removed as a downloadable image.
    """
    # Read the uploaded file
    frame = await read_image(file)
    # frame_tensor = tf.expand_dims(frame, axis=0)

    eye_points = [[307, 235], [218, 235]]  # Example default points

    # Ensure exactly two points are provided
    if len(eye_points) != 2:
        print("Error: Two eye points must be provided.")
        return {"error": "Invalid number of eye points provided"}

    # Load the SAM model
    print("Loading the SAM model.")
    model = SAM("sam2_b.pt")

    # Segment with the provided points (eyes)
    eye_labels = [1, 1]  # Label as foreground for both eyes
    eye_results = model(frame, points=eye_points, labels=eye_labels)

    # Access the mask data from the results
    eye_mask = eye_results[0].masks.data.sum(axis=0).cpu().numpy()  # Combine the two eye masks

    # Convert the mask to uint8 format and create an inverted mask
    eye_mask = (eye_mask > 0).astype('uint8')  # Binary mask where eyes are segmented
    inverse_mask = 1 - eye_mask  # Invert the mask to remove eyes from the image

    # Apply the inverse mask to the original frame to remove the eye segments
    result_image = frame * inverse_mask[:, :, np.newaxis]

    # Save the resulting image
    output_dir = "segmented_images"
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, file.filename)
    cv2.imwrite(output_filepath, result_image)

    # Send the saved image back to the client
    return FileResponse(output_filepath, media_type='image/jpeg', filename=file.filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8021)