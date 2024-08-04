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

app = FastAPI()

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

# Global variables for GAN components
eyes_gan = None
generator = None
discriminator = None
epoch_count = 0
prev_progress = -1
# Directories and settings
DATASETS_DIRECTORY = 'datasets'
EYE_TYPE = 'left'  # or 'left' or 'right'
EPOCHS = 1

@app.post("/load_model/")
async def load_model():
    """
    Initialize and load the EYES_GAN model from the latest checkpoint.
    
    This endpoint initializes the generator and discriminator models, 
    combines them into the EYES_GAN, and restores the model from the latest checkpoint.
    
    Returns:
        A JSON response indicating the success of the model loading operation.
    """
    global eyes_gan, generator, discriminator

    # Initialize the generator
    generator = EYES_GAN_GENERATOR(input_shape=(24, 50, 3))

    # Initialize the discriminator
    discriminator = EYES_GAN_DISCRIMINATOR(input_shape=(24, 50, 3))

    # Initialize EYES_GAN
    eyes_gan = EYES_GAN(generator, discriminator)

    # Restore the model from the latest checkpoint
    eyes_gan.restore()

    return {"status": "Model loaded successfully."}

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
    generated_image = generator.generate_and_save_image(img, save_path=output_filepath)

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

def unzip_dataset(file: UploadFile):
    """
    Unzip the uploaded dataset file.
    
    This function unzips the uploaded dataset file to the specified directory.
    
    Args:
        file: An uploaded dataset file in zip format.
    """
    file_path = file.filename
    folder, filename = os.path.split(file.filename)
    upload_directory = DATASETS_DIRECTORY

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(upload_directory)

    os.remove(file_path)
    logging.debug(f"Saved: Uploaded dataset at: {upload_directory}")


@app.post("/save_dataset_image")
async def save_dataset_image(file: UploadFile = File(...), path: str = Form(...)):
    save_path = os.path.join(DATASETS_DIRECTORY, 'Duncan', path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    logging.info(f"Saved file to: {save_path}")
    return JSONResponse(content={"message": "File saved successfully"})

# Define a Pydantic model for the input
class TrainRequest(BaseModel):
    dataset_path: str
    epochs: int
    learning_rate: float
    
@app.post("/train/")
async def train(request: TrainRequest):
    global eyes_gan, epoch_count, ACTIVE_DATASET_DIRECTORY

    if eyes_gan is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load_model/ first.")

    ACTIVE_DATASET_DIRECTORY = request.dataset_path
    logging.debug(f"Set active dataset: {ACTIVE_DATASET_DIRECTORY}")
    
    logging.debug(f"Splitting dataset...")
    split_folders(ACTIVE_DATASET_DIRECTORY)
    logging.debug(f"Dataset split successfully!")
    
    logging.info("Loading dataset...")
    eyes_gan_dataset = EYES_GAN_DATASET(dataset_path=ACTIVE_DATASET_DIRECTORY, debug=True)
    train_dataset, test_dataset, val_dataset = eyes_gan_dataset.prepare_datasets(eye_type=EYE_TYPE)
    logging.info("Dataset loaded successfully.")

    logging.info(f"Starting training loop... Total Epochs: {request.epochs}")

    def train_model():
        eyes_gan.fit(train_dataset, test_dataset, epochs=request.epochs, learning_rate=request.learning_rate)

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



@app.post("/get_checkpoint_image/")
async def get_checkpoint_image():
    global epoch_count, prev_progress, eyes_gan
    base_path = os.path.join("image_checkpoints")
    print("My epoch count: ", epoch_count)
    patterns = [
        f"image_at_epoch_{epoch_count}_step_*.png",
        f"image_input_at_epoch_{epoch_count}_step_*.png",
        f"image_target_at_epoch_{epoch_count}_step_*.png",
        f"image_predicted_at_epoch_{epoch_count}_step_*.png"
    ]

    max_retries = 200
    retries = max_retries
    print('DEBUG: Trying checkpoints...')
    while retries > 0:
        try:
            matched_files = []
            for pattern in patterns:
                matched_files.extend(glob.glob(os.path.join(base_path, pattern)))

            if matched_files:
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
            print('ERROR: Get checkpoints :', e)
            await asyncio.sleep(2)
            continue

        print('ERROR: Get checkpoints retrying in 5 seconds...')
        await asyncio.sleep(5)
        retries -= 1

    return {"error": "File not found or error occurred after max retries"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8021)

# from modules.eyes_gan_dataset import EYES_GAN_DATASET
# from modules.eyes_gan_generator import EYES_GAN_GENERATOR
# from modules.eyes_gan_discriminator import EYES_GAN_DISCRIMINATOR
# from modules.eyes_gan import EYES_GAN
# import os
# import tensorflow as tf
# import time
# import setproctitle
# import atexit

# def cleanup():
#     print("Running cleanup...")
#     if 'eyes_gan' in globals() and hasattr(eyes_gan, 'summary_writer'):
#         eyes_gan.summary_writer.close()

# def save_image(image, filepath):
#     # Squeeze the batch dimension if present
#     if len(image.shape) == 4:
#         image = tf.squeeze(image, axis=0)
#     # Convert the image tensor to a format suitable for saving
#     image = (image + 1.0) * 127.5  # Convert from [-1, 1] to [0, 255]
#     image = tf.cast(image, tf.uint8)  # Convert to uint8
#     encoded_image = tf.image.encode_jpeg(image)
#     tf.io.write_file(filepath, encoded_image)

# atexit.register(cleanup)

# process_name = "GAN_TRAINING"
# setproctitle.setproctitle(process_name)

# # Global variable to control training or restoring
# TRAIN = False  # Set to True to train, False to restore from checkpoint and validate

# if __name__ == "__main__":
#     try:
#         # Set the dataset path
#         DATASET_PATH = os.path.join(os.getcwd(), "Duncan")
        
#         # Load the dataset
#         print("Loading dataset...")
#         eyes_gan_dataset = EYES_GAN_DATASET(dataset_path=DATASET_PATH, debug=False)
#         train_dataset, test_dataset, val_dataset = eyes_gan_dataset.prepare_datasets(eye_type='left')
#         print("Dataset loaded successfully.")
        
#         # Initialize the generator
#         print("Initializing generator...")
#         generator = EYES_GAN_GENERATOR(input_shape=(24, 50, 3))
#         print(generator.generator.summary())

#         # Initialize the discriminator
#         print("Initializing discriminator...")
#         discriminator = EYES_GAN_DISCRIMINATOR(input_shape=(24, 50, 3))
#         print(discriminator.discriminator.summary())

#         # Initialize EYES_GAN
#         print("Initializing EYES_GAN...")
#         eyes_gan = EYES_GAN(generator, discriminator)
#         print("EYES_GAN initialized successfully.")
        
#         # Run the trained model on a few examples from the test set
#         print("Generating initial images from test set...")
#         i=0
#         for inp, tar in test_dataset.take(5):
#             # Save the input and target images
#             save_image(inp[0], f'initial_images/input/image_{i}.jpg')
#             save_image(tar[0], f'initial_images/target/image_{i}.jpg')
#             i = i+1
#             # Generate and save the predicted image
#             eyes_gan.generate_images(inp[0], tar[0], f'initial_images/generated/', step=i)

#         if TRAIN:
#             # Training loop
#             print("Starting training loop...")
#             eyes_gan.fit(train_dataset, test_dataset, epochs=13)
#             print("Training complete.")
#         else:
#             # Restore the model from the latest checkpoint
#             print("Restoring model from checkpoint...")
#             eyes_gan.restore()
#             print("Model restored successfully.")

#             # Run the model on the validation data
#             print("Running validation on validation set...")
#             eyes_gan.validate(val_dataset=val_dataset,validation_dataset_percentage=100)
#             print("Validation complete.")

#     except Exception as e:
#         print(f"An error occurred: {e}")