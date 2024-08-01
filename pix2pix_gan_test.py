from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from modules.eyes_gan_dataset import EYES_GAN_DATASET
from modules.eyes_gan_generator import EYES_GAN_GENERATOR
from modules.eyes_gan_discriminator import EYES_GAN_DISCRIMINATOR
from modules.eyes_gan import EYES_GAN
import os
import tensorflow as tf
import uvicorn

app = FastAPI()

# Initialize global variables
eyes_gan = None
generator = None
discriminator = None

@app.post("/load_model/")
async def load_model():
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
    global eyes_gan
    if eyes_gan is None:
        return {"error": "Model not loaded. Call /load_model/ first."}

    # Read the uploaded file
    contents = await file.read()
    img = tf.image.decode_image(contents, channels=3)
    img = tf.expand_dims(img, axis=0)

    # Generate image
    output_filepath = os.path.join("generated_images", file.filename)
    generated_image = generator.generate_and_save_image(img, save_path=output_filepath)
    print("my_generated_image: ", generated_image)
    
    return FileResponse(output_filepath)

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