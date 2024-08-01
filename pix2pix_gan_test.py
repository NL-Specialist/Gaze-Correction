from modules.eyes_gan_dataset import EYES_GAN_DATASET
from modules.eyes_gan_generator import EYES_GAN_GENERATOR
from modules.eyes_gan_discriminator import EYES_GAN_DISCRIMINATOR
from modules.eyes_gan import EYES_GAN
import os
import tensorflow as tf
import time
import setproctitle
import atexit

def cleanup():
    print("Running cleanup...")
    if 'eyes_gan' in globals() and hasattr(eyes_gan, 'summary_writer'):
        eyes_gan.summary_writer.close()

atexit.register(cleanup)

process_name = "GAN_TRAINING"
setproctitle.setproctitle(process_name)

# Global variable to control training or restoring
TRAIN = False  # Set to True to train, False to restore from checkpoint and validate

if __name__ == "__main__":
    try:
        # Set the dataset path
        DATASET_PATH = os.path.join(os.getcwd(), "Duncan")
        
        # Load the dataset
        print("Loading dataset...")
        eyes_gan_dataset = EYES_GAN_DATASET(dataset_path=DATASET_PATH)
        train_dataset, test_dataset, val_dataset = eyes_gan_dataset.prepare_datasets()
        print("Dataset loaded successfully.")
        
        # Initialize the generator
        print("Initializing generator...")
        generator = EYES_GAN_GENERATOR(input_shape=(24, 50, 3))
        print(generator.generator.summary())

        # Initialize the discriminator
        print("Initializing discriminator...")
        discriminator = EYES_GAN_DISCRIMINATOR(input_shape=(24, 50, 3))
        print(discriminator.discriminator.summary())

        # Initialize EYES_GAN
        print("Initializing EYES_GAN...")
        eyes_gan = EYES_GAN(generator, discriminator)
        print("EYES_GAN initialized successfully.")

        if TRAIN:
            # Run the trained model on a few examples from the test set
            print("Generating initial images from test set...")
            for inp, tar in test_dataset.take(5):
                eyes_gan.generate_images(inp, tar)

            # Training loop
            print("Starting training loop...")
            eyes_gan.fit(train_dataset, test_dataset, epochs=1)
            print("Training complete.")
        else:
            # Restore the model from the latest checkpoint
            print("Restoring model from checkpoint...")
            eyes_gan.restore()
            print("Model restored successfully.")

            # Run the model on the validation data
            print("Running validation on validation set...")
            eyes_gan.validate(val_dataset)
            print("Validation complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cleanup()