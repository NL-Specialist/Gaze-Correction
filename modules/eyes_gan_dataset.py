import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
import pandas as pd

class EYES_GAN_DATASET:
    def __init__(self, dataset_path, buffer_size=400, batch_size=1, debug=True):
        self.DATASET_PATH = dataset_path
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.DEBUG = debug
        self.IMG_WIDTH, self.IMG_HEIGHT = self.get_image_dimensions()

    def get_image_dimensions(self):
        sample_image_path = next(pathlib.Path(self.DATASET_PATH).rglob('*.jpg'))
        sample_image = tf.io.read_file(str(sample_image_path))
        sample_image = tf.image.decode_jpeg(sample_image)
        height, width = sample_image.shape[:2]
        if self.DEBUG:
            print(f"Determined image dimensions: {width}x{height}")
        return width, height

    def resize(self, input_image, real_image, height, width):
        """Resize images to the specified height and width."""
        if self.DEBUG:
            print(f"Resizing images to {height}x{width}")
        input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        if self.DEBUG:
            print(f"Resized images to {height}x{width}")
        
        return input_image, real_image

    def random_crop(self, input_image, real_image):
        """Randomly crop the images to the defined IMG_HEIGHT and IMG_WIDTH."""
        if self.DEBUG:
            print(f"Randomly cropping images to {self.IMG_HEIGHT}x{self.IMG_WIDTH}")
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        
        if self.DEBUG:
            print(f"Cropped images to {self.IMG_HEIGHT}x{self.IMG_WIDTH}")
        
        return cropped_image[0], cropped_image[1]

    def normalize(self, input_image, real_image):
        """Normalize the images to the [-1, 1] range."""
        if self.DEBUG:
            print("Normalizing images to [-1, 1] range")
        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        
        if self.DEBUG:
            print("Normalized images to [-1, 1] range")
        
        return input_image, real_image

    @tf.function()
    def random_jitter(self, input_image, real_image):
        """Apply random jitter to the images: resize, crop, and flip."""
        if self.DEBUG:
            print("Applying random jitter")
        # Resizing to a larger dimension than the target crop size
        resize_height = self.IMG_HEIGHT + 6  # Ensure the resized dimensions are larger
        resize_width = self.IMG_WIDTH + 6
        input_image, real_image = self.resize(input_image, real_image, resize_height, resize_width)
        input_image, real_image = self.random_crop(input_image, real_image)  # Random cropping to target size

        # if tf.random.uniform(()) > 0.5:
        #     if self.DEBUG:
        #         print("Randomly flipping images")
        #     # Random mirroring
        #     input_image = tf.image.flip_left_right(input_image)
        #     real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image


    def load(self, image_file):
        """Load input and real images from a given image file."""
        if self.DEBUG:
            print(f"Loading image from file: {image_file}")
        input_image = tf.io.read_file(image_file)
        input_image = tf.image.decode_jpeg(input_image)

        real_image = tf.io.read_file(image_file)
        real_image = tf.image.decode_jpeg(real_image)
        
        return input_image, real_image

    def load_image_train(self, image_file):
        if self.DEBUG:
            print(f"Loading and processing train image: {image_file}")
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_image_test(self, image_file):
        if self.DEBUG:
            print(f"Loading and processing test image: {image_file}")
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(input_image, real_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def get_image_paths(self, image_type, eye_type):
        """Get all image paths for the specified dataset, image type, and eye type."""
        if self.DEBUG:
            print(f"Getting image paths for {image_type} images ({eye_type} eye)")
        
        images_file_path = os.path.join(self.DATASET_PATH, "away") if image_type == "input" else os.path.join(self.DATASET_PATH, "at_camera")
        
        train_path = os.path.join(images_file_path, "train")
        test_path = os.path.join(images_file_path, "test")
        val_path = os.path.join(images_file_path, "validate")
        
        def load_images(path, eye_type):
            if self.DEBUG:
                print(f"Loading images from directory: {path}")
            image_paths = []
            for root, dirs, files in os.walk(path):
                for dir in dirs:
                    left_eye_path = os.path.join(root, dir, 'left_eye.jpg')
                    right_eye_path = os.path.join(root, dir, 'right_eye.jpg')

                    if eye_type == "left" and os.path.isfile(left_eye_path):
                        image_paths.append(left_eye_path)
                    elif eye_type == "right" and os.path.isfile(right_eye_path):
                        image_paths.append(right_eye_path)
                    elif eye_type == "both":
                        if os.path.isfile(left_eye_path):
                            image_paths.append(left_eye_path)
                        if os.path.isfile(right_eye_path):
                            image_paths.append(right_eye_path)

            return image_paths
        
        train_images = load_images(train_path, eye_type)
        test_images = load_images(test_path, eye_type)
        val_images = load_images(val_path, eye_type)
        
        if self.DEBUG:
            print(f"Number of train images loaded: {len(train_images)}")
            print(f"Number of test images loaded: {len(test_images)}")
            print(f"Number of val images loaded: {len(val_images)}")
    
        return train_images, test_images, val_images


    def prepare_datasets(self, eye_type='left'):
        ####### Split Input Images #######
        if eye_type == 'left' or eye_type == 'both':
            # Input Left Eye Split
            input_train_left_eye_images, input_test_left_eye_images, input_val_left_eye_images = self.get_image_paths("input", eye_type="left")
        else:
            input_train_left_eye_images = []
            input_test_left_eye_images = []
            input_val_left_eye_images = []
        
        if eye_type == 'right' or eye_type == 'both':
            # Input Right Eye Split
            input_train_right_eye_images, input_test_right_eye_images, input_val_right_eye_images = self.get_image_paths("input", eye_type="right")
        else:
            input_train_right_eye_images = []
            input_test_right_eye_images = []
            input_val_right_eye_images = []
        
        ####### Split Real Images #######
        if eye_type == 'left' or eye_type == 'both':
            # Real Left Eye Split
            real_train_left_eye_images, real_test_left_eye_images, real_val_left_eye_images = self.get_image_paths("real", eye_type="left")
        else:
            real_train_left_eye_images = []
            real_test_left_eye_images = []
            real_val_left_eye_images = []
        
        if eye_type == 'right' or eye_type == 'both':
            # Real Right Eye Split
            real_train_right_eye_images, real_test_right_eye_images, real_val_right_eye_images = self.get_image_paths("real", eye_type="right")
        else:
            real_train_right_eye_images = []
            real_test_right_eye_images = []
            real_val_right_eye_images = []
        
        # Combine the image paths for inputs and targets separately
        train_inputs = input_train_left_eye_images + input_train_right_eye_images
        test_inputs = input_test_left_eye_images + input_test_right_eye_images
        val_inputs = input_val_left_eye_images + input_val_right_eye_images
        
        train_targets = real_train_left_eye_images + real_train_right_eye_images
        test_targets = real_test_left_eye_images + real_test_right_eye_images
        val_targets = real_val_left_eye_images + real_val_right_eye_images


        if self.DEBUG:
            print(f"Total training inputs: {len(train_inputs)}")
            print(f"Total testing inputs: {len(test_inputs)}")
            print(f"Total validation inputs: {len(val_inputs)}")
            print(f"Total training targets: {len(train_targets)}")
            print(f"Total testing targets: {len(test_targets)}")
            print(f"Total validation targets: {len(val_targets)}")

        # Convert lists to TensorFlow datasets
        train_input_dataset = tf.data.Dataset.from_tensor_slices(train_inputs)
        test_input_dataset = tf.data.Dataset.from_tensor_slices(test_inputs)
        val_input_dataset = tf.data.Dataset.from_tensor_slices(val_inputs)

        train_target_dataset = tf.data.Dataset.from_tensor_slices(train_targets)
        test_target_dataset = tf.data.Dataset.from_tensor_slices(test_targets)
        val_target_dataset = tf.data.Dataset.from_tensor_slices(val_targets)

        # Map the datasets to the respective loading functions
        train_input_dataset = train_input_dataset.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        train_input_dataset = train_input_dataset.shuffle(self.BUFFER_SIZE)
        train_input_dataset = train_input_dataset.batch(self.BATCH_SIZE)

        test_input_dataset = test_input_dataset.map(self.load_image_test)
        test_input_dataset = test_input_dataset.batch(self.BATCH_SIZE)

        val_input_dataset = val_input_dataset.map(self.load_image_test)
        val_input_dataset = val_input_dataset.batch(self.BATCH_SIZE)

        train_target_dataset = train_target_dataset.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        train_target_dataset = train_target_dataset.shuffle(self.BUFFER_SIZE)
        train_target_dataset = train_target_dataset.batch(self.BATCH_SIZE)

        test_target_dataset = test_target_dataset.map(self.load_image_test)
        test_target_dataset = test_target_dataset.batch(self.BATCH_SIZE)

        val_target_dataset = val_target_dataset.map(self.load_image_test)
        val_target_dataset = val_target_dataset.batch(self.BATCH_SIZE)

        print("Training, testing, and validation datasets are ready.")

        # Combine input and target datasets into tuples
        train_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))
        test_dataset = tf.data.Dataset.zip((test_input_dataset, test_target_dataset))
        val_dataset = tf.data.Dataset.zip((val_input_dataset, val_target_dataset))

        # Summarize dataset details in a final table
        data_summary = {
            "Category": ["Training Inputs", "Testing Inputs", "Validation Inputs", "Training Targets", "Testing Targets", "Validation Targets"],
            "Total Images": [len(train_inputs), len(test_inputs), len(val_inputs), len(train_targets), len(test_targets), len(val_targets)]
        }

        summary_df = pd.DataFrame(data_summary)
        print(summary_df)

        return train_dataset, test_dataset, val_dataset

