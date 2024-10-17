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
        # This method remains the same
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


    # Modify the load method to accept two file paths
    def load(self, input_image_file, target_image_file):
        """Load input and target images from the given file paths."""
        if self.DEBUG:
            print(f"Loading input image from file: {input_image_file}")
            print(f"Loading target image from file: {target_image_file}")
        input_image = tf.io.read_file(input_image_file)
        input_image = tf.image.decode_jpeg(input_image)

        real_image = tf.io.read_file(target_image_file)
        real_image = tf.image.decode_jpeg(real_image)
        
        return input_image, real_image

    def load_image_train(self, input_image_file, target_image_file):
        if self.DEBUG:
            print(f"Loading and processing train images: {input_image_file}, {target_image_file}")
        input_image, real_image = self.load(input_image_file, target_image_file)
        # input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_image_test(self, input_image_file, target_image_file):
        if self.DEBUG:
            print(f"Loading and processing test images: {input_image_file}, {target_image_file}")
        input_image, real_image = self.load(input_image_file, target_image_file)
        input_image, real_image = self.resize(input_image, real_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def get_matched_image_paths(self, split, eye_type):
        """
        Get matched pairs of input and target image paths for the given split ('train', 'test', 'validate') and eye_type.
        """
        input_base_path = os.path.join(self.DATASET_PATH, "away", split)
        target_base_path = os.path.join(self.DATASET_PATH, "at_camera", split)

        # Get sorted list of directories in the input base path
        input_dirs = sorted([
            d for d in os.listdir(input_base_path)
            if os.path.isdir(os.path.join(input_base_path, d))
        ])

        input_image_paths = []
        target_image_paths = []

        for dir_name in input_dirs:
            input_dir_path = os.path.join(input_base_path, dir_name)
            target_dir_path = os.path.join(target_base_path, dir_name)
            if not os.path.exists(target_dir_path):
                continue

            # Collect images for the specified eye type
            eyes = []
            if eye_type == 'left':
                eyes = ['left_eye.jpg']
            elif eye_type == 'right':
                eyes = ['right_eye.jpg']
            elif eye_type == 'both':
                eyes = ['left_eye.jpg', 'right_eye.jpg']

            for eye in eyes:
                input_image_path = os.path.join(input_dir_path, eye)
                target_image_path = os.path.join(target_dir_path, eye)

                if os.path.isfile(input_image_path) and os.path.isfile(target_image_path):
                    input_image_paths.append(input_image_path)
                    target_image_paths.append(target_image_path)
                else:
                    if self.DEBUG:
                        print(f"Input or target image file does not exist for eye {eye} in directory {dir_name}")

        return input_image_paths, target_image_paths

    def prepare_datasets(self, eye_type='left'):
        # Get matched image paths for each dataset split
        train_inputs, train_targets = self.get_matched_image_paths('train', eye_type)
        test_inputs, test_targets = self.get_matched_image_paths('test', eye_type)
        val_inputs, val_targets = self.get_matched_image_paths('validate', eye_type)

        if self.DEBUG:
            print(f"Total training inputs: {len(train_inputs)}")
            print(f"Total testing inputs: {len(test_inputs)}")
            print(f"Total validation inputs: {len(val_inputs)}")
            print(f"Total training targets: {len(train_targets)}")
            print(f"Total testing targets: {len(test_targets)}")
            print(f"Total validation targets: {len(val_targets)}")

        # Convert lists to TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_targets))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets))

        # Map the datasets to the respective loading functions
        train_dataset = train_dataset.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE)
        train_dataset = train_dataset.batch(self.BATCH_SIZE)

        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(self.BATCH_SIZE)

        val_dataset = val_dataset.map(self.load_image_test)
        val_dataset = val_dataset.batch(self.BATCH_SIZE)

        print("Training, testing, and validation datasets are ready.")

        # Summarize dataset details in a final table
        data_summary = {
            "Category": ["Training Inputs", "Testing Inputs", "Validation Inputs", "Training Targets", "Testing Targets", "Validation Targets"],
            "Total Images": [len(train_inputs), len(test_inputs), len(val_inputs), len(train_targets), len(test_targets), len(val_targets)]
        }

        summary_df = pd.DataFrame(data_summary)
        print(summary_df)

        return train_dataset, test_dataset, val_dataset
