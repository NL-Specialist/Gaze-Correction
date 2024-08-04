import tensorflow as tf
import matplotlib.pyplot as plt
from modules.eyes_gan_dataset import EYES_GAN_DATASET
import os

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Concatenate

class ResizeLayer(Layer):
    def __init__(self, target_height, target_width):
        super(ResizeLayer, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, [self.target_height, self.target_width])


class EYES_GAN_GENERATOR:
    LAMBDA = 100
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __init__(self, input_shape=(24, 50, 3), output_channels=3):
        self.input_shape = input_shape
        self.OUTPUT_CHANNELS = output_channels
        self.generator = self.build_generator()

    def downsample(self, filters, size, apply_batchnorm=True):
        """Downsamples an input by a factor of 2."""
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def upsample(self, filters, size, apply_dropout=False):
        """Upsamples an input by a factor of 2."""
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer, use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def build_generator(self):
        """Builds the generator model using downsampling and upsampling layers."""
        inputs = Input(shape=self.input_shape)
        print(f"Input shape: {inputs.shape}")

        # Downsampling layers
        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 12, 25, 64)
            self.downsample(128, 4),  # (batch_size, 6, 13, 128)
            self.downsample(256, 4),  # (batch_size, 3, 7, 256)
            self.downsample(512, 4),  # (batch_size, 2, 4, 512)
        ]

        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            print(f"Downsampled to: {x.shape}")
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling layers
        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 3, 7, 512)
            self.upsample(256, 4, apply_dropout=True),  # (batch_size, 6, 13, 256)
            self.upsample(128, 4),  # (batch_size, 12, 25, 128)
            self.upsample(64, 4),  # (batch_size, 24, 50, 64)
        ]

        for up, skip in zip(up_stack, skips):
            x = up(x)
            print(f"Upsampled to: {x.shape}")
            if x.shape[1:3] == skip.shape[1:3]:
                x = Concatenate()([x, skip])
                print(f"Concatenated with skip connection: {x.shape}")

        # Final layer
        last = Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                               strides=1,
                               padding='same',
                               kernel_initializer=tf.random_normal_initializer(0., 0.02),
                               activation='tanh')  # (batch_size, 24, 50, OUTPUT_CHANNELS)
        x = last(x)
        x = ResizeLayer(24, 50)(x)  # Resize to (24, 50) using custom layer
        print(f"Final output shape: {x.shape}")

        return Model(inputs=inputs, outputs=x)


    def generate_and_save_image(self, input_image, save_path='generated_image.png'):
        """Generates an image using the generator and saves it to the specified path."""
        print("Predicting Images...")
        gen_output = self.generator(input_image, training=True)
        def save_image(image, save_path):
            if len(image.shape) == 4:
                image = tf.squeeze(image, axis=0)
            image = (image + 1.0) * 127.5
            image = tf.cast(image, tf.uint8)
            encoded_image = tf.image.encode_jpeg(image)
            tf.io.write_file(save_path, encoded_image)
            # print(f"Generated image saved to {save_path}")
            
        save_image(gen_output, save_path)
        return gen_output

    @staticmethod
    def generator_loss(disc_generated_output, gen_output, target):
        """Calculates the generator loss."""
        gan_loss = EYES_GAN_GENERATOR.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (EYES_GAN_GENERATOR.LAMBDA * l1_loss)
        print(f"Generator loss: {total_gen_loss}, GAN loss: {gan_loss}, L1 loss: {l1_loss}")
        return total_gen_loss, gan_loss, l1_loss

