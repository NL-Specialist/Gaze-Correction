import os
import tensorflow as tf
import datetime
import time
from tqdm import tqdm
import numpy as np
import imageio
import matplotlib.pyplot as plt
import shutil
import re
import asyncio
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras.mixed_precision import set_global_policy

# Set mixed precision policy
policy = Policy('mixed_float16')
set_global_policy(policy)

class EYES_GAN:
    def __init__(self, model_name, generator, discriminator, gpu_index='0'):        
        self.device = f'GPU:{gpu_index}'  # This will always refer to the first visible GPU after setting CUDA_VISIBLE_DEVICES
        print(f"[INFO] Using device: {self.device}")
        with tf.device(self.device):
            self.generator = generator
            self.discriminator = discriminator
            self.generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)
            self.model_save_dir = os.path.join('models', model_name)
            self.checkpoint_dir = os.path.join(self.model_save_dir, 'training_checkpoints')
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
            self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                                  discriminator_optimizer=self.discriminator_optimizer,
                                                  generator=self.generator.generator,
                                                  discriminator=self.discriminator.discriminator)
            self.log_dir = "logs/"
            self.summary_writer = tf.summary.create_file_writer(
                self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.gen_loss = []
            self.disc_loss = []
            self.progress = []
            self.step = 0

    async def predict(self, input_image, save_path='generated_image.png'):
        # Use asyncio.to_thread() to run the prediction in a non-blocking way
        return await asyncio.to_thread(self._run_prediction, input_image, save_path)

    def _run_prediction(self, input_image, save_path):
        with tf.device(self.device):
            prediction = self.generator.generator(input_image, training=True)
            self._save_image(prediction, save_path)
        return prediction

    def _save_image(self, image, save_path):
        if len(image.shape) == 4:
            image = tf.squeeze(image, axis=0)
        image = (image + 1.0) * 127.5
        image = tf.cast(image, tf.uint8)
        encoded_image = tf.image.encode_jpeg(image)
        tf.io.write_file(save_path, encoded_image)

    async def train_step(self, input_image, target, step):
        return await asyncio.to_thread(self._run_train_step, input_image, target, step)

    def _run_train_step(self, input_image, target, step):
        with tf.device(self.device):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = self.generator.generator(input_image, training=True)
                gen_output = tf.image.resize(gen_output, (24, 50))

                disc_real_output = self.discriminator.discriminator([input_image, target], training=True)
                disc_generated_output = self.discriminator.discriminator([input_image, gen_output], training=True)

                gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator.generator_loss(
                    disc_generated_output, gen_output, target)
                disc_loss = self.discriminator.discriminator_loss(
                    disc_real_output, disc_generated_output, tf.keras.losses.BinaryCrossentropy(from_logits=True))

            generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.discriminator.trainable_variables))

            with self.summary_writer.as_default():
                tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
                tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
                tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
                tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    async def generate_images(self, test_input, tar, save_dir=None, epoch=0, step=0):
        await asyncio.to_thread(self._run_generate_images, test_input, tar, save_dir, epoch, step)

    def _run_generate_images(self, test_input, tar, epoch, step, save_dir=None):
        with tf.device(self.device):
            if save_dir is None:
                save_dir = self.image_checkpoint_dir
            else:
                save_dir = os.path.join(self.model_save_dir, save_dir)

            print(f"Generating images at epoch {epoch}, step {step}")
            prediction = self.generator.generator(test_input, training=True)

            images = {
                'input': test_input[0],
                'target': tar[0],
                'predicted': prediction[0],
                'combined': [test_input[0], tar[0], prediction[0]]
            }

            titles = {
                'input': 'Input Image',
                'target': 'Ground Truth',
                'predicted': 'Predicted Image'
            }

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"Directory created: {save_dir}")

            for key, img in images.items():
                if key == 'combined':
                    plt.figure(figsize=(15, 5))
                    for i in range(3):
                        plt.subplot(1, 3, i + 1)
                        plt.title(titles[list(titles.keys())[i]])
                        plt.imshow(img[i] * 0.5 + 0.5)
                        plt.axis('off')
                    save_path = os.path.join(save_dir, f'image_at_epoch_{epoch}_step_{step}.png')
                    plt.savefig(save_path, facecolor='white')
                    plt.close()
                    print(f"Image saved to {save_path}")
                else:
                    img = img.numpy()  # Convert EagerTensor to NumPy array
                    img = (img * 0.5 + 0.5) * 255  # Convert from [-1, 1] to [0, 255]
                    img = img.astype(np.uint8)     # Convert to uint8
                    save_path = os.path.join(save_dir, f'image_{key}_at_epoch_{epoch}_step_{step}.png')
                    imageio.imwrite(save_path, img)
                    print(f"Image saved to {save_path}")

    def fit(self, model_name, train_ds, test_ds, epochs, learning_rate, checkpoint_interval=1000):
        with tf.device(self.device):
            self.model_name = model_name
            self.model_save_dir = os.path.join('models', self.model_name)
            print("[INFO] Model Save Directory Set: ", self.model_save_dir)

            # Clear Existing Training
            model_path = os.path.join('models', model_name)
            if os.path.exists(model_path):
                print(f"[INFO] Clearing existing training in path: ", model_path)
                for filename in os.listdir(model_path):
                    file_path = os.path.join(model_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'[ERROR] Failed to delete {file_path}. Reason: {e}')
            else:
                os.makedirs(model_path, exist_ok=True)

            self.checkpoint_dir = os.path.join(self.model_save_dir, 'training_checkpoints')
            print("[INFO] Training Checkpoint Directory Set: ", self.checkpoint_dir)

            self.image_checkpoint_dir = os.path.join(self.model_save_dir, 'image_checkpoints')
            print("[INFO] Image Checkpoint Directory Set: ", self.image_checkpoint_dir)

            example_input, example_target = next(iter(test_ds.take(1)))
            example_input, example_target = example_input[0], example_target[0]

            steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
            print(f'Steps per epoch: {steps_per_epoch}')

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            for epoch in range(epochs):
                print(f"Epoch {epoch}/{epochs}")
                start = time.time()

                progress_bar = tqdm(enumerate(train_ds), total=steps_per_epoch)

                for step, (input_image, target) in progress_bar:
                    global_step = epoch * steps_per_epoch + step
                    gen_total_loss, gen_gan_loss, gen_l1_loss, disc_total_loss = self._run_train_step(input_image, target, global_step)

                    if (step) % 100 == 0:
                        progress_bar.set_postfix({
                            'gen_total_loss': f'{gen_total_loss.numpy():.4f}',
                            'gen_gan_loss': f'{gen_gan_loss.numpy():.4f}',
                            'gen_l1_loss': f'{gen_l1_loss.numpy():.4f}',
                            'disc_loss': f'{disc_total_loss.numpy():.4f}'
                        })

                    if (step) % checkpoint_interval == 0:
                        self._run_generate_images(test_input=input_image, tar=target, epoch=epoch, step=step)
                        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                        print(f"Checkpoint saved at step {step} of epoch {epoch}")

                self.gen_loss.append(gen_total_loss)
                self.disc_loss.append(disc_total_loss)
                progress_percentage = ((epoch+1) / epochs) * 100
                self.progress.append(progress_percentage)

                print("[FIT] PROGRESS: ", progress_percentage)
                print("[FIT] GENERATOR LOSS: ", self.gen_loss[-1].numpy())
                print("[FIT] DISCRIMINATOR LOSS: ", self.disc_loss[-1].numpy())
                self._run_generate_images(test_input=input_image, tar=target, epoch=epoch, step=steps_per_epoch)
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print(f"Checkpoint saved at the end of epoch {epoch}")

                print(f'Time taken for epoch {epoch} is {time.time() - start:.2f} sec\n')

            self.summary_writer.close()

    async def restore(self, checkpoint_nr=-1):
        return await asyncio.to_thread(self._run_restore, checkpoint_nr)

    def _run_restore(self, checkpoint_nr=-1):
        with tf.device(self.device):
            if checkpoint_nr == -1:
                checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
            else:
                checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
                if checkpoint_path:
                    checkpoint_path = re.sub(r'ckpt-\d+', f'ckpt-{checkpoint_nr}', checkpoint_path)

            if checkpoint_path:
                self.checkpoint.restore(checkpoint_path)
                print(f"Restored from {checkpoint_path}")
            else:
                print(f"No checkpoint found in: {checkpoint_path}")

    async def validate(self, val_dataset, save_dir='validated_images', validation_dataset_percentage=10):
        await asyncio.to_thread(self._run_validate, val_dataset, save_dir, validation_dataset_percentage)

    def _run_validate(self, val_dataset, save_dir, validation_dataset_percentage):
        with tf.device(self.device):
            print("Running validation...")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"Directory created: {save_dir}")

            try:
                total_samples = sum(1 for _ in val_dataset)
                num_samples_to_validate = max(1, total_samples * validation_dataset_percentage // 100)

                val_dataset = val_dataset.take(num_samples_to_validate)

                for i, (inp, tar) in enumerate(val_dataset):
                    inp = inp[0]
                    tar = tar[0]
                    print(f"Processing data {i}")
                    print(f"Input tensor structure: {type(inp)}, shape: {inp.shape}")
                    print(f"Target tensor structure: {type(tar)}, shape: {tar.shape}")

                    self._run_generate_images(inp, tar, save_dir=save_dir, epoch=0, step=i)

                print(f"Validation complete. Images saved to {save_dir}.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            finally:
                self.summary_writer.close()

