import os
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from IPython import display
import time
from tqdm import tqdm

class EYES_GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator.generator,
                                              discriminator=self.discriminator.discriminator)
        self.log_dir = "logs/"
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def generate_images(self, test_input, tar, save_dir='image_checkpoints/', epoch=0, step=0):
        print(f"Generating images at epoch {epoch}, step {step}")
        prediction = self.generator.generator(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directory created: {save_dir}")

        save_path = os.path.join(save_dir, f'image_at_epoch_{epoch}_step_{step}.png')
        plt.savefig(save_path)
        plt.show()
        print(f"Image saved to {save_path}")

    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator.generator(input_image, training=True)

            # Ensure gen_output is reshaped to match target shape
            gen_output = tf.image.resize(gen_output, (24, 50))

            disc_real_output = self.discriminator.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator.discriminator_loss(disc_real_output, disc_generated_output, tf.keras.losses.BinaryCrossentropy(from_logits=True))

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
        
        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    def fit(self, train_ds, test_ds, epochs, checkpoint_interval=1000):
        example_input, example_target = next(iter(test_ds.take(1)))
        
        # Determine steps per epoch dynamically
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        print(f'Steps per epoch: {steps_per_epoch}')
    
        # Ensure the checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            start = time.time()
            
            progress_bar = tqdm(enumerate(train_ds), total=steps_per_epoch)
    
            for step, (input_image, target) in progress_bar:
                global_step = epoch * steps_per_epoch + step
                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(input_image, target, global_step)
    
                if (step + 1) % 100 == 0:
                    progress_bar.set_postfix({
                        'gen_total_loss': f'{gen_total_loss.numpy():.4f}',
                        'gen_gan_loss': f'{gen_gan_loss.numpy():.4f}',
                        'gen_l1_loss': f'{gen_l1_loss.numpy():.4f}',
                        'disc_loss': f'{disc_loss.numpy():.4f}'
                    })
    
                if (step + 1) % checkpoint_interval == 0:
                    self.generate_images(example_input, example_target, epoch=epoch, step=step + 1)
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                    print(f"Checkpoint saved at step {step + 1} of epoch {epoch + 1}")
    
            self.generate_images(example_input, example_target, epoch=epoch, step=steps_per_epoch)
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print(f"Checkpoint saved at the end of epoch {epoch + 1}")
    
            print(f'Time taken for epoch {epoch + 1} is {time.time()-start:.2f} sec\n')
    
        self.summary_writer.close()


    def restore(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if (latest_checkpoint):
            self.checkpoint.restore(latest_checkpoint)
            print(f"Restored from {latest_checkpoint}")
        else:
            print("No checkpoint found.")

    def validate(self, val_dataset, save_dir='validated_images'):
        print("Running validation...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directory created: {save_dir}")

        for i, (inp, tar) in enumerate(val_dataset):
            print(f"Processing data {i}")
            print(f"Input tensor structure: {type(inp)}, shape: {inp.shape}")
            print(f"Target tensor structure: {type(tar)}, shape: {tar.shape}")

            if not isinstance(inp, tf.Tensor):
                raise ValueError(f"Expected tensor for input, got {type(inp)}")
            if not isinstance(tar, tf.Tensor):
                raise ValueError(f"Expected tensor for target, got {type(tar)}")
            
            self.generate_images(inp, tar, save_dir=save_dir, epoch=0, step=i)
        
        print(f"Validation complete. Images saved to {save_dir}.")
        self.summary_writer.close()