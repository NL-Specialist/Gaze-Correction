import tensorflow as tf

class EYES_GAN_DISCRIMINATOR:
    def __init__(self, input_shape=(24, 50, 3)):
        self.input_shape = input_shape
        self.discriminator = self.build_discriminator()


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

    def build_discriminator(self):
        """Builds the discriminator model using downsampling layers."""
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=self.input_shape, name='input_image')
        tar = tf.keras.layers.Input(shape=self.input_shape, name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # Concatenate input and target images

        down1 = self.downsample(64, 4, False)(x)
        down2 = self.downsample(128, 4)(down1)
        down3 = self.downsample(256, 4)(down2)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    @staticmethod
    def discriminator_loss(disc_real_output, disc_generated_output, loss_object):
        """Calculates the discriminator loss."""
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        print(f"Discriminator loss: {total_disc_loss}, Real loss: {real_loss}, Generated loss: {generated_loss}")
        return total_disc_loss
