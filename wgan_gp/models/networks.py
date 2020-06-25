import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_dcgan_generator_32x32(latent_dims):
    model = keras.Sequential([
        layers.Input(shape=(latent_dims,), dtype='float32'),
        
        layers.Dense(4*4*4*latent_dims, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Reshape([4, 4, latent_dims * 4]),
        layers.Conv2DTranspose(filters=latent_dims * 4, 
            kernel_size=(5, 5), 
            strides=(1, 1), 
            padding='same', 
            use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        # size should be (batch, 4, 4, latent_dims * 4)

        layers.Conv2DTranspose(filters=latent_dims * 2, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same', 
            use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        # size should be (batch, 8, 8, latent_dims * 2)

        layers.Conv2DTranspose(filters=latent_dims, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same', 
            use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        # size should be (batch, 16, 16, latent_dims)

        layers.Conv2DTranspose(filters=3, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same', 
            use_bias=False),
        layers.Activation('tanh')
    ])
    return model


def make_dcgan_discriminator_32x32(data_shape, latent_dims):
    model = keras.Sequential([
        layers.Input(shape=data_shape, dtype='float32'),
        
        layers.Conv2D(latent_dims, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),

        layers.Conv2D(latent_dims * 2, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),

        layers.Conv2D(latent_dims * 4, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


def make_dcgan_generator_28x28(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_dcgan_discriminator(data_shape):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
            input_shape=data_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
