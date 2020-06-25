import functools
import random
import sys
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from wgan_gp.models.wgan_gp_trainer import train_wgan_gp
from wgan_gp.models.networks import make_dcgan_discriminator_32x32, make_dcgan_generator_32x32
from wgan_gp.utils.utils import create_dir_if_missing, current_timestamp

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
 
LATENT_DIMS = 100
DATA_SHAPE = [32, 32, 3]
NUM_ITER = 2
NUM_DISCRIM_UPDATE_PER_ITER = 5
BATCH_SIZE = 5
GRAD_PENALTY_COEF = 10.0

BASE_EXPORT_DIR = 'data/debug'
EXP_NAME = f'wgan_gp_{current_timestamp()}'
EXPORT_DIR = f'{BASE_EXPORT_DIR}/{EXP_NAME}'
DISCRIMINATOR_EXPORT_DIR = f'{EXPORT_DIR}/discriminator_model'
GENERATOR_EXPORT_DIR = f'{EXPORT_DIR}/generator_model'

create_dir_if_missing(EXPORT_DIR)

generator = make_dcgan_generator_32x32(LATENT_DIMS)
discriminator = make_dcgan_discriminator_32x32(DATA_SHAPE, LATENT_DIMS)

generator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.0, 
        beta_2=0.9)
discriminator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.0, 
        beta_2=0.9)

#(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
print(f'train_images: min={np.min(train_images)}, max={np.max(train_images)}, shape={train_images.shape}')
train_images = train_images.reshape([-1] + DATA_SHAPE).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
    .shuffle(train_images.shape[0]) \
    .batch(BATCH_SIZE) \
    .prefetch(1)

train_dataset = train_dataset.take(2)

epoch_logs = train_wgan_gp(
    discriminator, 
    generator, 
    dataset=train_dataset, 
    latent_dims=LATENT_DIMS,
    discriminator_optimizer=discriminator_optimizer, 
    generator_optimizer=generator_optimizer,
    num_epoch=NUM_ITER, 
    n_critic=NUM_DISCRIM_UPDATE_PER_ITER, 
    grad_penalty_coef=GRAD_PENALTY_COEF
)

create_dir_if_missing(GENERATOR_EXPORT_DIR)
generator.save(GENERATOR_EXPORT_DIR)
create_dir_if_missing(DISCRIMINATOR_EXPORT_DIR)
discriminator.save(DISCRIMINATOR_EXPORT_DIR)

epoch_logs_df = pd.DataFrame(epoch_logs)
epoch_logs_df.to_csv(f'{EXPORT_DIR}/epoch_logs.csv')

noise = tf.random.normal(shape=(5, LATENT_DIMS))
x_g = generator(noise).numpy()
x_g_rgb = ((x_g * 127.5) + 127.5).astype('int')

create_dir_if_missing(f'{EXPORT_DIR}/images')
for i in range(len(x_g)):
    plt.figure()
    plt.imshow(x_g_rgb[i])
    plt.savefig(f'{EXPORT_DIR}/images/image-{i}.png')
