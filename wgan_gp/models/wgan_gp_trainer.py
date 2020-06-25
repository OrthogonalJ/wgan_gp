import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm


def discriminator_loss(discriminator, generator, real_data, latent_vars, 
                       grad_penalty_coef, training=True):
    fake_data = generator(latent_vars, training=training)
    #print('fake_data.shape:', fake_data.shape)
    fake_scores = discriminator(fake_data, training=training)
    real_scores = discriminator(real_data, training=training)
    
    batch_size = real_data.shape[0]
    epsilon = tf.random.uniform(shape=(batch_size,))
    epsilon = tf.reshape(epsilon, [-1] + [1] * (real_data.shape.rank - 1))
    #print('epsilon.shape:', epsilon.shape)

    x_hat = epsilon * real_data + (1 - epsilon) * fake_data
    #print('x_hat.shape:', x_hat.shape)
    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        x_hat_scores = discriminator(x_hat, training=training)
    
    grads = tape.gradient(x_hat_scores, x_hat)
    grad_penalty = grad_penalty_coef * tf.math.square(tf.norm(grads, 2) - 1.0)

    loss = fake_scores - real_scores + grad_penalty
    return tf.reduce_mean(loss)


def generator_loss(generator, discriminator, latent_vars, training=True):
    fake_date = generator(latent_vars, training=training)
    fake_scores = discriminator(fake_date, training=training)
    return -tf.reduce_mean(fake_scores)


#@tf.function
def train_step(discriminator, generator, 
        discriminator_optimizer, generator_optimizer,
        real_data, latent_dims, n_critic, grad_penalty_coef):
    batch_size = real_data.shape[0]

    d_loss = tf.constant(1)
    for _ in range(n_critic):
        with tf.GradientTape() as d_tape:
            latent_vars = tf.random.normal(shape=(batch_size, latent_dims))
            d_loss = discriminator_loss(discriminator, generator, real_data,
                    latent_vars, grad_penalty_coef)
        
        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(d_grads, 
                discriminator.trainable_variables))
    
    with tf.GradientTape() as g_tape:
        latent_vars = tf.random.normal(shape=(batch_size, latent_dims))
        g_loss = generator_loss(generator, discriminator, latent_vars)
    
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(g_grads, 
            generator.trainable_variables))
    
    return d_loss, g_loss


def train_wgan_gp(discriminator, generator, 
        dataset, latent_dims,
        discriminator_optimizer, generator_optimizer,
        num_epoch, n_critic=5, grad_penalty_coef=10.0,
        log_freq=None, log_fn=None):
    n_critic = tf.constant(n_critic)
    grad_penalty_coef = tf.constant(grad_penalty_coef)
    
    d_loss_metric = tf.keras.metrics.Mean()
    g_loss_metric = tf.keras.metrics.Mean()

    epoch_logs = []
    for epoch_idx in tqdm(range(num_epoch)):
        d_loss_metric.reset_states()
        g_loss_metric.reset_states()

        for batch_data in tqdm(dataset):
            d_loss, g_loss = train_step(
                discriminator, generator, 
                discriminator_optimizer, generator_optimizer, 
                batch_data, latent_dims,
                n_critic, grad_penalty_coef
            )

            d_loss_metric(d_loss)
            g_loss_metric(g_loss)
        
        epoch_log_row = {
            'epoch': epoch_idx,
            'discriminator_loss': d_loss_metric.result().numpy(),
            'generator_loss': g_loss_metric.result().numpy(),
        }
        epoch_logs.append(epoch_log_row)

        if log_freq is not None and (epoch_idx % log_freq) == 0:
            assert log_fn is not None
            log_fn(generator, discriminator, epoch_log_row)
    
    return epoch_logs
