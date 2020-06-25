import os
import types
import datetime
import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout # pylint: disable=import-error


def min_max_normalize(x, new_min=0.0, new_max=1.0, axis=None):
    x_min = np.min(x, axis=axis)
    x_max = np.max(x, axis=axis)
    x_unif = (x - x_min) / (x_max - x_min)
    
    new_x = ((new_max - new_min) * x_unif) + new_min
    return new_x


def min_max_normalize_tf(x, new_min=0.0, new_max=1.0, axis=None):
    x_min = tf.reduce_min(x, axis=axis)
    x_max = tf.reduce_max(x, axis=axis)
    x_unif = (x - x_min) / (x_max - x_min)
    
    new_x = ((new_max - new_min) * x_unif) + new_min
    return new_x


def create_dir_if_missing(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def current_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')

def has_method(obj, method_name):
    method = getattr(obj, method_name, None)
    return callable(method)

def is_function_or_lambda(obj):
    return isinstance(obj, types.LambdaType) or isinstance(obj, types.FunctionType)

def flat_triangular_matrix_size(num_dims):
    """
    Returns: Number of elements required to construct an triangular matrix with shape [num_dims, num_dims]
    """
    return num_dims + sp.special.comb(num_dims, 2, exact=True)

def placeholder_like(array):
    tensor_shape = [None] * len(array.shape)
    placeholder = tf.compat.v1.placeholder(shape=tensor_shape, dtype=tf.as_dtype(array.dtype))
    return placeholder

def shape_list(tensor):
    tensor_shape = tf.shape(tensor)
    return [tensor_shape[i] for i in range(tensor.get_shape().ndims)]

def make_activation_layer(activation):
    if isinstance(activation, dict):
        return activation['factory'](*activation.get('args', []), 
                **activation.get('kwargs', dict()))
    return Activation(activation)

def make_fully_connected_block(input_layer, num_layers=1, layer_size=32,
        activation='relu', use_batch_norm=False, use_dropout=False, 
        dropout_rate=0.3):
    
    output_tensor = input_layer
    for _ in range(num_layers):
        output_tensor = Dense(layer_size)(output_tensor)
        if use_batch_norm:
            output_tensor = BatchNormalization()(output_tensor)
        output_tensor = make_activation_layer(activation)(output_tensor)
        if use_dropout:
            output_tensor = Dropout(dropout_rate)(output_tensor)
    
    return output_tensor

def make_mlp(input_layer, num_outputs, output_activation=None,
        num_layers=1, layer_size=32, activation='relu',
        use_batch_norm=False, use_dropout=False, 
        dropout_rate=0.3):
    
    output_tensor = make_fully_connected_block(input_layer,
            num_layers, layer_size, activation, use_batch_norm, 
            use_dropout, dropout_rate)
    
    output_tensor = Dense(num_outputs)(output_tensor)
    output_tensor = make_activation_layer(output_activation)(output_tensor)
    return output_tensor

def chain_functions(functions, chain_input):
    if len(functions) < 1:
        return chain_input
    
    value = functions[0](chain_input)
    for i in range(1, len(functions)):
        value = functions[i](value)

    return value

def gradient_variable_tuples(gradient_tape, tensor, variables):
    grads = gradient_tape.gradient(tensor, variables)
    return zip(grads, variables)

def unique_variables(variables):
    unique_vars = {var.experimental_ref(): var for var in variables}
    return list(unique_vars.values())

# def make_mlp(input_layer, num_outputs, output_activation=None, 
#         num_layers=1, layer_size=32, activation='relu', name_prefix='', output_name=None):
    
#     hidden_out = input_layer
#     for i in range(num_layers):
#         layer_name = '{}fc{}'.format(name_prefix, i)
#         hidden_out = Dense(layer_size, activation=activation, name=layer_name)(hidden_out)
    
#     output_tensor = Dense(num_outputs, activation=output_activation, name=output_name)(hidden_out)
#     return output_tensor

def reward_to_go(rewards, gamma):
    """
        Calculate discounted reward-to-go for one episode
    """
    trajectory_rewards = []
    for i in range(len(rewards)):
        num_steps_to_go = len(rewards) - i
        # pylint: disable=assignment-from-no-return
        gamma_coefs = np.power(np.full(num_steps_to_go, gamma), np.arange(num_steps_to_go))
        reward = np.sum(gamma_coefs * rewards[i:])
        trajectory_rewards.append(reward)
    return trajectory_rewards

def slice_back(tensor, size, begin=None, name='concat'):
    begin = begin if begin is not None else [0] * len(size)
    size_len = len(size)
    begin_len = len(begin)
    # size_len = size.get_shape()[-1]
    # begin_len = begin.get_shape()[-1]
    assert begin is None or size_len == begin_len, 'size and begin must have the same length'
    shape = tensor.get_shape()
    num_unsliced_dims = len(shape) - size_len
    size_ = [-1] * num_unsliced_dims
    begin_ = [0] * num_unsliced_dims
    size_.extend(size)
    begin_.extend(begin)
    # size_ = tf.constant([-1] * num_unsliced_dims)
    # begin_ = tf.constant([0] * num_unsliced_dims)
    # size_ = tf.concat([size_, size], axis=-1)
    # begin_ = tf.concat([begin_, begin], axis=-1)
    return tf.slice(tensor, size_, begin_, name)

class ConstantFunctor:
    def __init__(self, value):
        self._value = value
    def __call__(self, *args, **kwargs):
        return self._value

