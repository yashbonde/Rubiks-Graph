"""
common.py

This has some of the commonly used layers.

07.10.2019 - @yashbonde
"""

import tensorflow as tf

def shapes_list(inp):
    """
    cleaner handling of tensorflow shapes
    :param inp: input tensor
    :return: list of shapes combining dynamic and static shapes
    """
    shapes_static = inp.get_shape().as_list()
    shapes_dynamic = tf.shape(inp)
    cleaned_shape = [shapes_dynamic[i] if s is None else s for i, s in enumerate(shapes_static)]
    return cleaned_shape

def dense(inp, scope, num_features, weights_init_stddev=0.2):
    """
    dense block, first reshape input then matmul weights and then reshape

    :param inp: input tensor
    :param scope: tf variable scope
    :param num_features: number of output features
    :param weights_init_stddev: standard deviation value
    :return: processed output
    """
    with tf.variable_scope(scope):
        *start, nx = shapes_list(inp)
        weights = tf.get_variable('w', [1, nx, num_features],
                                  initializer=tf.random_normal_initializer(stddev=weights_init_stddev))
        bias = tf.get_variable('b', [num_features],
                               initializer=tf.constant_initializer(0))

        # reshape input and weights and perform matmul and add bias
        inp_reshaped = tf.reshape(inp, [-1, nx])
        w_reshaped = tf.reshape(weights, [-1, num_features])
        out = tf.matmul(inp_reshaped, w_reshaped) + bias

        out = tf.reshape(out, start + [num_features])
        return out

def expand_tile(value, size):
    """
    expand value to size

    :param value: input object to be tiles
    :param size: size to tile the object to
    :return: tiled output
    """
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    out = tf.expand_dims(value, axis=0)
    out = tf.tile(out, [size] + [1, ] * ndims)
    return out

def positions_for(tokens, past_length):
    """
    get positions only for a input tokens

    :param tokens: input tokens
    :param past_length: length of past object
    :return: output
    """
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    out = expand_tile(past_length + tf.range(nsteps), batch_size)
    return out