#!/usr/bin/python
# coding:utf-8

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layer_norm
import math
# paper reference: convolutional sequence to sequence learning
# fast reading comprehension with convnets

""" encoder with dilated convolutions, gated linear units and residual connection
1.  converts inputs from the embedding space to the kernel size (hidden) space.
2. 
"""

initializer = lambda:tf.contrib.layers.variance_scaling_initializer(
            factor=2.0,
            mode='FAN_AVG',
            uniform=False,
            dtype=tf.float32)

def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)

def conv(inputs, output_size, bias=True, activation=None, initializer=None,
         kernel_size=1, scope="conv", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                                  filter_shape,
                                  dtype=tf.float32,
                                  initializer=initializer)
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                                       bias_shape,
                                       initializer=tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs

def dila_conv(inputs, kernel_size, dilation, inchannel, outchannel, scope="dila_conv", reuse=None,
              initializer=None):
    """
    :param inputs: [batch, sequence_len, num_filters]
    :param dilate:
    :param num_filters:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.expand_dims(inputs, axis=2)
        kernel_ = tf.get_variable(
            name="dila_conv_kernel",
            shape=[kernel_size, 1, inchannel, outchannel],
            initializer=initializer,
            trainable=True,
            dtype=tf.float32
        )
        with tf.variable_scope(scope, reuse=reuse):
            outputs = tf.nn.atrous_conv2d(
                value=inputs,
                filters=kernel_,
                rate=dilation,
                padding="SAME",
            )
        outputs = tf.reshape(outputs, (-1, tf.shape(inputs)[1], outchannel))
        return outputs

def glu(inputs):
    """
    :param inputs: [batch ,sequence_len, 2*num_filters]
    :return:
    """
    x, y = tf.split(inputs, 2, axis=-1)
    return x * tf.sigmoid(y)

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def reidual_dims_block(inputs, num_filters, scope="res_dims_block", reuse=None):
    """
    :param inputs: [batch, sequence_len, embed_size]
    :param num_filters:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv(inputs, num_filters*2, scope="residual_conv1d")
        outputs = glu(outputs)
        return outputs

def residual_block(inputs, kernel_size, dilation, num_filters, dropout, scope=None, reuse=None, sublayer=(1,1),
                   initializer=None):
    with tf.variable_scope(scope, reuse=reuse):
        l, L = sublayer
        outputs = dila_conv(inputs, kernel_size, dilation, num_filters, num_filters * 2,
                            scope="dila_conv1", initializer=initializer)
        outputs = glu(outputs)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
        l += 1
        outputs = layer_norm(residual, scope="layer_norm")
        outputs = dila_conv(outputs, kernel_size, dilation, num_filters, num_filters * 2,
                            scope="dila_conv2", initializer=initializer)
        outputs = glu(outputs)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        return outputs

def conv_block(inputs, num_conv_block=5, kernel_size=3, dilation=2, num_filters=100, scope="conv_block", reuse=None,
               initializer=initializer(), dropout=0.0):
    with tf.variable_scope(scope, reuse=reuse):
        # reducte dimension
        inputs = add_timing_signal_1d(inputs)
        outputs = layer_norm(inputs, scope="layer_norm_red")
        outputs = reidual_dims_block(outputs, num_filters, scope="red_dim")
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)

        # redisual block
        sublayer = 1
        total_layer = num_conv_block * 2
        for i in range(num_conv_block):
            conv_dilation = dilation ** i
            outputs = layer_norm(outputs, scope="layer_norm-%d"%i)
            outputs = residual_block(outputs, kernel_size, conv_dilation, num_filters, dropout, initializer=initializer,
                                     scope="residual_block-%d"%i, reuse=reuse, sublayer=(sublayer, total_layer))

        # refinement
        for i in range(3):
            outputs = residual_block(outputs, kernel_size, 1, num_filters, dropout,
                                     scope="refine-%d"%i, reuse=reuse, sublayer=(sublayer, total_layer))

        return outputs


if __name__ == "__main__":
    # inputs = tf.random_normal(shape=[5,10, 32])
    inputs = tf.placeholder(tf.float32, [None, None, 32], "inputs")
    outputs = conv_block(inputs)
    for var in tf.trainable_variables():
        print(var)
    print(outputs)




