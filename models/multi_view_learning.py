# -*- coding: utf-8 -*-
"""
Created on 2021/5/22 20:37
@author: Jinxtan
email: tanyd20@fudan.edu.cn
PyCharm.py
"""
from __future__ import print_function
import tensorflow as tf

def dnn_layer(inputs, units=None,
              activation='relu',
              batch_normalization=True
              ):
    dnn = tf.keras.layers.Dense(units)
    x = dnn(inputs)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)
    # x = tf.keras.layers.Dropout(0.1, seed=0)(x)
    return x

def dnn_v1(input_shape = None, num_classes=2,
           x0 = None, x1 = None, x2 = None, x3 = None):
    x_inputs = tf.keras.Input(shape=input_shape)
    units = 20
    x_0 = dnn_layer(inputs=x_inputs[:, 0:x0, 0], units=units)
    x_1 = dnn_layer(inputs=x_inputs[:, 0:x1, 1], units=units)
    x_2 = dnn_layer(inputs=x_inputs[:, 0:x2, 2], units=units)
    x_3 = dnn_layer(inputs=x_inputs[:, 0:x3, 3], units=units)
    x = tf.concat((x_0, x_1, x_2, x_3),axis=1)
    y = dnn_layer(x, units=40)
    y = dnn_layer(y, units=30)
    y = dnn_layer(y, units=20)
    outputs = tf.keras.layers.Dense(num_classes,
                                    activation='sigmoid',
                                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = tf.keras.Model(inputs = x_inputs, outputs=outputs)
    return model