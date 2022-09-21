# -*- coding: utf-8 -*-
"""
Created on 2021/6/15 13:31
@author: Jinxtan
email: tanyd20@fudan.edu.cn
PyCharm.py
"""
import tensorflow as tf
def get_weight(weights):
    def mycrossentropy(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.ones_like(y_pred))
        loss = (1-weights)*tf.keras.backend.binary_crossentropy(y_true, y_pred)*pt_1+weights*tf.keras.backend.binary_crossentropy(y_true, y_pred)*pt_0
        return loss
    return mycrossentropy
def ks_():
    def ks(y_true,y_pred):
        y_true = tf.reshape(y_true,(-1,))
        y_pred = tf.reshape(y_pred,(-1,))
        length = tf.shape(y_true)[0]
        t = tf.math.top_k(y_pred,k = length,sorted = False)
        y_pred_sorted = tf.gather(y_pred,t.indices)
        y_true_sorted = tf.gather(y_true,t.indices)
        cum_positive_ratio = tf.truediv(
            tf.cumsum(y_true_sorted),tf.reduce_sum(y_true_sorted))
        cum_negative_ratio = tf.truediv(
            tf.cumsum(1 - y_true_sorted),tf.reduce_sum(1 - y_true_sorted))
        ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))
        return ks_value
    return ks