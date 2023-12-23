import tensorflow as tf

# 自定义一个KS metric
def ks(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    length = tf.shape(y_true)[0]
    t = tf.math.top_k(y_pred, k=length, sorted=True)
    y_pred_sorted = tf.gather(y_pred, t.indices)
    y_true_sorted = tf.gather(y_true, t.indices)
    cum_positive_ratio = tf.truediv(
        tf.cumsum(y_true_sorted), tf.reduce_sum(y_true_sorted))
    cum_negative_ratio = tf.truediv(
        tf.cumsum(1 - y_true_sorted), tf.reduce_sum(1 - y_true_sorted))
    ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))
    return ks_value
