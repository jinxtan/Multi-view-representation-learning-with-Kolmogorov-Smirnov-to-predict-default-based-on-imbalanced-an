import tensorflow as tf

def MV_DNN(positive_continuous_features,positive_discrete_features,negative_continuous_features,negative_discrete_features):
    """
    多视野神经网络
    :param positive_continuous_features:
    :param positive_discrete_features:
    :param negative_continuous_features:
    :param negative_discrete_features:
    :return:
    """
    # 设计多视野神经网络，将这些特征分别输入到不同的视野中
    positive_continuous_input = tf.keras.Input(shape=(len(positive_continuous_features),),
                                               name='positive_continuous_input')
    positive_discrete_input = tf.keras.Input(shape=(len(positive_discrete_features),), name='positive_discrete_input')
    negative_continuous_input = tf.keras.Input(shape=(len(negative_continuous_features),),
                                               name='negative_continuous_input')
    negative_discrete_input = tf.keras.Input(shape=(len(negative_discrete_features),), name='negative_discrete_input')

    positive_continuous_dense = tf.keras.layers.Dense(20, activation='relu')(positive_continuous_input)
    positive_discrete_dense = tf.keras.layers.Dense(20, activation='relu')(positive_discrete_input)
    negative_continuous_dense = tf.keras.layers.Dense(20, activation='relu')(negative_continuous_input)
    negative_discrete_dense = tf.keras.layers.Dense(20, activation='relu')(negative_discrete_input)

    x_concat = tf.keras.layers.Concatenate()(
        [positive_continuous_dense, positive_discrete_dense, negative_continuous_dense, negative_discrete_dense])
    x_concat_dense = tf.keras.layers.Dense(20, activation='relu')(x_concat)
    x_concat_dense = tf.keras.layers.Dense(20, activation='relu')(x_concat_dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x_concat_dense)

    model = tf.keras.Model(
        inputs=[positive_continuous_input, positive_discrete_input, negative_continuous_input, negative_discrete_input],
        outputs=output)

    return model
