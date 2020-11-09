from typing import Optional, Any

import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score


class MyMLP(tf.keras.Model):

    def __init__(self, dropout_rate=0.2,
                 hidden_activation=tf.nn.relu,
                 output_activation=tf.nn.softmax,
                 kernel_initializer='he_normal',
                 n_outputs=10):
        # Initialize the necessary components of tf.keras.Model
        super(MyMLP, self).__init__()
        # Now we initalize the needed layers
        # -----------------------------------------------------------
        # Flatten Layer
        self.flatten = tf.keras.layers.Flatten()
        # First Dense Layer
        self.dense_1 = tf.keras.layers.Dense(64,
                                             kernel_initializer=kernel_initializer,
                                             activation=hidden_activation)

        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)

        # Second Dense Layer
        self.dense_2 = tf.keras.layers.Dense(128,
                                             kernel_initializer=kernel_initializer,
                                             activation=hidden_activation)

        # Output Layer, if output activate was None (linear by defaut)) : logit True
        self.model_output = tf.keras.layers.Dense(n_outputs, activation=output_activation)

    # Forward pass of model - order does matter.
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense_1(x)
        x = self.dropout1(x, training=training)
        x = self.dense_2(x)

        return self.model_output(x)  # Return results of Output Layer


class MyCNN(tf.keras.Model):

    def __init__(self, dropout_rate=0.2,
                 hidden_activation=tf.nn.relu,
                 output_activation=tf.nn.softmax,
                 kernel_initializer='he_normal',
                 n_outputs=10):
        # Initialize the necessary components of tf.keras.Model
        super(MyCNN, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(filters=64,
                                             kernel_size=(5, 5),
                                             padding='valid',
                                             activation=hidden_activation,
                                             name='conv_1',
                                             input_shape=(28, 28, 1))
        self.norm_1 = tf.keras.layers.BatchNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.pooling_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))

        self.conv_2 = tf.keras.layers.Conv2D(filters=128,
                                             kernel_size=(5, 5),
                                             padding='same',
                                             activation=hidden_activation,
                                             name='conv_2')
        self.norm_2 = tf.keras.layers.BatchNormalization()
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.pooling_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                      name='pool_2')

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(100, activation=hidden_activation, kernel_initializer=kernel_initializer,
                                           name='dense_1')
        self.norm_3 = tf.keras.layers.BatchNormalization()
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)

        self.dense_2 = tf.keras.layers.Dense(128, activation=hidden_activation, name='dense_2')
        self.model_output = tf.keras.layers.Dense(units=n_outputs, activation=output_activation,name='dense_3')

    # Forward pass of model - order does matter.
    def call(self, inputs, training=False):
        #self.inputs = inputs
        # (batch_size, height, width, depth) with a filter = 64 + kernel of 5,5 + padding valid +=> (64, (x-5+1), (x-5+1), 64)
        conv_1 = self.conv_1(inputs)
        print('Convolution 1 has shape ' + str(conv_1.get_shape()))
        norm_1 = self.norm_1(conv_1, training=training)
        drp_1 = self.dropout_1(norm_1, training=training)
        pool_1 = self.pooling_1(drp_1)  # pooling of (2,2) => (64, x'/2, x'/2, 64)
        print('Pooling 1 has shape ' + str(pool_1.get_shape()))

        # Filter = 128 + kernel of 5,5 + padding same  => (64, x'', x'', 128)
        conv_2 = self.conv_2(pool_1)
        print('Convolution 2 has shape ' + str(conv_2.get_shape()))
        norm_2 = self.norm_2(conv_2, training=training)
        drp_2 = self.dropout_2(norm_2, training=training)
        pool_2 = self.pooling_2(drp_2)
        print('Pool 2 has shape ' + str(pool_2.get_shape()))

        flatten = self.flatten(pool_2)
        print('Flatten has shape ' + str(flatten.get_shape()))
        norm_3 = self.norm_3(flatten, training=training)
        dense = self.dense(norm_3)
        print('Dense has shape ' + str(dense.get_shape()))
        drp_3 = self.dropout_3(dense, training=training)

        outputs = self.dense_2(drp_3)
        outputs_model = self.model_output(outputs)

        return outputs_model, flatten


class LeNet(tf.keras.Model):
    def __init__(self,
                 input_shape=(28, 28, 1),
                 hidden_activation=tf.nn.tanh,
                 output_activation=tf.nn.softmax,
                 n_outputs=10):
        # Initialize the necessary components of tf.keras.Model
        super(LeNet, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(6, kernel_size=(5, 5),
                                             strides=(1, 1),
                                             activation=hidden_activation,
                                             input_shape=input_shape,
                                             padding="same",
                                             name='conv_1')

        self.pool_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                       strides=(2, 2),
                                                       padding='valid',
                                                       name = 'pool_1')

        self.conv_2 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5),
                                             strides=(1, 1),
                                             activation=hidden_activation,
                                             padding='valid',
                                             name='conv_2')

        self.pool_2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                       strides=(2, 2),
                                                       padding='valid',
                                                       name='pool_2')

        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(120, activation=hidden_activation)
        self.dense_2 = tf.keras.layers.Dense(84, activation=hidden_activation)

        self.model_output = tf.keras.layers.Dense(n_outputs, activation=output_activation)

    # Forward pass of model - order does matter.
    def call(self, inputs, training=None):
        conv_1 = self.conv_1(inputs)
        conv_1 = self.pool_1(conv_1)

        conv_2 = self.conv_2(conv_1)
        conv_2 = self.pool_2(conv_2)

        flatten = self.flatten(conv_2)
        dense_1 = self.dense_1(flatten)
        dense_2 = self.dense_2(dense_1)

        return self.model_output(dense_2), flatten