# *- coding: utf-8 -*

import os
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.text.crf import crf_log_likelihood


def get_data(data):
    if len(data) == 2:
        return data[0], data[1]
        # x, y
    elif len(data) == 3:
        return data
        # x, y, weights
    else:
        raise TypeError("Expect data is a tuple of size is 2 or 3")


class CrfLossModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs, training=None, mask=None):
        return self.base_model(inputs)

    def compute_loss(self, x, y, sample_weights, training=False):
        y_pred = self(x, training)
        _, potentials, sequence_length, chain_kernel = y_pred
        crf_loss = -1 * crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        if sample_weights is not None:
            crf_loss = crf_loss * sample_weights

        return tf.reduce_mean(crf_loss), sum(self.losses)

    def train_step(self, data):
        x, y, sample_weights = get_data(data)
        with tf.GradientTape() as tape:
            crf_loss, internal_losses = self.compute_loss(
                x, y, sample_weights, training=True
            )
            total_loss = crf_loss + internal_losses

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # 更新权重参数到模型上

    def test_step(self, data):
        x, y, sample_weights = get_data(data)
        crf_loss, internal_losses = self.compute_loss(
            x, y, sample_weights, training=False
        )
        return {"crf_loss_val": crf_loss, "internal_losses_val": internal_losses}


x_np, y_np = [None, None]

x_input = tf.keras.layers.Input(shape=x_np.shape[1:])
crf_outputs = tfa.layers.CRF(5)(x_input)
base_model = tf.keras.Model(x_input, crf_outputs)
model = CrfLossModel(base_model)

model.compile("adam")
model.fit(x=x_np, y=y_np)
model.evaluate(x_np, y_np)
model.predict(x_np)
model.save("my_model.tf")