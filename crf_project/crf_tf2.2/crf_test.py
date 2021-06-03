# *- coding: utf-8 -*

import os
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.text.crf import crf_log_likelihood
from bert4keras.models import build_transformer_model

path = '../../pretrain_models/chinese_wwm_ext_L-12_H-768_A-12/'

config_path = path + 'bert_config.json'
model_path = path + 'bert_model.ckpt'
vocab_path = path + 'vocab.txt'

num_class = len(['O', 'I-LOC', 'B-LOC', 'I-ORG', 'B-ORG', 'B-PER', 'I-PER']) + 3



def bert_build():
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=model_path,
        model='bert'
    )
    hidden = bert.model.output
    dropout = tf.keras.layers.Dropout(0.3)(hidden)
    dense = tf.keras.layers.Dense(num_class,
                                  activation='softmax',
                                  kernel_initializer=bert.initializer
                                  )(dropout)
    model = tf.keras.Model()
