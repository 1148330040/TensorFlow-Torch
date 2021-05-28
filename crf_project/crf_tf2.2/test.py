# *- coding: utf-8 -*

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import function
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model

path = '../../pretrain_models/chinese_wwm_ext_L-12_H-768_A-12/'
config_path = path + 'bert_config.json'
model_path = path + 'bert_model.ckpt'
vocab_path = path + 'vocab.txt'

tokenizer = Tokenizer(vocab_path, do_lower_case=True)

# vocab_size_bio = 0
#
# input_ids = tf.keras.Input(shape=(None,))
# input_mask = tf.keras.Input(shape=(None,))
# token_types_ids = tf.keras.Input(shape=(None,))
# labels = tf.keras.Input(shape=(None, ))
#
# input_seq_len = tf.reduce_sum(input_mask, axis=-1)
#
# bert = build_transformer_model(
#     config_path=config_path,
#     checkpoint_path=model_path,
#     model='bert',
#     return_keras_model=False
# )
#
#
# hidden = bert.model.output
# logistic_seq = tf.keras.layers.Dense(hidden, vocab_size_bio)
# prob_seq = tf.nn.softmax(logistic_seq)
#
#
#
# log_likelihood, transition_matrix = tfa.text.crf_log_likelihood(logistic_seq, labels, input_seq_len)
# predict_seq, crf_scores = tfa.layers.crf.crf_decode(logistic_seq, transition_matrix, input_seq_len)
#
# predict_label = predict_seq
#
# loss = -log_likelihood / tf.cast(input_seq_len, tf.float32)
# loss = tf.reduce_sum(loss)


def get_dataset():

    return None


class MyBertCrf(tf.keras.Model):
    def __init__(self, use_crf, vocab_size_bio, inputs_seq_len):
        super(MyBertCrf, self).__init__(use_crf, vocab_size_bio, inputs_seq_len)
        self.bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=model_path,
            model='bert',
            return_keras_model=False
        )
        self.softmax = tf.nn.softmax()
        self.use_crf = use_crf
        self.vocab_size_bio = vocab_size_bio

    @function
    def call(self, ids, mask, token_type_ids, labels):

        hidden = self.bert(ids, mask, token_type_ids).output

        logistic_seq = tf.keras.layers.Dense(hidden, self.vocab_size_bio)


        if self.use_crf:
            log_likelihood, transition_matrix = tfa.text.crf_log_likelihood(logistic_seq, labels, self.inputs_seq_len)
            predict_seq, crf_scores = tfa.layers.crf.crf_decode(logistic_seq, transition_matrix, self.inputs_seq_len)

            return [predict_seq, crf_scores, log_likelihood]
        else:
            prob_seq = tf.nn.softmax(logistic_seq)
            return prob_seq


def get_loss(log_likelihood, input_seq_len):
    loss = -log_likelihood / tf.cast(input_seq_len, tf.float32)
    loss = tf.reduce_sum(loss)
    return loss


def fit(ids, mask, token_type_ids, labels, use_crf, vocab_size_bio):

    inputs_seq_len = tf.reduce_sum(mask, axis=-1)
    params_bert = []
    params_other = []
    # bert层的学习率与其他层的学习率要区分开来
    bert_crf = MyBertCrf(use_crf, vocab_size_bio, inputs_seq_len)
    with tf.GradientTape() as tp:
        predict_seq, crf_scores, log_likelihood = bert_crf(ids, mask, token_type_ids, labels)
        loss = get_loss(log_likelihood, inputs_seq_len)

        for var in bert_crf.trainable_variables:
            model_name = var.name
            if str(model_name).startswith('bert'):
                params_bert.append(var)
            else:
                params_other.append(var)
        opti_bert = tf.keras.optimizers.Adam(learning_rate=1e-4)
        opti_other = tf.keras.optimizers.Adam(learning_rate=1e-3)
        gradients_bert = tp.gradients(loss, params_bert)
        gradients_other = tp.gradients(loss, params_other)

        gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
        gradients_other_clipped, norm_other = tf.clip_by_global_norm(gradients_other, 5.0)

        opti_bert.apply_gradients(zip(gradients_bert_clipped, params_bert))
        opti_other.apply_gradients(zip(gradients_other_clipped, params_other))





