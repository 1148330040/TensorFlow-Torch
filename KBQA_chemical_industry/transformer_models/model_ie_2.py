# -*- coding: utf-8 -*

import os
import random
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa
from bert4keras.snippets import sequence_padding

from transformers import TFBertModel, BertTokenizer

os.environ["CUDA_VISIBLE_DEVICES"]="1"

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-12_H-768")

max_len = 64
batch_size = 20

# EINECS号: 欧洲化学管理局编码
# InChI: 国际化合物标识
# CAS号: 美国物质编号

def dataset_generator():

    train_dataset = eval(open('../dataset_kbqa_ci/spider_product/fit_dataset.json').read())

    # random.sample(train_dataset, len(train_dataset))

    def get_keyword_id_seat(keyword_id):
        len_keyword_id = len(keyword_id)
        for i in range(len(input_id)):
            if input_id[i:(i+len_keyword_id)] == keyword_id:
                return i
        return -1

    for ds in train_dataset:
        text = ds['text']

        if len(text) > max_len - 1:
            continue

        product_names = ds['product_names']
        product_values = ds['product_values']

        inputs = tokenizer.encode_plus(text)

        input_id = inputs['input_ids']
        input_mask = inputs['attention_mask']
        input_token = inputs['token_type_ids']

        label = np.zeros((max_len, 2, 2))

        for pro_name in product_names:
            pro_name_id = tokenizer.encode(pro_name)[1:-1]
            pro_name_seat = get_keyword_id_seat(pro_name_id)
            if pro_name_seat != -1:
                label[pro_name_seat, 0, 0] = 1
                label[pro_name_seat+len(pro_name_id)-1, 0, 1] = 1

        for pro_value in product_values:
            pro_value_id = tokenizer.encode(pro_value)[1:-1]
            pro_value_seat = get_keyword_id_seat(pro_value_id)
            if pro_value_seat != -1:
                label[pro_value_seat, 1, 0] = 2
                label[pro_value_seat+len(pro_value_id)-1, 1, 1] = 2

        input_id = sequence_padding([input_id], length=max_len)[0]
        input_mask = sequence_padding([input_mask], length=max_len)[0]
        input_token = sequence_padding([input_token], length=max_len)[0]

        yield input_id, input_mask, input_token, label


def get_text_token(content):
    inputs = tokenizer.encode_plus(content)

    input_id = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_token = inputs['token_type_ids']

    input_id = tf.constant(sequence_padding([input_id], length=max_len), dtype=tf.int32)
    input_mask = tf.constant(sequence_padding([input_mask], length=max_len), dtype=tf.int32)
    input_token = tf.constant(sequence_padding([input_token], length=max_len), dtype=tf.int32)

    return input_id, input_mask, input_token



class BertCrf4Ie(tf.keras.Model):
    def __init__(self, output_dim):
        super(BertCrf4Ie, self).__init__(output_dim)
        self.output_dim = output_dim

        self.bert = TFBertModel.from_pretrained('uer/chinese_roberta_L-12_H-768', output_hidden_states=True)

        self.dropout = tf.keras.layers.Dropout(0.3)

        self.dense = tf.keras.layers.Dense(2 * self.output_dim)


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len], name='input_id', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='input_mask', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='input_token', dtype=tf.int32))])
    def call(self, batch_data):

        input_ids, masks, tokens = batch_data

        hidden = self.bert(input_ids, masks, tokens)

        dropout_inputs = self.dropout(hidden[0], 1)

        predict = self.dense(dropout_inputs)

        # predict = tf.keras.layers.Lambda(lambda x: x**2)(predict)

        predict = tf.keras.layers.Reshape((-1, self.output_dim, 2))(predict)

        return predict


def loss4value(t, p, mask):

    loss_value = tf.keras.losses.binary_crossentropy(y_true=t, y_pred=p)

    loss_value = tf.cast(loss_value, dtype=tf.float32)
    loss_value = tf.reduce_sum(loss_value, axis=2)

    mask = tf.cast(mask, dtype=tf.float32)

    loss_value = tf.reduce_sum(loss_value * mask) / tf.reduce_sum(mask)

    return loss_value


def get_str_value(seats, input_id):
    str_values = []

    for seat in seats:
        token_value = np.array(input_id[0])[seat]
        decode_token_value = tokenizer.decode(token_value)
        name_value = ''.join(decode_token_value.split(' '))

        for mask in ['[SEP]', '[PAD]', '[UNK]', '[UNK]']:
            if mask in name_value:
                name_value = name_value.replace(mask, '')

        str_values.append(name_value)

    return str_values


def valid_ie(valid_ds):

    model = tf.saved_model.load('../model_save/model_mid_2/')

    len_lab = 1e-10
    len_pre = 1e-10
    len_pre_is_true = 1e-10

    for ds in valid_ds:
        text = ds['text']

        input_id, input_mask, inputs_token = get_text_token(text)

        predict = model((input_id, input_mask, inputs_token))
        predict = predict[0]

        seat_start = np.where(predict[:, :, 0] > 0.3)
        seat_end = np.where(predict[:, :, 1] > 0.3)

        start_value = seat_start[0]
        start_kind = seat_start[1]
        end_value = seat_end[0]
        end_kind = seat_end[1]

        seats = []
        for num_s, a in enumerate(start_value):
            for num_e, b in enumerate(end_value):
                if start_kind[num_s] == end_kind[num_e] and a <= b:
                    # 逐一匹配, 匹配到最合适的时候直接中止
                    # 考虑到问句关键词较少的情况下可以使用该种方法
                    # 如果关键词再多的话建议根据对应的位置进行匹配
                    # for num, (a, b) in enumerate(zip(obj_start_pos, obj_end_pos)):
                    #     if obj_start_pre[num] == obj_end_pre[num] and a <= b:
                    seat = np.arange(a, b+1)
                    seats.append(seat)
                    continue

        predict_value = get_str_value(seats, input_id)
        label_value = ds['product_names'] + ds['product_values']

        len_lab += len(label_value)
        len_pre += len(predict_value)
        len_pre_is_true += len(set(label_value) & set(predict_value))

    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    return f1_value


def fit_ie():

    model4ie = BertCrf4Ie(2)

    optimizer_bert = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.95, beta_2=0.99)

    f1_value = 1e-10

    valid_dataset = eval(open('../dataset_kbqa_ci/spider_product/valid_dataset.json').read())
    random.sample(valid_dataset, len(valid_dataset))

    def fit_step(inputs, label):

        with tf.GradientTape() as tape:

            predict = model4ie(inputs)

            loss_value = loss4value(label, predict, inputs[1])

            params_bert = []
            params_crf = []

            for var in model4ie.trainable_variables:
                model_name = var.name
                none_bert_layer = ['tf_bert_model/bert/pooler/dense/kernel:0',
                                   'tf_bert_model/bert/pooler/dense/bias:0']
                if model_name in none_bert_layer:
                    continue
                if model_name.startswith('tf_bert_model'):
                    params_bert.append(var)
                else:
                    params_crf.append(var)

        params = tape.gradient(loss_value, params_bert)

        optimizer_bert.apply_gradients(zip(params, params_bert))

        return loss_value, predict

    fit_dataset = tf.data.Dataset.from_generator(
        dataset_generator,
        (tf.int32, tf.int32, tf.int32, tf.int8),
        (tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len, 2, 2]))
    )

    fit_dataset = fit_dataset.batch(batch_size=batch_size)

    for _ in range(5):
        for num, batch_ds in enumerate(fit_dataset):
            batch_inputs = batch_ds[:-1]
            batch_label = batch_ds[-1]
            loss_f, predict_p = fit_step(batch_inputs, batch_label)

            if num % 500 == 0:
                print("loss_f: ", loss_f)

        model4ie.save('../model_save/model_mid_2/')

        f1_score = valid_ie(valid_ds=valid_dataset[:3000])

        print("f1_score: ", f1_score)

        if f1_score > f1_value:
            model4ie.save('../model_save/model_best_2/')
            f1_value = f1_score

fit_ie()


# model = tf.saved_model.load('../model_save/model_mid_2/')
def predict_text(text):

    input_id, input_mask, inputs_token = get_text_token(text)

    label = tf.zeros(shape=(1, max_len), dtype=tf.int8)

    predict, _ = model((input_id, input_mask, inputs_token, label))

    predict = np.array(predict[0])[:len(text)]

    x_seat = np.where(predict == 1)[0]
    y_seat = np.where(predict == 2)[0]

    x_seats = seat_cut(x_seat)
    y_seats = seat_cut(y_seat)

    x_values = get_str_value(x_seats, input_id)
    y_values = get_str_value(y_seats, input_id)

    return x_values, y_values


