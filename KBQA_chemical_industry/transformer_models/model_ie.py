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

os.environ["CUDA_VISIBLE_DEVICES"]="0"

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-12_H-768")

max_len = 64
batch_size = 50

# EINECS号: 欧洲化学管理局编码
# InChI: 国际化合物标识
# CAS号: 美国物质编号

def dataset_generator():

    train_dataset = eval(open('../dataset_kbqa_ci/spider_product/fit_dataset.json').read())

    random.sample(train_dataset, len(train_dataset))


    def get_keyword_id_seat(keyword_id):
        len_keyword_id = len(keyword_id)
        for i in range(len(input_id)):
            if input_id[i:(i+len_keyword_id)] == keyword_id:
                return i
        return -1


    for ds in train_dataset:
        text = ds['text']
        product_names = ds['product_names']
        product_values = ds['product_values']

        inputs = tokenizer.encode_plus(text)

        input_id = inputs['input_ids']
        input_mask = inputs['attention_mask']
        input_token = inputs['token_type_ids']

        label = np.zeros(max_len)

        for pro_name in product_names:
            pro_name_id = tokenizer.encode(pro_name)[1:-1]
            pro_name_seat = get_keyword_id_seat(pro_name_id)
            if pro_name_seat != -1:
                label[pro_name_seat: (pro_name_seat+len(pro_name_id))] = 1

        for pro_value in product_values:
            pro_value_id = tokenizer.encode(pro_value)[1:-1]
            pro_value_seat = get_keyword_id_seat(pro_value_id)
            if pro_value_seat != -1:
                label[pro_value_seat: (pro_value_seat+len(pro_value_id))] = 2

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

        self.dense = tf.keras.layers.Dense(self.output_dim)

        self.other_params = tf.Variable(tf.random.uniform(shape=(output_dim, output_dim)))


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len], name='input_id', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='input_mask', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='input_token', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='label', dtype=tf.int8))])
    def call(self, batch_data):

        input_ids, masks, tokens, target = batch_data

        hidden = self.bert(input_ids, masks, tokens)
        print("hidden: ", hidden)

        input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)

        dropout_inputs = self.dropout(hidden[0], 1)
        print("dropout_inputs: ", dropout_inputs)
        trigger_predict = self.dense(dropout_inputs)
        print("trigger_predict: ", trigger_predict)
        log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(trigger_predict,
                                                                            target,
                                                                            input_seq_len,
                                                                            self.other_params )

        decode_predict, crf_scores = tfa.text.crf_decode(trigger_predict, self.other_params , input_seq_len)

        return decode_predict, log_likelihood


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


def seat_cut(value):
    """用于将获取的数据切割分配
    比如: x的位置信息[1, 2, 3, 6, 7, 8, 13, 14]
    需要将其变为[[1, 2, 3], [6, 7, 8], [13, 14]]
    """
    cut_all = []
    cut = [value[0]]
    for i in value[1:]:
        if i - 1 == cut[-1]:
            cut.append(i)
        else:
            cut_all.append(cut)
            cut = [i]

    cut_all.append(cut)

    return cut_all


def valid_ie(valid_ds):

    model = tf.saved_model.load('../model_save/model_mid_robert/')

    len_lab = 1e-10
    len_pre = 1e-10
    len_pre_is_true = 1e-10

    for ds in valid_ds[:2000]:
        text = ds['text']

        input_id, input_mask, inputs_token = get_text_token(text)

        label = tf.zeros(shape=(1, max_len), dtype=tf.int8)

        predict, _ = model((input_id, input_mask, inputs_token, label))

        predict = np.array(predict[0])[:len(text)]

        xy_seats = np.where(predict > 0)[0]
        xy_seats = seat_cut(xy_seats)

        predict_value = get_str_value(xy_seats, input_id)

        label_value = ds['product_names'] + ds['product_values']

        len_lab += len(label_value)
        len_pre += len(predict_value)
        len_pre_is_true += len(set(label_value) & set(predict_value))

    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    return f1_value



def fit_ie():

    model4ie = BertCrf4Ie(3)

    optimizer_bert = tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.95, beta_2=0.99)
    optimizer_crf = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.95, beta_2=0.99)

    f1_value = 1e-10

    valid_dataset = eval(open('../dataset_kbqa_ci/spider_product/valid_dataset.json').read())
    # random.sample(valid_dataset, len(valid_dataset))

    def fit_step(inputs):

        with tf.GradientTape() as tape:

            predict, log_likelihood = model4ie(inputs)

            loss_value = -tf.reduce_mean(log_likelihood)

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

        params = tape.gradient(loss_value, [params_bert, params_crf])
        var_bert = params[0]
        var_crf = params[1]

        optimizer_bert.apply_gradients(zip(var_bert, params_bert))
        optimizer_crf.apply_gradients(zip(var_crf, params_crf))

        return loss_value, predict

    fit_dataset = tf.data.Dataset.from_generator(
        dataset_generator,
        (tf.int32, tf.int32, tf.int32, tf.int8),
        (tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len]))
    )

    fit_dataset = fit_dataset.batch(batch_size=batch_size)

    for _ in range(1):
        for num, batch_ds in enumerate(fit_dataset):

            loss_f, predict_p = fit_step(batch_ds)
            if num % 600 == 0:
                print("loss_f: ", loss_f)
                print("predict_p: ", predict_p[0])
                print("batch_ds[-1]: ", batch_ds[-1][0])

        model4ie.save('../model_save/model_mid_robert/')

        f1_score = valid_ie(valid_ds=valid_dataset[:3000])
        print("f1_score: ", f1_score)

        if f1_score > f1_value:
            model4ie.save('../model_save/model_best_robert/')
            f1_value = f1_score

fit_ie()
# f1_score:  0.8225995941471789   best bert
# f1_score:  0.8224738303781282   best robert


# model = tf.saved_model.load('../model_save/model_mid/')
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


