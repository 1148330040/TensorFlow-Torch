import os
import re

import warnings
from datetime import datetime

import jieba
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa
from bert4keras.snippets import sequence_padding

from transformers import BertTokenizer, TFBertModel

import dataset_process as dp

os.environ["CUDA_VISIBLE_DEVICES"]="0"

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-2_H-768")

max_len = 52
batch_size = 30


valid_dataset = pd.read_excel('../dataset/valid.xlsx')
test_dataset = pd.read_excel('../dataset/test.xlsx')


def get_text_token(content):
    inputs = tokenizer.encode_plus(content)

    input_id = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_token = inputs['token_type_ids']

    input_id = tf.constant(sequence_padding([input_id], length=max_len), dtype=tf.int32)
    input_mask = tf.constant(sequence_padding([input_mask], length=max_len), dtype=tf.int32)
    input_token = tf.constant(sequence_padding([input_token], length=max_len), dtype=tf.int32)

    return input_id, input_mask, input_token



class BertCrf4SPO(tf.keras.Model):
    def __init__(self, output_dim):
        super(BertCrf4SPO, self).__init__(output_dim)
        self.output_dim = output_dim

        self.bert = TFBertModel.from_pretrained('uer/chinese_roberta_L-12_H-768', output_hidden_states=True)

        self.dropout = tf.keras.layers.Dropout(0.3)

        self.dense = tf.keras.layers.Dense(self.output_dim)

        self.other_params = tf.Variable(tf.random.uniform(shape=(output_dim, output_dim)))


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len], name='input_ids', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='masks', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='tokens', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='target', dtype=tf.int32))])
    def call(self, batch_data):

        input_ids, masks, tokens, target = batch_data

        hidden = self.bert(input_ids, masks, tokens)

        input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)

        dropout_inputs = self.dropout(hidden[0], 1)

        trigger_predict = self.dense(dropout_inputs)

        log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(trigger_predict,
                                                                            target,
                                                                            input_seq_len,
                                                                            self.other_params )

        decode_predict, crf_scores = tfa.text.crf_decode(trigger_predict, self.other_params , input_seq_len)

        return decode_predict, log_likelihood


def valid():
    model = tf.saved_model.load(f'../model4spo_1/save_model_mid/')
    special_mark = ['x', 'y', 'cvt', 'time', 'distance', 'price']

    len_lab = 1e-10
    len_pre = 1e-10
    len_pre_is_true = 1e-10

    for _, ds in valid_dataset.iterrows():
        text = ds['content']
        text = text.replace('《', '')
        text = text.replace('》', '')

        all_spo = ds['spo']

        input_id, input_mask, input_token = get_text_token(text)

        labels = []
        predicts = []

        for spo in all_spo:
            for w in spo:
                if w not in special_mark:
                    labels.append(w)


        target = tf.constant(np.array([tf.zeros(shape=max_len, dtype=tf.int32)]), dtype=tf.int32)

        predict, _ = model((input_id, input_mask, input_token, target))
        value = np.array(predict)[0][:len(text)]
        for num in [1, 2, 3]:
            site = np.where(value == num)[0]
            if len(site) > 0:
                word = ''.join(tokenizer.decode(np.array(input_id[0])[site]).split(' '))
                for mask in ['[SEP]', '[PAD]', '[UNK]', '[UNK]']:
                    if mask in word:
                        word.replace(mask, '')
                if len(word) > 0 :
                    predicts.append(word)
        len_lab += len(labels)
        len_pre += len(predicts)
        len_pre_is_true += len(set(labels) & set(predicts))

    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    return f1_value


def spo_cut(value):
    """用于将获取的数据切割分配
    比如: s的位置信息[1, 2, 3, 6, 7, 8, 13, 14]
    需要将其变为[[1, 2, 3], [6, 7, 8], [13, 14]]
    """
    cut_all = []
    cut = [value[0]]
    for i in value[1:]:
        if i-1 == cut[-1]:
            cut.append(i)
        else:
            cut_all.append(cut)
            cut = [i]

    cut_all.append(cut)

    return cut_all


def fit():
    model4spo = BertCrf4SPO(4)

    optimizer_bert = tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.95, beta_2=0.99)
    optimizer_crf = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.95, beta_2=0.99)

    def fit_step(inputs):

        with tf.GradientTape() as tape:

            predict, log_likelihood = model4spo(inputs)

            loss_value = -tf.reduce_mean(log_likelihood)

            params_bert = []
            params_crf = []

            for var in model4spo.trainable_variables:
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

    f1_score = 1e-10

    train_ds = tf.data.Dataset.from_generator(
        dp.data_generator(),
        (tf.int32, tf.int32, tf.int32, tf.int32),
        (tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len]))
    )

    train_ds = train_ds.shuffle(buffer_size=4000).batch(batch_size=batch_size)

    for i in range(10):
        for num, batch_ds in enumerate(train_ds):
            loss_v, predict = fit_step(batch_ds)
            target = batch_ds[-1]
            if (num + 1) % 20 == 0:
                print(f"time: {datetime.now()}, loss: {loss_v}, target: {target[0]}, predict: {predict[0]}")

        model4spo.save(f'../model4spo_1/save_model_mid/')

        new_f1_score = valid()

        print(f"old f1 value: {f1_score}, new f1 value: {new_f1_score}")
        if new_f1_score > f1_score:
            f1_score = new_f1_score
            model4spo.save('../model4spo_1/save_model_best/')


model = tf.saved_model.load('../model4spo_1/save_model_best/')

def predict(content):

    input_id, input_mask, input_token = get_text_token(content)

    none_labels = tf.constant(np.array([tf.zeros(shape=max_len, dtype=tf.int32)]), dtype=tf.int32)

    predict_value, _ = model((input_id, input_mask, input_token, none_labels))
    predict_value = np.array(predict_value[0])[:len(content)]
    s = []
    p = []
    o = []

    break_num = 0

    for num in [1, 2, 3]:
        # 获取spo对应的位置信息
        sites = np.where(predict_value == num)[0]
        if len(sites) > 0:
            # 拆分spo

            sites = spo_cut(sites)

            for site in sites:
                word = ''.join(tokenizer.decode(np.array(input_id[0])[site]).split(' '))
                for mask in ['[SEP]', '[PAD]', '[UNK]', '[UNK]']:
                    if mask in word:
                        word = word.replace(mask, '')

                if num == 1:
                    s.append(word)
                if num == 2:
                    o.append(word)
                if num == 3:
                    p.append(word)
        else:
            break_num += 1
            if break_num == 3:
                return None

    return [s, p, o]
