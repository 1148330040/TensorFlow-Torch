# coding:gbk
import heapq
import os
import json
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa

from xpinyin import Pinyin
from Pinyin2Hanzi import dag
from Pinyin2Hanzi import DefaultDagParams

from bert4keras.snippets import sequence_padding
from transformers import BertTokenizer, TFBertModel


dagparams = DefaultDagParams()
pinyin = Pinyin()

max_len = 48
batch_size = 32
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
print(tokenizer.vocab_size)

def data_process():
    ds_path = 'industry.json'
    industry_ds = eval(open(ds_path).read())

    fit_ds = []

    def get_text2pinyin(names):
        # 获取文本拼音
        names_pinyin = []
        for name in names:
            names_pinyin.append(pinyin.get_pinyin(name).split('-'))
        return names_pinyin

    def get_pinyin2text(names):
        # 获取同音词
        names_text = []
        for name in names:
            result = dag(dagparams, name, path_num=1)
            if len(result) == 0:
                names_text.append([])
            else:
                names_text.append(result[0].path)
        return names_text

    for ds in industry_ds:

        text = ds['text']
        ds_names = ds['product_names']
        ds_values = ds['product_values']

        mid_ds = {'text': text}

        ds_names = ds_names + ds_values

        new_ds_names = []
        right_words = []
        wrong_words = []

        for ds_name in ds_names:
            if len(ds_name) == 0:
                continue
            num = 1 if len(ds_name) // random.choice([2, 3, 4]) == 0 else len(ds_name) // random.choice([2, 3, 4])
            words = random.sample(ds_name, num)

            right_words.append(words)

            new_words = get_text2pinyin(words)
            new_words = get_pinyin2text(new_words)

            wrong_words.append(new_words)

            if len(new_words) == 0:
                new_ds_names.append([])
                continue

            for word, new_word in zip(words, new_words):
                if len(new_word) == 0:
                    continue
                ds_name = ds_name.replace(word, new_word[0])
            new_ds_names.append(ds_name)

        for ds_name, new_ds_name in zip(ds_names, new_ds_names):
            if len(new_ds_name) == 0:
                continue
            text = text.replace(ds_name, new_ds_name)

        mid_ds['wrong_text'] = text
        mid_ds['right_words'] = right_words
        mid_ds['wrong_words'] = wrong_words
        fit_ds.append(mid_ds)

    with open('fit_dataset.json', 'w') as f:
        f.write(json.dumps(fit_ds, ensure_ascii=False))


ds_path = 'fit_dataset.json'
fit_ds = eval(open(ds_path).read())


def get_word2id():

    if os.path.exists('word2id.json'):
        word2id = json.loads(open('word2id.json').read())
    else:
        all_word = []
        for ds in fit_ds:
            right_text = list(set(ds['text']))
            wrong_text = list(set(ds['wrong_text']))
            all_word = list(set(all_word + right_text + wrong_text))

        word2id = {word:num for num, word in enumerate(all_word)}
        vocabs = open('word2id.json', 'w')
        vocabs.write(json.dumps(word2id))

    id2word = dict(zip(word2id.values(), word2id.keys()))
    word4py = {}

    for w in word2id.keys():
        if '\u4e00' <= w <= '\u9fff':
            py = pinyin.get_pinyin(w)
            if py in word4py:
                word4py[py].append(w)
            else:
                word4py[py] = [w]

    return word2id, id2word, word4py

w2id, id2w, w4py = get_word2id()


def data_generator_fw():
    for ds in fit_ds[:42000]:

        right_text = ds['text']
        wrong_text = ds['wrong_text']

        inputs = tokenizer.encode_plus(wrong_text)

        input_id = inputs['input_ids']
        input_mask = inputs['attention_mask']
        input_token = inputs['token_type_ids']

        right_id = tokenizer.encode(right_text)

        labels_ie = np.zeros(max_len)

        wrong_positions = np.where((np.array(input_id) == np.array(right_id)) == False)[0]
        wrong_positions = [p-1 for p in wrong_positions]
        # np.array处理后的编号是从1开始计数, 但是从列表的角度计算则是从0开始计数的
        # 因此需要为列表内的所有数据减1处理
        for p in wrong_positions:
            if p > max_len-2:
                continue
            labels_ie[p] = 1

        input_id = sequence_padding([input_id], max_len)[0]
        input_mask = sequence_padding([input_mask], max_len)[0]
        input_token = sequence_padding([input_token], max_len)[0]

        yield input_id, input_mask, input_token, labels_ie


def get_text_token(content):
    inputs = tokenizer.encode_plus(content)

    input_id = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_token = inputs['token_type_ids']

    input_id = tf.constant(sequence_padding([input_id], length=max_len), dtype=tf.int32)
    input_mask = tf.constant(sequence_padding([input_mask], length=max_len), dtype=tf.int32)
    input_token = tf.constant(sequence_padding([input_token], length=max_len), dtype=tf.int32)

    return input_id, input_mask, input_token


class BertCrf4Fw(tf.keras.Model):
    """用于获取文本中的错字"""
    def __init__(self, output_dim):
        super(BertCrf4Fw, self).__init__(output_dim)
        self.output_dim = output_dim

        self.bert = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext', output_hidden_states=True)

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
        # 使用hidden[2]意思是返回的hidden_states, -2指的是attention层中的倒数第二层(也就是LayerNormal)
        # 因此使用-2层就可以自己对其进行LayerNormalization(苏神构建), 这里不进行自主操作

        return decode_predict, log_likelihood, hidden[2][-1]


class BertCrf4Tec(tf.keras.Model):
    """用于纠错"""
    def __init__(self, output_dim):
        super(BertCrf4Tec, self).__init__(output_dim)
        self.output_dim = output_dim
        self.dense = tf.keras.layers.Dense(units=output_dim * 2, activation='sigmoid')

        self.other_params = tf.Variable(tf.random.uniform(shape=(2, 2)))

    @tf.function(input_signature=[(tf.TensorSpec([None, 768], name='hidden', dtype=tf.float32),
                                   tf.TensorSpec([None, 2], name='target', dtype=tf.float32))])
    def call(self, inputs):
        tec_hidden, target = inputs

        tec_predict = self.dense(tec_hidden)

        tec_predict = tf.keras.layers.Reshape(target_shape=(self.output_dim, 2))(tec_predict)

        return tec_predict


def get_positions_hidden(positions, bert_hidden):
    """用于提取给定位置处的张量
    """
    batch_weights = tf.ones(shape=(1, 768))
    if bert_hidden.shape[0] == 1:
        for pos in positions:
            weight = tf.gather(bert_hidden, [pos], batch_dims=1)
            # 提取对应的张量
            batch_weights = tf.concat([batch_weights, weight], axis=0)

        batch_weights = tf.expand_dims(batch_weights, 0)

        return tf.reduce_mean(batch_weights, 1)

    weights = tf.ones(shape=(1, 768))
    for num, poses in enumerate(positions):
        hidden = tf.expand_dims(bert_hidden[num], 0)
        # 首先为hidden添加一个维度
        # bert_hidden[num]->(x, 768)变为(1, x, 768)目的是为了提取对应位置信息的张量
        for pos in poses:
            if pos == 0:
                # pos=0则表示错字处理完毕, 到了padding部分, 因此张量融合均值
                weights = tf.expand_dims(weights, 0)
                # 添加一个维度然后进行均值处理, 获取所有错字位置hidden的均值
                weights = tf.reduce_mean(weights, axis=1)
                break
            weight = tf.gather(hidden, [pos], batch_dims=1)
            # 提取对应的张量
            weights = tf.concat([weights, weight], axis=0)
            # 将提取后的张量重新合并成一个batch

        batch_weights = tf.concat([batch_weights, weights], axis=0)
        # 将每个input的错字均值重新拼接为一个batch

    trainable_weights = batch_weights[1:]
    # shape: (batch_size, 768)
    return trainable_weights


def valid_fw():
    model_fw = tf.saved_model.load(f'SavedModel/model_fw/mid_model/')

    len_lab = 1e-10
    len_pre = 1e-10
    len_pre_is_true = 1e-10

    for num, ds in enumerate(fit_ds):
        text = ds['wrong_text']

        label_fw = re.findall('[\u4e00-\u9fa5]+', str(ds['wrong_words']))

        predict_fw_words = []

        ids, mask, tokens = get_text_token(text)

        target = tf.constant(np.array([tf.zeros(shape=max_len, dtype=tf.int32)]), dtype=tf.int32)

        predict_fw, _, hidden = model_fw((ids, mask, tokens, target))
        predict_fw = predict_fw[0]

        # 防止预测值标记在padding部分
        predict_fw = predict_fw[:len(text)]

        sites = np.where(predict_fw == 1)[0]

        tec_sites = []

        # 筛选错字的位置(已出现的错字则不放入该列表内)
        for site in sites:
            predict_id = [np.array(ids[0])[site]]

            predict_fw = tokenizer.decode(predict_id)

            if predict_fw not in predict_fw_words:
                tec_sites.append(site)

            predict_fw_words.append(predict_fw)

        predict_fw_words = list(set(predict_fw_words))
        labels = list(set(label_fw))

        len_lab += len(labels)
        len_pre += len(predict_fw_words)
        len_pre_is_true += len(set(labels) & set(predict_fw_words))

    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    return f1_value


def fit_fw_tec():
    model4fw = BertCrf4Fw(2)

    optimizer_bert = tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.95, beta_2=0.99)
    optimizer_crf = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.95, beta_2=0.99)

    def fit_step(inputs):
        fw_ds = inputs[:4]

        with tf.GradientTape() as tape:

            predict_fw, log_likelihood, hidden = model4fw(fw_ds)

            loss_value_fw = -tf.reduce_mean(log_likelihood)

            loss_value = loss_value_fw

            params_bert = []
            params_crf = []

            for var in model4fw.trainable_variables:
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

        return loss_value_fw, predict_fw

    f1 = 1e-10

    train_ds = tf.data.Dataset.from_generator(
        data_generator_fw,
        (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
        (tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len]),
         tf.TensorShape([len(w2id)]), tf.TensorShape([max_len]))
    )

    train_ds = train_ds.batch(batch_size)

    for i in range(10):
        for num, batch_ds in enumerate(train_ds):
            loss_fw, predict_fw = fit_step(batch_ds)
            if (num + 1) % 100 == 0:
                print(f"time: {datetime.now()}, loss_fw: {loss_fw}")

        # model4fw.save(f'SavedModel/model_fw/mid_model/')
        #
        # new_f1_score = valid_fw()
        #
        # print(f"old f1 value: {f1}, new f1 value: {new_f1_score}")
        # if new_f1_score > f1:
        #     f1 = new_f1_score
        #     model4fw.save(f'SavedModel/model_fw/best_model/')

# fit_fw_tec()

# fw_train_ds = tf.data.Dataset.from_generator(
#     data_generator,
#     (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
#     (tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len]),
#      tf.TensorShape([len(w2id)]), tf.TensorShape([max_len]))
# )
#
# fw_train_ds = fw_train_ds.batch(batch_size)

# for i in fw_train_ds:
#     print(i)
