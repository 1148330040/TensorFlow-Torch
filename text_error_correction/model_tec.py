# coding:

"""
使用句子维度进行识别效果不错
"""

import os
import json

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa

from xpinyin import Pinyin
from datetime import datetime

from bert4keras.snippets import sequence_padding
from transformers import BertTokenizer, TFBertModel


pinyin = Pinyin()

max_len = 96
batch_size = 24

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

ds_path = 'fit_dataset.json'
fit_ds = eval(open(ds_path).read())


def get_word2id():
    """
    用于获取同音的所有字, 目的是将其添加额input_id的后缀, 将其从一个生成问题变成抽取问题
    """
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


def data_generator_tec():
    for ds in fit_ds[:99999]:

        wrong_text = ds['wrong_text']
        right_words = ds['right_words']
        wrong_words = ds['wrong_words']

        right_words = sum(right_words, [])
        wrong_words = sum(sum(wrong_words, []), [])

        labels_tec = np.zeros(max_len)
        # np.array处理后的编号是从1开始计数, 但是从列表的角度计算则是从0开始计数的
        # 因此需要为列表内的所有数据减1处理
        for r, w in zip(right_words, wrong_words):
            new_wrong_text = wrong_text + '[SEP]'
            if '\u4e00' <= r <= '\u9fff':
                pass
            else:
                continue
            r_py = pinyin.get_pinyin(r)

            r_py_w = w4py[r_py]
            for r_w in r_py_w:
                new_wrong_text += r_w

            inputs = tokenizer.encode_plus(new_wrong_text)

            input_id = inputs['input_ids']
            input_mask = inputs['attention_mask']
            input_token = inputs['token_type_ids']

            right_word_id = tokenizer.encode(r)[1]
            wrong_word_id = tokenizer.encode(w)[1]

            # 考虑到原本的文本中存在目标文字, 因此直接获取最后一个
            try:
                right_word_id_index = [i for i, x in enumerate(input_id) if x == right_word_id][-1]
                wrong_word_id_index = [i for i, x in enumerate(input_id) if x == wrong_word_id][0]
            except:
                continue

            if right_word_id_index > (max_len-2):
                continue

            labels_tec[right_word_id_index] = 1

            input_id = sequence_padding([input_id], max_len)[0]
            input_mask = sequence_padding([input_mask], max_len)[0]
            input_token = sequence_padding([input_token], max_len)[0]

            yield input_id, input_mask, input_token, labels_tec, wrong_word_id_index



def get_text_token(content):
    inputs = tokenizer.encode_plus(content)

    input_id = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_token = inputs['token_type_ids']

    input_id = tf.constant(sequence_padding([input_id], length=max_len), dtype=tf.int32)
    input_mask = tf.constant(sequence_padding([input_mask], length=max_len), dtype=tf.int32)
    input_token = tf.constant(sequence_padding([input_token], length=max_len), dtype=tf.int32)

    return input_id, input_mask, input_token


class BertCrf4Tec(tf.keras.Model):
    """用于获取文本中的错字"""
    def __init__(self, output_dim):
        super(BertCrf4Tec, self).__init__(output_dim)
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

        hidden = self.bert(input_ids, masks, tokens)[0]

        input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)

        dropout_inputs = self.dropout(hidden, 1)

        tec_predict = self.dense(dropout_inputs)

        log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(tec_predict,
                                                                            target,
                                                                            input_seq_len,
                                                                            self.other_params )

        decode_predict, crf_scores = tfa.text.crf_decode(tec_predict, self.other_params , input_seq_len)
        # 使用hidden[2]意思是返回的hidden_states, -2指的是attention层中的倒数第二层(也就是LayerNormal)
        # 因此使用-2层就可以自己对其进行LayerNormalization(苏神构建), 这里不进行自主操作

        return decode_predict, log_likelihood


def get_positions_hidden(positions, bert_hidden):
    """用于提取给定位置处的张量
    """
    weights = tf.ones(shape=(1, 768))
    if bert_hidden.shape[0] == 1:
        weight = tf.gather(bert_hidden, [positions], batch_dims=1)
        weight = tf.reduce_mean(weight, axis=1)
        return weight

    for num, pos in enumerate(positions):
        hidden = tf.expand_dims(bert_hidden[num], 0)
        # 首先为hidden添加一个维度
        # bert_hidden[num]->(max_len, 768)变为(1, max_len, 768)目的是为了提取对应位置信息的张量

        weight = tf.gather(hidden, [pos], batch_dims=1)
        # 提取对应的张量

        weights = tf.concat([weights, weight], axis=0)
        # 将提取后的张量重新合并成一个batch

    trainable_weights = weights[1:]
    # shape: (batch_size, 768)
    return trainable_weights


def valid_tec():
    model_tec = tf.saved_model.load(f'SavedModel/model_tec_1/mid_model/')

    len_lab = 1e-10
    len_pre = 1e-10
    len_pre_is_true = 1e-10

    for num, ds in enumerate(fit_ds):
        if num > 99999:
            wrong_text = ds['wrong_text']

            right_words = ds['right_words']
            wrong_words = ds['wrong_words']

            right_words = sum(right_words, [])
            tec_labels = right_words

            wrong_words = sum(sum(wrong_words, []), [])

            input_ids = []
            input_masks = []
            input_tokens = []
            input_targets = []

            if len(wrong_words) == 0:
                continue

            for w in wrong_words:
                try:
                    input_text = wrong_text
                    if '\u4e00' <= w <= '\u9fff':
                        pass
                    else:
                        continue
                    r_py = pinyin.get_pinyin(w)

                    r_py_w = w4py[r_py]
                    for r_w in r_py_w:
                        input_text += r_w

                    ids, mask, tokens = get_text_token(input_text)

                    target = tf.constant(np.array(tf.zeros(shape=max_len, dtype=tf.int32)), dtype=tf.int32)

                    input_ids.append(ids[0])
                    input_masks.append(mask[0])
                    input_tokens.append(tokens[0])
                    input_targets.append(target)
                except:
                    continue

            input_ids = tf.cast(input_ids, dtype=tf.int32)
            input_masks = tf.cast(input_masks, dtype=tf.int32)
            input_tokens = tf.cast(input_tokens, dtype=tf.int32)
            input_targets = tf.cast(input_targets, dtype=tf.int32)

            predict_tec, log_likelihood = model_tec((input_ids, input_masks, input_tokens, input_targets))

            predict_tec_pos = [list(np.where(p == 1)[0])[:1] for p in predict_tec]
            predict_tec = []
            for input_id, poses in zip(input_ids, predict_tec_pos):
                if len(poses) == 0:
                    continue
                for pos in poses:
                    pos_id = input_id[pos]
                    predict_tec.append(tokenizer.decode(pos_id))

            predict_tec = list(set(predict_tec))

            masks = ["[ S E P ]", "[ U N K ]", "[ C L S ]", "[ M A S K ]", "[ P A D ]"]
            for mask in masks:
                if mask in predict_tec:
                    predict_tec.remove(mask)

            predict_tec_words = list(set(predict_tec))
            labels = list(set(tec_labels))

            len_lab += len(labels)
            len_pre += len(predict_tec_words)
            len_pre_is_true += len(set(labels) & set(predict_tec_words))

    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    return recall



def fit_tec():
    model4tec = BertCrf4Tec(2)

    optimizer_bert = tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.95, beta_2=0.99)
    optimizer_crf = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.95, beta_2=0.99)

    def fit_step(inputs):
        ids, masks, tokens, targets, poses = inputs

        with tf.GradientTape() as tape:

            predict_tec, log_likelihood = model4tec((ids, masks, tokens, targets))

            loss_value_tec = -tf.reduce_mean(log_likelihood)

            params_bert = []
            params_crf = []

            for var in model4tec.trainable_variables:
                model_name = var.name
                none_bert_layer = ['tf_bert_model/bert/pooler/dense/kernel:0',
                                   'tf_bert_model/bert/pooler/dense/bias:0']
                if model_name in none_bert_layer:
                    continue
                if model_name.startswith('tf_bert_model'):
                    params_bert.append(var)
                else:
                    params_crf.append(var)

            # params_crf = model4tecs.trainable_variables

        params = tape.gradient(loss_value_tec, [params_bert, params_crf])
        var_bert = params[0]
        var_crf = params[1]

        optimizer_bert.apply_gradients(zip(var_bert, params_bert))
        optimizer_crf.apply_gradients(zip(var_crf, params_crf))

        return loss_value_tec, predict_tec

    f1 = 1e-10

    train_ds = tf.data.Dataset.from_generator(
        data_generator_tec,
        (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
        (tf.TensorShape([max_len]), tf.TensorShape([max_len]), tf.TensorShape([max_len]),
         tf.TensorShape([max_len]), tf.TensorShape(None))
    )

    train_ds = train_ds.batch(batch_size=batch_size).shuffle(buffer_size=200)


    for i in range(10):
        for num, batch_ds in enumerate(train_ds):
            loss_tec, predict = fit_step(batch_ds)
            if (num + 1) % 100 == 0:
                print(f"time: {datetime.now()}, loss_tec: {loss_tec}")

        model4tec.save('SavedModel/model_tec_1/mid_model/')

        f1_value = valid_tec()

        print(f"old_f1 : {f1}, new_f1: {f1_value}")

        if f1_value > f1:
            f1 = f1_value
            model4tec.save('SavedModel/model_tec_1/best_model/')

fit_tec()

