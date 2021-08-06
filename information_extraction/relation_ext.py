# *- coding: utf-8 -*
# =================================
# time: 2021.7.27
# author: @唐志林
# function: 模型适配
# =================================

import os
import json

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa
from bert4keras.models import build_transformer_model

from transformers import BertTokenizer, TFBertModel

pd.set_option('display.max_columns', None)


tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

relation_ext_path = 'dataset/duie/duie_train.json/duie_train.json'

word_count_min = 1
max_len = 256


def get_dataset(path):
    """将数据处理成:
    {'text': text,  'spo_list': [(s, p, o)]}
    s-subject-实体, p-predicate-关系, o-object-客实体
    """
    dataset = open(path)
    spo_ds = []
    for i in range(100):
        ds = dataset.readline()
        ds = json.loads(ds)

        text = ds['text']
        spo_list = ds['spo_list']

        spo_ds.append({'text': text,
                       'spo_list': spo_list})

    spo_ds = pd.DataFrame(spo_ds)
    return spo_ds


def predicate2seq(dataset):
    spo_predicates = []
    for spo_ls in dataset['spo_list']:
        for spo in spo_ls:
            spo_predicates.append(spo['predicate'])
    spo_predicates = list(set(spo_predicates))

    predicate_vocabs = {word: num for num, word in enumerate(spo_predicates)}

    vocabs = open('dataset/predicate_vocabs.json', 'w')
    vocabs.write(json.dumps(predicate_vocabs))

    return predicate_vocabs


def get_predict_vocab(dataset):
    if os.path.exists('dataset/predicate_vocabs.json'):
        vocabs = json.loads(open('dataset/predicate_vocabs.json').read())
    else:
        vocabs = predicate2seq(dataset)

    p2id = vocabs
    id2p = {value: key for key, value in vocabs.items()}

    return p2id, id2p


spo_dataset = get_dataset(path=relation_ext_path)
predicate2id, id2predicate = get_predict_vocab(spo_dataset)
batch_size = 25


def data_generator(dataset):
    """最终会输出一下内容:
    1: ids, masks, tokens 来自bert的tokenizer
    2: subject_local_mask: subject在text的位置信息标注
    3: object_local_mask: object所处的类型以及对应在text的位置size: (len(predict), len(input_id))
    predict指关系数据的数目
    4: position_predict: subject和object的位置信息以及predict的编码(通俗理解为类型)信息
    """
    ids, masks, tokens = [], [], []
    subject_position_mask, object_position_mask, position_predict = [], [], []

    for _, ds in dataset.iterrows():
        text = ds['text']
        spo_lists = ds['spo_list']

        inputs = tokenizer.encode_plus(text,
                                       add_special_tokens=True,
                                       max_length=max_len,
                                       padding='max_length',
                                       return_token_type_ids=True)

        input_id = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        ids.append(input_id)
        masks.append(attention_mask)
        tokens.append(token_type_ids)

        mask4sub = np.ones((len(text)), dtype=np.int32)
        mask4obj = np.zeros(shape=(len(predicate2id), len(input_id), 2), dtype=np.int32)
        sub_obj_predicate = []
        for spo_list in spo_lists:

            sub = spo_list['subject']
            obj = spo_list['object']['@value']

            predicate = spo_list['predicate']
            predicate = predicate2id[predicate]

            sub_position = (text.index(sub)+1, text.index(sub)+len(sub)+1)
            obj_position = (text.index(obj)+1, text.index(obj)+len(obj)+1)
            # +1的原因是因为token会在text前加一个开始编码101

            sub_obj_predicate.append((sub_position, obj_position, predicate))

            mask4sub = list(mask4sub) + (max_len - len(mask4sub)) * [0]
            mask4sub = np.array(mask4sub)
            mask4sub[sub_position[0]:sub_position[1]] = np.ones((sub_position[1]-sub_position[0])) + 1


            mask4obj[predicate, obj_position[0], 0] = 1
            mask4obj[predicate, obj_position[1], 1] = 1
            # 由于一个s可能对应多个o且关系不同, 因此构建一个多维数组:
            # batch_size (len(predicate2id), len(inputs['input_ids']))
            # 其中行号代表一个关系类型, 比如说'主演'这个关系在predicate2id的编码是4
            # 那么如果预测的object在第四行则表示他们的关系是'主演'
            # 这样的做法是因为考虑一个subject可能对应多个object且关系也不同,因此将所有关系
            # 作为候选, 出现在哪一行则代表关系是哪一种同时一行数据表示编码后的输入,标记在哪一个位置
            # 则该位置对应的词就位object, 通过此方法可以解决:
            # 1: 同时找到predicate和object
            # 2: 解决了一个输入可能存在多组三元组关系数据

            subject_position_mask.append(mask4sub)
            object_position_mask.append(mask4obj)
            position_predict.append(sub_obj_predicate)

        if len(ids) == batch_size or _ == len(dataset):
            yield {
                'ids': tf.constant(ids, dtype=tf.int32),
                'masks': tf.constant(masks, dtype=tf.int32),
                'tokens': tf.constant(tokens, dtype=tf.int32),
                'subject_position_mask': tf.constant(subject_position_mask, dtype=tf.int32),
                'object_position_mask': tf.constant(object_position_mask, dtype=tf.int32),
                'position_predict': np.array(position_predict)
            }
            ids, masks, tokens = [], [], []
            subject_position_mask, object_position_mask, position_predict = [], [], []


# data = data_generator(spo_dataset)
# example_data = next(iter(data))
#
# positions = example_data['position_predict']
# position_start = np.array([[pos[0][0][0]] for pos in positions])
# position_end = np.array([[pos[0][0][1]] for pos in positions])
#
# model = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
# dropout = tf.keras.layers.Dropout(0.3)
# dense = tf.keras.layers.Dense(2)
# other_params = tf.Variable(tf.random.uniform(shape=(2, 2)))
#
# ids = example_data['ids']
# masks = example_data['masks']
# tokens = example_data['tokens']
# target = example_data['subject_position_mask']
#
# input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)

# for var in model.trainable_weights:
#     print(var.name)
# print(model.trainable_weights[-5])

# hidden = model(ids, masks, tokens)[0]
#
#
# hidden_drop = dropout(hidden, 1)
# logistic_seq = dense(hidden_drop)
# logistic_seq = tf.keras.layers.Lambda(lambda x: x**2)(logistic_seq)
#
#
# log_likelihood, other_params = tfa.text.crf.crf_log_likelihood(logistic_seq,
#                                                                target,
#                                                                input_seq_len,
#                                                                other_params)
# decode_predict, _ = tfa.text.crf_decode(logistic_seq, other_params , input_seq_len)


# layer_normal = tf.keras.layers.LayerNormalization()
# dense = tf.keras.layers.Dense(len(predicate2id)*2,
#                               activation='sigmoid',
#                               kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.02))
# weights = model.trainable_weights[-5]
# start = tf.gather(weights, position_start)
# end = tf.gather(weights, position_end)
# new_hidden = tf.keras.layers.concatenate(inputs=[start, end], axis=-1)
# reshape = tf.keras.layers.Reshape((-1, len(predicate2id), 2))
# normal = layer_normal(new_hidden)
# out = dense(normal)
# print(reshape(out))

class BertCrf4Sub(tf.keras.Model):
    """该类用以获取subject
    使用bert&crf模型, 以序列标注的方法进行训练
    """
    def __init__(self, output_dim):
        super(BertCrf4Sub, self).__init__(output_dim)
        # 如果使用embedding需要用到此参数, input_dim=len(vocabs)
        self.output_dim = output_dim

        self.bert = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(self.output_dim)
        self.other_params = tf.Variable(tf.random.uniform(shape=(output_dim, output_dim)))

    @tf.function
    def call(self, batch_data):
        ids, masks, tokens, subject_target = batch_data
        input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)
        hidden = self.bert(ids, masks, tokens)[0]
        dropout_inputs = self.dropout(hidden, 1)
        sub_predict = self.dense(dropout_inputs)
        log_likelihood = None
        if self.use_crf:
            log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(sub_predict,
                                                                                subject_target,
                                                                                input_seq_len,
                                                                                self.other_params)
            sub_predict, _ = tfa.text.crf_decode(sub_predict, self.other_params , input_seq_len)

            # 获取到了预测的subject和对应的crf损失值
            # crf损失值用以迭代优化crf模型
            # return decode_predict, log_likelihood
        # sub_predict: 预测的subject, weights: bert的倒数第5层feed&forward层用于共享编码层, log_like: 用于计算crf层的损失函数
        return sub_predict, self.bert.trainable_weights[-5], log_likelihood


class Bert4Obj(tf.keras.Model):
    """该类用以获取obj
    """
    def __init__(self, output_dim):
        super(Bert4Obj, self).__init__(output_dim)
        self.dense = tf.keras.layers.Dense(output_dim,
                                           activation='sigmoid',
                                           kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.02))
        self.layer_normal = tf.keras.layers.LayerNormalization(conditional=True)


    @tf.function
    def call(self, inputs):
        trainable_weights = inputs
        normal_inputs = self.layer_normal(trainable_weights)
        obj_predict = self.dense(normal_inputs)
        obj_predict = tf.keras.layers.Lambda(lambda x: x**4)(obj_predict)
        obj_predict = tf.keras.layers.Reshape((-1, len(predicate2id), 2))(obj_predict)
        return obj_predict


def concatenate_weights(trainable_weight, positions):
    # todo 处理positions内部一个输入对应多个subject的问题
    position_start, position_end = positions
    trainable_weight_start = tf.gather(trainable_weight, position_start)
    trainable_weight_end = tf.gather(trainable_weight, position_end)
    trainable_weights = tf.keras.layers.concatenate(trainable_weight_start, trainable_weight_end)
    return trainable_weights


def loss4sub_crf(log):
    # todo loss with crf
    loss = -tf.reduce_mean(log)

    return loss


def loss4sub(t, p):
    # todo loss with nan crf
    loss_value = tf.keras.losses.sparse_categorical_crossentropy(y_true=t, y_pred=p)
    loss_value = tf.reduce_mean(loss_value)

    return loss_value



def fit_step(dataset, fit=True):
    dataset = data_generator(dataset)

    model_sub = BertCrf4Sub(2)
    opti_bert_sub = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.95)
    opti_other_sub = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

    def fit_subject(inputs, crf=True):

        weights_bert = []         # bert参数权重获取
        weights_other = []        # crf和其它层参数权重

        with tf.GradientTape() as tape:
            predict, weights, log = model_sub(inputs)

            # todo 计算损失函数
            if crf:
                loss = loss4sub_crf(log)
            else:
                loss = loss4sub(predict, inputs[-1])

            for var in model_sub.trainable_variables:
                model_name = var.name
                none_bert_layer =  ['tf_bert_model/bert/pooler/dense/kernel:0',
                                    'tf_bert_model/bert/pooler/dense/bias:0']

                if model_name in none_bert_layer:
                    pass
                elif model_name.startswith('tf_bert_model'):
                    weights_bert.append(var)
                else:
                    weights_other.append(var)

        params_all = tape.gradient(loss, [weights_bert, weights_other])
        gradients_bert = params_all[0]
        gradients_other = params_all[1]

        opti_other_sub.apply_gradients(zip(gradients_other, weights_other))

        opti_bert_sub.apply_gradients(zip(gradients_bert, weights_bert))


        return predict, weights

    # todo
    def fit_object(inputs):

        return None


    for _, data in enumerate(dataset):
        ids = data['ids']
        masks = data['masks']
        tokens = data['tokens']
        sub_labels = data['subject_position_mask']
        obj_labels = data['object_position_mask']
        positions = data['position_predict']
        predict, weights = fit_subject(inputs=[ids, masks, tokens, sub_labels])
        if fit:
            weights = concatenate_weights(weights, positions)
            fit_object()



