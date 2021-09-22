# *- coding: utf-8 -*
# =================================
# time: 2021.7.27
# author: @唐志林
# function: 模型适配
# =================================

import os
import json
import random

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import recompute_grad
from tensorflow.keras import initializers, activations

from datetime import datetime

from transformers import BertTokenizer, TFBertModel

pd.set_option('display.max_columns', None)

tf.config.experimental_run_functions_eagerly(True)

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

relation_ext_path = 'dataset/duie/duie_train.json/duie_train.json'
valid_ds_path = 'dataset/duie/duie_dev.json/duie_dev.json'
test_ds_path = 'dataset/duie/duie_test2.json/duie_test2.json'

max_len = 128

model_sub_path = f'models_save/model_sub/best_model/'
model_obj_path = f'models_save/model_obj/best_model/'
model_pre_path = f'models_save/model_pre/best_model/'


def get_dataset(path):
    """将数据处理成:
    {'text': text,  'spo_list': [(s, p, o)]}
    s-subject-实体, p-predicate-关系, o-object-客实体
    """
    dataset = open(path)
    spo_ds = []

    for ds in dataset.readlines():
        ds = json.loads(ds)

        text = ds['text']
        if len(text) > max_len - 3:
            continue

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
valid_dataset = get_dataset(path=valid_ds_path)
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
    subject_labels, object_labels, predicate_labels, positions = [], [], [], []
    dataset = dataset.sample(frac=1.0)
    for _, ds in dataset.iterrows():
        text = ''.join(ds['text'])
        spo_lists = ds['spo_list']

        inputs = tokenizer.encode_plus(text,
                                       add_special_tokens=True,
                                       max_length=max_len,
                                       padding='max_length',
                                       return_token_type_ids=True)

        input_id = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        if len(input_id) > max_len:
            continue

        mask4sub = np.zeros((len(text)), dtype=np.int32)
        mask4obj = np.zeros((len(text)), dtype=np.int32)
        mask4pre = np.zeros((len(predicate2id), 2), dtype=np.int32)

        # 此处保存的位置信息仅用于标注obj时随机从多组s-p-o中抽出对应的位置信息
        sub_obj_predicate = {}
        choice_position = []

        for spo_list in spo_lists:

            sub = spo_list['subject']
            obj = spo_list['object']['@value']

            predicate = spo_list['predicate']
            if len(predicate) < 1:
                continue
            predicate = predicate2id[predicate]

            try:
                sub_position = (text.index(sub), text.index(sub) + len(sub))
                obj_position = (text.index(obj), text.index(obj) + len(obj))
            except:
                continue

            # +1的原因是因为token会在text前加一个开始编码101

            if sub_position not in sub_obj_predicate:
                sub_obj_predicate[sub_position] = []
            sub_obj_predicate[sub_position].append((obj_position, predicate))

            # 对于sub, 可以直接将对应在text的位置处标注出来

            mask4sub = list(mask4sub) + (max_len - len(mask4sub)) * [0]

            mask4sub = np.array(mask4sub)

            mask4sub[sub_position[0]:sub_position[1]] = 1

        # 此处的位置信息是用于保存从多组s-p-o中抽取的那一组s-p-o的位置信息
        # 该位置信息后续将会用于同bert的encoder编码向量交互用于预测obj和predicate

        if bool(sub_obj_predicate):
            sub = random.choice(list(sub_obj_predicate.keys()))

            mask4obj = list(mask4obj) + (max_len - len(mask4obj)) * [0]
            mask4obj = np.array(mask4obj)
            for obj_pre in sub_obj_predicate[sub]:
                obj = obj_pre[0]
                mask4pre[obj_pre[1], 1] = 1
                # 将对应的predicate位置处置为1
                mask4obj[obj[0]:obj[1]] = 1
            mask4pre = np.array([[1, 0] if sum(mp)==0 else mp for mp in mask4pre])

            choice_position.append((sub, sub_obj_predicate[sub]))

        if len(choice_position) >= 1:
            subject_labels.append(mask4sub)
            object_labels.append(mask4obj)
            predicate_labels.append(mask4pre)
            positions.append(choice_position)
            ids.append(input_id[:max_len])
            masks.append(attention_mask[:max_len])
            tokens.append(token_type_ids[:max_len])
        else:
            continue

        if len(ids) == batch_size or _ == len(dataset):
            yield {
                'input_ids': tf.constant(ids, dtype=tf.int32),
                'masks': tf.constant(masks, dtype=tf.int32),
                'tokens': tf.constant(tokens, dtype=tf.int32),
                'subject_labels': tf.constant(subject_labels, dtype=tf.int32),
                'object_labels': tf.constant(object_labels, dtype=tf.int32),
                'predicate_labels': tf.constant(predicate_labels),
                'positions': np.array(positions)
            }
            tf.keras.Sequential()
            ids, masks, tokens = [], [], []
            subject_labels, object_labels, predicate_labels, positions = [], [], [], []


class LayerNormalization(tf.keras.layers.Layer):
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """

    def __init__(
            self,
            center=True,
            scale=True,
            epsilon=None,
            conditional=False,
            hidden_units=None,
            hidden_activation='linear',
            hidden_initializer='glorot_uniform',
            **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = mask if mask is not None else []
            masks = [m[None] for m in masks if m is not None]
            if len(masks) == 0:
                return None
            else:
                return tf.keras.backend.all(tf.keras.layers.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = tf.keras.layers.Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = tf.keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = tf.keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )

    @tf.function
    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(tf.keras.backend.ndim(inputs) - tf.keras.backend.ndim(cond)):
                cond = tf.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = tf.keras.backend.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = tf.keras.backend.mean(tf.keras.backend.square(outputs), axis=-1, keepdims=True)
            std = tf.keras.backend.sqrt(variance + self.epsilon)
            outputs = outputs / std * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer':
                initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ModelBertCrf4Sub(tf.keras.Model):
    """该类用以获取subject
    使用bert&crf模型, 以序列标注的方法进行训练
    """

    def __init__(self, output_dim, use_crf, _=None):
        super(ModelBertCrf4Sub, self).__init__(output_dim, use_crf, _)
        self.use_crf = use_crf
        self.output_dim = output_dim
        self.bert = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(self.output_dim)
        self.other_params = tf.Variable(tf.random.uniform(shape=(self.output_dim, self.output_dim)))


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len], name='input_ids', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='masks', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='tokens', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='target', dtype=tf.int32)),])
    def call(self, batch_data):
        input_ids, masks, tokens, target = batch_data

        input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)

        hidden = self.bert(input_ids, masks, tokens)[0]

        dropout_inputs = self.dropout(hidden, 1)

        sub_predict = self.dense(dropout_inputs)

        sub_predict = tf.keras.layers.Lambda(lambda x: x ** 2)(sub_predict)

        if self.use_crf:
            log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(sub_predict,
                                                                                target,
                                                                                input_seq_len,
                                                                                self.other_params)
            decode_predict, crf_scores = tfa.text.crf_decode(sub_predict, self.other_params, input_seq_len)

            return decode_predict, hidden, log_likelihood
        else:
            prob_seq = tf.nn.sigmoid(sub_predict)
            return prob_seq, hidden, None

    def return_crf(self):
        return  self.use_crf


class ModelCrf4Obj(tf.keras.Model):
    """该类用以获取object
    """
    def __init__(self, output_dim, use_crf, _=None):
        super(ModelCrf4Obj, self).__init__(output_dim, use_crf, _)
        self.use_crf = use_crf

        self.dense = tf.keras.layers.Dense(output_dim)

        self.layer_normal = LayerNormalization(conditional=True)

        self.dropout = tf.keras.layers.Dropout(0.1)

        self.other_params = tf.Variable(tf.random.uniform(shape=(output_dim, output_dim)))


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len, 768], name='hidden', dtype=tf.float32),
                                   tf.TensorSpec([None, 768], name='weights_sub', dtype=tf.float32),
                                   tf.TensorSpec([None, max_len], name='obj_target', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='masks', dtype=tf.int32))])
    def call(self, inputs):
        hidden, weights_sub, obj_target, masks = inputs

        input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)

        weights_sub = self.layer_normal([hidden, weights_sub])

        weights_obj = self.dropout(weights_sub, 1)

        obj_predict = self.dense(weights_obj)

        obj_predict = tf.keras.layers.Lambda(lambda x: x**2)(obj_predict)

        if self.use_crf:
            log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(obj_predict,
                                                                                obj_target,
                                                                                input_seq_len,
                                                                                self.other_params)

            obj_predict, _ = tfa.text.crf_decode(obj_predict, self.other_params, input_seq_len)

            return obj_predict, log_likelihood
        else:
            obj_predict = tf.nn.sigmoid(obj_predict)
            return obj_predict, None

    def return_crf(self):
        return self.use_crf


class Model4Pre(tf.keras.Model):
    """该类用以获取predicate
    如果predicate是固定的话，那么可以使用分类模型
    如果predicate不是固定的话，那么可以使用seq2seq文本生成模型
    """
    def __init__(self, predicate_sorts):
        super(Model4Pre, self).__init__(predicate_sorts)

        # self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=predicate_sorts))

        self.dense = tf.keras.layers.Dense(units=predicate_sorts * 2,
                                           activation='sigmoid')

        self.lambdas = tf.keras.layers.Lambda(lambda x: x[:, 0])
        # 为了做分类问题需要将shape(batch_size, max_len, 768)->(batch_size, 768)
        # 由于attention的缘故不需要对(max_len, 768)做操作直接取768就可以, 因此使用lambda x: x[:, 0]即可
        self.dropout = tf.keras.layers.Dropout(0.1)

        self.reshape = tf.keras.layers.Reshape((-1, predicate_sorts, 2))

        self.layer_normal = LayerNormalization(conditional=True)


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len, 768], name='hidden', dtype=tf.float32),
                                   tf.TensorSpec([None, 1536], name='trainable_weights', dtype=tf.float32))])
    def call(self, inputs):
        hidden, weights_sub_obj = inputs
        # 此处trainable_weights包含(weights_sub, weights_obj)

        weights_sub_obj = self.layer_normal([hidden, weights_sub_obj])

        weights_sub_obj = self.dropout(weights_sub_obj, 1)

        weights_sub_obj = self.lambdas(weights_sub_obj)

        predict = self.dense(weights_sub_obj)

        predict = tf.keras.layers.Lambda(lambda x: x**4)(predict)

        predict = self.reshape(predict)

        predict = tf.squeeze(predict)

        # predict = tf.nn.softmax(predict)

        return predict



model_sub = ModelBertCrf4Sub(output_dim=2, use_crf=True)
opti_bert_sub = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.95)
opti_other_sub = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

model_obj = ModelCrf4Obj(output_dim=2, use_crf=True)
opti_obj = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

# 此处predicate种类是确定的因此作为分类模型考虑
model_pre = Model4Pre(len(predicate2id))
opti_pre = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.95)


def loss4sub_obj_crf(log):
    loss = -tf.reduce_mean(log)

    return loss


def loss4sub_obj(t, p):
    loss_value = tf.keras.losses.sparse_categorical_crossentropy(y_true=t, y_pred=p)
    loss_value = tf.reduce_mean(loss_value)

    return loss_value


def loss4pre(t, p):
    # p = tf.round(p)
    loss_value = tf.keras.losses.binary_crossentropy(y_true=t, y_pred=p)
    loss_value = tf.reduce_mean(loss_value)
    return loss_value


def get_positions_hidden(positions, bert_hidden):
    """用于提取给定位置处的张量
    """

    weights = tf.ones(shape=(1, 768))

    if bert_hidden.shape[0] == 1:
        pos = (np.arange(positions[0], positions[1] + 1))
        weight = tf.gather(bert_hidden, [pos], batch_dims=1)
        weight = tf.reduce_mean(weight, axis=1)

        return weight

    for num, pos in enumerate(positions):
        hidden = tf.expand_dims(bert_hidden[num], 0)
        # 首先为hidden添加一个维度
        # bert_hidden[num]->(128, 768)变为(1, 128, 768)目的是为了提取对应位置信息的张量
        pos = (np.arange(pos[0], pos[1] + 1))
        # 将起始位置张量变为一个range

        weight = tf.gather(hidden, [pos], batch_dims=1)
        # 提取对应的张量

        weight = tf.reduce_mean(weight, axis=1)
        # 去掉增加的维度

        weights = tf.concat([weights, weight], axis=0)
        # 将提取后的张量重新合并成一个batch

    trainable_weights = weights[1:]
    # shape: (batch_size, 768)


    return trainable_weights


def concatenate_weights_sub(bert_encoder_hidden, positions):
    # 将subject对应位置的起始信息张量拼接起来

    trainable_weights = get_positions_hidden(positions, bert_encoder_hidden)

    return trainable_weights


def concatenate_weights_sub_obj(bert_encoder_hidden, positions):
    positions_sub, positions_obj = positions

    trainable_weights_sub = get_positions_hidden(positions_sub, bert_encoder_hidden)
    trainable_weights_obj = get_positions_hidden(positions_obj, bert_encoder_hidden)

    trainable_weights = tf.keras.layers.concatenate([trainable_weights_sub, trainable_weights_obj])
    # shape: (batch_size, 768 * 2)

    return trainable_weights


def get_predict_data(content):
    """将输入的问句初步token处理
    """
    inputs = tokenizer.encode_plus(content,
                                   add_special_tokens=True,
                                   max_length=max_len,
                                   padding='max_length',
                                   return_token_type_ids=True)

    input_id = tf.constant([inputs['input_ids'][:128]], dtype=tf.int32)
    input_mask = tf.constant([inputs['attention_mask'][:128]], dtype=tf.int32)
    token_type_ids = tf.constant([inputs["token_type_ids"][:128]], dtype=tf.int32)

    label = tf.constant([max_len * [0]], dtype=tf.int32)

    return input_id, input_mask, token_type_ids, label


def predict(content):
    """预测函数, 步骤如下：
    1: 预测subject
    2: 获取预测到的subject对应的位置信息, 并以此拆分判断预测到了几个subject
    3: 按照循环的方式针对每一个subject进行object的预测
    4: 步骤同上获取object位置信息, 拆分, 按照每一组(subject, object)预测predicate, 并放入到spo列表中
    5: 整理spo列表传递到evaluate函数中"""

    input_id, input_mask, token_type_ids, label = get_predict_data(content=content)

    # 需要以全局变量的形式出现

    def get_positions(value):
        position = []
        positions = []

        value = np.where(value[0] > 0)[0]
        for i in value:
            if len(position) == 0:
                position.append(i)
                continue
            if i - 1 == position[-1]:
                position.append(i)
            else:
                positions.append(position)
                position = [i]
            if i == value[-1]:
                positions.append(position)

        positions = [(pos[0], pos[-1]) for pos in positions]

        return positions

    predict_sub, bert_hidden, _ = model_sub.call((input_id, input_mask, token_type_ids, label))

    positions_sub = get_positions(predict_sub)

    all_spo = []

    if len(positions_sub) > 0:
        for pos_sub in positions_sub:
            obj_weights = concatenate_weights_sub(bert_encoder_hidden=bert_hidden,
                                                  positions=pos_sub)
            predict_obj, _ = model_obj.call((bert_hidden, obj_weights, label, input_mask))

            positions_obj = get_positions(predict_obj)

            if len(positions_obj) == 0:
                continue

            for pos_obj in positions_obj:
                pos_sub_obj = [pos_sub, pos_obj]

                pre_weights = concatenate_weights_sub_obj(bert_encoder_hidden=bert_hidden,
                                                          positions=pos_sub_obj)

                predict_pre = model_pre.call((bert_hidden, pre_weights))
                predicate = tf.argmax(predict_pre, axis=1)

                predicate = np.where(predicate > 0)[0]
                if len(predicate) == 0:
                    continue

                predicate = predicate[0]
                if pos_sub[-1] >= 127 or pos_obj[-1] >= 127:
                    all_spo.append(([]))
                else:
                    all_spo.append(([pos_sub[0], pos_sub[1]],
                                    predicate,
                                    [pos_obj[0], pos_obj[1]]))

        all_spo = [(content[spo[0][0]:(spo[0][1] + 1)],
                    id2predicate[spo[1]],
                    content[spo[2][0]:(spo[2][1] + 1)]) if len(spo) > 0 else () for spo in all_spo]

    return all_spo


def evaluate():
    """"""
    len_lab = 1e-10
    len_pre = 1e-10
    len_pre_is_true = 1e-10
    num = 0
    val_dataset = valid_dataset

    for _, ds in val_dataset.iterrows():
        text = ds['text']

        spo_t = ds['spo_list']
        spo_labels = set([(spo['subject'], spo['predicate'], spo['object']['@value']) for spo in spo_t])
        spo_predicts = set(predict(content=text))
        len_lab += len(spo_labels)
        len_pre += len(spo_predicts)
        len_pre_is_true += len(spo_labels & spo_predicts)
        num += 1
        if num % 200==0:
            print(spo_predicts)
            print(spo_labels)
        # & 该运算的作用是返回两个输入内部相同的值
        # a: [(1, 2), (3, 4), (5, 6)],  b: [(3, 4), (1, 2), (6, 5)]
        # 返回 [(1, 2), (3, 4)]

    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    return f1_value, precision, recall


def fit_step():

    def fit_models(inputs_model, inputs_labels, inputs_other):
        positions_sub, positions_obj = inputs_other
        sub_label, obj_label, pre_label = inputs_labels
        ids, masks, tokens = inputs_model
        weights_bert = []  # bert参数权重获取
        weights_other = []  # crf和其它层参数权重

        with tf.GradientTape() as tape:
            predict_sub, bert_hidden, logs_sub = model_sub(batch_data=(ids, masks, tokens, sub_label))

            obj_weights = concatenate_weights_sub(bert_encoder_hidden=bert_hidden,
                                                  positions=positions_sub)

            predict_obj, logs_obj = model_obj(inputs=(bert_hidden, obj_weights, obj_label, masks))

            pre_weights = concatenate_weights_sub_obj(bert_encoder_hidden=bert_hidden,
                                                      positions=[positions_sub, positions_obj])

            predict_pre = model_pre(inputs=(bert_hidden, pre_weights))

            if model_sub.return_crf():
                loss_sub = loss4sub_obj_crf(logs_sub)
            else:
                loss_sub = loss4sub_obj(t=sub_label, p=predict_sub)

            if model_obj.return_crf():
                loss_obj = loss4sub_obj_crf(logs_obj)
            else:
                loss_obj = loss4sub_obj(t=obj_label, p=predict_obj)

            loss_pre = loss4pre(t=pre_label, p=predict_pre)

            loss = loss_obj + loss_sub + loss_pre

            for var in model_sub.trainable_variables:
                model_name = var.name
                none_bert_layer = ['tf_bert_model/bert/pooler/dense/kernel:0',
                                   'tf_bert_model/bert/pooler/dense/bias:0']

                if model_name in none_bert_layer:
                    pass
                elif model_name.startswith('tf_bert_model'):
                    weights_bert.append(var)
                else:
                    weights_other.append(var)

            weights_obj = model_obj.trainable_variables
            weights_pre = model_pre.trainable_variables

        params_all = tape.gradient(loss, [weights_bert, weights_other, weights_obj, weights_pre])

        params_bert_sub = params_all[0]
        params_other_sub = params_all[1]
        params_obj = params_all[2]
        params_pre = params_all[3]

        opti_bert_sub.apply_gradients(zip(params_bert_sub, weights_bert))
        opti_other_sub.apply_gradients(zip(params_other_sub, weights_other))

        opti_obj.apply_gradients(zip(params_obj, weights_obj))

        opti_pre.apply_gradients(zip(params_pre, weights_pre))

        # predict: 预测获取的sub, weights: inputs的bert编码层feed&forward层向量
        return predict_sub, predict_obj, predict_pre, loss_sub, loss_obj, loss_pre

    f1_value_mid, precision_mid, recall_mid = 0.370280600354258, 0.4655414908579591, 0.30738275808699417

    for num in range(2, 15):
        dataset = data_generator(spo_dataset)
        for _, data in enumerate(dataset):
            ids = data['input_ids']
            masks = data['masks']
            tokens = data['tokens']
            sub_labels = data['subject_labels']
            obj_labels = data['object_labels']
            pre_labels = data['predicate_labels']
            positions = data['positions']

            sub_positions = [ps[0][0] for ps in positions]  # subject的位置信息
            obj_predicate = [ps[0][1] for ps in positions]  # object和predicate的信息
            obj_positions = [ps[0][0] for ps in obj_predicate]  # object的位置信息

            # 位置信息: 用于在预测object和predicate时同bert-encoder编码信息交互

            p_sub, p_obj, p_pre, loss_sub, loss_obj, loss_pre = fit_models(
                inputs_model=[ids, masks, tokens],
                inputs_labels=[sub_labels, obj_labels, pre_labels],
                inputs_other=[sub_positions, obj_positions]
            )

            if _ % 500 == 0:
                with open('models_save/fit_logs.txt', 'a') as f:
                    # 'a'  要求写入字符
                    # 'wb' 要求写入字节(str.encode(str))
                    log = f"times: {datetime.now()}, " \
                          f"num: {_}, sub_loss: {loss_sub}, loss_obj: {loss_obj}, loss_pre: {loss_pre}, \n" \
                          f"sub_labels[:1]: {sub_labels[:1]}, \n sub_predict[:1]: {p_sub[:1]}, \n" \
                          f"obj_labels[:1]: {obj_labels[:1]}, \n obj_predict[:1]: {p_obj[:1]}, \n" \
                          f"pre_labels[:3]: {[tf.argmax(i, axis=1) for i in pre_labels[:3]]}, \n " \
                          f"pre_predict[:3]: {[tf.argmax(i, axis=1) for i in p_pre[:3]]}  \n"
                    f.write(log)
                print(f"times: {datetime.now()}, num: {_}, sub_loss: {loss_sub}, loss_obj: {loss_obj}, loss_pre: {loss_pre}")
                print(f"pre_predict: {tf.argmax(p_pre[0], axis=1)}")
                print(f"pre_labels : {tf.argmax(pre_labels[0], axis=1)}")

        model_sub.save(model_sub_path+'current/' + str(num) + '/n')
        model_obj.save(model_obj_path+'current/' + str(num) + '/n')
        model_pre.save(model_pre_path+'current/' + str(num) + '/n')

        f1_value, precision, recall = evaluate()

        with open('models_save/fit_logs.txt', 'a') as f:
            f.write(f"第{num}轮训练结束后, f1_value: {f1_value}, precision_value: {precision}, recall: {recall}")

        if f1_value > f1_value_mid:
            f1_value_mid = f1_value
            model_sub.save(model_sub_path)
            model_obj.save(model_obj_path)
            model_pre.save(model_pre_path)
            with open('models_save/fit_logs.txt', 'a') as f:
                f.write(f"当前最优模型在第{num}轮保存！")

# fit_step()





