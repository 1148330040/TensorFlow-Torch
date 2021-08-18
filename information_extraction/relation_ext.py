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
# from bert4keras.layers import LayerNormalization
# from bert4keras.models import build_transformer_model
# from tensorflow import recompute_grad
from tensorflow import recompute_grad
from tensorflow.keras import initializers, activations

from datetime import datetime
from transformers import BertTokenizer, TFBertModel

pd.set_option('display.max_columns', None)

tf.config.experimental_run_functions_eagerly(True)

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

relation_ext_path = 'dataset/duie/duie_train.json/duie_train.json'

word_count_min = 1
max_len = 128


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
        if len(text) > 126:
            continue

        spo_list = ds['spo_list']

        spo_ds.append({'text': text,
                       'spo_list': spo_list})

    spo_ds = pd.DataFrame(spo_ds)
    return spo_ds



get_dataset(relation_ext_path)


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
batch_size = 10


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
                sub_position = (text.index(sub), text.index(sub)+len(sub))
                obj_position = (text.index(obj), text.index(obj)+len(obj))
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

            choice_position.append((sub, sub_obj_predicate[sub]))

        if len(choice_position) >= 1:
            subject_labels.append(mask4sub)
            object_labels.append(mask4obj)
            predicate_labels.append(mask4pre)
            positions.append(choice_position)
            ids.append(input_id)
            masks.append(attention_mask)
            tokens.append(token_type_ids)
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


data = data_generator(spo_dataset)
# example_data = next(iter(data))
# positions = example_data['positions']
#
# # print(example_data['object_position_mask'])
# # print(example_data['object_position_mask'].shape)
# sub_positions = [ps[0][0] for ps in positions]
# obj_predicate = [ps[0][1] for ps in positions]
# obj_positions = [ps[0][0] for ps in obj_predicate]
# predicate = [ps[0][1] for ps in obj_predicate]
# # sub_position_start = tf.constant([[ps[0]] for ps in sub_positions], dtype=tf.int32)
# # sub_position_end = tf.constant([[ps[1]] for ps in sub_positions], dtype=tf.int32)
#
# sub_position_start = np.array([[ps[0]] for ps in sub_positions])
# sub_position_end = np.array([[ps[1]] for ps in sub_positions])
#
# obj_positions_start = np.array([[ps[0]] for ps in obj_positions])
# obj_positions_end = np.array([[ps[1]] for ps in obj_positions])
#
# positions_sub = [sub_position_start, sub_position_end]
# positions_obj = [obj_positions_start, obj_positions_end]
#
# def concatenate_weights_sub(bert_encoder_hidden, positions):
#     # 将subject对应位置的起始信息张量拼接起来
#     position_start, position_end = positions
#     positions_se = np.array([np.arange(position_start[i], position_end[i]) for i in range(len(position_start))])
#
#     # trainable_weight_start = tf.gather(bert_encoder_hidden, tf.convert_to_tensor(position_start), batch_dims=1)
#     trainable_weight_end = tf.gather(bert_encoder_hidden, positions_se, batch_dims=1)
#     print(trainable_weight_end)
#     # trainable_weights = tf.keras.layers.concatenate([trainable_weight_start, trainable_weight_end])
#     # trainable_weights = tf.squeeze(trainable_weights, 1)
#
#     return None
#
#
#
# model = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
# dropout = tf.keras.layers.Dropout(0.3)
# dense = tf.keras.layers.Dense(2, activation='sigmoid')
# layer_normal2 = LayerNormalization(conditional=True)
# layer_normal1 = tf.keras.layers.LayerNormalization()
#
# ids = example_data['input_ids']
# masks = example_data['masks']
# tokens = example_data['tokens']
# target = example_data['subject_labels']
# target2 = example_data['object_labels']
# target3 = example_data['predicate_labels']
#
# input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)
#
#
# def test_predict_sub():
#     hidden = model(ids, masks, tokens)[0]
#     other_params = tf.Variable(tf.random.uniform(shape=(2, 2)))
#     input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)
#     dropout_inputs = dropout(hidden, 1)
#     sub_predict = dense(dropout_inputs)
#     log_likelihood, other_params = tfa.text.crf.crf_log_likelihood(sub_predict,
#                                                                    target2,
#                                                                    input_seq_len,
#                                                                    other_params)
#     sub_predict, _ = tfa.text.crf_decode(sub_predict, other_params , input_seq_len)
#     print(sub_predict)
#
# def test_predict_obj():
#     import keras
#     from bert4keras.backend import K
#     hidden = model(ids, masks, tokens)[0]
#     other_params = tf.Variable(tf.random.uniform(shape=(2, 2)))
#     input_seq_len2 = tf.cast(tf.reduce_sum(tf.where(target2 < 1, x=0, y=1), axis=1), dtype=tf.int32)
#
#     weights = concatenate_weights_sub(hidden, positions_sub)
#     weights = layer_normal2([hidden, weights])
#
#     obj_predict = dense(weights)
#     # log_likelihood, other_params = tfa.text.crf.crf_log_likelihood(obj_predict,
#     #                                                                target2,
#     #                                                                input_seq_len2,
#     #                                                                other_params)
#     # sub_predict, _ = tfa.text.crf_decode(obj_predict, other_params , input_seq_len)
# test_predict_obj()


class ModelBertCrf4Sub(tf.keras.Model):
    """该类用以获取subject
    使用bert&crf模型, 以序列标注的方法进行训练
    """
    def __init__(self, output_dim, use_crf, _=None):
        super(ModelBertCrf4Sub, self).__init__(output_dim, use_crf, _)
        self.use_crf = use_crf
        # 如果使用embedding需要用到此参数, input_idm = len(vocabs)
        self.output_dim = output_dim

        self.bert = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')

        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(self.output_dim)
        self.other_params = tf.Variable(tf.random.uniform(shape=(self.output_dim, self.output_dim)))


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len], name='input_ids', dtype=tf.int32),
                                  tf.TensorSpec([None, max_len], name='masks', dtype=tf.int32),
                                  tf.TensorSpec([None, max_len], name='tokens', dtype=tf.int32),
                                  tf.TensorSpec([None, max_len], name='target', dtype=tf.int32)),
                                  tf.TensorSpec([None, None], name='input_seq_len', dtype=tf.int32)])
    def call(self, batch_data):
        input_ids, masks, tokens, target, input_seq_len = batch_data

        hidden = self.bert(input_ids, masks, tokens)[0]
        dropout_inputs = self.dropout(hidden, 1)
        logistic_seq = self.dense(dropout_inputs)
        if self.use_crf:
            log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(logistic_seq,
                                                                                target,
                                                                                input_seq_len,
                                                                                self.other_params)
            decode_predict, crf_scores = tfa.text.crf_decode(logistic_seq, self.other_params , input_seq_len)

            return decode_predict, hidden, log_likelihood
        else:
            prob_seq = tf.nn.softmax(logistic_seq)
            return prob_seq, hidden, None


class ModelCrf4Obj(tf.keras.Model):
    """该类用以获取object
    """
    def __init__(self, output_dim, use_crf, _=None):
        super(ModelCrf4Obj, self).__init__(output_dim, use_crf, _)
        self.use_crf = use_crf

        self.dense = tf.keras.layers.Dense(output_dim)

        self.layer_normal = LayerNormalization(conditional=True)
        # LN由于是源自于bert4keras无法在梯度训练下于tf共存因此放弃使用
        # self.layer_normal = tf.keras.layers.LayerNormalization()

        self.dropout = tf.keras.layers.Dropout(0.1)

        self.lstm = tf.keras.layers.LSTM(units=2)

        self.other_params = tf.Variable(tf.random.uniform(shape=(output_dim, output_dim)))


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len, 768], name='hidden', dtype=tf.float32),
                                   tf.TensorSpec([None, max_len, 1536], name='weights_sub', dtype=tf.float32),
                                   tf.TensorSpec([None, max_len], name='obj_target', dtype=tf.int32),
                                   tf.TensorSpec([None, None], name='input_seq_len', dtype=tf.int32))])
    def call(self, inputs):
        hidden, weights_sub, obj_target, input_seq_len = inputs

        weights_sub = self.layer_normal([hidden, weights_sub])

        weights_obj = self.dropout(weights_sub, 1)

        obj_predict = self.dense(weights_obj)

        log_likelihood = None

        log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(obj_predict,
                                                                            obj_target,
                                                                            input_seq_len,
                                                                            self.other_params)

        obj_predict, _ = tfa.text.crf_decode(obj_predict, self.other_params, input_seq_len)

        return obj_predict, log_likelihood


class Model4Pre(tf.keras.Model):
    """该类用以获取predicate
    如果predicate是固定的话，那么可以使用分类模型
    如果predicate不是固定的话，那么可以使用seq2seq文本生成模型
    """
    def __init__(self, pre_sorts):
        super(Model4Pre, self).__init__(pre_sorts)

        self.dense1 = tf.keras.layers.Dense(units=768)

        self.dense2 = tf.keras.layers.Dense(units=pre_sorts * 2, activation='sigmoid')

        self.cnn = tf.keras.layers.Conv1D(filters=768, kernel_size=1)

        self.lambdas = tf.keras.layers.Lambda(lambda x: x[:, 0])
        # 为了做分类问题需要将shape(batch_size, max_len, 768)->(batch_size, 768)
        # 由于attention的缘故不需要对(max_len, 768)做操作直接取768就可以, 因此使用lambda x: x[:, 0]即可
        self.dropout = tf.keras.layers.Dropout(0.5)

        self.reshape = tf.keras.layers.Reshape((-1, pre_sorts, 2))

        # self.layer_normal = LayerNormalization(conditional=True)


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len, 768], name='hidden', dtype=tf.float32),
                                   tf.TensorSpec([None, max_len, 3072], name='trainable_weights', dtype=tf.float32))])
    def call(self, inputs):
        hidden, trainable_weights = inputs
        # 此处trainable_weights包含(weights_sub, weights_obj)

        trainable_weights = self.dense1(trainable_weights)

        weights3 = tf.keras.layers.concatenate([hidden, trainable_weights])

        weights_pre = self.cnn(weights3)

        dropout_inputs = self.dropout(weights_pre, 1)

        lambdas_inputs = self.lambdas(dropout_inputs)

        predict = self.dense2(lambdas_inputs)

        predict = self.reshape(predict)

        return predict


def loss4sub_obj_crf(log):
    loss = -tf.reduce_mean(log)

    return loss


def loss4sub_obj(t, p):
    loss_value = tf.keras.losses.sparse_categorical_crossentropy(y_true=t, y_pred=p)
    loss_value = tf.reduce_mean(loss_value)

    return loss_value


def loss4pre(t, p):
    loss_value = tf.keras.losses.binary_crossentropy(y_true=t, y_pred=p)
    loss_value = tf.reduce_mean(loss_value)

    return loss_value


def concatenate_weights_sub(bert_encoder_hidden, positions):
    # 将subject对应位置的起始信息张量拼接起来
    position_start, position_end = positions
    trainable_weight_start = tf.gather(bert_encoder_hidden, position_start, batch_dims=1)
    trainable_weight_end = tf.gather(bert_encoder_hidden, position_end, batch_dims=1)
    trainable_weights = tf.keras.layers.concatenate([trainable_weight_start, trainable_weight_end])
    # trainable_weights = tf.squeeze(trainable_weights, 1)

    return trainable_weights[:, 0]


def concatenate_weights_sub_obj(bert_encoder_hidden, positions):
    positions_sub, positions_obj = positions
    position_start_sub, position_end_sub = positions_sub
    position_start_obj, position_end_obj = positions_obj

    trainable_weight_start_sub = tf.gather(bert_encoder_hidden, position_start_sub, batch_dims=1)
    trainable_weight_end_sub = tf.gather(bert_encoder_hidden, position_end_sub, batch_dims=1)

    trainable_weight_start_obj = tf.gather(bert_encoder_hidden, position_start_obj, batch_dims=1)
    trainable_weight_end_obj = tf.gather(bert_encoder_hidden, position_end_obj, batch_dims=1)

    trainable_weight_sub = tf.keras.layers.concatenate([trainable_weight_start_sub, trainable_weight_end_sub])
    trainable_weight_obj = tf.keras.layers.concatenate([trainable_weight_start_obj, trainable_weight_end_obj])

    trainable_weights = tf.keras.layers.concatenate([trainable_weight_sub, trainable_weight_obj])
    # trainable_weights = tf.squeeze(trainable_weights, 1)

    # todo 处理sub以及obj的向量同bert编码层的交互
    return trainable_weights[:, 0]


def fit_step(dataset):
    # dataset = data_generator(dataset)
    model_sub = ModelBertCrf4Sub(output_dim=2, use_crf=True)
    opti_bert_sub = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.95)
    opti_other_sub = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.95)

    model_obj = ModelCrf4Obj(output_dim=2, use_crf=True)
    opti_obj = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.95)

    # 此处predicate种类是确定的因此作为分类模型考虑
    model_pre = Model4Pre(len(predicate2id))
    opti_pre = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.95)

    def fit_subject(inputs):

        weights_bert = []         # bert参数权重获取
        weights_other = []        # crf和其它层参数权重

        with tf.GradientTape() as tape:
            predict, weights, log = model_sub(inputs)

            # todo 计算损失函数
            if log is not None:
                loss = loss4sub_obj_crf(log)
            else:
                loss = loss4sub_obj(inputs[-1], predict)
                # inputs[-1]: subject的target

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

        # predict: 预测获取的sub, weights: inputs的bert编码层feed&forward层向量
        return predict, weights, loss

    # todo
    def fit_object(inputs):

        with tf.GradientTape() as tape:
            predict, log = model_obj(inputs)
            if len(log)>0:
                loss = loss4sub_obj_crf(log=log)
            else:
                loss = loss4sub_obj(t=inputs[-1], p=predict)

            params_all = tape.gradient(loss, model_obj.trainable_variables)
            opti_obj.apply_gradients(zip(params_all, model_obj.trainable_variables))

        return predict, loss

    # todo
    def fit_predicate(inputs):
        target, weights = inputs
        with tf.GradientTape() as tape:
            predict = model_pre(weights)
            loss = loss4pre(t=target, p=predict)
            params_all = tape.gradient(loss, model_pre.trainable_weights)
            opti_pre.apply_gradients(zip(params_all, model_pre.trainable_weights))

        return predict, loss

    for _, data in enumerate(dataset):
        ids = data['input_ids']
        masks = data['masks']
        tokens = data['tokens']
        sub_labels = data['subject_labels']
        obj_labels = data['object_labels']
        pre_labels = data['predicate_labels']
        positions = data['positions']

        sub_positions = [ps[0][0] for ps in positions]      # subject的位置信息
        obj_predicate = [ps[0][1] for ps in positions]      # object和predicate的信息
        obj_positions = [ps[0][0] for ps in obj_predicate]  # object的位置信息

        # subject位置信息: 用于在预测object和predicate时同bert-encoder编码信息交互
        # object位置信息: 用于在预测predicate时同bert-encoder编码信息交互
        # predicate信息: 用于查验

        sub_position_start = [[ps[0]] for ps in sub_positions]
        sub_position_end = [[ps[1]] for ps in sub_positions]

        obj_positions_start = [[ps[0]] for ps in obj_positions]
        obj_positions_end = [[ps[1]] for ps in obj_positions]

        positions_sub = [sub_position_start, sub_position_end]
        positions_obj = [obj_positions_start, obj_positions_end]

        input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)

        # 首先训练subject模型
        predict_sub, bert_encoder_hidden, loss_sub = fit_subject(inputs=(ids, masks, tokens, sub_labels, input_seq_len))

        # 其次训练object模型
        weights = concatenate_weights_sub(bert_encoder_hidden, positions_sub)
        predict_obj, loss_obj = fit_object(inputs=[bert_encoder_hidden, weights, obj_labels, input_seq_len])

        # 最后训练predicate模型
        # weights = concatenate_weights_sub_obj(bert_encoder_hidden, [positions_sub, positions_obj])
        # predict_pre, loss_pre = fit_predicate(inputs=[pre_labels, [bert_encoder_hidden, weights]])

        if _ % 10 == 0:
            with open('models_save/fit_logs.txt', 'a') as f:
                # 'a'  要求写入字符
                # 'wb' 要求写入字节(str.encode(str))
                log = f"times: {datetime.now()}, " \
                      f"num: {_}, sub_loss: {loss_sub}, loss_obj: {loss_obj}, loss_pre: {None}, \n" \
                      f"sub_labels[:3]: {sub_labels[:1]}, \n sub_predict: {predict_sub[:1]}, \n" \
                      f"obj_labels[:3]: {obj_labels[:1]}, \n obj_predict: {predict_obj[:1]}, \n" \
                      # f"pre_labels[:3]: {pre_labels[:5]}, \n pre_predict: {predict_pre[:5]}  \n"
                f.write(log)
            print(f"times: {datetime.now()}, num: {_}, sub_loss: {loss_sub}, loss_obj: {loss_obj}, loss_pre: {None}")


    # model_sub.save(filepath='models_save/model_sub/1/')
    # model_obj.save(filepath='models_save/model_obj/1/')
    # model_pre.save(filepath='models_save/model_pre/1/')


fit_step(data)