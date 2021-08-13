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
from tensorflow import recompute_grad
from tensorflow.keras import initializers, activations

from transformers import BertTokenizer, TFBertModel

pd.set_option('display.max_columns', None)

tf.config.experimental_run_functions_eagerly(True)

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
        mask4obj = np.ones((len(text)), dtype=np.int32)
        mask4pre = np.zeros((len(predicate2id), 2), dtype=np.int32)

        # 此处保存的位置信息仅用于标注obj时随机从多组s-p-o中抽出对应的位置信息
        sub_obj_predicate = {}
        choice_position = []

        for spo_list in spo_lists:

            sub = spo_list['subject']
            obj = spo_list['object']['@value']

            predicate = spo_list['predicate']
            predicate = predicate2id[predicate]

            sub_position = (text.index(sub)+1, text.index(sub)+len(sub)+1)
            obj_position = (text.index(obj)+1, text.index(obj)+len(obj)+1)
            # +1的原因是因为token会在text前加一个开始编码101

            if sub_position not in sub_obj_predicate:
                sub_obj_predicate[sub_position] = []
            sub_obj_predicate[sub_position].append((obj_position, predicate))

            # 对于sub, 可以直接将对应在text的位置处标注出来
            mask4sub = list(mask4sub) + (max_len - len(mask4sub)) * [0]
            mask4sub = np.array(mask4sub)

            mask4sub[sub_position[0]:sub_position[1]] = 2

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
                mask4obj[obj[0]:obj[1]] = 2

            choice_position.append((sub, sub_obj_predicate[sub]))
            # 由于一个s可能对应多个o且关系不同, 因此构建一个多维数组:
            # batch_size (len(predicate2id), len(inputs['input_ids']))
            # 其中行号代表一个关系类型, 比如说'主演'这个关系在predicate2id的编码是4
            # 那么如果预测的object在第四行则表示他们的关系是'主演'
            # 这样的做法是因为考虑一个subject可能对应多个object且关系也不同,因此将所有关系
            # 作为候选, 出现在哪一行则代表关系是哪一种同时一行数据表示编码后的输入,标记在哪一个位置
            # 则该位置对应的词就位object, 通过此方法可以解决:
            # 1: 同时找到predicate和object
            # 2: 解决了一个输入可能存在多组三元组关系数据

        subject_labels.append(mask4sub)
        object_labels.append(mask4obj)
        predicate_labels.append(mask4pre)
        positions.append(choice_position)

        if len(ids) == batch_size or _ == len(dataset):
            yield {
                'ids': tf.constant(ids, dtype=tf.int32),
                'masks': tf.constant(masks, dtype=tf.int32),
                'tokens': tf.constant(tokens, dtype=tf.int32),
                'subject_labels': tf.constant(subject_labels, dtype=tf.int32),
                'object_labels': tf.constant(object_labels, dtype=tf.int32),
                'predicate_labels': tf.constant(predicate_labels),
                'positions': np.array(positions)
            }
            ids, masks, tokens = [], [], []
            subject_labels, object_labels, predicate_labels, positions = [], [], [], []


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
# sub_position_start = [[ps[0]] for ps in sub_positions]
# sub_position_end = [[ps[1]] for ps in sub_positions]
#
# obj_positions_start = [[ps[0]] for ps in obj_positions]
# obj_positions_end = [[ps[1]] for ps in obj_positions]
#
# positions_sub = [sub_position_start, sub_position_end]
# positions_obj = [obj_positions_start, obj_positions_end]
#
# def concatenate_weights_sub(bert_encoder_hidden, positions):
#     # 将subject对应位置的起始信息张量拼接起来
#     position_start, position_end = positions
#     trainable_weight_start = tf.gather(bert_encoder_hidden, position_start)
#     trainable_weight_end = tf.gather(bert_encoder_hidden, position_end)
#     trainable_weights = tf.keras.layers.concatenate([trainable_weight_start, trainable_weight_end])
#     trainable_weights = tf.squeeze(trainable_weights, 1)
#
#     return trainable_weights
#
# model = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
# dropout = tf.keras.layers.Dropout(0.3)
# dense = tf.keras.layers.Dense(3)
# layer_normal = LayerNormalization(conditional=True)
# dense1 = tf.keras.layers.Dense(units=768)
#
# #
# ids = example_data['ids']
# masks = example_data['masks']
# tokens = example_data['tokens']
# target = example_data['subject_labels']
# target2 = example_data['object_labels']
# target3 = example_data['predicate_labels']
#
# input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)
#
# hidden = model(ids, masks, tokens)[0]

# def test_predict_subj():
#     other_params = tf.Variable(tf.random.uniform(shape=(3, 3)))
#     input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)
#     dropout_inputs = dropout(hidden, 1)
#     sub_predict = dense(dropout_inputs)
#     log_likelihood, other_params = tfa.text.crf.crf_log_likelihood(sub_predict,
#                                                                    target2,
#                                                                    input_seq_len,
#                                                                    other_params)
#     sub_predict, _ = tfa.text.crf_decode(sub_predict, other_params , input_seq_len)
#     print(sub_predict)
#     test_predict_subj()

# def test_predict_obj():
#     other_params = tf.Variable(tf.random.uniform(shape=(3, 3)))
#     input_seq_len2 = tf.cast(tf.reduce_sum(tf.where(target2 < 1, x=0, y=1), axis=1), dtype=tf.int32)
#
#     # dropout_inputs = dropout(hidden, 1)
#     weights2 = concatenate_weights_sub(hidden, positions_sub)
#     weights2 = dense1(weights2)
#     input_weights2 = layer_normal([hidden, weights2])
#
#     obj_predict = dense(input_weights2)
#     log_likelihood, other_params = tfa.text.crf.crf_log_likelihood(obj_predict,
#                                                                    target2,
#                                                                    input_seq_len2,
#                                                                    other_params)
#     sub_predict, _ = tfa.text.crf_decode(obj_predict, other_params , input_seq_len)
#     print(sub_predict)
#
#     test_predict_obj()


class ModelBertCrf4Sub(tf.keras.Model):
    """该类用以获取subject
    使用bert&crf模型, 以序列标注的方法进行训练
    """
    def __init__(self, output_dim, use_crf, _=None):
        super(ModelBertCrf4Sub, self).__init__(output_dim, use_crf, _)
        # 如果使用embedding需要用到此参数, input_dim=len(vocabs)
        self.use_crf = use_crf

        self.bert = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')

        self.dropout = tf.keras.layers.Dropout(0.3)

        self.dense = tf.keras.layers.Dense(output_dim)

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

        # sub_predict: 预测的subject
        # weights: bert的倒数第5层feed&forward层用于共享编码层
        # log_like: 用于计算crf层的损失函数
        return sub_predict, hidden, log_likelihood


class ModelCrf4Obj(tf.keras.Model):
    """该类用以获取object
    """
    def __init__(self, output_dim, use_crf, _=None):
        super(ModelCrf4Obj, self).__init__(output_dim, use_crf, _)
        self.use_crf = use_crf

        self.dense1 = tf.keras.layers.Dense(units=768)

        self.dense2 = tf.keras.layers.Dense(output_dim)

        # self.layer_normal = LayerNormalization(conditional=True)
        # LN由于是源自于bert4keras无法在梯度训练下于tf共存因此放弃使用
        self.cnn = tf.keras.layers.Conv1D(filters=768, kernel_size=1)

        self.dropout = tf.keras.layers.Dropout(0.3)

        self.reshape = tf.keras.layers.Reshape((-1, len(predicate2id), 1))

        self.other_params = tf.Variable(tf.random.uniform(shape=(output_dim, output_dim)))


    @tf.function
    def call(self, inputs):
        hidden, weights_sub, obj_target = inputs

        input_seq_len2 = tf.cast(tf.reduce_sum(tf.where(obj_target < 1, x=0, y=1), axis=1), dtype=tf.int32)


        # weights2 = self.dense1(weights_sub)

        weights2 = tf.keras.layers.concatenate([hidden, weights_sub])

        weights_obj = self.cnn(weights2)
        # input_weights2 = self.layer_normal([hidden, weights2])
        weights_obj = self.dropout(weights_obj, 1)

        obj_predict = self.dense2(weights_obj)

        log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(obj_predict,
                                                                            obj_target,
                                                                            input_seq_len2,
                                                                            self.other_params)

        obj_predict, _ = tfa.text.crf_decode(obj_predict, self.other_params, input_seq_len2)

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


    @tf.function
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
    trainable_weight_start = tf.gather(bert_encoder_hidden, position_start)
    trainable_weight_end = tf.gather(bert_encoder_hidden, position_end)
    trainable_weights = tf.keras.layers.concatenate([trainable_weight_start, trainable_weight_end])
    trainable_weights = tf.squeeze(trainable_weights, 1)

    return trainable_weights


def concatenate_weights_sub_obj(bert_encoder_hidden, positions):
    positions_sub, positions_obj = positions
    position_start_sub, position_end_sub = positions_sub
    position_start_obj, position_end_obj = positions_obj

    trainable_weight_start_sub, trainable_weight_end_sub = tf.gather(bert_encoder_hidden, position_start_sub), \
                                                           tf.gather(bert_encoder_hidden, position_end_sub)

    trainable_weight_start_obj, trainable_weight_end_obj = tf.gather(bert_encoder_hidden, position_start_obj), \
                                                           tf.gather(bert_encoder_hidden, position_end_obj)

    trainable_weight_sub = tf.keras.layers.concatenate([trainable_weight_start_sub, trainable_weight_end_sub])
    trainable_weight_obj = tf.keras.layers.concatenate([trainable_weight_start_obj, trainable_weight_end_obj])

    trainable_weights = tf.keras.layers.concatenate([trainable_weight_sub, trainable_weight_obj])
    trainable_weights = tf.squeeze(trainable_weights, 1)

    # todo 处理sub以及obj的向量同bert编码层的交互
    return trainable_weights


def fit_step(dataset, fit=True):
    # dataset = data_generator(dataset)
    model_sub = ModelBertCrf4Sub(output_dim=3, use_crf=True)
    opti_bert_sub = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.95)
    opti_other_sub = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

    model_obj = ModelCrf4Obj(output_dim=3, use_crf=True)
    opti_obj = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.95)

    # 此处predicate种类是确定的因此作为分类模型考虑
    model_pre = Model4Pre(len(predicate2id))
    opti_pre = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

    def fit_subject(inputs, crf=True):

        weights_bert = []         # bert参数权重获取
        weights_other = []        # crf和其它层参数权重

        with tf.GradientTape() as tape:
            predict, weights, log = model_sub(inputs)

            # todo 计算损失函数
            if crf:
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
        return predict, weights

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

        return predict

    # todo
    def fit_predicate(inputs):
        target, weights = inputs
        with tf.GradientTape() as tape:
            predict = model_pre(weights)
            loss = loss4pre(t=target, p=predict)
            params_all = tape.gradient(loss, model_pre.trainable_weights)
            opti_pre.apply_gradients(zip(params_all, model_pre.trainable_weights))

        return predict

    for _, data in enumerate(dataset):
        ids = data['ids']
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

        # 训练过程开始:

        # 首先训练subject模型
        predict_sub, bert_encoder_hidden = fit_subject(inputs=[ids, masks, tokens, sub_labels])
        print("fit sub is success!")
        # 其次训练object模型
        weights = concatenate_weights_sub(bert_encoder_hidden, positions_sub)
        predict_obj = fit_object(inputs=[bert_encoder_hidden, weights, obj_labels])
        # predict_obj = tf.squeeze(predict_obj, axis=1)
        print("fit obj is success!")
        # todo fit predicate

        # 最后训练predicate模型
        weights = concatenate_weights_sub_obj(bert_encoder_hidden, [positions_sub, positions_obj])
        predict_pre = fit_predicate(inputs=[pre_labels, [bert_encoder_hidden, weights]])


fit_step(data, True)