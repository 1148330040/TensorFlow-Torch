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
from tensorflow.keras import initializers, activations

from datetime import datetime

from transformers import BertTokenizer, TFBertModel

pd.set_option('display.max_columns', None)

tf.config.experimental_run_functions_eagerly(True)

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

relation_ext_path = 'dataset/duie/duie_train.json/duie_train.json'
valid_ds_path = 'dataset/duie/duie_dev.json/duie_dev.json'
test_ds_path = 'dataset/duie/duie_test2.json/duie_test2.json'
schema_ds_path = 'dataset/duie/duie_schema/duie_schema.json'



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
    subject_labels, object_labels, positions = [], [], []
    # len_text: 用于后续处理padding的问题:
    # 计算loss的时候去除掉padding_sequence的影响确保只计算文本长度范围内的损失值
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

        if len(input_id) > 128:
            continue

        mask4sub = np.zeros((max_len, 2), dtype=np.int32)
        mask4obj = np.zeros((max_len, len(predicate2id), 2), dtype=np.int32)

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

            mask4sub[sub_position[0], 0] = 1
            mask4sub[sub_position[1], 1] = 1

        # 此处的位置信息是用于保存从多组s-p-o中抽取的那一组s-p-o的位置信息
        # 该位置信息后续将会用于同bert的encoder编码向量交互用于预测obj和predicate

        if bool(sub_obj_predicate):
            sub = random.choice(list(sub_obj_predicate.keys()))

            for obj_pre in sub_obj_predicate[sub]:
                # mask4pre[obj_pre[1], 1] = 1
                # 将对应的predicate位置处置为1
                obj = obj_pre[0]
                pre = obj_pre[1]
                mask4obj[obj[0], pre, 0] = 1
                mask4obj[obj[1], pre, 1] = 1

            choice_position.append((sub, sub_obj_predicate[sub]))

        if len(choice_position) >= 1:
            subject_labels.append(mask4sub)
            object_labels.append(mask4obj)
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
                'subject_labels': tf.constant(subject_labels, dtype=tf.float32),
                'object_labels': tf.constant(object_labels, dtype=tf.float32),
                'positions': np.array(positions),
            }
            ids, masks, tokens = [], [], []
            subject_labels, object_labels, positions  = [], [], []


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

    def __init__(self, output_dim):
        super(ModelBertCrf4Sub, self).__init__(output_dim)
        self.output_dim = output_dim
        self.bert = TFBertModel.from_pretrained("hfl/chinese-bert-wwm-ext", output_hidden_states=True)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(self.output_dim, activation='sigmoid')

    @tf.function(input_signature=[(tf.TensorSpec([None, max_len], name='input_ids', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='masks', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='tokens', dtype=tf.int32))])
    def call(self, batch_data):
        input_ids, masks, tokens = batch_data

        hidden = self.bert(input_ids, masks, tokens)

        dropout_inputs = self.dropout(hidden[0], 1)

        sub_predict = self.dense(dropout_inputs)

        sub_predict = tf.keras.layers.Lambda(lambda x: x**2)(sub_predict)

        return sub_predict, hidden[2][-2]


class ModelCrf4Obj(tf.keras.Model):
    """该类用以获取object
    """
    def __init__(self, output_dim):
        super(ModelCrf4Obj, self).__init__(output_dim)

        self.output_dim = output_dim

        self.dense = tf.keras.layers.Dense(output_dim * 2, activation='sigmoid')

        self.layer_normal = LayerNormalization(conditional=True)

        self.dropout = tf.keras.layers.Dropout(0.3)


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len, 768], name='hidden', dtype=tf.float32),
                                   tf.TensorSpec([None, 768], name='weights_sub', dtype=tf.float32))])
    def call(self, inputs):
        hidden, weights_sub = inputs

        weights_sub = self.layer_normal([hidden, weights_sub])

        weights_obj = self.dropout(weights_sub, 1)

        obj_predict = self.dense(weights_obj)

        obj_predict = tf.keras.layers.Lambda(lambda x: x ** 4)(obj_predict)

        obj_predict = tf.keras.layers.Reshape((-1, self.output_dim, 2))(obj_predict)

        return obj_predict


def loss4model(t, p, mask, obj=False):
    """t: 实际labels, p: 预测labels, mask: shape-(Batch_size, max_len) bool矩阵用于解决padding的问题
    subject的labels: [batch_size, max_len, 2]
    object的labels: [batch_size, max_len, len(predicate2id), 2]
    因此在进行loss操作时obj需要多进行一步reduce_mean的操作
    """
    if obj:
        loss_value = tf.keras.losses.binary_crossentropy(y_true=t, y_pred=p)
        loss_value = tf.reduce_sum(loss_value, axis=2)
    else:
        loss_value = tf.keras.losses.binary_crossentropy(y_true=t, y_pred=p)

    loss_value = tf.cast(loss_value, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    loss_value = tf.reduce_sum(loss_value * mask) / tf.reduce_sum(mask)

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


def get_predict_data(content):
    """将输入的问句初步token处理
    """
    inputs = tokenizer.encode_plus(content,
                                   add_special_tokens=True,
                                   max_length=max_len,
                                   padding='max_length',
                                   return_token_type_ids=True)

    input_id = tf.constant([inputs['input_ids']], dtype=tf.int32)
    input_mask = tf.constant([inputs['attention_mask']], dtype=tf.int32)
    token_type_ids = tf.constant([inputs["token_type_ids"]], dtype=tf.int32)

    return input_id, input_mask, token_type_ids


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


model_sub = tf.saved_model.load('models_save/model_sub/best_model/current_new_loss_new_lr_new_hidden/1/')
model_obj = tf.saved_model.load('models_save/model_obj/best_model/current_new_loss_new_lr_new_hidden/1/')
def predict(content):
    """预测函数, 步骤如下：
    1: 预测subject
    2: 获取预测到的subject对应的位置信息, 并以此拆分判断预测到了几个subject
    3: 按照循环的方式针对每一个subject进行object的预测
    4: 步骤同上获取object位置信息, 拆分, 按照每一组(subject, object)预测predicate, 并放入到spo列表中
    5: 整理spo列表传递到evaluate函数中"""
    spos = []
    input_id, input_mask, token_type_ids = get_predict_data(content=content)

    predict_sub, bert_hidden = model_sub.call((input_id, input_mask, token_type_ids))
    predict_sub = predict_sub[0]
    sub_pos_start = np.where(predict_sub[:, 0] > 0.4)[0]
    sub_pos_end = np.where(predict_sub[:, 1] > 0.4)[0]

    if len(sub_pos_start) == 0 or len(sub_pos_end) == 0:
        return spos

    for sub_s, sub_e in zip(sub_pos_start, sub_pos_end):
        if sub_s > sub_e or sub_e > len(content):
            continue
        sub_pos = [sub_s, sub_e]
        obj_weights = concatenate_weights_sub(bert_hidden, sub_pos)

        predict_obj = model_obj.call((bert_hidden, obj_weights))
        predict_obj = predict_obj[0]

        obj_pos_start = np.where(predict_obj[:, :, 0] >= 0.4)
        obj_pos_end = np.where(predict_obj[:, :, 1] >= 0.4)

        obj_start_pos = obj_pos_start[0]
        obj_start_pre = obj_pos_start[1]

        obj_end_pos = obj_pos_end[0]
        obj_end_pre = obj_pos_end[1]

        for num, (a, b) in enumerate(zip(obj_start_pos, obj_end_pos)):
            if obj_start_pre[num] == obj_end_pre[num] and a <= b:

                if b > len(content):
                    continue

                spos.append(
                    (
                        content[sub_pos[0]:sub_pos[1]],
                        id2predicate[obj_start_pre[num]],
                        content[a:b]
                    )
                )

    return spos


def evaluate(path):
    """"""
    dataset = open(path)

    len_lab = 1e-10
    len_pre = 1e-10
    len_pre_is_true = 1e-10
    num = 0
    for ds in dataset.readlines():
        ds = json.loads(ds)
        text = ds['text']
        get_predict_data(text)
        if len(text) > 125:
            continue
        num += 1
        spo_t = ds['spo_list']

        spo_labels = [(spo['subject'], spo['predicate'], spo['object']['@value']) for spo in spo_t]
        spo_labels = set([SPO(spo) for spo in spo_labels])
        spo_predicts = predict(content=text)
        spo_predicts = set([SPO(spo) for spo in spo_predicts])

        if num % 200 == 0:
            print("spo_labels: ", spo_labels)
            print("spo_predicts: ", spo_predicts)
        len_lab += len(spo_labels)
        len_pre += len(spo_predicts)
        len_pre_is_true += len(spo_labels & spo_predicts)
        # & 该运算的作用是返回两个输入内部相同的值
        # a: [(1, 2), (3, 4), (5, 6)],  b: [(3, 4), (1, 2), (6, 5)]
        # 返回 [(1, 2), (3, 4)]

    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    return f1_value, precision, recall

print(evaluate(path=valid_ds_path))

# 4: (0.5821507858982343, 0.688787089182148, 0.5041063024822383)
# 5: (0.5825210760199712, 0.7183384448854423, 0.4898957275998908)
# 9: (0.5848143281930724, 0.6676666798247358, 0.5202546830303605)

def predict_test():
    dataset = open(test_ds_path)

    schema_ds = open(schema_ds_path)

    schema_ds = pd.DataFrame(schema_ds, columns=['schema'])

    schema_ds['predicate'] = schema_ds['schema'].apply(
        lambda x: json.loads(x)['predicate']
    )
    schema_ds['subject_type'] = schema_ds['schema'].apply(
        lambda x: json.loads(x)['subject_type']
    )
    schema_ds['object_type'] = schema_ds['schema'].apply(
        lambda x: json.loads(x)['object_type']['@value']
    )

    for ds in dataset.readlines():
        # 物流作业方法
        # 数据分析方法五种
        # 比较新闻传播学
        spo_test = {'spo_list': []}
        ds = json.loads(ds)
        text = ds['text']
        spo_predict = set(predict(text))
        spo_test['text'] = text

        for spo in spo_predict:
            spos = {}
            s = spo[0]
            p = spo[1]
            o = spo[2]
            spos['predicate'] = p
            spos['subject'] = s
            spos['object'] = {'@value': o}
            spos['subject_type'] = schema_ds.loc[schema_ds['predicate']==p]['subject_type'].values[0]
            spos['object_type'] = schema_ds.loc[schema_ds['predicate']==p]['object_type'].values[0]
            spo_test['spo_list'].append(spos)


def fit_step():

    model_sub = ModelBertCrf4Sub(output_dim=2)

    model_obj = ModelCrf4Obj(output_dim=len(predicate2id))

    opti_sub = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.90, beta_2=0.95)

    @tf.function()
    def fit_models(inputs_model, inputs_labels, inputs_other):
        positions_sub, positions_obj = inputs_other
        sub_label, obj_label = inputs_labels
        ids, masks, tokens = inputs_model
        weights_sub = []
        with tf.GradientTape() as tape:
            predict_sub, bert_hidden = model_sub(batch_data=(ids, masks, tokens))

            obj_weights = concatenate_weights_sub(bert_encoder_hidden=bert_hidden,
                                                  positions=positions_sub)

            predict_obj = model_obj(inputs=(bert_hidden, obj_weights))

            loss_sub = loss4model(t=sub_label, p=predict_sub, mask=masks)

            loss_obj = loss4model(t=obj_label, p=predict_obj, mask=masks, obj=True)

            loss = loss_sub + loss_obj


            for var in model_sub.trainable_variables:
                model_name = var.name
                none_bert_layer = ['tf_bert_model/bert/pooler/dense/kernel:0',
                                   'tf_bert_model/bert/pooler/dense/bias:0',
                                   'Variable:0']
                if model_name not in none_bert_layer:
                    weights_sub.append(var)

            weights_obj = model_obj.trainable_variables

        params_all = tape.gradient(loss, [weights_sub, weights_obj])

        params_sub = params_all[0]
        params_obj = params_all[1]

        opti_sub.apply_gradients(zip(params_sub, weights_sub))

        opti_sub.apply_gradients(zip(params_obj, weights_obj))

        # predict: 预测获取的sub, weights: inputs的bert编码层feed&forward层向量
        return predict_sub, predict_obj, loss_sub, loss_obj


    for num in range(0, 15):
        spo_datasets = pd.DataFrame(spo_dataset).sample(frac=1.0)
        dataset = data_generator(spo_datasets)
        for _, data in enumerate(dataset):
            ids = data['input_ids']
            masks = data['masks']
            tokens = data['tokens']
            sub_labels = data['subject_labels']
            obj_labels = data['object_labels']
            positions = data['positions']

            sub_positions = [ps[0][0] for ps in positions]  # subject的位置信息
            obj_predicate = [ps[0][1] for ps in positions]  # object和predicate的信息
            obj_positions = [ps[0][0] for ps in obj_predicate]  # object的位置信息

            # 位置信息: 用于在预测object和predicate时同bert-encoder编码信息交互

            p_sub, p_obj, loss_sub, loss_obj = fit_models(
                inputs_model=[ids, masks, tokens],
                inputs_labels=[sub_labels, obj_labels],
                inputs_other=[sub_positions, obj_positions]
            )

            if _ % 500 == 0:
                with open('models_save/fit_logs.txt', 'a') as f:
                    # 'a'  要求写入字符
                    # 'wb' 要求写入字节(str.encode(str))
                    log = f"times: {datetime.now()}, " \
                          f"num: {_}, sub_loss: {loss_sub}, loss_obj: {loss_obj}\n" \

                    f.write(log)
                    print(log)
        model_sub.save(model_sub_path + 'current_new_loss_new_lr_new_hidden/' + str(num) + '/')
        model_obj.save(model_obj_path + 'current_new_loss_new_lr_new_hidden/' + str(num) + '/')


# fit_step()
