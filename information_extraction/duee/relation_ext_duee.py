# *- coding: utf-8 -*
# =================================
# time: 2021.7.27
# author: @唐志林
# function: 模型适配
# =================================

import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf

from bert4keras.snippets import sequence_padding
from tensorflow.keras import activations, initializers
from transformers import BertTokenizer, TFBertModel


os.environ["CUDA_VISIBLE_DEVICES"]="1"
tf.get_logger().setLevel('ERROR')

pd.set_option('display.max_columns', None)

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

train_ds_path = '../dataset/duee/duee_train.json/duee_train.json'
valid_ds_path = '../dataset/duee/duee_dev.json/duee_dev.json'
test_ds_path = '../dataset/duee/duee_test2.json/duee_test2.json'
schema_ds_path = '../dataset/duee/duee_event_schema.json'

max_len = 128
batch_size = 25

model_trigger_path = f'models_save/model_trigger/'
model_event_role_path = f'models_save/model_event_role/'


def get_dataset(path):
    """将数据处理成:
    {'text': text,  'event_type': event_type, 'trigger': trigger,
    'arguments': [(role0, role0_value),
                  (role1, role1_value),
                  (role2, role2_value), ...]]}
    event_type: 事件类型
    trigger: 出发对应事件的关键词
    arguments: 事件所对应的相关描述(event_type内部的role-固定)
    """
    dataset = open(path)
    events = []

    for ds in dataset.readlines():
        ds = json.loads(ds)

        text = ds['text']

        text = ''.join([
            "，" if t == " " or t == "\n" or t == "\t" else "\"" if t == "“" else t
            for t in list(text.lower())
        ])

        event_list_es = ds['event_list']

        event = []

        for event_list in event_list_es:
            event_type = event_list['event_type']
            trigger = event_list['trigger']
            arguments = [(argument['role'], argument['argument']) for argument in event_list['arguments']]
            event.append((event_type, trigger, arguments))

        events.append({
            'text': text,
            'events': event
        })

    events = pd.DataFrame(events)

    return events


def get_schema():
    """构建schema字典表:
    key: event_type
    value: event_type对应的role转化的id表
    exp: '财经/交易-出售/收购': {'时间': 0, '出售方': 1, '交易物': 2, '出售价格': 3, '收购方': 4}
    """
    dataset = open(schema_ds_path)
    schema = {}
    for ds in dataset.readlines():
        ds = json.loads(ds)
        event_type = ds['event_type']
        if event_type not in schema:
            schema[event_type] = {}
            role_list = ds['role_list']
            for num, role in enumerate(role_list):
                schema[event_type][role['role']] = num

    return schema

def seq2id():
    schema = get_schema()
    events2id = {event:num for num, event in enumerate(schema.keys())}

    all_roles = []
    for schema_value in schema.values():
        for role in schema_value.keys():
            if role not in all_roles:
                all_roles.append(role)
    roles2id = {role:num for num, role in enumerate(all_roles)}

    return events2id, roles2id, schema

events2id, roles2id, schema= seq2id()
id2events = {id:event for event, id in events2id.items()}
id2roles = {id:role for role, id in roles2id.items()}


# 模型多输出问题
def data_generator(dataset):
    """最终会输出一下内容:
    1: ids, masks, tokens 来自bert的tokenizer
    2: triggers_labels: 找到所有关键词 shape: (max_len, 2)
    3: events_labels: trigger对应的event(一个trigger对应一个event)
    4: role_labels: event对应的论元role及其论元的位置信息
    """
    ids, masks, tokens = [], [], []
    triggers_labels, events_labels, roles_labels, triggers_positions = [], [], [], []

    def get_seq_pos(seq):
        """不要通过词在text中的位置信息找，因为有可能部分符号或者特殊字符编码后的长度与text长度不一致
        """

        seq_len = len(seq)
        for i in range(len(input_id)):
            if input_id[i:i + seq_len] == seq:
                return i
        return -1

    for _, ds in dataset.iterrows():
        text = ''.join(ds['text'])

        inputs = tokenizer.encode_plus(text)

        input_id = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        if len(input_id) > 128:
            continue

        mask4triggers = np.zeros((max_len, len(events2id), 2), dtype=np.int32)
        # 对于triggers的label只需要找到trigger在编码后的位置并对应上去即可

        masks4roles = np.zeros((max_len, len(roles2id), 2))
        # shape: (max_len, len(roles2id), 2) 相当于是一个多分类问题, 获取论元role的同时获取一个

        # event = random.choice(ds['events'])
        # 随机抽取一个event, 如果考虑添加负样本的话, 可以将event_type和对应的trigger以及arguments打乱顺序
        all_arguments = []

        for event in ds['events']:

            event_type = event[0]
            triggers = event[1]
            argument = event[2]

            triggers_seq = tokenizer.encode(triggers)[1:-1]
            triggers_pos = get_seq_pos(triggers_seq)

            if triggers_pos == -1:
                continue

            triggers_pos = (triggers_pos, triggers_pos + len(triggers_seq))
            event_id = events2id[event_type]
            # event_id获取
            mask4triggers[triggers_pos[0], event_id, 0] = 1
            mask4triggers[triggers_pos[1], event_id, 1] = 1

            all_arguments.append((argument, triggers_pos))

        if bool(all_arguments) is False:
            continue

        argument_pos = random.choice(all_arguments)

        arguments = argument_pos[0]
        pos = argument_pos[1]
        for argument in arguments:
            if len(argument) > 0:
                role = argument[0]
                role_id = roles2id[role]
                # role id获取
                role_text = argument[1]
                role_seq = tokenizer.encode(role_text)[1:-1]
                role_pos = get_seq_pos(role_seq)
                # role_text 位置获取

                if role_pos == -1:
                    continue

                role_pos = (role_pos, role_pos + len(role_seq))
                masks4roles[role_pos[0], role_id, 0] = 1
                masks4roles[role_pos[1], role_id, 1] = 1

        ids.append(input_id)
        masks.append(attention_mask)
        tokens.append(token_type_ids)
        triggers_labels.append(mask4triggers)
        roles_labels.append(masks4roles)
        triggers_positions.append(pos)

        if len(ids) == batch_size or _ == len(dataset):
            ids = sequence_padding(ids, max_len)
            masks = sequence_padding(masks, max_len)
            tokens = sequence_padding(tokens, max_len)
            triggers_labels = sequence_padding(triggers_labels, max_len)
            roles_labels = sequence_padding(roles_labels, max_len)

            yield {
                'ids': tf.constant(ids, dtype=tf.int32),
                'masks': tf.constant(masks, dtype=tf.int32),
                'tokens': tf.constant(tokens, dtype=tf.int32),
                'triggers_labels': tf.constant(triggers_labels, dtype=tf.float32),
                'roles_labels': tf.constant(roles_labels, dtype=tf.float32),
                'triggers_positions': np.array(triggers_positions),
            }

            ids, masks, tokens = [], [], []
            triggers_labels, roles_labels, triggers_positions = [], [], []



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


class ModelBert4Trigger(tf.keras.Model):
    """该类用以获取Trigger
    """
    def __init__(self, output_dim):
        super(ModelBert4Trigger, self).__init__(output_dim)
        self.output_dim = output_dim
        self.bert = TFBertModel.from_pretrained("hfl/chinese-bert-wwm-ext", output_hidden_states=True)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(self.output_dim * 2, activation='sigmoid')

    @tf.function(input_signature=[(tf.TensorSpec([None, max_len], name='input_ids', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='masks', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='tokens', dtype=tf.int32))])
    def call(self, batch_data):
        input_ids, masks, tokens = batch_data

        hidden = self.bert(input_ids, masks, tokens)

        dropout_inputs = self.dropout(hidden[0], 1)

        sub_predict = self.dense(dropout_inputs)

        sub_predict = tf.keras.layers.Reshape((-1, self.output_dim, 2))(sub_predict)

        return sub_predict, hidden[2][-2]


class Model4EventRole(tf.keras.Model):
    """该类用以获取event和role及其论元
    """
    def __init__(self, roles):
        super(Model4EventRole, self).__init__(roles)

        self.roles = roles

        self.dense_roles = tf.keras.layers.Dense(roles * 2, activation='sigmoid')

        self.layer_normal = LayerNormalization(conditional=True)

        self.dropout = tf.keras.layers.Dropout(0.1)


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len, 768], name='hidden', dtype=tf.float32),
                                   tf.TensorSpec([None, 768], name='weights_trigger', dtype=tf.float32))])
    def call(self, inputs):
        hidden, weights_trigger = inputs

        # weights_trigger = self.layer_normal([hidden, weights_trigger])

        weights_trigger = hidden

        weights_trigger = self.dropout(weights_trigger, 1)

        roles_predict = self.dense_roles(weights_trigger)

        roles_predict = tf.keras.layers.Reshape((-1, self.roles, 2))(roles_predict)

        return roles_predict


def loss4model(t, p, mask):
    """event: shape(batch_size, max_len) 无需考虑padding的问题
    role: shape(batch_size, max_len, len(roles2id), 2)
    trigger: shape(batch_size, max_len, 2)
    因此role需要多进行一轮reduce_sum之后与mask进行点乘以解决padding的问题
    """

    loss_value = tf.keras.losses.binary_crossentropy(y_true=t, y_pred=p)

    loss_value = tf.reduce_sum(loss_value, axis=2)

    loss_value = tf.cast(loss_value, dtype=tf.float32)

    mask = tf.cast(mask, dtype=tf.float32)

    loss_value = tf.reduce_sum(loss_value * mask) / tf.reduce_sum(mask)

    return loss_value


def get_positions_hidden(positions, bert_hidden):
    """用于提取给定位置处的张量
    """
    weights = tf.ones(shape=(1, 768))
    if bert_hidden.shape[0] == 1:
        poses = (np.arange(positions[0], positions[1]))
        if len(poses) == 0:
            poses = ([positions[0]])

        weight = tf.gather(bert_hidden, [poses], batch_dims=1)
        weight = tf.reduce_mean(weight, axis=1)

        return weight

    for num, pos in enumerate(positions):
        hidden = tf.expand_dims(bert_hidden[num], 0)
        # 首先为hidden添加一个维度
        # bert_hidden[num]->(128, 768)变为(1, 128, 768)目的是为了提取对应位置信息的张量

        poses = (np.arange(pos[0], pos[1]))
        if len(poses) == 0:
            poses = ([pos[0]])

        # 将起始位置张量变为一个range
        weight = tf.gather(hidden, [poses], batch_dims=1)
        # 提取对应的张量
        weight = tf.reduce_mean(weight, axis=1)
        # 去掉增加的维度
        weights = tf.concat([weights, weight], axis=0)
        # 将提取后的张量重新合并成一个batch

    trainable_weights = weights[1:]
    # shape: (batch_size, 768)

    return trainable_weights


def concatenate_weights_trigger(bert_encoder_hidden, positions):
    # 将subject对应位置的起始信息张量拼接起来
    trainable_weights = get_positions_hidden(positions, bert_encoder_hidden)

    return trainable_weights


def get_predict_seq(content):
    inputs = tokenizer.encode_plus(content)

    input_id = tf.constant([inputs['input_ids']], dtype=tf.int32)
    input_mask = tf.constant([inputs['attention_mask']], dtype=tf.int32)
    token_type_ids = tf.constant([inputs["token_type_ids"]], dtype=tf.int32)

    if len(input_id) > 128:
        return None

    id_length = len(input_id[0])

    input_id = sequence_padding(input_id, length=128)
    input_mask = sequence_padding(input_mask, length=128)
    token_type_ids = sequence_padding(token_type_ids, length=128)

    return [input_id, input_mask, token_type_ids, id_length]


def predict(content, model_trigger, model_event_role):

    inputs = get_predict_seq(content)

    input_predicts = []
    input_predict = {}

    if bool(inputs):
        ids, mask, token, id_length = inputs

        trigger_predict, hidden = model_trigger((ids, mask, token))

        trigger_predict = trigger_predict[0]

        trigger_start = np.where(trigger_predict[:, :, 0] > 0.5)
        trigger_end = np.where(trigger_predict[:, :, 1] > 0.5)

        pos_trigger_start = trigger_start[0]
        pos_trigger_end = trigger_end[0]

        event_trigger_start = trigger_start[1]
        event_trigger_end = trigger_start[1]

        trigger_poses = []
        # 获取trigger的位置信息和对应的event_id
        for num, (a, b) in enumerate(zip(pos_trigger_start, pos_trigger_end)):
            if event_trigger_start[num] == event_trigger_end[num] and a <= b:
                trigger_poses.append((a, b, event_trigger_start[num]))
        print("trigger_poses: ", trigger_poses)
        for trigger in trigger_poses:
            trigger_pos = [trigger[0], trigger[1]]

            event_id = trigger[2]
            input_predict['event_type'] = id2events[event_id]


            if trigger_pos[0] > trigger_pos[1] or trigger_pos[1] > id_length:
                continue

            trigger = ''.join(tokenizer.decode(ids[0][trigger_pos[0]:trigger_pos[1]]).split(' '))
            print("trigger: ", trigger)
            weights_trigger = concatenate_weights_trigger(hidden, trigger_pos)

            role_predict = model_event_role((hidden, weights_trigger))
            role_predict = role_predict[0]

            role_pos_start = np.where(role_predict[:, :, 0] > 0.1)
            role_pos_end = np.where(role_predict[:, :, 1] > 0.1)

            role_start_pos = role_pos_start[0]
            role_end_pos = role_pos_end[0]
            # role的论元位置信息

            role_start_id = role_pos_start[1]
            role_end_id = role_pos_end[1]
            # role对应的id信息

            input_roles = {}

            for num, (a, b) in enumerate(zip(role_start_pos, role_end_pos)):
                if b > id_length + 1:
                    continue

                if role_start_id[num] == role_end_id[num] and a <= b:
                    argument = ''.join(tokenizer.decode(ids[0][a:b]).split(' '))

                    for mask in ['[SEP]', '[PAD]', '[UNK]', '[UNK]']:
                        while mask in argument:
                            argument = argument.replace(mask, '')

                    role = id2roles[role_start_id[num]]
                    if role not in input_roles.keys():
                        input_roles['role'] = role
                    input_roles['argument'] = argument
            input_predict['arguments'] = [input_roles]
            input_predicts.append(input_predict)
            input_predict = {}

    return input_predicts


def valid():
    model_trigger = tf.saved_model.load(model_trigger_path + 'mid_model/')
    model_event_role = tf.saved_model.load(model_event_role_path + 'mid_model/')

    valid_ds = open(valid_ds_path)

    len_lab = 1e-10
    len_pre = 1e-10
    len_pre_is_true = 1e-10

    def get_spo_values(events):

        spo = []
        for event in events:
            event_type = event['event_type']
            event_arguments = event['arguments']
            for event_argument in event_arguments:
                if len(event_argument) == 0:
                    continue
                role = event_argument['role']
                argument = event_argument['argument']
                spo.append(
                    (
                        event_type,
                        role,
                        argument
                    )
                )
        return spo

    for ds in valid_ds.readlines():
        result = {}
        ds = json.loads(ds)
        content = ds['text']
        valid_id = ds['id']
        valid_events = ds['event_list']

        result['id'] = valid_id
        labels = get_spo_values(valid_events)
        print("labels: ", labels)
        if len(labels) == 0:
            continue

        predicts = predict(content, model_trigger, model_event_role)
        result['event_list'] = predicts
        predicts = get_spo_values(predicts)
        print("predicts: ", predicts)
        len_lab += len(labels)
        len_pre += len(predicts)
        len_pre_is_true += len(set(labels) & set(predicts))

    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    return f1_value


def fit():
    events = len(events2id)
    model_trigger = ModelBert4Trigger(output_dim=events)
    optimizer_trigger = tf.keras.optimizers.Adam(lr=2e-5, beta_1=0.9, beta_2=0.95)

    roles = len(roles2id)
    model_role = Model4EventRole(roles=roles)
    optimizer_event_role = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.95)

    def fit_step(inputs, labels, positions):
        ids, masks, tokens = inputs
        trigger_labels, role_labels = labels
        pos = positions
        model_trigger_weights = []

        with tf.GradientTape() as tape:

            trigger_predict, hidden = model_trigger((ids, masks, tokens))

            weights_trigger = concatenate_weights_trigger(hidden, pos)

            role_predict = model_role((hidden, weights_trigger))

            loss_trigger = loss4model(t=trigger_labels, p=trigger_predict, mask=masks)

            loss_role = loss4model(t=role_labels, p=role_predict, mask=masks)

            loss = loss_trigger + loss_role

            for var in model_trigger.trainable_variables:
                model_name = var.name
                none_bert_layer = ['tf_bert_model/bert/pooler/dense/kernel:0',
                                   'tf_bert_model/bert/pooler/dense/bias:0',
                                   'Variable:0']
                if model_name not in none_bert_layer:
                    model_trigger_weights.append(var)

            model_role_weights = model_role.trainable_weights

        params_all = tape.gradient(loss, [model_trigger_weights, model_role_weights])

        params_trigger = params_all[0]
        params_event_role = params_all[1]

        optimizer_trigger.apply_gradients(zip(params_trigger, model_trigger_weights))
        optimizer_event_role.apply_gradients(zip(params_event_role, model_role_weights))

        return trigger_predict, role_predict, [loss_trigger, loss_role]

    f1_value = 1e-10

    for num in range(100):
        dataset = get_dataset(path=train_ds_path)
        dataset = dataset.sample(frac=1.0)
        dataset = data_generator(dataset)

        for _, ds in enumerate(dataset):
            input_id, input_mask, input_tokens = ds['ids'], ds['masks'], ds['tokens']
            trigger_label = ds['triggers_labels']
            role_label = ds['roles_labels']
            pos = ds['triggers_positions']

            trigger_predict, role_predict, loss = fit_step(
                inputs=[input_id, input_mask, input_tokens],
                labels=[trigger_label, role_label],
                positions=pos
            )

            if _ % 200 == 0:
                print(f"times: {datetime.now()}, epoch: {_}, loss trigger: {loss[0]}, loss event: {loss[1]}")
                for i in range(2):
                    print("trigger predict start: ", np.where(trigger_predict[i][:, :, 0] > 0.5))
                    print("trigger predict end: ", np.where(trigger_predict[i][:, :, 1] > 0.5))
                    print("trigger label start: ", np.where(trigger_label[i][:, :, 0] > 0.5))
                    print("trigger label end: ", np.where(trigger_label[i][:, :, 1] > 0.5))

                    print("role predict start: ", np.where(role_predict[i][:, :, 0] > 0.5))
                    print("role predict end: ", np.where(role_predict[i][:, :, 1] > 0.5))
                    print("role label: ", np.where(role_label[i][:, :, 0] > 0.5))
                    print("role label: ", np.where(role_label[i][:, :, 1] > 0.5))

        model_trigger.save(model_trigger_path + 'mid_model/')
        model_role.save(model_event_role_path + 'mid_model/')
        f1 = valid()
        print("last f1 value: ", f1_value)
        print("new f1 value: ", f1)
        if f1 > f1_value:
            f1_value = f1
            model_trigger.save(model_trigger_path + 'best_model/')
            model_role.save(model_event_role_path + 'best_model/')


def predict_result():
    model_trigger = tf.saved_model.load(model_trigger_path + 'best_model/')
    model_role = tf.saved_model.load(model_event_role_path + 'best_model/')

    test_ds = open(test_ds_path)

    with open('duee_result.json', 'a') as f:
        for ds in test_ds.readlines():
            result = {}
            ds = json.loads(ds)
            content = ds['text']
            valid_id = ds['id']
            result['id'] = valid_id
            predicts = predict(content, model_trigger, model_role)
            result['event_list'] = predicts
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
