import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf

from bert4keras.snippets import sequence_padding
from tensorflow.keras import activations, initializers
import tensorflow_addons as tfa
from transformers import BertTokenizer, TFBertModel


os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.get_logger().setLevel('ERROR')

pd.set_option('display.max_columns', None)

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

train_ds_path = '../dataset/duee/duee_train.json/duee_train.json'
valid_ds_path = '../dataset/duee/duee_dev.json/duee_dev.json'
test_ds_path = '../dataset/duee/duee_test2.json/duee_test2.json'
schema_ds_path = '../dataset/duee/duee_event_schema.json'

max_len = 128
batch_size = 25

model_trigger_path = f'models_save/model_trigger_crf/'
model_role_path = f'models_save/model_role_crf/'


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
    events2id = {event:num+1 for num, event in enumerate(schema.keys())}

    all_roles = []
    for schema_value in schema.values():
        for role in schema_value.keys():
            if role not in all_roles:
                all_roles.append(role)
    roles2id = {role:num+1 for num, role in enumerate(all_roles)}

    return events2id, roles2id, schema

events2id, roles2id, schema= seq2id()
id2events = {id:event for event, id in events2id.items()}
id2roles = {id:role for role, id in roles2id.items()}


def data_generator(path):

    dataset = open(path)

    def get_seq_pos(seq):
        """不要通过词在text中的位置信息找，因为有可能部分符号或者特殊字符编码后的长度与text长度不一致
        """
        seq_len = len(seq)
        for i in range(len(input_id)):
            if input_id[i:i + seq_len] == seq:
                return i
        return -1

    ids, masks, tokens = [], [], []
    trigger_labels, role_labels, trigger_position = [], [], []

    for ds in dataset.readlines():
        ds = json.loads(ds)

        text = ds['text']
        events = ds['event_list']

        text = ''.join([
            "，" if t == " " or t == "\n" or t == "\t" else "\"" if t == "“" else t
            for t in list(text.lower())
        ])

        inputs = tokenizer.encode_plus(text)

        input_id = inputs['input_ids']
        input_mask = inputs['attention_mask']
        input_token = inputs['token_type_ids']

        if len(input_id) > 128:
            continue

        mask_trigger_label = np.zeros(shape=max_len)
        mask_role_label = np.zeros(shape=max_len)
        all_arguments = []
        for event in events:
            arguments = []
            event_type = event['event_type']
            trigger = event['trigger']
            argument = event['arguments']

            if bool(argument) is False:
                continue

            event_type_id = events2id[event_type]

            triggers_seq = tokenizer.encode(trigger)[1:-1]
            triggers_pos = get_seq_pos(triggers_seq)

            if triggers_pos == -1:
                continue

            triggers_pos = (triggers_pos, triggers_pos + len(triggers_seq))

            mask_trigger_label[triggers_pos[0]:triggers_pos[1]] = event_type_id

            for arg in argument:
                role = arg['role']
                role_argument = arg['argument']
                arguments.append((role, role_argument))

            all_arguments.append((arguments, triggers_pos))

        if bool(all_arguments) is False:
            continue

        argument_pos = random.choice(all_arguments)
        arguments = argument_pos[0]
        pos = argument_pos[1]

        for argument in arguments:

            role = argument[0]
            role_word = argument[1]

            role_id = roles2id[role]

            role_word_seq = tokenizer.encode(role_word)[1:-1]
            role_word_pos = get_seq_pos(role_word_seq)

            if role_word_pos == -1:
                continue

            role_word_pos = (role_word_pos, role_word_pos + len(role_word_seq))

            mask_role_label[role_word_pos[0]:role_word_pos[1]] = role_id

        ids.append(input_id)
        masks.append(input_mask)
        tokens.append(input_token)
        trigger_labels.append(mask_trigger_label)
        role_labels.append(mask_role_label)
        trigger_position.append(pos)

        if len(ids) == batch_size:
            ids = sequence_padding(ids, max_len)
            masks = sequence_padding(masks, max_len)
            tokens = sequence_padding(tokens, max_len)
            trigger_labels = sequence_padding(trigger_labels, max_len)
            role_labels = sequence_padding(role_labels, max_len)

            yield {
                'ids': tf.constant(ids, dtype=tf.int32),
                'masks': tf.constant(masks, dtype=tf.int32),
                'tokens': tf.constant(tokens, dtype=tf.int32),
                'trigger_labels': tf.constant(trigger_labels, dtype=tf.float32),
                'role_labels': tf.constant(role_labels, dtype=tf.float32),
                'trigger_position': np.array(trigger_position),
            }

            ids, masks, tokens = [], [], []
            trigger_labels, role_labels, trigger_position = [], [], []


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



class BertCrf4Trigger(tf.keras.Model):
    def __init__(self, output_dim):
        super(BertCrf4Trigger, self).__init__(output_dim)
        self.output_dim = output_dim

        self.bert = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext', output_hidden_states=True)

        self.dropout = tf.keras.layers.Dropout(0.3)

        self.dense = tf.keras.layers.Dense(self.output_dim)

        self.other_params = tf.Variable(tf.random.uniform(shape=(output_dim, output_dim)))


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len], name='input_ids', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='masks', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='tokens', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='target', dtype=tf.float32))])
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

        return decode_predict, hidden[2][-2], log_likelihood


class BertCrf4Role(tf.keras.Model):
    def __init__(self, output_dim):
        super(BertCrf4Role, self).__init__(output_dim)
        self.output_dim = output_dim

        self.dense = tf.keras.layers.Dense(self.output_dim)

        self.dropout = tf.keras.layers.Dropout(0.3)

        self.layer_normal = LayerNormalization(conditional=True)

        self.other_params = tf.Variable(tf.random.uniform(shape=(output_dim, output_dim)))


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len, 768], name='hidden', dtype=tf.float32),
                                   tf.TensorSpec([None, 768], name='weights_trigger', dtype=tf.float32),
                                   tf.TensorSpec([None, max_len], name='target', dtype=tf.float32),
                                   tf.TensorSpec([None, max_len], name='masks', dtype=tf.int32),)])
    def call(self, inputs):
        hidden, weights_trigger, target, masks = inputs

        input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)

        weights_trigger = self.layer_normal([hidden, weights_trigger])

        weights_trigger = self.dropout(weights_trigger, 1)

        roles_predict = self.dense(weights_trigger)

        log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(roles_predict,
                                                                            target,
                                                                            input_seq_len,
                                                                            self.other_params)

        decode_predict, crf_scores = tfa.text.crf_decode(roles_predict, self.other_params, input_seq_len)

        return decode_predict, log_likelihood


def get_positions_hidden(positions, bert_hidden):
    """用于提取给定位置处的张量
    """
    weights = tf.ones(shape=(1, 768))
    if bert_hidden.shape[0] == 1:
        poses = (np.arange(positions[0], positions[-1]+1))
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

def predict(content, model_trigger, model_role):

    def get_poses(value):
        value = np.where(value > 0)[0]

        poses = []
        pos = []
        for num in range(len(value)):
            if len(pos)==0 or pos[-1]+1 == value[num]:
                pos.append(value[num])
            else:
                poses.append(pos)
                pos = [value[num]]
        poses.append(pos)

        return poses

    input_predicts = []
    input_predict = {}

    ids, masks, tokens, id_length = get_predict_seq(content)

    none_labels = np.zeros(shape=(1, max_len))
    none_labels = tf.constant(none_labels, dtype=tf.float32)
    # 为crf做一个假label满足形状(None, max_len)即可

    trigger_predict, hidden, _ = model_trigger((ids, masks, tokens, none_labels))
    trigger_predict = trigger_predict[0]
    trigger_poses = get_poses(np.array(trigger_predict))

    predicts = []

    for pos in trigger_poses:

        if len(pos) == 0 or pos[-1] > id_length + 1:
            continue

        event = trigger_predict[pos[0]]
        event = id2events[int(event)]
        input_predict['event_type'] = event
        weights_trigger = concatenate_weights_trigger(hidden, [pos[0], pos[-1]])

        role_predict, _ = model_role((hidden, weights_trigger, none_labels, masks))
        role_predict = role_predict[0]
        role_poses = get_poses(np.array(role_predict))

        input_roles = {}
        input_predict['arguments'] = []
        for pos2 in role_poses:

            if len(pos2) == 0 or pos2[-1] > id_length + 1:
                continue

            role = role_predict[pos2[0]]
            role = id2roles[int(role)]

            argument = np.array(ids[0][pos2[0]:(pos2[-1] + 1)])
            argument = ''.join(tokenizer.decode(argument).split(' ')).strip()

            if role not in input_roles.keys():
                input_roles['role'] = role

            input_roles['argument'] = argument
            input_predict['arguments'].append(input_roles)
            input_roles = {}
        input_predicts.append(input_predict)
        input_predict = {}
    return input_predicts


def valid():
    dataset = open(valid_ds_path)

    model_trigger = tf.saved_model.load(model_trigger_path + 'mid_model/')
    model_role = tf.saved_model.load(model_role_path + 'mid_model/')

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
                        argument.lower()
                    )
                )
        return spo

    len_lab = 1e-10
    len_pre = 1e-10
    len_pre_is_true = 1e-10

    for ds in dataset.readlines():
        ds = json.loads(ds)
        text = ds['text']
        events = ds['event_list']

        labels = get_spo_values(events)

        text = ''.join([
            "，" if t == " " or t == "\n" or t == "\t" else "\"" if t == "“" else t
            for t in list(text.lower())
        ])
        predicts = get_spo_values(predict(text, model_trigger, model_role))
        len_lab += len(labels)
        len_pre += len(predicts)
        len_pre_is_true += len(set(labels) & set(predicts))

    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    return f1_value


def fit():
    model_trigger = BertCrf4Trigger(output_dim=len(events2id)+1)
    opi_bert = tf.keras.optimizers.Adam(lr=2e-5, beta_1=0.9, beta_2=0.95)
    opi_crf = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.95)

    model_role = BertCrf4Role(output_dim=len(roles2id)+1)
    opi_crf_2 = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.95)

    def fit_step(inputs, labels, positions):
        ids, masks, tokens = inputs
        trigger_labels, role_labels = labels
        pos = positions
        params_trigger_bert = []
        params_trigger_crf = []
        with tf.GradientTape() as tape:
            trigger_predict, hidden, log_likelihood = model_trigger((ids, masks, tokens, trigger_labels))
            weights_trigger = concatenate_weights_trigger(hidden, pos)

            role_predict, log_likelihood_2 = model_role((hidden, weights_trigger, role_labels, masks))

            loss_trigger = -tf.reduce_mean(log_likelihood)

            loss_role = -tf.reduce_mean(log_likelihood_2)

            loss = loss_trigger + loss_role

            for var in model_trigger.trainable_variables:
                model_name = var.name
                none_bert_layer = ['tf_bert_model/bert/pooler/dense/kernel:0',
                                   'tf_bert_model/bert/pooler/dense/bias:0']

                if model_name in none_bert_layer:
                    pass
                elif model_name.startswith('tf_bert_model'):
                    params_trigger_bert.append(var)
                else:
                    params_trigger_crf.append(var)

            params_role = model_role.trainable_weights

        params_all = tape.gradient(loss, [params_trigger_bert, params_trigger_crf, params_role])

        trigger_bert = params_all[0]
        trigger_crf = params_all[1]
        role_crf = params_all[2]

        opi_bert.apply_gradients(zip(trigger_bert, params_trigger_bert))
        opi_crf.apply_gradients(zip(trigger_crf, params_trigger_crf))
        opi_crf_2.apply_gradients(zip(role_crf, params_role))

        return trigger_predict, role_predict, [loss_trigger, loss_role]

    f1_value = 1e-10
    for num in range(100):
        dataset = data_generator(path=train_ds_path)

        for _, ds in enumerate(dataset):
            input_id, input_mask, input_tokens = ds['ids'], ds['masks'], ds['tokens']
            trigger_label = ds['trigger_labels']
            role_label = ds['role_labels']
            pos = ds['trigger_position']
            trigger_predict, role_predict, loss = fit_step(
                inputs=[input_id, input_mask, input_tokens],
                labels=[trigger_label, role_label],
                positions=pos
            )

            if _ % 400 == 0:
                print(f"times: {datetime.now()}, epoch: {_}, loss trigger: {loss[0]}, loss role: {loss[1]}")
                print("trigger_predict: ", trigger_predict[0])
                print("trigger_label: ", trigger_label[0])

                print("role_predict: ", role_predict[0])
                print("role_label: ", role_label[0])

        model_trigger.save(model_trigger_path + f'mid_model/')
        model_role.save(model_role_path + f'mid_model/')

        f1_v = valid()

        print("last f1: ", f1_value)
        print("new f1: ", f1_v)

        if f1_v > f1_value:
            model_trigger.save(model_trigger_path + 'best_model/')
            model_role.save(model_role_path + 'best_model/')
            f1_value = f1_v


def get_predict_result():

    model_trigger = tf.saved_model.load(model_trigger_path + f'best_model/')
    model_role = tf.saved_model.load(model_role_path + f'best_model/')

    test_ds = open(test_ds_path)

    with open('duee_result_crf.json', 'a') as f:
        for ds in test_ds.readlines():
            result = {}
            ds = json.loads(ds)
            content = ds['text']
            valid_id = ds['id']
            result['id'] = valid_id
            predicts = predict(content, model_trigger, model_role)
            result['event_list'] = predicts
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

# get_predict_result()