import os
import re
import json
import math
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

train_ds_path = '../dataset/duee_fn/duee_fin_train.json/duee_fin_train.json'
valid_ds_path = '../dataset/duee_fn/duee_fin_dev.json/duee_fin_dev.json'
test_ds_path = '../dataset/duee_fn/duee_fin_test2.json/duee_fin_test2.json'

sample_ds_path = '../dataset/duee_fn/duee_fin_sample.json'
schema_ds_path = '../dataset/duee_fn/duee_fin_event_schema.json'

batch_size = 12
max_len = 256
# 超出最大长度的将进行滑动处理
mid_len = 128
# 滑动处理的长度
min_len = 68
# 低于最小长度的将进行连接处理
slide_len = 68
# 滑动窗口大小

model_path = f'models_save/model_crf/'


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
    events_link_roles = []
    for event, roles in schema.items():
        for role in roles.keys():
            events_link_roles.append(event+'&'+role)

    return events_link_roles

all_event_role = seq2id()

seq2ids = {key: num for num, key in enumerate(all_event_role)}
ids2seq = {num: key for num, key in enumerate(all_event_role)}


def data_clean(content):

    pattern_1 = "http://[a-zA-Z0-9.?/&=:-_]*"
    content = re.sub(pattern_1, '', content)
    pattern_2 = "【\S+】"
    content = re.sub(pattern_2, '', content)
    pattern_3 = "\.{2,7}"
    content = re.sub(pattern_3, '。', content)
    pattern_4 = "www.[a-zA-Z0-9.?/&=:-]*"
    content = re.sub(pattern_4, '', content)

    punctuations = ['？', '！', '?', '!']
    for pun in punctuations:
        content = content.replace(pun, '。')

    for pun in ['●', '■', '\n', '\t', '\u2003']:
        content = content.replace(pun, ' ')

    content = content[:-1] if content[-1] == '。' else content

    pattern_5 = '[\s]+'
    content = re.sub(pattern_5, ' ', content)


    return content.strip()


def get_seq_pos(seq, input_id):
    """不要通过词在text中的位置信息找，因为有可能部分符号或者特殊字符编码后的长度与text长度不一致
    """
    seq_len = len(seq)
    for i in range(len(input_id)):
        if input_id[i:i + seq_len] == seq:
            return i
    return -1


def slide_contents(contents):

    contents = contents.strip().split('。')
    end_contents = []
    mid_content = ''

    for content in contents:
        if len(content) >= max_len:
            # 长度大于max_len, 进行滑动处理
            num = int(len(content) // slide_len)
            for i in range(num):
                end_contents.append(content[i * slide_len:128 + (i * slide_len)])
            continue
        elif len(content) <= min_len:
            # 长度小于min_len, 进行拼接处理
            if len(mid_content) < min_len:
                mid_content += content + ' '
            else:
                end_contents.append(mid_content)
                mid_content = ''
        else:
            end_contents.append(mid_content + content)
            mid_content = ''
    end_contents.append(mid_content)

    if len(end_contents) > 1:
        if len(end_contents[-1]) < min_len:
            end_contents[-2] = end_contents[-2] + ' ' + end_contents[-1]
            end_contents = end_contents[:-1]

    return end_contents


def data_generator(path):
    dataset = open(path)


    ids, masks, tokens = [], [], []
    role_labels = []

    for ds in dataset.readlines():
        ds = json.loads(ds)
        contents = ds['text']

        try:
            event_lists = ds['event_list']
        except:
            continue

        contents = data_clean(contents)

        role_arguments = {}

        for event_list in event_lists:
            event_type = event_list['event_type']
            arguments = event_list['arguments']
            for argument in arguments:
                role = argument['role']
                argument_text = argument['argument']
                event_role = event_type + '&' + role
                role_arguments[seq2ids[event_role]] = argument_text

        contents = slide_contents(contents)
        #-----------------------------------------
        # encode

        for content in contents:

            mask4roles = np.zeros(max_len, dtype=np.float32)

            inputs = tokenizer.encode_plus(content)

            input_id = inputs['input_ids']
            input_mask = inputs['attention_mask']
            input_token = inputs['token_type_ids']

            for role, argument in role_arguments.items():
                if argument in content:
                    role_id = role
                    argument_seq = tokenizer.encode(argument)[1:-1]
                    argument_seq_pos = get_seq_pos(argument_seq, input_id)

                    if argument_seq_pos == -1:
                        continue

                    argument_seq_pos = (argument_seq_pos, argument_seq_pos + len(argument_seq))

                    mask4roles[argument_seq_pos[0]:argument_seq_pos[1]] = role_id

            ids.append(input_id)
            masks.append(input_mask)
            tokens.append(input_token)
            role_labels.append(mask4roles)

            if len(ids) == batch_size:
                ids = sequence_padding(ids, max_len)
                masks = sequence_padding(masks, max_len)
                tokens = sequence_padding(tokens, max_len)

                yield {
                    'ids': tf.constant(ids, dtype=tf.int32),
                    'masks': tf.constant(masks, dtype=tf.int32),
                    'tokens': tf.constant(tokens, dtype=tf.int32),
                    'role_labels': tf.constant(role_labels, dtype=tf.float32)
                }
                ids, masks, tokens = [], [], []
                role_labels = []



class BertCrf4Role(tf.keras.Model):
    def __init__(self, output_dim):
        super(BertCrf4Role, self).__init__(output_dim)
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

        return decode_predict, log_likelihood


def get_predict_seq(content):
    inputs = tokenizer.encode_plus(content)

    input_id = tf.constant([inputs['input_ids']], dtype=tf.int32)
    input_mask = tf.constant([inputs['attention_mask']], dtype=tf.int32)
    token_type_ids = tf.constant([inputs["token_type_ids"]], dtype=tf.int32)

    if len(input_id) > max_len - 1:
        return None

    id_length = len(input_id[0])

    input_id = sequence_padding(input_id, length=max_len)
    input_mask = sequence_padding(input_mask, length=max_len)
    token_type_ids = sequence_padding(token_type_ids, length=max_len)

    return [input_id, input_mask, token_type_ids, id_length]


def get_poses(predict_value):
    value = np.where(predict_value > 0)[0]

    poses = []
    pos = []
    for num in range(len(value)):
        if len(pos) == 0:
            pos.append(value[num])
        elif pos[-1] + 1 == value[num] and predict_value[pos[-1]] == predict_value[value[num]]:
            pos.append(value[num])
        else:
            poses.append(pos)
            pos = [value[num]]
    poses.append(pos)

    return poses


def predict(content, model):

    ids, masks, tokens, id_length = get_predict_seq(content)

    none_labels = np.zeros(shape=(1, max_len))
    none_labels = tf.constant(none_labels, dtype=tf.float32)

    predict_value, _ = model((ids, masks, tokens, none_labels))
    predict_value = predict_value[0]
    predict_poses = get_poses(np.array(predict_value))

    predict_spo = []

    for p_pos in predict_poses:

        if len(p_pos) == 0 or p_pos[-1] > id_length + 1:
            continue

        event_role_id = predict_value[p_pos[0]]

        event_role =  ids2seq[int(event_role_id)]
        event = event_role.split('&')[0]
        role = event_role.split('&')[1]

        argument = np.array(ids[0][p_pos[0]:(p_pos[-1] + 1)])
        argument = ''.join(tokenizer.decode(argument).split(' ')).strip()

        predict_spo.append(
            (
             event,
             role,
             argument
            )
        )

    return predict_spo


def valid():

    dataset = open(valid_ds_path)

    model = tf.saved_model.load(model_path + f'mid_model/')

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
        contents = ds['text']
        try:
            event_lists = ds['event_list']
            label_spo = get_spo_values(event_lists)
        except:
            continue
        contents = data_clean(contents)
        predict_spo = predict(contents, model)

        len_lab += len(label_spo)
        len_pre += len(predict_spo)
        len_pre_is_true += len(set(label_spo) & set(predict_spo))

    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    return f1_value


def fit():
    model = BertCrf4Role(output_dim=len(seq2ids)+1)
    opi_bert = tf.keras.optimizers.Adam(lr=2e-5, beta_1=0.9, beta_2=0.95)
    opi_crf = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.95)

    def fit_step(inputs, labels):
        ids, masks, tokens = inputs
        labels = labels

        params_bert = []
        params_crf = []
        with tf.GradientTape() as tape:
            predict, log_likelihood = model((ids, masks, tokens, labels))

            loss = -tf.reduce_mean(log_likelihood)

            for var in model.trainable_variables:
                model_name = var.name
                none_bert_layer = ['tf_bert_model/bert/pooler/dense/kernel:0',
                                   'tf_bert_model/bert/pooler/dense/bias:0']

                if model_name in none_bert_layer:
                    pass
                elif model_name.startswith('tf_bert_model'):
                    params_bert.append(var)
                else:
                    params_crf.append(var)

        params_all = tape.gradient(loss, [params_bert, params_crf])

        model_bert = params_all[0]
        model_crf = params_all[1]

        opi_bert.apply_gradients(zip(model_bert, params_bert))
        opi_crf.apply_gradients(zip(model_crf, params_crf))

        return predict, loss

    f1_value = 1e-10

    for num in range(100):

        dataset = data_generator(path=train_ds_path)

        for _, ds in enumerate(dataset):

            input_id, input_mask, input_tokens = ds['ids'], ds['masks'], ds['tokens']

            role_label = ds['role_labels']

            predict, loss = fit_step(
                inputs=[input_id, input_mask, input_tokens],
                labels=role_label,
            )

            if _ % 400 == 0:
                print(f"times: {datetime.now()}, epoch: {_}, loss: {loss}")
                print("role_predict: ", predict[0])
                print("role_label: ", role_label[0])

        model.save(model_path + f'mid_model/')

        f1_v = valid()

        print("last f1: ", f1_value)
        print("new f1: ", f1_v)

        if f1_v > f1_value:
            model.save(model_path + 'best_model/')
            f1_value = f1_v

fit()