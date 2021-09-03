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

from tensorflow.keras import initializers, activations

from datetime import datetime

from transformers import BertTokenizer, TFBertModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
    num = 0
    for ds in dataset.readlines():
        ds = json.loads(ds)

        text = ds['text']
        if len(text) > max_len - 3:
            continue
        num += 1
        if num > 200:
            break
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
        mask4obj = np.zeros((max_len, len(predicate2id), 2), dtype=np.int32)
        # mask4pre = np.zeros((len(predicate2id), 2), dtype=np.int32)

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

            for obj_pre in sub_obj_predicate[sub]:
                obj = obj_pre[0]
                pre = obj_pre[1]
                mask4obj[obj[0]:obj[1], pre, 1] = 1

            choice_position.append((sub, sub_obj_predicate[sub]))

        if len(choice_position) >= 1:
            subject_labels.append(mask4sub)
            object_labels.append(mask4obj)
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
        self.dense = tf.keras.layers.Dense(self.output_dim,
                                           activation='sigmoid')

        self.lambdas = tf.keras.layers.Lambda(lambda x: x ** 2)

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

        sub_predict = self.lambdas(sub_predict)

        log_likelihood = tf.constant(1)

        if self.use_crf:
            log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(sub_predict,
                                                                                target,
                                                                                input_seq_len,
                                                                                self.other_params)
            decode_predict, crf_scores = tfa.text.crf_decode(sub_predict, self.other_params, input_seq_len)

            return decode_predict, hidden, log_likelihood
        else:
            return sub_predict, hidden, log_likelihood

    def return_crf(self):
        return  self.use_crf


class ModelCrf4Obj(tf.keras.Model):
    """该类用以获取object
    """
    def __init__(self, output_dim):
        super(ModelCrf4Obj, self).__init__(output_dim)

        self.output_dim = output_dim

        self.dense = tf.keras.layers.Dense(output_dim * 2,
                                           activation='sigmoid')

        self.layer_normal = LayerNormalization(conditional=True)

        self.dropout = tf.keras.layers.Dropout(0.1)

        self.lambdas = tf.keras.layers.Lambda(lambda x: x**4)


    @tf.function(input_signature=[(tf.TensorSpec([None, max_len, 768], name='hidden', dtype=tf.float32),
                                   tf.TensorSpec([None, 768], name='weights_sub', dtype=tf.float32))])
    def call(self, inputs):
        hidden, weights_sub = inputs

        weights_sub = self.layer_normal([hidden, weights_sub])
        weights_obj = self.dropout(weights_sub, 1)

        obj_predict = self.dense(weights_obj)
        obj_predict = self.lambdas(obj_predict)

        obj_predict = tf.keras.layers.Reshape((-1, self.output_dim, 2))(obj_predict)

        return obj_predict


def loss4sub_crf(log):
    loss = -tf.reduce_mean(log)

    return loss


def loss4sub(t, p):
    loss_value = tf.keras.losses.sparse_categorical_crossentropy(y_true=t, y_pred=p)
    loss_value = tf.reduce_mean(loss_value)

    return loss_value


def loss4obj(t, p):
    loss_value = tf.keras.losses.binary_crossentropy(y_true=t, y_pred=p)
    loss_value = tf.reduce_mean(loss_value)

    return loss_value


def get_positions_hidden(positions, bert_hidden):
    """用于提取给定位置处的张量
    """

    weights = tf.ones(shape=(1, 768))

    if bert_hidden.shape[0] == 1:
        pos = (np.arange(positions[0], positions[1] + 1))
        weights = tf.gather(bert_hidden, [pos], batch_dims=1)
        weights = tf.reduce_mean(weights, axis=1)

        return weights

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

    input_id = tf.constant([inputs['input_ids'][:128]], dtype=tf.int32)
    input_mask = tf.constant([inputs['attention_mask'][:128]], dtype=tf.int32)
    token_type_ids = tf.constant([inputs["token_type_ids"][:128]], dtype=tf.int32)

    label = tf.constant([max_len * [0]], dtype=tf.int32)

    return input_id, input_mask, token_type_ids, label


# def predict(content):
#     """预测函数, 步骤如下：
#     1: 预测subject
#     2: 获取预测到的subject对应的位置信息, 并以此拆分判断预测到了几个subject
#     3: 按照循环的方式针对每一个subject进行object的预测
#     4: 步骤同上获取object位置信息, 拆分, 按照每一组(subject, object)预测predicate, 并放入到spo列表中
#     5: 整理spo列表传递到evaluate函数中"""
#
#     model_sub = tf.saved_model.load('models_save/model_sub/best_model/current_no_crf/12/')
#     model_obj = tf.saved_model.load('models_save/model_obj/best_model/current_no_crf/12/')
#     model_pre = tf.saved_model.load('models_save/model_pre/best_model/current_no_crf/12/')
#
#     input_id, input_mask, token_type_ids, label = get_predict_data(content=content)
#
#     # 需要以全局变量的形式出现
#
#     def get_positions(value):
#         position = []
#         positions = []
#
#         value = np.where(value > 0)[0]
#         for i in value:
#             if len(position) == 0:
#                 position.append(i)
#                 continue
#             if i - 1 == position[-1]:
#                 position.append(i)
#             else:
#                 positions.append(position)
#                 position = [i]
#             if i == value[-1]:
#                 positions.append(position)
#
#         positions = [(pos[0], pos[-1]) for pos in positions]
#
#         return positions
#
#     predict_sub, bert_hidden, _ = model_sub.call((input_id, input_mask, token_type_ids, label))
#
#     predict_sub = tf.argmax(predict_sub[0], axis=1)
#
#     positions_sub = get_positions(predict_sub)
#
#     all_spo = [(content[spo[0][0]:(spo[0][1] + 1)],
#                 id2predicate[spo[1]],
#                 content[spo[2][0]:(spo[2][1] + 1)]) if len(spo) > 0 else () for spo in all_spo]
#
#     return all_spo


# def evaluate():
#
#     len_lab = 1e-10
#     len_pre = 1e-10
#     len_pre_is_true = 1e-10
#     num = 0
#     val_dataset = valid_dataset
#
#     for _, ds in val_dataset.iterrows():
#         text = ds['text']
#         spo_t = ds['spo_list']
#         spo_labels = set([(spo['subject'], spo['predicate'], spo['object']['@value']) for spo in spo_t])
#
#         spo_predicts = set(predict(content=text))
#
#         len_lab += len(spo_labels)
#         len_pre += len(spo_predicts)
#         len_pre_is_true += len(spo_labels & spo_predicts)
#         num += 1
#         if num % 200==0:
#             print(text)
#             print(spo_predicts)
#             print(spo_labels)
#         # & 该运算的作用是返回两个输入内部相同的值
#         # a: [(1, 2), (3, 4), (5, 6)],  b: [(3, 4), (1, 2), (6, 5)]
#         # 返回 [(1, 2), (3, 4)]
#
#     f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
#     precision = len_pre_is_true / len_pre
#     recall = len_pre_is_true / len_lab
#
#     return f1_value, precision, recall

# print(evaluate())
# 2: 0.29
# 5: 0.4078515565097396, 0.4050070049339118, 0.4107363479120354
# 7: 0.41278370715632456, 0.39735932096825066, 0.42945391648134595
# 8: 0.4177088087631236, 0.4115848174132056, 0.42401779095626563
# 10: 0.4129040236192496, 0.4132809357304223, 0.41252779836916414

def get_predict_obj_pre(predict):
    """获取预测值的位置信息和predicate"""
    predict = tf.reduce_sum(predict, axis=2)
    predict = tf.argmax(predict, axis=1)

    value = np.where(predict > 0)[0]

    position = []
    positions = []
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
    predictae = [int(predict[pos[0]:pos[-1]].numpy().mean()) for pos in positions]

    return positions, predictae



def fit_step():

    model_sub = ModelBertCrf4Sub(output_dim=2, use_crf=False)
    opti_bert_sub = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.95)
    opti_other_sub = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

    model_obj = ModelCrf4Obj(output_dim=len(predicate2id))
    opti_obj = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

    def fit_models(inputs_model, inputs_labels, pos_subject):
        positions_sub = pos_subject
        sub_label, obj_label = inputs_labels
        ids, masks, tokens = inputs_model
        weights_bert = []  # bert参数权重获取
        weights_other = []  # crf和其它层参数权重

        with tf.GradientTape() as tape:
            predict_sub, bert_hidden, logs_sub = model_sub(batch_data=(ids, masks, tokens, sub_label))

            obj_weights = concatenate_weights_sub(bert_encoder_hidden=bert_hidden,
                                                  positions=positions_sub)

            predict_obj = model_obj(inputs=(bert_hidden, obj_weights))

            if model_sub.return_crf():
                loss_sub = loss4sub_crf(logs_sub)
            else:
                loss_sub = loss4sub(t=sub_label, p=predict_sub)

            loss_obj = loss4obj(t=obj_label, p=predict_obj)

            loss = loss_obj + loss_sub

            for var in model_sub.trainable_variables:
                model_name = var.name
                none_bert_layer = ['tf_bert_model/bert/pooler/dense/kernel:0',
                                   'tf_bert_model/bert/pooler/dense/bias:0',
                                   'Variable:0']

                if model_name in none_bert_layer:
                    pass
                elif model_name.startswith('tf_bert_model'):
                    weights_bert.append(var)
                else:
                    weights_other.append(var)

            weights_obj = model_obj.trainable_variables

        params_all = tape.gradient(loss, [weights_bert, weights_other, weights_obj])

        params_bert_sub = params_all[0]
        params_other_sub = params_all[1]
        params_obj = params_all[2]

        opti_bert_sub.apply_gradients(zip(params_bert_sub, weights_bert))
        opti_other_sub.apply_gradients(zip(params_other_sub, weights_other))

        opti_obj.apply_gradients(zip(params_obj, weights_obj))

        return predict_sub, predict_obj, loss_sub, loss_obj

    f1_value_mid, precision_mid, recall_mid = 0.370280600354258, 0.4655414908579591, 0.30738275808699417

    for num in range(1, 15):
        dataset = data_generator(spo_dataset)
        for _, data in enumerate(dataset):
            ids = data['input_ids']
            masks = data['masks']
            tokens = data['tokens']
            sub_labels = data['subject_labels']
            obj_labels = data['object_labels']
            positions = data['positions']

            sub_positions = [ps[0][0] for ps in positions]  # subject的位置信息

            # 位置信息: 用于在预测object和predicate时同bert-encoder编码信息交互

            p_sub, p_obj, loss_sub, loss_obj = fit_models(
                inputs_model=[ids, masks, tokens],
                inputs_labels=[sub_labels, obj_labels],
                pos_subject=sub_positions
            )

            if _ % 500 == 0:
                with open('models_save/fit_logs.txt', 'a') as f:
                    # 'a'  要求写入字符
                    # 'wb' 要求写入字节(str.encode(str))
                    obj_labels_pos, obj_labels_pre = get_predict_obj_pre(obj_labels[0])
                    obj_pre_pos, obj_pre_pre = get_predict_obj_pre(p_obj[0])
                    log = f"times: {datetime.now()}, " \
                          f"num: {_}, sub_loss: {loss_sub}, loss_obj: {loss_obj} \n" \
                          f"sub_labels[:1]: {sub_labels[:1]}, \n sub_predict[:1]: {tf.argmax(p_sub[:1][0], axis=1)}, \n" \
                          f"obj_labels: {obj_labels_pos, obj_labels_pre}, obj_predict: {obj_pre_pos, obj_pre_pre}"
                    print(log)
                    f.write(log)

        model_sub.save(model_sub_path+'current_no_crf/' + str(num) + '/')
        model_obj.save(model_obj_path+'current_no_crf/' + str(num) + '/')


fit_step()


