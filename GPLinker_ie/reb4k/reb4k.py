#! -*- coding:utf-8 -*-
# 三元组抽取任务，基于GlobalPointer的仿TPLinker设计
# 文章介绍：https://kexue.fm/archives/8888
# 数据集：http://ai.baidu.com/broad/download?dataset=sked
# 最优f1=0.827
# 说明：由于使用了EMA，需要跑足够多的步数(5000步以上）才生效，如果
#      你的数据总量比较少，那么请务必跑足够多的epoch数，或者去掉EMA。

import json
import numpy as np
import tensorflow as tf

from bert4keras.backend import keras, K, batch_gather
from bert4keras.tokenizers import Tokenizer
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from tqdm import tqdm
# 原keras 0.10.6
import bert4keras

print(bert4keras.__version__)

maxlen = 128
batch_size = 40
epochs = 1
config_path = '../../pretrain_models/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../pretrain_models/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../pretrain_models/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def normalize(text):
    """简单的文本格式化函数
    """
    return ' '.join(text.split())


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': normalize(l['text']),
                'spoes': [(
                    normalize(spo['subject']), spo['predicate'],
                    normalize(spo['object'])
                ) for spo in l['spo_list']]
            })
    return D


# 加载数据集
train_data = load_data('root/kg/datasets/train_data.json')
valid_data = load_data('root/kg/datasets/dev_data.json')
predicate2id, id2predicate = {}, {}

with open('root/kg/datasets/all_50_schemas') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器



class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        for is_end, d in self.sample(random):
            text = d['text'].lower()

            tokens = tokenizer.tokenize(text, maxlen=maxlen)
            mapping = tokenizer.rematch(text, tokens)
            head_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            tail_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            # 整理三元组 {(s, o, p)}
            spoes = set()
            for s, p, o in d['spoes']:
                s, o = s.lower(), o.lower()
                if s not in text or o not in text:
                    continue
                sh, oh = text.index(s), text.index(o)
                st, ot = sh + len(s) - 1, oh + len(o) - 1
                if sh in head_mapping and st in tail_mapping:
                    if oh in head_mapping and ot in tail_mapping:
                        sh, st = head_mapping[sh], tail_mapping[st]
                        oh, ot = head_mapping[oh], tail_mapping[ot]
                        if sh <= st and oh <= ot:
                            spoes.add((sh, st, predicate2id[p], oh, ot))
            # 构建标签
            entity_labels = [set() for _ in range(2)]
            head_labels = [set() for _ in range(len(predicate2id))]
            tail_labels = [set() for _ in range(len(predicate2id))]
            for sh, st, p, oh, ot in spoes:
                entity_labels[0].add((sh, st))
                entity_labels[1].add((oh, ot))
                head_labels[p].add((sh, oh))
                tail_labels[p].add((st, ot))
            for label in entity_labels + head_labels + tail_labels:
                if not label:  # 至少要有一个标签
                    label.add((0, 0))  # 如果没有则用0填充

            entity_labels = sequence_padding([list(l) for l in entity_labels])
            head_labels = sequence_padding([list(l) for l in head_labels])
            tail_labels = sequence_padding([list(l) for l in tail_labels])
            # 构建batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_entity_labels = sequence_padding(
                    batch_entity_labels, seq_dims=2
                )
                batch_head_labels = sequence_padding(
                    batch_head_labels, seq_dims=2
                )
                batch_tail_labels = sequence_padding(
                    batch_tail_labels, seq_dims=2
                )
                yield [batch_token_ids, batch_segment_ids], [
                    batch_entity_labels, batch_head_labels, batch_tail_labels
                ]
                batch_token_ids, batch_segment_ids = [], []
                batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []


def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred = K.concatenate([y_pred, zeros], axis=-1)

    if mask_zero:
        infs = zeros + K.infinity()
        y_pred = K.concatenate([infs, y_pred[..., 1:]], axis=-1)

    y_pos_2 = batch_gather(y_pred, y_true)
    y_pos_1 = K.concatenate([y_pos_2, zeros], axis=-1)
    print(y_pos_2)
    print(zeros)
    if mask_zero:
        y_pred = K.concatenate([-infs, y_pred[..., 1:]], axis=-1)
        y_pos_2 = batch_gather(y_pred, y_true)
    # print("y_pos_2 2: ", y_pos_2)
    pos_loss = K.logsumexp(-y_pos_1, axis=-1)
    all_loss = K.logsumexp(y_pred, axis=-1)
    aux_loss = K.logsumexp(y_pos_2, axis=-1) - all_loss
    aux_loss = K.clip(1 - K.exp(aux_loss), K.epsilon(), 1)
    neg_loss = all_loss + K.log(aux_loss)
    return pos_loss + neg_loss


def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    shape = K.shape(y_pred)
    # print

    y_true = y_true[..., 0] * K.cast(shape[2], K.floatx()) + y_true[..., 1]
    # print("y_true2: ", y_true)
    # print("y_pred: ", y_pred)
    y_pred = K.reshape(y_pred, (shape[0], -1, K.prod(shape[2:])))

    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, True)

    return K.mean(K.sum(loss, axis=1))


# 加载预训练模型
base = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False
)


# 预测结果
entity_output = GlobalPointer(heads=2, head_size=64)(base.model.output)
print("entity_output: ", entity_output)
head_output = GlobalPointer(
    heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
)(base.model.output)
print("head_output: ", head_output)
tail_output = GlobalPointer(
    heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
)(base.model.output)
print("tail_output: ", tail_output)
outputs = [entity_output, head_output, tail_output]

# 构建模型
model = keras.models.Model(base.model.inputs, outputs)
model.compile(loss=globalpointer_crossentropy, optimizer=Adam(2e-5))


def extract_spoes(text, threshold=0):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    outputs = model.predict([token_ids, segment_ids])
    outputs = [o[0] for o in outputs]
    # 抽取subject和object
    subjects, objects = set(), set()
    print(outputs[0].shape)
    print(outputs[0])
    print(outputs[0][:, [0, -1]])
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)
            for p in ps:
                spoes.add((
                    text[mapping[sh][0]:mapping[st][-1] + 1], id2predicate[p],
                    text[mapping[oh][0]:mapping[ot][-1] + 1]
                ))
    return list(spoes)


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


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spoes']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('best_model.weights')
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

# train_generator = data_generator(train_data, batch_size)
# for i in train_generator:
#     print(i[1][0])
# if __name__ == '__main__':
#
#     train_generator = data_generator(train_data, batch_size)
#     evaluator = Evaluator()
#
#     model.fit(
#         train_generator.forfit(),
#         steps_per_epoch=len(train_generator),
#         epochs=epochs,
#         callbacks=[evaluator]
#     )
#
# else:
#
#     model.load_weights('best_model.weights')
model.load_weights('best_model.weights')
extract_spoes('《离开》是由张宇谱曲，演唱')

# for num, ds in enumerate(data_generator(valid_data, 1)):
#     token_ids = ds[0][0]
#     segment_ids = ds[0][1]
#     print(segment_ids)
#     labels = ds[1]
#     ent_labels = labels[0]
#
#     hea_labels = labels[1]
#     tai_labels = labels[2]
#     print(ent_labels)
#     print(hea_labels)
#     print(tai_labels)
#
#     outputs = model.predict([token_ids, segment_ids])
#     ent_predict = outputs[0]
#     hea_predict = outputs[1]
#     tai_predict = outputs[2]
#
#     e_l = globalpointer_crossentropy(y_true=ent_labels, y_pred=ent_predict)
#     h_l = globalpointer_crossentropy(y_true=hea_labels, y_pred=hea_predict)
#     t_l = globalpointer_crossentropy(y_true=tai_labels, y_pred=tai_predict)
#
#     loss = e_l + h_l + t_l
#     print(ent_predict.shape, hea_predict.shape, tai_predict.shape)
#     print(K.eval(e_l), K.eval(h_l), K.eval(t_l), K.eval(loss))
#     if num == 1:
#         break

# y_p = K.constant([[
#   [[-0.2592256 , 1.0454894 ,-0.9758351 , 1.9407908 , 1.7157809 ],
#    [-3.4356883 , 0.84522855, 0.5913627 , 0.78976464, 1.2029325 ],
#    [-0.7979082 ,-1.1207677 , 1.6385337 , 1.3763942 , 1.0052536 ],
#    [ 0.5486523 , 0.17679814, 2.8184495 ,-0.73059773, 0.45005438],
#    [-0.45639783,-0.08298386,-1.4082115 , 0.66634065, 0.83254117]],
#
#   [[-2.8197222 , 0.2805484 , 0.18243302,-0.06270654, 0.3187103 ],
#    [-0.8903727 , 0.41078323, 1.7204309 , 0.1767501 ,-0.86140645],
#    [-1.6703016 ,-1.0925517 ,-0.76689625,-0.1190173 ,-0.85877097],
#    [ 1.8303224 ,-0.5352632 , 0.20828131,-0.02431715,-0.4060021 ],
#    [-0.88904095, 0.8088768 ,-0.6907064 ,-2.7910483 ,-0.3981254 ]],]])
#
#
# y_t = K.constant([[
#     [[ 0.8072489,  -0.11576363], [-0.42092773,  2.0847497 ],
#      [ 1.223819,   -1.2086424 ], [-0.6146781,   0.43398544],],
#     [[-1.3319923,  -0.2822943 ], [ 0.49303085,  0.9643995 ],
#      [ 0.97373784,  0.33546585], [ 1.0708929,  -0.5091826 ],]
# ]])
# y_t = K.random_normal(shape=(2, 4, 2))
# y_p = K.random_normal(shape=(2, 55, 55))
# print(y_t)
# print(y_p)
#
# print(K.eval(globalpointer_crossentropy(y_t, y_p)))
# test_hidden = K.ones(shape=(1, 24, 768))
# entity_output = GlobalPointer(heads=2, head_size=64)(test_hidden)
# print("entity_output: ", K.eval(entity_output))
# head_output = GlobalPointer(
#     heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
# )(test_hidden)
# print("head_output: ", K.eval(head_output))
# tail_output = GlobalPointer(
#     heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
# )(test_hidden)
# print(tail_output)
# print("tail_output: ", K.eval(tail_output))