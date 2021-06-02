# *- coding: utf-8 -*

import os
import warnings

from bert4keras.snippets import DataGenerator, sequence_padding
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

import codecs
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import torch

from torch import tensor, long


from tensorflow import function
from transformers import BertTokenizer, TFBertModel, BertModel, AutoModel

tf.config.experimental_run_functions_eagerly(True)

path = '../../pretrain_models/chinese_wwm_ext_L-12_H-768_A-12/'

config_path = path + 'bert_config.json'
model_path = path + 'bert_model.ckpt'
vocab_path = path + 'vocab.txt'

max_len = 128
batch_size = 16
epochs = 50
hidden_dim = 200

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
vocab_size = len(tokenizer.get_vocab())
embedding_dim = 300
num_class = len(['O', 'I-LOC', 'B-LOC', 'I-ORG', 'B-ORG', 'B-PER', 'I-PER']) + 1

def load_iob2(file_path):
    """加载 IOB2 格式的数据"""
    token_seq = []
    label_seq = []
    tokens = []
    labels = []
    with codecs.open(file_path) as f:
        for index, line in enumerate(f):
            items = line.strip().split()
            if len(items) == 2:
                token, label = items
                tokens.append(token)
                labels.append(label)
            elif len(items) == 0:
                if tokens:
                    token_seq.append(tokens)
                    label_seq.append(labels)
                    tokens = []
                    labels = []
            else:
                continue

    if tokens:  # 如果文件末尾没有空行，手动将最后一条数据加入序列的列表中
        token_seq.append(tokens)
        label_seq.append(labels)

    return np.array(token_seq), np.array(label_seq)

def labels2seq(data, id2seq=False):
    if not id2seq:
        label_seq = {
            'O': 1, 'I-LOC': 2, 'B-LOC': 3, 'I-ORG': 4, 'B-ORG': 5, 'B-PER': 6, 'I-PER': 7
        }
    else:
        label_seq = {
            1: 'O', 2: 'I-LOC', 3: 'B-LOC', 4: 'I-ORG', 5: 'B-ORG', 6: 'B-PER', 7: 'I-PER'
        }

    label = [label_seq[i] for i in data]
    return label


def dataset_generator(data):
    ids, masks, tokens, labels = [], [], [], []
    data = data.sample(frac=1.0)

    for num, (_, d) in enumerate(data.iterrows()):
        content = ''.join(d['content'])
        inputs = tokenizer.encode_plus(content,
                                       add_special_tokens=True,
                                       max_length=max_len,
                                       pad_to_max_length=True,
                                       return_token_type_ids=True)

        input_id = inputs['input_ids']
        input_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        label = labels2seq(d['label'])
        label = label + (max_len - len(label)) * [0]

        ids.append(input_id)
        masks.append(input_mask)
        tokens.append(token_type_ids)
        labels.append(label)

        if len(ids) == batch_size or _ == len(data):
            yield {
                'ids': tf.constant(ids, dtype=tf.int32),
                'masks': tf.constant(masks, dtype=tf.int32),
                'tokens': tf.constant(tokens, dtype=tf.int32),
                'labels': tf.constant(labels, dtype=tf.int32),
            }
            ids, masks, tokens, labels = [], [], [], []


def get_dataset():
    token_seqs, label_seqs = load_iob2('../dataset/dh_msaa.txt')
    content_ = pd.DataFrame(token_seqs, columns=['content'])
    labels_ = pd.DataFrame(label_seqs, columns=['label'])
    dataset = pd.concat([content_, labels_], axis=1)

    return dataset


class MyBertCrf(tf.keras.Model):
    def __init__(self, use_crf, vocab_size, output_dim):
        super(MyBertCrf, self).__init__(use_crf, vocab_size, output_dim)
        self.use_crf = use_crf
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        self.bert = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.bil_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))
        self.dense = tf.keras.layers.Dense(self.output_dim)
        self.other_params = tf.Variable(tf.random.uniform(shape=(output_dim, output_dim)))

    @tf.function
    def call(self, ids, masks, tokens, target, input_seq_len):
        # print(self.bert(ids, masks, tokens))
        hidden = self.bert(ids, masks, tokens)[0]
        # embedding = self.embedding(hidden)
        dropout_inputs = self.dropout(hidden, 1)
        # bil_lstm_outputs = self.bil_lstm(dropout_inputs)
        logistic_seq = self.dense(dropout_inputs)
        if self.use_crf:
            log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(logistic_seq,
                                                                                target,
                                                                                input_seq_len,
                                                                                self.other_params )
            predict_seq, crf_scores = tfa.text.crf_decode(logistic_seq, self.other_params , input_seq_len)
            # print("log_likelihood: ", log_likelihood)
            # print("target: ", target)
            # print("predict_seq: ", predict_seq)
            return predict_seq, log_likelihood, crf_scores
        else:
            prob_seq = tf.nn.softmax(logistic_seq)
            return prob_seq


def get_loss(log_likelihood):
    # loss = -log_likelihood / tf.cast(input_seq_len, tf.float32)
    loss = -tf.reduce_mean(log_likelihood)
    return loss


def get_acc():
    return None


def train_steps(use_crf, vocab_size, output_dim):
    # bert层的学习率与其他层的学习率要区分开来
    dataset = get_dataset()
    dataset = dataset_generator(dataset)
    bert_crf = MyBertCrf(use_crf, vocab_size, output_dim)
    opti_bert = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.95)
    opti_other = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

    def fit_models(batch_data):
        params_bert = []
        params_other = []
        ids = batch_data['ids']
        masks = batch_data['masks']
        tokens = batch_data['tokens']
        target = batch_data['labels']
        input_seq_len = tf.reduce_sum(masks, axis=1)
        # input_seq_len = tf.math.count_nonzero(masks, 1)与reduce_sum() 计算结果一致

        with tf.GradientTape() as tp:
            predict_seq, log_likelihood, crf_scores = bert_crf(ids, masks, tokens, target, input_seq_len)
            loss = -tf.reduce_mean(log_likelihood)
            for var in bert_crf.trainable_variables:
                model_name = var.name
                none_bert_layer =  ['tf_bert_model/bert/pooler/dense/kernel:0',
                                    'tf_bert_model/bert/pooler/dense/bias:0']

                if model_name in none_bert_layer:
                    pass
                elif model_name.startswith('tf_bert_model'):
                    params_bert.append(var)
                else:
                    params_other.append(var)

        params_all = tp.gradient(loss, [params_bert, params_other])
        gradients_bert = params_all[0]
        gradients_other = params_all[1]

        # gradients_other_clipped, norm_other = tf.clip_by_global_norm(gradients_other, 5.0)
        # clip的操作是为了解决梯度爆炸或者消失的问题, 但是在tf2中优化器可以通过衰减速率在控制学习率的变化
        opti_other.apply_gradients(zip(gradients_other, params_other))

        # gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
        opti_bert.apply_gradients(zip(gradients_bert, params_bert))

        return loss, predict_seq, target

    for epoch in range(epochs + 1):
        for _, data in enumerate(dataset):
            loss, predict, labels = fit_models(batch_data=data)
            if _ % 5 == 0:
                print(f"epoch: {epoch}, step: {_}, loss_value: {loss}")
            if _ % 200 == 0:
                print(f"predict: {predict[0]}, target: {labels[0]}")

        if (epoch+1) % 10 == 0:
            checkpoint = tf.train.Checkpoint(model=bert_crf)
            checkpoint_manager = tf.train.CheckpointManager(
                checkpoint, directory=f'model_save/bert_crf.ckpt.epoch_{epoch}', max_to_keep=3)
            checkpoint_manager.save()
# 训练一个epoch是46m

model = MyBertCrf(True, vocab_size, num_class)
print(model.trainable_variables[-1])
checkpoint = tf.train.Checkpoint(model=model)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=f'model_save/bert_crf.ckpt.epoch_49', max_to_keep=3)
checkpoint.restore(checkpoint_manager.latest_checkpoint)

str = '我现在在杭州的下沙区域上班, 是一个打工人'

inputs = tokenizer.encode_plus(str,
                               add_special_tokens=True,
                               max_length=max_len,
                               pad_to_max_length=True,
                               return_token_type_ids=True)

input_id = inputs['input_ids']
input_mask = inputs['attention_mask']
token_type_ids = inputs["token_type_ids"]
targets = np.zeros(shape=len(input_id))
input_seq_len = tf.reduce_sum(input_mask, axis=1)
predict_seq, log_likelihood, crf_scores = model.predict(input_id, input_mask, token_type_ids)
print(predict_seq)