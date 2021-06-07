# *- coding: utf-8 -*


import codecs
import re
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import f1_score

from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")

tf.config.experimental_run_functions_eagerly(True)

path = '../../pretrain_models/chinese_wwm_ext_L-12_H-768_A-12/'

config_path = path + 'bert_config.json'
model_path = path + 'bert_model.ckpt'
vocab_path = path + 'vocab.txt'

epochs = 4
max_len = 128
batch_size = 16
hidden_dim = 200

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

vocab_size = len(tokenizer.get_vocab())

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

    if tokens:  # 如果文件末尾没有空行, 手动将最后一条数据加入序列的列表中
        token_seq.append(tokens)
        label_seq.append(labels)

    return np.array(token_seq), np.array(label_seq)


def labels4seq(data, id2seq=False):
    if not id2seq:
        label_seq = {
            'O': 1, 'I-LOC': 2, 'B-LOC': 3, 'I-ORG': 4, 'B-ORG': 5, 'B-PER': 6, 'I-PER': 7}
    else:
        label_seq = {
            '1': 'O', '2': 'I-LOC', '3': 'B-LOC', '4': 'I-ORG', '5': 'B-ORG', '6': 'B-PER', '7': 'I-PER', '0': ''}

    label = [label_seq[i] for i in data]

    return label


def dataset_generator(data):
    ids, masks, tokens, labels, labels_length = [], [], [], [], []

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
        label_length = len(d['label'])
        # 加入label_length的目的仅限于后续将content, label和predict值切分出来，在训练过程中不涉及

        label = labels4seq(d['label'])
        label = label + (max_len - len(label)) * [0]

        ids.append(input_id)
        masks.append(input_mask)
        tokens.append(token_type_ids)
        labels.append(label)
        labels_length.append([label_length])

        if len(ids) == batch_size or _ == len(data):
            yield {
                'ids': tf.constant(ids, dtype=tf.int32),
                'masks': tf.constant(masks, dtype=tf.int32),
                'tokens': tf.constant(tokens, dtype=tf.int32),
                'labels': tf.constant(labels, dtype=tf.int32),
                'labels_length': tf.constant(labels_length, dtype=tf.int32)
            }
            ids, masks, tokens, labels, labels_length = [], [], [], [], []


def get_dataset():
    token_seqs, label_seqs = load_iob2('../dataset/dh_msaa.txt')

    content_ = pd.DataFrame(token_seqs, columns=['content'])
    labels_ = pd.DataFrame(label_seqs, columns=['label'])

    dataset = pd.concat([content_, labels_], axis=1)

    return dataset


class MyBertCrf(tf.keras.Model):
    def __init__(self, use_crf, input_dim, output_dim):
        super(MyBertCrf, self).__init__(use_crf, input_dim, output_dim)
        self.use_crf = use_crf
        self.input_dim = input_dim
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
        dropout_inputs = self.dropout(hidden, 1)
        # bil_lstm_outputs = self.bil_lstm(dropout_inputs)
        logistic_seq = self.dense(dropout_inputs)
        if self.use_crf:
            log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(logistic_seq,
                                                                                target,
                                                                                input_seq_len,
                                                                                self.other_params )
            decode_predict, crf_scores = tfa.text.crf_decode(logistic_seq, self.other_params , input_seq_len)
            # print("log_likelihood: ", log_likelihood)
            # print("target: ", target)
            # print("predict_seq: ", predict_seq)

            return decode_predict, log_likelihood, crf_scores
        else:
            prob_seq = tf.nn.softmax(logistic_seq)
            # prob_seq -> shape(batch_size, max_len, num_class)
            # arg_max很好理解就是为了获取每个字节(包含mask)对应的num_class值
            # 正确的输出应该在此处加一个argmax的,但是考虑后续的loss计算这一步移动到其他位置

            return prob_seq, None, None


def get_loss(log_likelihood):
    loss = -tf.reduce_mean(log_likelihood)
    return loss


def get_loss_2(t, p):
    # no crf
    loss_value = tf.keras.losses.sparse_categorical_crossentropy(y_true=t, y_pred=p)
    loss_value = tf.reduce_mean(loss_value)
    return loss_value


def start(dataset, use_crf, input_dim, output_dim, fit=True):
    # bert层的学习率与其他层的学习率要区分开来
    dataset = dataset_generator(dataset)

    bert_crf = MyBertCrf(use_crf, input_dim, output_dim)

    opti_bert = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.95)
    opti_other = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

    checkpoint = tf.train.Checkpoint(model=bert_crf)
    if use_crf:
        model_ckpt = f'model_save/bert_crf_checkpoint'
    else:
        model_ckpt = f'model_save/bert_checkpoint'

    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    directory=model_ckpt,
                                                    max_to_keep=3)

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
            if use_crf:
                loss_value = get_loss(log_likelihood)
            else:
                loss_value = get_loss_2(target, predict_seq)

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

        # 不适用crf的话下面的other参数为空不需要做多余的修改
        params_all = tp.gradient(loss_value, [params_bert, params_other])
        gradients_bert = params_all[0]
        gradients_other = params_all[1]

        # gradients_other_clipped, norm_other = tf.clip_by_global_norm(gradients_other, 5.0)
        # clip的操作是为了解决梯度爆炸或者消失的问题, 但是在tf2中优化器可以通过衰减速率在控制学习率的变化
        opti_other.apply_gradients(zip(gradients_other, params_other))

        # gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
        opti_bert.apply_gradients(zip(gradients_bert, params_bert))

        return loss_value, predict_seq, target

    if fit:
        # fit
        for _, data in enumerate(dataset):
            loss, predict, labels = fit_models(batch_data=data)
            if _ % 5 == 0:
                print(f"step: {_}, loss_value: {loss}")
            if _ % 20 == 0:
                print(f"predict: {tf.argmax(predict, axis=-1)[0]}, target: {labels[0]}, ids: {data['ids'][0]}")
        checkpoint_manager.save()

        return None
    else:
        # valid
        valid_pre_label = pd.DataFrame()
        checkpoint.restore(tf.train.latest_checkpoint('model_save/bert_crf_checkpoint'))
        for _, inputs in enumerate(dataset):
            mid = pd.DataFrame()
            valid_id = inputs['ids']
            valid_mask = inputs['masks']
            valid_token = inputs['tokens']
            valid_target = inputs['labels']
            valid_target_length = inputs['labels_length']
            valid_seq_len = tf.reduce_sum(valid_mask, axis=1)

            valid_pred, _, _ = bert_crf(valid_id, valid_mask, valid_token, valid_target, valid_seq_len)

            if not use_crf:
                valid_pred = tf.argmax(valid_pred, axis=-1)

            valid_pred = pd.DataFrame(np.array(valid_pred), dtype=str)

            valid_pred['pre'] = valid_pred[valid_pred.columns].apply(
                lambda x: list(x), axis=1
            )

            valid_target = pd.DataFrame(np.array(valid_target), dtype=str)
            valid_target['label'] = valid_target[valid_target.columns].apply(
                lambda x: list(x), axis=1
            )

            valid_id = pd.DataFrame(np.array(valid_id), dtype=str)
            valid_id['ids'] = valid_id[valid_id.columns].apply(
                lambda x: list(x), axis=1
            )

            mid['length'] = pd.DataFrame(np.array(valid_target_length), columns=['label_length'])['label_length']
            mid['label'] = valid_target['label']
            mid['pre'] = valid_pred['pre']
            mid['ids'] = valid_id['ids']

            valid_pre_label = pd.concat([valid_pre_label, mid])

        return valid_pre_label


# 训练一个epoch是46m
train_path = '../dataset/train_dataset.xlsx'
valid_path = '../dataset/valid_dataset.xlsx'

def fit_model():
    train_dataset = pd.read_excel(train_path, usecols=['content', 'label'])

    train_dataset['content'] = train_dataset['content'].apply(
        lambda x: x.split('|')
    )
    train_dataset['label'] = train_dataset['label'].apply(
        lambda x: x.split('|')
    )
    start(dataset=train_dataset, use_crf=True, input_dim=vocab_size, output_dim=num_class, fit=True)


def valid_model():
    valid_dataset = pd.read_excel(valid_path, usecols=['content', 'label'])

    valid_dataset['content'] = valid_dataset['content'].apply(
        lambda x: x.split('|')
    )

    valid_dataset['label'] = valid_dataset['label'].apply(
        lambda x: x.split('|')
    )

    valid = start(dataset=valid_dataset, use_crf=True, input_dim=vocab_size, output_dim=num_class, fit=False)

    valid.dropna(inplace=True)

    def get_real_content(data, length):
        return data[1:1 + length]

    valid['content'] = valid.apply(
        lambda x: get_real_content(x['ids'], x['length']), axis=1
    )

    def get_real_label_pre(data, length):
        return data[:length]

    valid['label'] = valid.apply(
        lambda x: get_real_label_pre(x['label'], x['length']), axis=1
    )

    valid['pre'] = valid.apply(
        lambda x: get_real_label_pre(x['pre'], x['length']), axis=1
    )

    valid.to_excel('../dataset/valid_bert.xlsx')


def get_f1score():
    valid = pd.read_excel('../dataset/valid_bert.xlsx', usecols=['label', 'pre'])

    valid['label'] = valid['label'].apply(
        lambda x: re.findall(pattern='[0-9]', string=x)
    )
    valid['pre'] = valid['pre'].apply(
        lambda x: re.findall(pattern='[0-9]', string=x)
    )

    def f1score(t, p):
        t = np.array(t)
        p = np.array(p)
        return f1_score(y_true=t, y_pred=p, average="macro")

    valid['score'] = valid.apply(
        lambda x: f1score(x['label'], x['pre']), axis=1
    )

    print(valid['score'].mean())
