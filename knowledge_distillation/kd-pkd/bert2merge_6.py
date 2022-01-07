# *- coding: utf-8 -*
import json
import os
import re

import warnings

import numpy as np
import pandas as pd

import tensorflow as tf

from bert4keras.snippets import sequence_padding
from sklearn.metrics import f1_score
from transformers import BertTokenizer, TFBertModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-12_H-768")

batch_size = 20
max_len = 68


def data_cleans():
    dataset = pd.read_csv('dataset/simp/online_shopping_10_cats.csv')

    dataset = dataset.sample(frac=1.0)

    dataset['review'] = dataset['review'].apply(
        lambda x: re.sub(u'\W{2,}', ' ', str(x))
    )

    dataset['review'] = dataset['review'].apply(
        lambda x: re.sub(u'\s{2,}', ' ', str(x))
    )

    dataset['review'] = dataset['review'].apply(
        lambda x: np.NAN if len(str(x)) < 2 or len(str(x)) > 68 else x
    )

    dataset.dropna(inplace=True)

    kind_sort = {
        '计算机': 0, '酒店': 2, '蒙牛': 4, '平板': 6, '热水器': 8, '书籍': 10, '水果': 12, '洗发水': 14, '衣服': 16, '手机': 18}

    dataset['cat'] = dataset['cat'].apply(
        lambda x: kind_sort[x]
    )

    dataset['label'] = dataset['label'] + dataset['cat']

    return dataset


dataset = pd.read_excel('dataset/dataset_sample.xlsx')
dataset.dropna(inplace=True)
train_dataset = dataset[:41000]
test_dataset = dataset[41000:44000]
valid_dataset = dataset[44000:]


def data_generator():
    for _, ds in train_dataset.iterrows():
        text = ds['review']

        inputs = tokenizer.encode_plus(text)

        input_id = inputs['input_ids']
        input_mask = inputs['attention_mask']
        input_token = inputs['token_type_ids']

        label = np.zeros(shape=20)

        label[ds['label']] = 1

        input_id = sequence_padding([input_id], length=max_len)[0]
        input_mask = sequence_padding([input_mask], length=max_len)[0]
        input_token = sequence_padding([input_token], length=max_len)[0]

        yield (input_id, input_mask, input_token, label)


class Bert4Teacher(tf.keras.Model):

    def __init__(self, output_dim):
        super(Bert4Teacher, self).__init__(output_dim)
        self.output_dim = output_dim

        self.bert = TFBertModel.from_pretrained('uer/chinese_roberta_L-12_H-768', output_hidden_states=True)

        self.dense = tf.keras.layers.Dense(self.output_dim, activation='softmax')

    @tf.function(input_signature=[(tf.TensorSpec([None, max_len], name='input_ids', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='masks', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='tokens', dtype=tf.int32))])
    def call(self, batch_data):
        input_ids, masks, tokens = batch_data

        hidden = self.bert(input_ids, masks, tokens)

        dense_hidden = tf.keras.layers.Lambda(lambda x: x[:, 0])(hidden[0])

        trigger_predict = self.dense(dense_hidden)

        return trigger_predict


class Bert4student(tf.keras.Model):
    def __init__(self, output_dim):
        super(Bert4student, self).__init__(output_dim)
        self.output_dim = output_dim

        self.bert = TFBertModel.from_pretrained('uer/chinese_roberta_L-6_H-768', output_hidden_states=True)

        self.dense = tf.keras.layers.Dense(self.output_dim, activation='softmax')

    @tf.function(input_signature=[(tf.TensorSpec([None, max_len], name='input_ids', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='masks', dtype=tf.int32),
                                   tf.TensorSpec([None, max_len], name='tokens', dtype=tf.int32))])
    def call(self, batch_data):
        input_ids, masks, tokens = batch_data

        hidden = self.bert(input_ids, masks, tokens)

        dense_hidden = tf.keras.layers.Lambda(lambda x: x[:, 0])(hidden[0])

        trigger_predict = self.dense(dense_hidden)

        return trigger_predict


def get_seq_ds(content):
    ids, masks, tokens = [], [], []

    inputs = tokenizer.encode_plus(content)

    input_id = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_token = inputs['token_type_ids']

    ids.append(tf.constant(input_id, dtype=tf.int32))
    masks.append(tf.constant(input_mask, dtype=tf.int32))
    tokens.append(tf.constant(input_token, dtype=tf.int32))

    ids = sequence_padding(ids, max_len)
    masks = sequence_padding(masks, max_len)
    tokens = sequence_padding(tokens, max_len)

    return ids, masks, tokens


def valid_process(path):
    model = tf.saved_model.load(path)

    predicts = []

    for num, ds in valid_dataset.iterrows():
        text = ds['review']
        labels = np.zeros(20)
        labels[ds['label']] = 1

        ids, mask, tokens = get_seq_ds(text)

        predict = model((ids, mask, tokens))

        predict = np.nanargmax(np.array(predict[0]))

        predicts.append(predict)

    labels = np.array(valid_dataset['label'])

    f1 = f1_score(y_true=labels, y_pred=np.array(predicts), average='weighted')

    return f1


def test_process(path):
    model = tf.saved_model.load(path)

    predicts = []

    for num, ds in test_dataset.iterrows():
        text = ds['review']
        labels = np.zeros(20)
        labels[ds['label']] = 1

        ids, mask, tokens = get_seq_ds(text)

        predict = model((ids, mask, tokens))

        predict = np.nanargmax(np.array(predict[0]))

        predicts.append(predict)

    labels = np.array(test_dataset['label'])

    f1 = f1_score(y_true=labels, y_pred=np.array(predicts), average='weighted')

    return f1

path = ['student_save_merge/best_model_merge_6_last/', 'student_save_merge/mid_model_merge_6_last/', 'student_save_merge/mid_model_merge_6_epoch5_last/']

# skpi_2_best: 0.8607966106259194
# skpi_2_mid_last: 0.8223111150082856
# skpi_2_mid_epoch_5: 0.8172534203258751

# last_best: 0.858555568236078
# last_mid_last: 0.7735008037006058
# last_mid_epoch_5: 0.79941028009999


for pth in path:
    print("pth: ", pth)
    print(test_process(pth))


def fit():
    teacher_model = Bert4Teacher(20)
    student_model = Bert4student(20)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.99, beta_2=0.95)

    def fit_step(inputs, target):

        with tf.GradientTape() as tape:

            predict_tea = teacher_model(inputs)

            loss_value_tea = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=target, y_pred=predict_tea))

            predict_stu = student_model(inputs)

            loss_value_stu = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=target, y_pred=predict_stu))

            loss_value = loss_value_tea + loss_value_stu

        teacher_var = []

        for var in teacher_model.trainable_variables:
            none_bert_layer = ['tf_bert_model/bert/pooler/dense/kernel:0',
                               'tf_bert_model/bert/pooler/dense/bias:0']
            if var.name in none_bert_layer:
                continue
            else:
                teacher_var.append(var)

        # -----------------------------------------
        # 对student模型进行蒸馏操作

        stu_attention_var = []
        stu_other_var = []
        for var in student_model.trainable_variables:
            none_bert_layer = ['tf_bert_model_1/bert/pooler/dense/kernel:0',
                               'tf_bert_model_1/bert/pooler/dense/bias:0']
            if var.name in none_bert_layer:
                continue
            elif 'attention' in var.name:
                stu_attention_var.append(var)
            else:
                stu_other_var.append(var)

        params = tape.gradient(loss_value, [teacher_var, stu_other_var])

        teacher_params = params[0]

        stu_other_params = params[1]

        optimizer.apply_gradients(zip(teacher_params, teacher_var))

        optimizer.apply_gradients(zip(stu_other_params, stu_other_var))

        # 对student模型的attention层进行蒸馏处理
        # 与teacher模型层编号为 2 * stu_layer_num + 1(skip操作即stu编号为2的attention层与tea编号为5的attention层进行均值操作)
        stu_attention_params = []
        for var in student_model.trainable_variables:

            stu_var_name = ''.join(var.name.split('/')[1:])
            stu_var_num = re.findall(pattern=u'\d', string=stu_var_name)[0]
            stu_var_num = int(stu_var_num) + 6
            # stu_var_num = int(stu_var_num) + 6  ----last
            # stu_var_num = int(stu_var_num) * 2 + 1  --------skip

            name1 = re.sub(u'\d', str(stu_var_num), stu_var_name, 1)

            for var2 in teacher_model.trainable_variables:
                name2 = ''.join(var2.name.split('/')[1:])
                # print("name2: ", name2)
                if 'attention' in var2.name:
                    if name2 == name1:
                        stu_attention_params.append(tf.Variable((var + var2) / 2))

        optimizer.apply_gradients(zip(stu_attention_params, stu_attention_var))

        return loss_value

    f1_value_start_stu = 1e-5
    f1_value_start_tea = 1e-5

    train_ds = tf.data.Dataset.from_generator(
        data_generator,
        (tf.int32, tf.int32, tf.int32, tf.float32),
        (tf.TensorShape([68]), tf.TensorShape([68]), tf.TensorShape([68]), tf.TensorShape([20]))
    )

    train_ds = train_ds.shuffle(buffer_size=batch_size * 20).batch(batch_size=batch_size)

    for num in range(10):
        for _, ds in enumerate(train_ds):
            ids, masks, tokens, labels = ds[0], ds[1], ds[2], ds[3]

            loss = fit_step([ids, masks, tokens], labels)

            if (_ + 1) % 400 == 0:
                print(loss)

        student_model.save('student_save_merge/mid_model_merge_6_last/')

        if num == 4:
            student_model.save('student_save_merge/mid_model_merge_6_epoch5_last/')

        f1_stu = valid_process(path='student_save_merge/mid_model_merge_6_last/')

        print("new_f1_stu: ", f1_stu)

        print("last_f1_stu: ", f1_value_start_stu)

        if f1_stu > f1_value_start_stu:
            f1_value_start_stu = f1_stu
            teacher_model.save('student_save_merge/best_model_merge_6_last/')


fit()