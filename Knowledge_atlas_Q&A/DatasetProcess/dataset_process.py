# *- coding: utf-8 -*
import json
import re

import warnings

import jieba
import requests

import numpy as np
import pandas as pd

import tensorflow as tf

from transformers import BertTokenizer
from bert4keras.snippets import sequence_padding

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

path = '../dataset/'

train_path = path + 'ccks2021_task13_train.txt'
test_path1 = path + 'ccks2021_task13_valid_only_questions.txt'
test_path2 = path + 'test_b_questions.txt'

# 问句中获取的关键词需要转换为标准图数据库实体

max_len = 52
batch_size = 30

tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-12_H-768")

train_dataset = pd.read_excel('../dataset/train.xlsx')
# e2w = json.load(open('../dataset/entity2words.json'))


def process_train():
    train_ds = open(train_path)

    content = []
    query  = []
    answer = []
    kind = []

    def process_keywords(text):
        if '>?' in text:
            text = text.replace('>?', '> ?')
        if 'x<' in text:
            text = text.replace('x<', 'x <')
        if 'y<' in text:
            text = text.replace('y<', 'y <')
        if '><' in text:
            text = text.replace('><', '> <')
        text = text.replace('\t', ' ')
        compile1 = re.compile(pattern=u'[{](.+?)[}]')
        # 获取对应查询语句中的{}内的内容, 这里面包含了查询语句涉及到的关键词及其关系

        keywords = re.findall(compile1, text)[0]
        keywords = keywords.split('.')[:-1]
        replace_pun = ['<', '>', '"', '?']

        for pun in replace_pun:
            keywords = [keyword.replace(pun, '') for keyword in keywords]


        keywords = [keyword.split(' ') for keyword in keywords]

        for keyword in keywords:
            while '' in keyword:
                keyword.remove('')
            if len(re.findall(r'[\u4e00-\u9fa5]', ''.join(keyword))) == 0:
                keywords.remove(keyword)

        sort = len(keywords)

        return sort, keywords

    def process_answer(text):
        text = text.strip()
        answer2 = []
        compile1 = re.compile(pattern=u'[<](.+?)[>]')
        compile2 = re.compile(pattern=u'["](.+?)["]')
        answer1 = re.findall(compile1, text)
        if "\"" in ds:
            answer2 = re.findall(compile2, text)

        answer1 += answer2

        for ans in answer1:
            if 'http' in ans:
                answer1.remove(ans)

        return answer1


    for ds in train_ds.readlines():

        if ds == '\n':
            continue
        elif ds[0] == 'q':
            ds = ds.split(':')[-1]
            content.append(ds.strip())
        elif ds[0] in ['<', '"']:
            answer.append(process_answer(ds))
        else:
            text_kind, spo_words = process_keywords(ds)
            query.append(spo_words)
            kind.append(text_kind)

    train_ds = pd.DataFrame({
        'content': content,
        'spo': query,
        'answer': answer,
        'kind': kind
    })

    return train_ds


def get_text_token(content):
    inputs = tokenizer.encode_plus(content)

    input_id = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_token = inputs['token_type_ids']

    input_id = tf.constant(sequence_padding([input_id], length=max_len), dtype=tf.int32)
    input_mask = tf.constant(sequence_padding([input_mask], length=max_len), dtype=tf.int32)
    input_token = tf.constant(sequence_padding([input_token], length=max_len), dtype=tf.int32)

    return input_id, input_mask, input_token


def get_most_similarity_word(text, label_words, train=0, top=1):
    """ 两个目标:
    1: 在训练阶段已知label_word想要获取在text中对应的keyword
    2: 在预测阶段已知text中的keyword想要获取label_word, 这里的keyword可以使三元组中的任意目标(s, p, o)均可
    3: train=1时为训练阶段数据处理, train=0是为预测阶段
    """

    post_ds = {
        'train': train,
        'top': top,
        'text': text,
        'predicate': label_words
    }

    get_ds = requests.post(url='http://192.168.0.131:8001/', json=post_ds)
    # 处理训练数据, 由于predicate已经获取无需在进行操作因此只获取与predicate最相关的关键词即可
    outputs = get_ds.json()

    if train == 1:
        keyword = outputs['word']
        label_word = outputs['predicate']

        return keyword, label_word
    else:
        sim_value = outputs['sim_value']
        top_words = outputs['top_words']

        return sim_value, top_words


def data_generator():
    # 1: subject, 2: object, 3: predicate

    special_mark = ['{', 'x', 'y', 'cvt', 'time', 'distance', 'price']

    def get_seq_pos(seq):
        """不要通过词在text中的位置信息找，因为有可能部分符号或者特殊字符编码后的长度与text长度不一致
        """
        seq_len = len(seq)
        for i in range(len(input_id)):
            if input_id[i:i + seq_len] == seq:
                return i
        return -1

    for _, ds in train_dataset.iterrows():
        text = ds['content']

        all_spo = ds['spo']

        labels = np.zeros(max_len)

        inputs = tokenizer.encode_plus(text)

        input_id = inputs['input_ids']
        input_mask = inputs['attention_mask']
        input_token = inputs['token_type_ids']

        for spo in all_spo:
            s = spo[0]
            p = spo[1]
            o = spo[2]

            # 先把关键词转为普通数据用于标识text中对应的数据
            for num, w in enumerate([s, o]):
                # subject 标记为1 object 标记为2
                if w not in special_mark:
                    try:
                        if w in text:
                            pass
                        else:
                            w = list(requests.post(url=f'http://192.168.0.131:8000/words_entity/{w}').json())[0]
                        w_token_id = tokenizer.encode(w)[1:-1]
                        w_start_pos = get_seq_pos(w_token_id)
                        if w_start_pos != -1:
                            labels[w_start_pos:(w_start_pos+len(w_token_id))] = num + 1
                    except:
                        break

            # predicate通过simbert进行相似度匹配
            # 如果p在文本中出现, 则直接进行标记操作
            # 如果p不在文本中出现, 则需要用当前的p和文本中相似度值匹配最高的词进行匹配操作

            if p in text:
                p_token_id = tokenizer.encode(p)[1:-1]
                p_start_pos = get_seq_pos(p_token_id)
                if p_start_pos != -1:
                    labels[p_start_pos:(p_start_pos+len(p_token_id))] = 3
            else:
                input_text_words = list(jieba.cut(text))
                input_predicate_words = [p]
                # 待训练数据已经给出了最合适的p因此不需要接受p-word, 但如果是待预测数据则只需要获取p-word不需要keyword

                # 将获取的文本数据和相关实体的predicate数据使用simbert处理
                # 获取最接近predicate的文本关键词
                keyword, _ = get_most_similarity_word(input_text_words, input_predicate_words, train=1, top=1)
                p_token_id = tokenizer.encode(keyword)[1:-1]
                p_start_pos = get_seq_pos(p_token_id)

                if p_start_pos != -1:
                    labels[p_start_pos:(p_start_pos+len(p_token_id))] = 3

        input_id = sequence_padding([input_id], length=max_len)[0]
        input_mask = sequence_padding([input_mask], length=max_len)[0]
        input_token = sequence_padding([input_token], length=max_len)[0]
        labels = sequence_padding([labels], length=max_len)[0]

        yield (input_id, input_mask, input_token, labels)
