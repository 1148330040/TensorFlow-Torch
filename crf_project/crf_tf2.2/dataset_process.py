# *- coding: utf-8 -*


import codecs
import random
import numpy as np
import pandas as pd

from bert4keras.tokenizers import Tokenizer


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
                print('格式错误。行号：{} 内容：{}'.format(index, line))
                continue

    if tokens:  # 如果文件末尾没有空行，手动将最后一条数据加入序列的列表中
        token_seq.append(tokens)
        label_seq.append(labels)

    return np.array(token_seq), np.array(label_seq)


token_seqs, label_seqs = load_iob2('../dataset/dh_msaa.txt')

print(token_seqs)
print(label_seqs)

path = '../../pretrain_models/chinese_wwm_ext_L-12_H-768_A-12/'

config_path = path + 'bert_config.json'
model_path = path + 'bert_model.ckpt'
vocab_path = path + 'vocab.txt'


tokenizer = Tokenizer(vocab_path, do_lower_case=True)

token_link_seq = [''.join(i) for i in token_seqs[:10]]

print(len(token_seqs[0]))
print(len(token_link_seq[0]))
t = tokenizer.encode(token_link_seq[0])[0]
t = tokenizer.decode(t)
print(t)
