# *- coding: utf-8 -*-
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sanic import Sanic
from sanic.response import HTTPResponse

from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

import scipy.spatial as sp

os.environ["CUDA_VISIBLE_DEVICES"]="1"

gpus = tf.config.experimental.list_physical_devices('GPU')
# 对需要进行限制的GPU进行设置
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2345)])


app = Sanic(__name__)

pd.set_option('display.max_columns', None)

max_len = 12

pretrained_path = '../SimilarityProcess/chinese_simbert_L-6_H-384_A-12/'

bert_config_path = pretrained_path + 'bert_config.json'
bert_model_path = pretrained_path + 'bert_model.ckpt'
bert_vocab_path = pretrained_path + 'vocab.txt'

bert = build_transformer_model(
    bert_config_path,
    bert_model_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

bert = keras.models.Model(bert.model.inputs, bert.model.outputs[0])

tokenizer = Tokenizer(token_dict=bert_vocab_path, do_lower_case=True)


def train_sim_bert_solve_top_1(input_text_words, input_label_words):
    """此函数用于处理词表关键词同词表的实体词匹配
    """

    # input_text_words: 输入文本相关的关键词
    # input_predicate_words: 实体对应相关predicate词

    input_text_ids = sequence_padding([tokenizer.encode(v)[0] for v in input_text_words])
    input_label_words_ids = sequence_padding([tokenizer.encode(v)[0] for v in input_label_words])

    # 获取输入文本中切割后每个词的词向量
    input_text_vec = bert.predict([input_text_ids, np.zeros_like(input_text_ids)],
                         verbose=True)

    # 获取输入实体对应predicate每个词的词向量
    input_label_words_vec = bert.predict([input_label_words_ids, np.zeros_like(input_label_words_ids)],
                         verbose=True)

    similarity_value_table = 1 - sp.distance.cdist(input_text_vec, input_label_words_vec, metric='cosine')
    sim_max_value = pd.DataFrame(similarity_value_table).stack().idxmax()
    text_max_word = sim_max_value[0]
    label_words_max_word = sim_max_value[1]

    return input_text_words[text_max_word], input_label_words[label_words_max_word]


def predict_sim_bert_solve_top_k(input_text, input_label_words, top=1):
    """此函数用于处理文本的关键词进行匹配
    """

    input_text_ids = sequence_padding([tokenizer.encode(input_text)[0]])
    input_label_words_ids = sequence_padding([tokenizer.encode(v)[0] for v in input_label_words])

    # 获取输入文本中切割后每个词的词向量
    input_text_vec = bert.predict([input_text_ids, np.zeros_like(input_text_ids)],
                         verbose=True)

    # 获取输入实体对应predicate每个词的词向量
    input_label_words_vec = bert.predict([input_label_words_ids, np.zeros_like(input_label_words_ids)],
                         verbose=True)

    similarity_value_table = 1 - sp.distance.cdist(input_text_vec, input_label_words_vec, metric='cosine')

    similarity_value_table = np.array(similarity_value_table[0])

    # 获取top-k排名的相似度label_words
    top_k_words = similarity_value_table.argsort()[::-1][:top]

    # 返回最大相似度值及其对应的word
    return similarity_value_table[top_k_words], np.array(input_label_words)[top_k_words]


@app.route('/', methods=['POST'])
async def deploy(request):
    inputs = request.json
    train = inputs['train']
    top = inputs['top']
    input_text = inputs['text']
    input_predicate_words = inputs['predicate']

    if train == 1:
        word, predicate = train_sim_bert_solve_top_1(input_text, input_predicate_words)
        information = dict()
        information['word'] = word
        information['predicate'] = predicate

    else:
        value, top_words = predict_sim_bert_solve_top_k(input_text, input_predicate_words, top)
        information = dict()
        information['sim_value'] = list(value)
        information['top_words'] = list(top_words)
        print(information)
    return HTTPResponse(json.dumps(information))



if __name__ == '__main__':

    app.run(host="0.0.0.0", port='8001')
