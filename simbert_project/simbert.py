# *- coding: utf-8 -*-

# =================================
# time: 2021.4.9
# author: @唐志林
# function: 处理title和desc连接后的句子
# =================================

import os
import numpy as np
import pandas as pd

from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

pd.set_option('display.max_columns', None)


def sim_bert_solve(dataset):
    dataset.index = np.arange(len(dataset))

    pretrained_path = \
        '/chinese_simbert_L-6_H-384_A-12/'

    bert_config_path = os.path.join(pretrained_path, 'bert_config.json')
    bert_model_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    bert_vocab_path = os.path.join(pretrained_path, 'vocab.txt')

    bert = build_transformer_model(
        bert_config_path,
        bert_model_path,
        with_pool='linear',
        application='unilm',
        return_keras_model=False,
    )

    bert = keras.models.Model(bert.model.inputs, bert.model.outputs[0])

    tokenizer = Tokenizer(token_dict=bert_vocab_path, do_lower_case=True)

    dataset['index'] = dataset.index

    dataset['index'] = dataset['index'].apply(lambda x: str(x))

    max_length = 150

    dataset['title_link_desc'] = dataset[['index', 'title', 'desc']].apply(
        lambda x: list(x), axis=1
    )

    dataset['title_link_desc'] = dataset['title_link_desc'].apply(
        lambda x: '||'.join([str(v).strip() for v in x])
    )

    texts = dataset['title_link_desc'].values

    title_link_desc_a = dataset['title_link_desc'].values

    token_ids_a = sequence_padding([tokenizer.encode(v, maxlen=max_length)[0] for v in title_link_desc_a])

    token_ids_vec_a = bert.predict(
        [token_ids_a, np.zeros_like(token_ids_a)], verbose=True, batch_size=128
    )

    a_vec = token_ids_vec_a / (token_ids_vec_a**2).sum(axis=1, keepdims=True)**0.5

    vec_all = a_vec.reshape(-1, 384)

    def most_similar(text, top_n):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_length)
        vec = bert.predict([[token_ids], [segment_ids]])[0]
        vec /= (vec**2).sum()**0.5
        sims = np.dot(vec_all, vec)

        return [(texts[i].split('||')[:2], sims[i]) for i in sims.argsort()[::-1][1:(top_n+1)]]

    dataset['similarity_index_title_value'] = dataset['title_link_desc'].apply(
        lambda x: most_similar(text=str(x), top_n=10)
    )

    dataset['Top10_similarity_values_id'] = dataset['similarity_index_title_value'].apply(
        lambda x: [list(dataset.loc[dataset['index'] == v[0][0], '_id'])[0] for v in x]
    )

    dataset['Top10_similarity_title'] = dataset['similarity_index_title_value'].apply(
        lambda x: [v[0][1].strip() for v in x]
    )

    dataset['Top10_similarity_values'] = dataset['similarity_index_title_value'].apply(
        lambda x: [v[1] for v in x]
    )

    columns = ['_id', 'title', 'Top10_similarity_title', 'Top10_similarity_values_id']

    dataset = dataset[columns]

    return dataset
