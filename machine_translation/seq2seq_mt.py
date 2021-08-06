# -* coding: utf-8 -*

import os
import re
import json
import pandas as pd

import tensorflow as tf


dataset = pd.DataFrame(open('dataset/spa.txt', encoding='utf-8'), columns=['ds'])
dataset = dataset.head(1000)


def dataset_process():
    punctuation = r"[^a-zA-Z?.!,¿]+"

    dataset['ds'] = dataset['ds'].apply(
        lambda x: re.sub(pattern=r"([?.!,¿])", repl=r" \1 ", string=str(x))
    )

    dataset['english'] = dataset['ds'].apply(
        lambda x: x.split('\t')[0]
    )

    dataset['spanish'] = dataset['ds'].apply(
        lambda x: x.split('\t')[1].split('\n')[0]
    )


    for col in ['english', 'spanish']:
        dataset[col] = dataset[col].apply(
            lambda x: re.sub(pattern=punctuation, repl=' ', string=str(x)).strip()
        )
        dataset[col] = dataset[col].apply(
            lambda x: re.sub(pattern="[' ']+", repl=' ', string=str(x)).strip()
        )

    dataset.pop('ds')

dataset_process()


def make_vocabs():
    word_vocabs_eng = ' '.join(list(dataset['english'].value_counts().index))
    word_vocabs_eng = word_vocabs_eng.split(' ')
    word_vocabs_eng = list(set(word_vocabs_eng))
    while '' in word_vocabs_eng:
        word_vocabs_eng.remove('')

    word_vocabs_sp = ' '.join(list(dataset['spanish'].value_counts().index))
    word_vocabs_sp = word_vocabs_sp.split(' ')
    word_vocabs_sp = list(set(word_vocabs_sp))
    while '' in word_vocabs_sp:
        word_vocabs_sp.remove('')

    word_vocabs = word_vocabs_eng + word_vocabs_sp

    num = 1
    chars2id = {}
    for word in word_vocabs:
        num = num + 1
        chars2id[word] = num

    chars2id['<start>'] = 0
    chars2id['<end>'] = 1

    id2chars = {v:k for k,v in chars2id.items()}

    return chars2id, id2chars


def word2id(contents, vocabs):
    print(contents)
    contents = contents.split(' ')
    contents = [vocabs[w] for w in contents]
    contents = [0] + contents + [1]
    return contents


def id2word(contents, vocabs):
    return ' '.join([vocabs[w] for w in contents[1:-1]])


c2id, id2c = make_vocabs()
emb_dim = 64
gru_unit = 1024
word_vocab_size = len(c2id)
batch_size = 64
max_len = 64

def dataset_generator(data):

    eng_ids, spn_ids = [], []
    data = data.sample(frac=1.0)

    for num, (_, d) in enumerate(data.iterrows()):

        eng_content =  d['english']
        spn_content = d['spanish']

        eng_id = word2id(eng_content, c2id)
        spn_id = word2id(spn_content, c2id)

        eng_ids.append(eng_id)
        spn_ids.append(spn_id)

        if len(eng_ids) == batch_size or _ == len(data):
            print(eng_ids)
            yield {
                'eng_ids': tf.constant(eng_ids, dtype=tf.int32),
                'spn_ids': tf.constant(spn_ids, dtype=tf.int32)
            }
            eng_ids, spn_ids = [], []


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, gru_units):
        super(Encoder, self).__init__()
        self.units = gru_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                   output_dim=self.embedding_dim)
        self.gru = tf.keras.layers.GRU(
            units=self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    @tf.function
    def call(self, inputs, hidden):
        inputs = self.embedding_dim(inputs)
        outputs, states = self.gru(inputs, initial_state=hidden)
        return outputs, states


def get_initial_hidden_states():
    return tf.zeros_like((batch_size, gru_unit))


dataset = dataset_generator(dataset)
for i in dataset:
    print(i)

