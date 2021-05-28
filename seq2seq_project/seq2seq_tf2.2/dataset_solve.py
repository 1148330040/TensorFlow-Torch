# *- coding: utf-8 -*

import re
import os
import json
import pymongo
import numpy as np
import pandas as pd
from io import StringIO

from sklearn.model_selection import train_test_split

def dataset_clean(dataset):
    # 将title或者content空白的数据去除
    dataset.dropna(inplace=True)

    dataset['title'] = dataset['title'].astype(str)
    dataset['content'] = dataset['content'].astype(str)

    # 不存在中文的title和content去除
    pattern = u"[\u4e00-\u9fa5]"
    dataset['title'] = dataset['title'].apply(
        lambda x: np.NAN if len(re.findall(pattern=u'[\u4e00-\u9fa5]+', string=x)) == 0 else x
    )

    dataset['content'] = dataset['content'].apply(
        lambda x: np.NAN if len(re.findall(pattern=u'[\u4e00-\u9fa5]+', string=x)) == 0 else x
    )
    dataset.dropna(inplace=True)

    # 将title长度大于content的去除
    def cont_len(t, c):
        return np.NAN if len(t) >= len(c) else t

    dataset['title'] = dataset.apply(
        lambda x: cont_len(x['title'], x['content']), axis=1
    )
    dataset.dropna(inplace=True)

    # 只保留数据中的常规字符和中文字符
    pattern = r"[\u4e00-\u9fa50-9a-zA-Z，。.、？！“”（）()]+"

    dataset['title'] = dataset['title'].apply(
        lambda x: ''.join([text if len(text) > 2 else '' for text in re.findall(pattern=pattern, string=x)])
    )

    dataset['content'] = dataset['content'].apply(
        lambda x: ''.join([text if len(text) > 2 else '' for text in re.findall(pattern=pattern, string=x)])
    )

    dataset['title'] = dataset['title'].apply(
        lambda x: np.NAN if len(x) <= 1 else x
    )

    dataset['content'] = dataset['content'].apply(
        lambda x: np.NAN if len(x) <= 1 else x
    )
    dataset.dropna(inplace=True)

    return dataset


def solve_dataset():
    path = '../../../news2016ch/dataset/'

    for file in os.listdir(path):
        if file == 'link_dataset.xlsx' or file == 'seq2seq_config.json':
            pass
        else:
            print(file)
            path_ = path + file
            data = pd.read_excel(path_)
            print(f"处理前: {len(data)}")
            data = data[['news_id', 'content', 'title']].copy()
            dataset = dataset_clean(dataset=data)
            print(f"处理前: {len(data)}, 处理后: {len(dataset)}")
            # dataset.to_excel('../../../news2016ch/dataset_clean/'+file)

solve_dataset()

def link_dataset():
    path = '../news_dataset_clean/'
    dataset = pd.DataFrame({})
    for file in os.listdir(path):
        print(file)
        data = pd.read_excel(path + file, usecols=['content', 'title'])
        data['content'] = data['content'].astype(str)
        data['title'] = data['title'].astype(str)
        dataset = pd.concat([dataset, data])
    dataset.to_excel(path + 'link_dataset.xlsx')


