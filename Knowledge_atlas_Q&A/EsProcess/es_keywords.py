# -*- coding: utf-8 -*
import json
from datetime import datetime

from elasticsearch import Elasticsearch
es = Elasticsearch(hosts="121.41.47.36", port=19230)


def delete():
    try:
        es.indices.delete('huagong_es')
    except:
        pass


def make_es_dic():
    """在构建之前先进行删除指定库"""
    delete()
    print("数据删除完毕, 开始构建新的es数据")
    dataset = json.load(open('../dataset/s4po.json'))
    # 使用化工产品数据
    keys = list(set(dataset.keys()))
    for num, key in enumerate(keys):
        es_json = {
            'subject': key,
        }
        if (num+1) % 1000 == 0:
            print(num)
        es.index(index='huagong_es', doc_type='politics', body=es_json)


def get_es_top_words(word, top):
    dsl = {
            'query': {
                'match': {
                    'subject': word
                }
            },
            'size': top
        }

    es_json = es.search(index='rmyy_es', doc_type='politics', body=dsl)
    # es_json = json.dumps(es_json, indent=2, ensure_ascii=False)
    es_json = es_json['hits']['hits']

    results = []
    for ej in es_json:
        value = ej["_source"]['subject']
        results.append(value)

    return results
