# -*- coding: utf-8 -*
import json
from datetime import datetime

from elasticsearch import Elasticsearch

es = Elasticsearch(hosts="x.x.x.x", port=0)



def delete_all():
    try:
        es.indices.delete('attribute_match')
        es.indices.delete('product_match')
    except:
        pass


def create_attribute():
    """为所有属性名称构建一个库用以对属性的相似度匹配
    """

    all_attribute = ['中文名称', '中文别名', '英文名称', '英文别名', '美国物质编号', '欧洲化学管理局编码', '分子式', '分子量',
                     '国际化合物标识', '分子结构', '密度', '熔点', '沸点', '闪点', '水溶性', '蒸汽压', '物化性质', '产品用途']

    for att in all_attribute:
        es_json = {
            'attribute': att,
        }

        es.index(index='attribute_match', doc_type='politics', body=es_json)


def create_product():
    """为所有的化工产品构建一个库, 用以化工产品的相似度匹配
    """
    product = json.load(open('../dataset_kbqa_ci/spider_product/product_name_values.json'))
    product = list(product.keys())

    for pro in product:
        es_json = {
            'product': pro
        }

        es.index(index='product_match', doc_type='politics', body=es_json)

# create_product()
# create_attribute()

# ['八烷基二甲基氯化铵', 'RS[环丙基氟苯基喹啉基]二羟基庚酸+苯乙胺', 'NN二乙基癸酰胺', 'N乙基吡咯烷酮', '异喹啉硼酸盐酸盐', '甲基噻酚甲醛', '甲基溴嘧啶', '异丙基异氰酸酯', '喹尼氟', 'NBoc氧代脯氨酸']

def get_top1_attribute(word, kinds, top=1):

    index = 'product_match'
    match_index = 'product'

    if kinds == 1:
        index = 'attribute_match'
        match_index = 'attribute'

    dsl = {
        'query': {
            'match': {
                match_index: word
            }
        },
        'size': top
    }

    es_json = es.search(index=index, doc_type='politics', body=dsl)
    es_json = es_json['hits']['hits']

    results = []
    for ej in es_json:
        value = ej["_source"][match_index]
        results.append(value)

    return results
