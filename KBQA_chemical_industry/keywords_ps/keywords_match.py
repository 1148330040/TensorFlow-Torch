# *- coding: utf-8 -*

import json

import es_keywords


ds = json.load(open('../dataset_kbqa_ci/spider_product/product_name_values.json'))

def keywords_match(x_values, y_values):
    """考虑化工产品关键词都应用同一套,
    采用最简单直白的循环匹配
    x_values: 化工产品
    y_values: 化工产品对应的属性
    """
    all_attribute = ['中文名称', '中文别名', '英文名称', '英文别名', '美国物质编号', '欧洲化学管理局编码', '分子式', '分子量',
                     '国际化合物标识', '分子结构', '密度', '熔点', '沸点', '闪点', '水溶性', '蒸汽压', '物化性质', '产品用途']
    answer = []

    for x in x_values:
        if x not in ds.keys():
            x = es_keywords.get_top1_attribute(x, kinds=0)[0]

        for y in y_values:
            if y not in all_attribute:
                # 如果获得的y不在所有的属性列表
                # 需要用es的方式来找出匹配程度最高的属性
                # 比如用户问的是分子架构 需要为其匹配出分子结构
                y = es_keywords.get_top1_attribute(y, kinds=1)[0]

            if y in ds[x]:
                v = ds[x][y]
                if type(v) != str:
                    v= v[0]

                answer.append((x, y , v))

    return answer

