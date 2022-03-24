# *- coding: utf-8 -*
import re
import os
import json
import random
import pandas as pd



def process_product():
    """将爬取的化工产品进行初步的处理
    """
    product_ds = eval(open('../dataset_kbqa_ci/spider_product/product_value_ts.json').read())

    product_ds_supply = eval(open('../dataset_kbqa_ci/spider_product/product_value_supply_ts.json').read())

    product_value = {}

    product_value_supply = {}

    pattern = re.compile(u'[^\u4e00-\u9fa5]')  # 中文的范围为\u4e00-\u9fa5

    for ds in product_ds:

        if '中文名称' not in ds:
            continue

        keys = list(ds.keys())

        # 由于页面解析的问题, 部分key出现于value对应不上的问题, 因此先在此处删除这部分key
        # 然后额外爬取之后再拼接起来
        for key in keys:
            if len(key) > 4 and key not in ['InChI', 'EINECS号']:
                ds.pop(key)
        if '物化性质' in keys:
            ds.pop('物化性质')
        if '产品用途' in keys:
            ds.pop('产品用途')

        if 'EINECS号' in ds:
            ds['欧洲化学管理局编号'] = ds.pop('EINECS号')
        if 'InChI' in ds:
            ds['国际化合物标识'] = ds.pop('InChI')
        if 'CAS号' in ds:
            ds['美国物质编号'] = ds.pop('CAS号')

        if '中文名称' not in ds and '英文名称' not in ds:
            continue

        # if '中文名称' not in ds:
        #     ds['英文名称'] = re.sub(r'[0-9,.，。()（）\'-]+', '', ds['英文名称'])
        #     product_value[ds['英文名称']] = ds

        else:
            # ds['中文名称'] = re.sub(r'[0-9,.，。()（）\'-]+', '', ds['中文名称'])

            ds['中文名称'] = re.sub(pattern, '', ds['中文名称'])

            if ds['中文名称'] in product_value:
                continue
            product_value[ds['中文名称']] = ds

    for ds in product_ds_supply:
        if len(ds) == 1:
            continue
        if '中文名称' not in ds:
            continue
        ds['中文名称'] = re.sub(pattern, '', ds['中文名称'])
        ds_name = ds['中文名称']
        ds.pop('中文名称')
        product_value_supply[ds_name] = ds

    keys = list(set(product_value.keys()))

    # 完整版化工产品
    product_value_full = []
    for key in keys:

        if key in product_value_supply:
            product_value_full.append({**product_value[key], **product_value_supply[key]})
        else:
            product_value_full.append(product_value[key])

    with open('../dataset_kbqa_ci/spider_product/product_value_full.json', 'w') as f:
        f.write(str(product_value_full))


def create_product_name_values():
    """将化工产品及其对应属性构建一个完整的字典
    {'name1': [xxx], 'name2': [xxx], ...}
    用于后续获取到了产品名称是索引对应属性值
    """
    product_values = eval(open('../dataset_kbqa_ci/spider_product/product_value_full.json').read())
    product_name_values = {}
    for pv in product_values:
        if '中文名称' in pv:
            name = pv['中文名称']
            pv.pop('中文名称')
            product_name_values[name] = pv
        # else:
        #     name = pv['英文名称']
        #     pv.pop('英文名称')
        #     product_name_values[name] = pv

    with open('../dataset_kbqa_ci/spider_product/product_name_values.json', 'w') as f:
        f.write(json.dumps(product_name_values))


def fill_text_mould():
    # 语气助词
    yqzc_words = ['阿', '啊', '啦', '唉', '呢', '吧', '了',
                  '哇', '呀', '吗', '哦', '哈', '哟', '么', '']
    # 知道的近义词
    know_words = ['晓得', '晓畅', '了然', '分明', '明确', '认识', '理解',
                  '明了', '了解', '懂得', '明白', '清楚', '明晰', '知晓', '']
    #
    de_words = ['的', '得', '地', '']
    # 化工产品形容词
    xrc_words = ['易燃品', '易爆品', '化学危险品', '危险化学品', '危化品', '剧毒物品',
                 '腐蚀品', '违禁品', '放射性物品', '禁运物品', '', '', '', '', '', '', '', '']
    #
    dx_words = ['东西', '玩意', '玩意儿', '货色', '成色', '物品', '内容', '细节', '']
    # 疑问句开头引导词
    start_words = ['嘿', '喂', '嗨', '在', '哈喽', 'hi', 'hello', '问下',
                   '了解下', '沟通下', '打扰下', '打断一下', '请问', '你好', '']

    moulds = open('../dataset_kbqa_ci/问句模板.txt').read()

    moulds = moulds.split('\n')

    product_values = eval(open('../dataset_kbqa_ci/spider_product/product_value_full.json').read())
    product_name_keys = {}
    for pv in product_values:

        if '中文名称' in pv:
            name = pv['中文名称']
            pv.pop('中文名称')
        else:
            name = pv['英文名称']
            pv.pop('英文名称')
        product_name_keys[name] = list(pv.keys())

    while '' in moulds:
        moulds.remove('')


    # x: 化工产品中文名称, y: 化工产品相关属性
    x_values = list(product_name_keys.keys())

    fit_dataset = []
    for _ in range(1):
        # 构建5w * 10 组数据
        for num in range(len(x_values)):
            mould = random.choice(moulds)
            x_num = mould.count('x')
            y_num = mould.count('y')
            x_backend = random.sample(x_values, k=x_num)
            y_backend = []

            text = random.choice(start_words) + ',' + mould + random.choice(dx_words) + random.choice(yqzc_words)
            text = text.replace('的', random.choice(de_words))
            ds = {'product_names': [], 'product_values': []}

            for x in x_backend:
                # 考虑到部分化工产品本身包含了很多特殊符号, 因此从这一点上增强它的数据丰富度

                x_value = random.choice(xrc_words) + x
                text = text.replace('x', x_value, 1)
                text = text.replace('知道', random.choice(know_words))
                # 将涉及的化工产品名称保存
                ds['product_names'].append(x)
                y_backend += product_name_keys[x]

            # 考虑到多数情况下的化工产品对应的属性基本上是固定的, 因此在对化工产品替换过程中将其对应的属性合并到一起
            # 根据属性的数目进行随机抽取, 然后替换
            y_backend = list(set(y_backend))
            y_backend = random.sample(y_backend, k=y_num)

            for y in y_backend:
                text = text.replace('y', y, 1)
                # 将涉及的化工产品属性值保存
                ds['product_values'].append(y)

            ds['text'] = text
            fit_dataset.append(ds)


    with open('../dataset_kbqa_ci/spider_product/valid_dataset.json', 'w') as f:
        f.write(str(fit_dataset))


process_product()
create_product_name_values()
fill_text_mould()

# s = ['a', 'sd', 'd', 'f', 'w']
# print(random.sample(s, k=3))
# print(len(eval(open('../dataset_kbqa_ci/spider_product/fit_dataset.json').read())))