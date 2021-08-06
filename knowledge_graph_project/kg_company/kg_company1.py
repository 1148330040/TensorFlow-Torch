# *- coding: utf-8 -*

import re
import json

import pandas as pd
import numpy as np

from py2neo import Graph, Relationship, Node

pd.set_option('display.max_columns', None)


# 数据库内部命名混乱重新设置
neo4j_name = {'绩效评级': 'performance_description',
              '警告级别': 'warning_level',
              '绩效评估': 'performance_rating',
              '是否错峰': 'staggering_peak',
              '错峰措施': 'staggering_measures',
              '错峰方式': 'peak_shifting_method',
              '区域工业局座机号码': 'territorial_industry_office',
              '区域环境局编号': 'territorial_environmental_bureau',
              '区域领导': 'territory_leader',
              '企业领导': 'enterprise_leader',
              '技术领导': 'technology_leader',
              '业务时限': 'time_limit_for_business',
              '注册登记机关': 'registration_authority',
              '公司类型': 'company_type',
              '公司状态': 'operating_state',
              '批准日期': 'approve_date',
              '营收': 'paid_capital',
              '注册资本': 'registered_capital',
              '位置-经度': 'longitude',
              '位置-纬度': 'latitude',
              '地址': 'address',
              '县/区': 'county',
              '市': 'city',
              '省': 'province',
              '联系方式': 'contact_information',
              '网址': 'website',
              '行业': 'industry',
              '公司名称': 'company',
              '企业认证编号': 'social_code',
              '公司缩写': 'company_name_short',
              '法人': 'legal_person',
              '工业园区规模': 'park_type',
              '工业园区名称': 'park_name',
              '公司规模': 'scale',
              '工业园区面积': 'area_industry',
              '企业标签': 'label',
              '投产日期': 'date_production'}
col_name1 = {
    'performance_description': '绩效评级',
    'warning_level': '警告级别',
    'performance_rating': '绩效评估',
    'staggering_peak': '是否错峰',
    'staggering_measures': '错峰措施',
    'peak_shifting_method': '错峰方式',
    'territorial_industry_office': '区域工业局座机号码',
    'territorial_environmental_bureau': '区域环境局编号',
    'territory_leader': '区域领导',
    'enterprise_leader': '企业领导',
    'technology_leader': '技术领导',
    'time_limit_for_business': '业务时限',
    'registration_authority': '注册登记机关',
    'company_type': '公司类型',
    'operating_state': '公司状态',
    'approve_date': '批准日期',
    'paid_capital': '营收',
    'registered_capital': '注册资本',
    'longitude': '位置-经度',
    'latitude': '位置-纬度',
    'addr': '地址',
    'region': '县/区',
    'city': '市',
    'province': '省',
    'telphone': '联系方式',
    'website': '网址',
    'industry': '行业',
    'company': '公司名称',
    'socialcode': '企业认证编号',
    'companyabbr': '公司缩写',
    'legal_person': '法人'}
col_names2 = {
    'address': '地址',
    'city': '市',
    'company': '公司名称',
    'industry': '行业',
    'latitude': '位置-纬度',
    'longitude': '位置-经度',
    'legal_person': '法人',
    'province': '省',
    'socialcode': '企业认证编号',
    'county': '县/区',
    'park_type': '工业园区规模',
    'park_name': '工业园区名称',
    'scale': '公司规模',
    'area_industry': '工业园区面积',
    'label': '企业标签',
    'website': '网址',
    'email': '联系方式',
    'performance': '绩效评级',
    'date_production': '投产日期'
}


def get_dataset():

    dataset_path = '../dataset/'

    use_col1 = ['performance_description', 'warning_level', 'performance_rating', 'staggering_peak',
               'staggering_measures', 'peak_shifting_method', 'territorial_industry_office',
               'territorial_environmental_bureau', 'territory_leader', 'enterprise_leader',
               'technology_leader', 'time_limit_for_business', 'registration_authority',
               'company_type', 'operating_state', 'approve_date', 'paid_capital', 'registered_capital',
               'longitude', 'latitude', 'addr', 'region', 'city', 'province', 'telphone', 'industry',
               'company', 'socialcode', 'companyabbr', 'legal_person', 'website']

    dataset1 = pd.DataFrame(pd.read_excel(dataset_path + 'company1.xls', usecols=use_col1, nrows=100))
    dataset1.rename(columns=col_name1, inplace=True)

    def process_ds2(ds2):

        use_col = ['park_type', 'park_name', 'scale', 'area_industry', 'label', 'website',
                   'email', 'performance', 'date_production']

        pattern = "\"base\".*?}"
        pattern2 = "{.*}"

        ds2['info_new'] = ds2['info_new'].apply(
            lambda x: re.findall(pattern=pattern, string=x)[0]
        )

        ds2['info_new'] = ds2['info_new'].apply(
            lambda x: json.loads(re.findall(pattern=pattern2, string=x)[0])
        )

        ds2['info_new'] = ds2['info_new'].apply(
            lambda x: x if bool(x) else np.nan
        )

        def get_col_value(d, c):
            if type(d) == dict:
                if c in d.keys():
                    if d[c] != '-':
                        return d[c]
            return np.nan

        for col in use_col:
            ds2[col] = ds2['info_new'].apply(
                lambda x: get_col_value(x, col)
            )
        return ds2

    use_cols2 = ['address', 'city', 'company', 'industry', 'latitude', 'legal_person',
                 'longitude', 'province', 'socialcode', 'county', 'park_type', 'park_name',
                 'scale', 'area_industry', 'label', 'website', 'email', 'performance', 'date_production']

    dataset2 = pd.read_excel(dataset_path + 'company2.xls', nrows=100)
    dataset2 = process_ds2(dataset2)
    dataset2 = dataset2[use_cols2]
    dataset2.rename(columns=col_names2, inplace=True)

    dataset1 = dataset1.append(dataset2, ignore_index=True)

    return dataset1


dataset = get_dataset()

graph = Graph("http://192.168.0.140:7474", auth=("neo4j", "123456"))

node_col = ['公司名称', '行业', '公司类型', '省', '市', '绩效评级', '网址']



def get_nodes():
    nodes_dict = {}

    for col in node_col:
        nodes_dict[col] = list(set(dataset[col].value_counts().index))

    return nodes_dict


def create_nodes():
    nodes = get_nodes()

    attribute_col = ['警告级别', '绩效评估', '是否错峰', '错峰措施', '错峰方式', '区域工业局座机号码','技术领导',
                     '区域环境局编号', '业务时限', '注册登记机关', '公司状态', '批准日期', '营收', '注册资本', '企业领导',
                     '位置-经度', '位置-纬度', '地址', '县/区', '联系方式', '企业认证编号', '公司缩写', '法人', '区域领导',
                     '工业园区规模', '工业园区名称', '工业园区面积', '公司规模', '企业标签', '投产日期']
    ds = dataset.copy()
    ds.fillna('-', inplace=True)

    for key, value in nodes.items():
        for v in value:
            if v != '-':
                node = Node(key, name=str(v).strip())
                if key == '公司名称':
                    dsc = ds.loc[ds['公司名称'] == v]
                    for col in attribute_col:
                        attribute_v = list(dsc[col])[0]
                        node[col] = str(attribute_v)

                graph.create(node)


def get_triple():
    dataset_company = dataset[node_col].copy()
    triple4ds = pd.DataFrame()

    for _, dsc in dataset_company.iterrows():
        ""
        data = pd.DataFrame(dsc.iloc[1:])
        data.columns = ['node2']
        data['relationship'] = data.index
        data['node1'] = dsc['公司名称']

        data['name1'] = '公司名称'
        data['name2'] = data['relationship']
        triple4ds = pd.concat([triple4ds, data])

    triple4ds.index = np.arange(len(triple4ds))

    triple4ds['node1'] = triple4ds['node1'].apply(
        lambda x: np.nan if x == '-' else x
    )
    triple4ds['node2'] = triple4ds['node2'].apply(
        lambda x: np.nan if x == '-' else x
    )

    triple4ds.dropna(inplace=True)

    return triple4ds


def make_relation(df_data):
    """建立关系
    """
    for _, data in df_data.iterrows():
        # print(f"node1: {str(data['node1'])}----relationship: {data['relationship']}-----node2: {str(data['node2'])}")

        node1 = graph.nodes.match(data['name1']).where(name=str(data['node1']).strip()).first()
        # 直接通过graph查找节点
        relationship = data['relationship']
        node2 = graph.nodes.match(data['name2']).where(name=str(data['node2']).strip()).first()
        # 通过NodeMatcher创建节点
        graph.create(Relationship(node1, relationship, node2))


def get_node_attribute(name):
    return dict(graph.nodes.match().where(name=name))


create_nodes()
triple = get_triple()
make_relation(triple)

# print(type(list(dataset.loc[dataset['公司名称']=='河北万生保温材料有限公司']['公司缩写'])[0]))
# print(list(dataset.loc[dataset['公司名称']=='河北万生保温材料有限公司']['公司缩写'])[0] is np.nan)
# print(list(dataset.loc[dataset['公司名称']=='河北万生保温材料有限公司']['工业园区规模'])[0])
# print(type(list(dataset.loc[dataset['公司名称']=='河北万生保温材料有限公司']['工业园区规模'])[0]))
# print(list(dataset.loc[dataset['公司名称']=='河北万生保温材料有限公司']['工业园区规模'])[0] is np.nan)
# print(dataset.loc[dataset['公司名称']=='河北万生保温材料有限公司'].fillna('-'))