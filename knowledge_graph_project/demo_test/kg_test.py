# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from py2neo import Node, Graph, Relationship, NodeMatcher

invoice_data = pd.read_excel('Invoice_data_Demo.xls', header=0)

neo4j_link = Graph("http://192.168.0.140:7474", auth=("neo4j", "123456"))

node1_name = '发票名称'
node2_name = '发票值'

def get_node_dataset():
    """节点数据抽取以及三元组数据构建
    node1: 原始节点
    node2: 原始节点对应的属性值
    """

    data_relation = pd.DataFrame()
    for _, d in invoice_data.iterrows():
        s = pd.DataFrame(d.iloc[1:])
        s.columns = ['value']
        s['relationship'] = s.index
        s['name'] = d['发票名称']
        data_relation = pd.concat([data_relation, s])

    data_relation.index = np.arange(len(data_relation))
    data_relation = data_relation[['name', 'relationship', 'value']]
    node1 = list(set(data_relation['name'].values))
    node2 = list(set(data_relation['value'].values))
    node1 = [str(i) for i in node1]
    node2 = [str(i) for i in node2]

    return node1, node2, data_relation


node1_value, node2_value, ds = get_node_dataset()
# node1_value-发票名称, node2_value-发票对应的值, 三元组数据


def create_node(node_list_key, node_list_value):
    """建立节点"""
    for name in node_list_key:
        name_node = Node('发票名称', name=name)
        neo4j_link.create(name_node)
    for name in node_list_value:
        value_node = Node('发票值', name=name)
        neo4j_link.create(value_node)


def delete_node(dell_all=False, del_name=None):
    if dell_all:
        neo4j_link.delete_all()

    if del_name:
        node_name = node2_name
        if del_name in node1_value:
            node_name = node1_name
        node_graph = neo4j_link.nodes.match(node_name).where(name=del_name).graph
        neo4j_link.delete(subgraph=node_graph)


def create_relation(df_data):
    """建立关系, 为何不在创建节点的同时直接建立联系的原因是因为
    假如某一列数据里面含有多组重复值，那么如果直接按照数据迭代创建节点的会，会重复创建重复值节点多次
    """
    node_match = NodeMatcher(neo4j_link)

    for _, data in df_data.iterrows():
        # node1 = neo4j_link.match('发票名称').where(mame=str(data['name']))
        node1 = neo4j_link.nodes.match(node1_name).where(name=str(data['name']))
        # 直接通过graph查找节点
        relationship = data['relationship']
        node2 = node_match.match(node2_name).where(name=str(data['value']))
        # 通过NodeMatcher创建节点
        neo4j_link.create(Relationship(node1.first(), relationship, node2.first()))


print(invoice_data)
# delete_node(True)
# create_node(n1, n2)
# create_relation(ds)
# node1 = neo4j_link.nodes.match('发票名称').where(name='山东增值税电子普通发票').graph
# print(node1)
# test_n1 = neo4j_link.run(cypher="MATCH (n:发票名称) where n.name='山东增值税电子普通发票' return n").to_subgraph()
# neo4j_link.delete(test_n1)