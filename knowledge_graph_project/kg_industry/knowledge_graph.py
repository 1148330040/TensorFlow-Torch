import pandas as pd
import numpy as np

from py2neo import Node, Graph, Relationship

graph = Graph("http://192.168.0.140:7474", auth=("neo4j", "123456"))

node1_name = '行业'
node2_name = '问题类型'
node3_name = '工艺类型'
node4_name = '工艺'

dataset = pd.read_excel('../dataset/ds55.xls')
dataset.pop('answer')
print(dataset)

def get_nodes():
    """获取node值的两个标准:
    1: 去重   2: 区分
    """
    ind = list(dataset['industry'].value_counts().index)
    qt = list(dataset['question_type'].value_counts().index)
    pt = list(dataset['process_type'].value_counts().index)
    p = list(dataset['process'].value_counts().index)

    # 使用字典作为存储node值是一个比较好的办法
    # 使用key作为它们的node值类型, value作为对应去重值的列表
    nodes = {
        '行业': ind,
        '问题类型': qt,
        '工艺类型': pt,
        '工艺': p
    }

    return nodes


def creat_nodes(nodes):
    for key, values in nodes.items():
        for value in values:
            node = Node(key, name=value)
            graph.create(node)


def get_triple4data():
    """三元组数据即: 节点-关系-节点
    使用DataFrame包含需要确保有对应的三列即两列节点一列关系值
    构建三元组数据需要确保原始数据中列与列之间的关系，
    是一列直接对应多列还是一列对应一列处于递进的关系需要梳理清楚
    这里的数据是递进关系即：
    行业->工艺类型/工艺, 工艺类型->工艺, 工艺->问题类型
    """
    node_dataset = pd.DataFrame()

    def get_data(data, relation):
        data.columns = ['node1', 'node2']
        data['relationship'] = relation
        return data

    ind4pt = dataset[['industry', 'process_type']].copy()
    ind4pt.dropna(inplace=True)
    ind4pt = get_data(ind4pt, relation='工艺类型')
    ind4pt['name1'] = '行业'
    ind4pt['name2'] = '工艺类型'

    ind4p = dataset[['industry', 'process_type', 'process']].copy()
    ind4p['get_nan'] = ind4p['process_type'].apply(
        lambda x: np.nan if type(x) == str else 1
    )
    ind4p.dropna(subset=['get_nan'], inplace=True)
    ind4p = ind4p[['industry', 'process']]
    ind4p = get_data(ind4p, relation='工艺')
    ind4p['name1'] = '行业'
    ind4p['name2'] = '工艺'

    pt4p = dataset[['process_type', 'process']].copy()
    pt4p.dropna(inplace=True)
    pt4p = get_data(pt4p, relation='工艺')
    pt4p['name1'] = '工艺类型'
    pt4p['name2'] = '工艺'

    p4qt = dataset[['process', 'question_type']].copy()
    p4qt = get_data(p4qt, relation='问题类型')
    p4qt['name1'] = '工艺'
    p4qt['name2'] = '问题类型'

    node_dataset = pd.concat([node_dataset, ind4pt])
    node_dataset = pd.concat([node_dataset, ind4p])
    node_dataset = pd.concat([node_dataset, pt4p])
    node_dataset = pd.concat([node_dataset, p4qt])

    node_dataset = node_dataset.drop_duplicates(subset=['node1', 'relationship', 'node2'], keep='first')

    return node_dataset


def create_relation(triple_data):
    """建立关系
    """
    for _, data in triple_data.iterrows():
        # node1 = neo4j_link.match('发票名称').where(mame=str(data['name']))
        node1 = graph.nodes.match(data['name1']).where(name=str(data['node1']))
        relationship = data['relationship']
        node2 = graph.nodes.match(data['name2']).where(name=str(data['node2']))
        # print(data['node1'], data['node2'], data['name1'], data['name2'])
        # print(node1.first())
        # print(node2.first())
        # print(relationship)
        graph.create(Relationship(node1.first(), relationship, node2.first()))

def delete_node(dell_all=False, del_name=None):
    if dell_all:
        graph.delete_all()

    if del_name:
        graph.run(cypher=f'match (n) where n.name = {del_name} detach delete n')


# nodes_data = get_nodes()
# print(dataset)
# print(nodes_data)
# creat_nodes(nodes_data)
# triple4data = get_triple4data()
# create_relation(triple4data)
delete_node(True)