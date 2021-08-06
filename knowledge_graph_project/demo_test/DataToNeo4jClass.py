# -*- coding: utf-8 -*-
from py2neo import Node, Graph, Relationship


class DataToNeo4j(object):
    """将excel中数据存入neo4j"""

    def __init__(self):
        """建立连接"""
        link = Graph("http://192.168.0.140:7474", auth=("neo4j", "123456"))
        self.graph = link
        # 定义label
        self.invoice_name = '发票名称'
        self.invoice_value = '发票值'
        self.graph.delete_all()

    def create_node(self, node_list_key, node_list_value):
        """建立节点"""
        for name in node_list_key:
            name_node = Node(self.invoice_name, name=name)
            self.graph.create(name_node)
        for name in node_list_value:
            value_node = Node(self.invoice_value, name=name)
            self.graph.create(value_node)

    def create_relation(self, df_data):
        """建立联系"""
        print(df_data)
        for _, data in df_data.iterrows():
            print(data['name'])
            node1 = self.graph.match(nodes=(data['name'], ), r_type='mame')
            relationship = data['relation']
            node2 = self.graph.match(nodes=(data['name2'], ), r_type='name')
            print(node1)
            print(node2)