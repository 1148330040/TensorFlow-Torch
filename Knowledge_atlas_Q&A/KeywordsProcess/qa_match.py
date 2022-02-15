# *- coding: utf-8 -*

import requests

from dataset_process import get_most_similarity_word


def qa_process(present_kw):
    """当前的信息不足以支持关键词匹配逻辑的进行时
    将会切换到问答流程中以补充当前信息
    """
    message = f"无法获取您的意图，请问您是否想要了解以下相关内容信息？如果是的话请输入对应内容: " \
              f"\n{present_kw}\n"
    # 建议在前端加一个否定按钮
    print(message)

    input_mes = input("我想了解的是: ")

    while input_mes not in message:
        input_mes = input("请输入提示信息内的相关内容: ")

    present_spo = list(requests.post(url=f'http://192.168.0.131:8000/triple/{input_mes}').json())

    present_p = list(set([p_spo[0] for p_spo in present_spo]))
    # 获取当前spo列表中的p值来筛选用户的兴趣相关的内容
    print(present_p)
    input_mes = input("请问您想了解哪方面的信息: ")

    if input_mes in present_p:
        # 如果用户进一步了解的信息存在于备选的spo中则返回相关信息, 不存在则需要进行相似度计算选出top相关信息
        output_mes = [p_spo if input_mes in p_spo else '' for p_spo in present_spo]
    else:
        _, present_back_p = get_most_similarity_word(input_mes, present_p, top=1)
        present_back_p = present_back_p[0]
        output_mes = [p_spo if present_back_p in p_spo else '' for p_spo in present_spo]

    while '' in output_mes:
        output_mes.remove('')

    print(output_mes)

    return output_mes


