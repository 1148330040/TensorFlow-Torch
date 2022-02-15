# *- coding: utf-8 -*

import os
import json
import redis

from sanic import Sanic
from datetime import datetime
from sanic.response import HTTPResponse
from information_ext_bert import predict
from keywords_match import  process_e_ep,       \
                            process_ep_e,       \
                            process_so_p,       \
                            process_so_p_es,    \
                            process_ep_e_es,    \
                            process_e_ep_es
app = Sanic(__name__)


def keywords_process(content):

    outputs = predict(content)
    # 获取三元组信息
    value = None
    if bool(outputs) is False:
        # 进行qa处理
        return value
    print(outputs)

    s, p, o = outputs
    # 两种处理方向
    # 1: 处理单一实体
    # 2: 处理多实体
    if len(s) <=1 and len(o) <= 1 and len(p) <= 1:
        if len(s) == 0:
            value = process_ep_e(o[0], p[0], content)
        elif len(o) == 0:
            value = process_ep_e(s[0], p[0], content)
        elif len(s) == 0 and len(p) == 0:
            value = process_e_ep(o[0], content)
        elif len(o) == 0 and len(p) == 0:
            value = process_e_ep(s[0], content)
        else:
            value = process_so_p(s[0], o[0], content)
    else:
        if len(p) == 0:
            value = process_so_p_es(s, o, content)
        if len(s) == 0 and len(p) != 0:
            value = process_ep_e_es(o, p, content)
        if len(o) == 0 and len(p) != 0:
            value = process_ep_e_es(s, p, content)

        if len(s) == 0 and len(p) == 0:
            value = process_e_ep_es(o, content)
        if len(o) == 0 and len(p) == 0:
            value = process_e_ep_es(s, content)

    return value


def test():
    content = input("请输入你想了解的信息:")
    value = keywords_process(content)
    print(value)
    return test()


test()
# 女词人柳如是和歌手许嵩的主要作品是什么？
# @app.route("/start/<content:string>", methods=['POST'])
# async def deploy(request, content):
#     start_time = datetime.now()
#     value = keywords_process(content)
#     information = {'content:': content}
#     if value is None:
#         information['value'] = '信息获取失败！'
#     else:
#         information['value'] = value
#
#     information['cost_time'] = datetime.now() - start_time
#
#     return HTTPResponse(json.dumps(information))
#
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port='8002')
