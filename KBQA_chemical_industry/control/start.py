# *- coding: utf-8 -*
import json

import model_ie
import es_keywords
import keywords_match

from sanic import Sanic
from sanic.response import HTTPResponse

app = Sanic(__name__)


@app.route("input_text/<text:string>", methods=['POST'])
async def input_text(request, text):

    product_name, product_attribute = model_ie.predict_text(text)
    # 如果没有获取到关键词则依据文本内容直接匹配
    if len(product_name) == 0:
        product_name = es_keywords.get_top1_attribute(word=text, kinds=0, top=3)
    if len(product_attribute) == 0:
        product_attribute = es_keywords.get_top1_attribute(word=text, kinds=1, top=2)

    answers = keywords_match.keywords_match(product_name, product_attribute)

    t = ''
    for answer in answers:
        t += answer[0] + '的' + answer[1] + '是' + answer[-1] + '。\n'

    result = {'answer': t}

    return HTTPResponse(json.dumps(result))

# input_text('氢化可松和酸甘油酯的物化型质以及分子式是什么？')
if __name__ == '__main__':
    app.run('0.0.0.0')