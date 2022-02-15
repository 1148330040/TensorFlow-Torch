# *- coding: utf-8 -*
import json

from sanic import Sanic
from sanic.response import HTTPResponse

from json_process import process_triple_spo, process_words_entity

app = Sanic(__name__)

# words2entity, entity2words = process_words_entity()

# s4po, o4ps = process_triple_spo()
# entity4triple = {**s4po, **o4ps}

s4po = json.load(open('../dataset/train_s4po.json'))
o4ps = json.load(open('../dataset/train_o4ps.json'))
entity4triple = {**s4po, **o4ps}

words2entity = json.load(open('../dataset/words2entity.json'))
entity2words = json.load(open('../dataset/entity2words.json'))

@app.route("/triple/<word:string>", methods=['POST'])
async def deploy_triple_ds(request, word):
    return HTTPResponse(json.dumps(entity4triple[word]))


@app.route("/words_entity/<word:string>", methods=['POST'])
async def deploy_words_entity_ds(request, word):
    return HTTPResponse(json.dumps(words2entity[word]))


@app.route("/entity_words/<word:string>", methods=['POST'])
async def deploy_words_entity_ds(request, word):
    return HTTPResponse(json.dumps(entity2words[word]))


@app.route("/words_entity_keys", methods=['POST'])
async def deploy_words_entity_ds(request):
    return HTTPResponse(json.dumps(list(words2entity.keys())))



if __name__ == '__main__':
    app.run(host="0.0.0.0")