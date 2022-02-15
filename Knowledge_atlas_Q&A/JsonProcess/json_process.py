import re
import json

path = '../dataset/'
words_entity_path = path + 'pkubase-mention2ent.txt'
triple_path = path + 'triple.txt'

def process_triple_spo():

    dataset = open(triple_path)

    try:
        s4po = json.load(open('../dataset/s4po.json'))
        o4ps = json.load(open('../dataset/o4ps.json'))
    except:
        s4po = {}
        o4ps = {}
        compile1 = re.compile(pattern=u'[<](.+?)[>]')

        for ds in dataset.readlines():
            ds = ds.replace('<', '"').replace('>', '"').strip()
            keywords = re.findall(compile1, ds)
            try:
                s = keywords[0]
                p = keywords[1]
                o = keywords[-1]
                if s not in s4po:
                    s4po[s] = [(p, o)]
                else:
                    s4po[s].append((p, o))
                if o not in o4ps:
                    o4ps[o] = [(p, s)]
                else:
                    o4ps[o].append((p, s))
            except:
                continue

        with open('../dataset/s4po.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(s4po, ensure_ascii=False))

        with open('../dataset/o4ps.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(o4ps, ensure_ascii=False))

    return s4po, o4ps


def process_words_entity():
    """考虑到后续会较为频繁的进行实体词和普通词相互的转化
    因此建立两个字典表
    """
    try:
        words2entity = json.load(open('../dataset/words2entity.json'))
        entity2words = json.load(open('../dataset/entity2words.json'))

    except:
        words_entity = open(words_entity_path)
        words2entity = {}
        entity2words = {}

        for ds in words_entity.readlines():
            ds = ds.split('\t')
            word = ds[0]
            entity = ds[1]

            if word not in words2entity:
                words2entity[word] = [entity]
            else:
                words2entity[word].append(entity)

            if entity not in entity2words:
                entity2words[entity] = [word]
            else:
                entity2words[entity].append(word)

        with open('../dataset/words2entity.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(words2entity, ensure_ascii=False))

        with open('../dataset/entity2words.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(entity2words, ensure_ascii=False))

    return words2entity, entity2words