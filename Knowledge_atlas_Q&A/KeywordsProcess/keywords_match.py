# *- coding: utf-8 -*

import json
import requests

from dataset_process import get_most_similarity_word

import qa_match as qa
import es_keywords as es


word2entity_keys = list(requests.post(url='http://192.168.0.131:8000/words_entity_keys').json())


def get_entity_word(word):
    """获取关键词对应的实体"""
    return list(requests.post(url=f'http://192.168.0.131:8000/words_entity/{word}').json())


def get_triple_word(word):
    """获取实体对应的相关spo值"""
    return list(requests.post(url=f'http://192.168.0.131:8000/triple/{word}').json())


def get_entity(entity, content):
    """用以处理单个entity时的back候选实体
    """
    label_words = []
    if entity in word2entity_keys:
        # 关键词存在于词表词表中则直接给出相关实体词候选
        label_words = get_entity_word(entity)
    else:
        for wk in word2entity_keys:
            # 关键词不存在与词表中, 则判断是否存在有包含该关键词的实体词候选
            if entity in wk:
                if wk not in label_words:
                    label_words.append(wk)
        if len(label_words) == 0:
            # 不存在包含改关键词的实体则需要通过ES与储存库的数据进行初步匹配
            label_words = es.get_es_top_words(entity, top=1)

        if len(label_words) > 5:
            _, label_words = get_most_similarity_word(content, label_words, top=5)

    return label_words


def process_e_ep(entity, content):
    """意味着只获取到了一个实体
    """
    entity_back = get_entity(entity, content)
    spo_back = qa.qa_process(entity_back)

    return spo_back


def process_ep_e(entity, p, content):
    """1: 根据获取到的实体(s/o)关键词去匹配对应的实体
    2: 如果关键词可以直接找到实体, 则跳到第4步否则获取到top-k的实体
    3: 根据关键词和对应的 关键词-实体表 进行相似度匹配获取top-k的实体列表
    4: 使用text和实体列表获取top-1的实体
    5: 通过关键词获取predicate列表借用text获取相似度最高的文本
    """

    def e_match_p():
        back_spo = []
        for e in entity_back:
            try:
                e_back_ep = get_triple_word(e)
            except:
                continue
            for eep in e_back_ep:
                if p in eep:
                    back_spo.append([e, p, eep[-1]])

        if len(back_spo) == 0:
            # 说明获取到的p不存在与现有后备实体中, 对话流程介入
            back_spo = qa.qa_process(entity_back)

        return back_spo

    entity_back = get_entity(entity, content)
    back_spo = e_match_p()
    print("back_spo2: ", back_spo)

    return back_spo


def process_so_p(s, o, content):
    """已知两个实体s和o获取关系p
    """

    subject_back = get_entity(s, content)
    object_back = get_entity(o, content)
    entity_back = []
    for sub_back in subject_back:
        try:
            sub_back_op = get_triple_word(sub_back)
        except:
            continue
        for obj_back in object_back:
            for op in sub_back_op:
                if obj_back in op:
                    entity_back.append([sub_back, obj_back, op[0]])

    if len(entity_back) == 1:
        return entity_back[-1]

    if len(entity_back) == 0:
        # 此时说明无法获取到任何spo相关信息
        # 问答流程介入
        spo_back = []
        if len(subject_back) != 0:
            spo_back += qa.qa_process(subject_back)
        if len(object_back) != 0:
            spo_back += qa.qa_process(object_back)
        return spo_back

    entity_back = [' '.join(ent_back) for ent_back in entity_back]
    print(entity_back)
    _, entity_back = get_most_similarity_word(content, entity_back)
    print(entity_back)
    predicate = entity_back[-1]

    return predicate


def get_entity_es(entity, content):
    """用以获取多个entity存在时的back候选实体
    """
    label_words = []
    for ent in entity:
        if ent in word2entity_keys:
            # 关键词存在于词表词表中则直接给出相关实体词候选
            label_word = get_entity_word(ent)
        else:
            label_word = []
            for wk in word2entity_keys:
                # 关键词不存在与词表中, 则判断是否存在有包含该关键词的实体词候选
                if ent in wk:
                    if wk not in label_words:
                        label_word.append(wk)
            if len(label_word) == 0:
                # 不存在包含改关键词的实体则需要与全部实体词进行匹配
                # 该做法会导致关键词与所有实体进行相似度匹配计算
                # 如果图数据库中的实体数目过于庞大则不建议这样做
                label_word = word2entity_keys

        if len(label_word) > 5:
            _, label_word = get_most_similarity_word(content, label_word, top=5)

        label_words.append(label_word)

    return label_words


def process_so_p_es(s, o, content):
    """处理获取到多个s/o时的情况"""

    def s_match_o():
        # 实体匹配另一实体
        s_o_match_back = []

        for sub_back in subject_back:
            for s_back in sub_back:
                try:
                    s_back_op = get_triple_word(s_back)
                except:
                    continue

                for obj_back in object_back:
                    for o_back in obj_back:
                        for s_op in s_back_op:
                            if o_back in s_op:
                                s_o_match_back.append([s_back, s_op[0], o_back])
                                # [s, p, o]

        return s_o_match_back

    # 处理 entity
    subject_back = get_entity_es(s, content)
    object_back = get_entity_es(o, content)

    spo_back = s_match_o()
    # 一般来说两个实体的关系通常有多种比如A和B既可以是师生关系也可以是朋友关系或者亲子,亲戚关系
    # 但是在问句中往往只会关注一种关系, 因此会结合当前content筛选出最符合当前语境的top-1 spo

    if len(spo_back) == 0:
        # 说明当前阶段无法获取到标准实体, 需要借助问答流程的帮助
        present_subject_back = []
        for s_back_ in subject_back:
            present_subject_back += s_back_
            spo_back = qa.qa_process(present_subject_back)

    return spo_back


def process_ep_e_es(e, p, content):
    """当获取到多个s/o以及p时, 获取o/s
    e: 代表s/o实体
    """
    def e_match_p():
        e_p_match_back = []
        for entity in entity_back:
            for e_back in entity:
                try:
                    e_back_sp = get_triple_word(e_back)
                except:
                    continue
                for p_back in p:
                    for esp in e_back_sp:
                        if p_back in esp:
                            e_p_match_back.append([e_back_sp[-1], p, e_back])

        if len(e_p_match_back) == 0:
            # 说明获取的p关键词不是实体predicate, 需要进行相似度处理
            for entity in entity_back:
                for e_back in entity:
                    try:
                        e_back_ep = get_triple_word(e_back)
                    except:
                        continue
                    e_back_p = [ep[0] for ep in e_back_ep]
                    for p_back in p:
                        # 使用for循环索引p是因为不同的实体对应的p可能不是一致
                        # 因此需要确保entity和每一个p_back进行索引
                        _, p_back_word = get_most_similarity_word(p_back, e_back_p, top=1)
                        p_back_word = p_back_word[0]
                        for eep in e_back_ep:
                            if p_back_word in eep:
                                e_p_match_back.append([e_back, p_back_word, eep[-1]])
                # 将所有候选的实体以及其当前候选实体最匹配的p和e2组合到一起
                # 假设获取到的entity_back: [e1, e2, e3]
                # 则e_p_match_back: [[s1, p1, o1], [s2, p2, o2], [s3, p3, o3]]
        return e_p_match_back

    entity_back = get_entity_es(e, content)
    spo_back = e_match_p()
    entity = spo_back

    return entity


def process_e_ep_es(e, content):
    """此函数用于处理关键词仅限于某一个实体的情况
    比如: 两个s或者两个o, 获取两者共同的predicate返回对应的实体
    """

    def e_match_p():
        # 需要获取最合适的p的同时, 也需要获取p对应的entity

        all_entity_back_p = []  # 用于记录每个实体备选项的p
        all_p = []  # 用于获取所有实体备选项中出现次数最多的p

        for entity in entity_back:
            entity_back_p = []
            for e_back in entity:
                try:
                    e_back_ep = get_triple_word(e_back)
                except:
                    entity_back_p.append([])
                    continue
                e_back_p = list(set([ep[0] for ep in e_back_ep]))
                _, back_p = get_most_similarity_word(content, e_back_p, top=2)
                entity_back_p.append(back_p)
                all_p += back_p

            all_entity_back_p.append(entity_back_p)
        # 获取到每个entity备选的top-2predicate
        # 然后对比每个entity备选项获取重叠的predicate
        p = max(all_p, key=all_p.count)
        # 根据每个p出现的次数来获取最合适的predicate
        # 获取predicate对应的entity
        entity_site = []
        if all_p.count(p) >= 2:
            # 说明两者间有共同的predicate
            for num1, e1 in enumerate(all_entity_back_p):
                # num1 用于标记这是哪一个entity
                for num2, e2 in enumerate(e1):
                    # num2 用于标记这是哪一个entity备选
                    if p in e2:
                        entity_site.append([num1, num2])

            e = []
            for es in entity_site:
                e.append(entity_back[es[0]][es[1]])

            return p, e
        else:
            # 说明两者没有共同的predicate, 将两者的top-1 predicate作为选择
            spo = []

            for e_back, e_back_p in zip(entity_back, all_entity_back_p):
                mid_back_p = []
                for ebp in e_back_p:
                    mid_back_p += ebp

                mid_back_p = list(set(mid_back_p))

                _, top_p = get_most_similarity_word(content, mid_back_p)

                back_p = top_p[0]

                for eb in e_back:
                    try:
                        e_back_ep = get_triple_word(eb)
                    except:
                        continue

                    for ebep in e_back_ep:
                        if back_p in ebep:
                            spo.append([eb, back_p, ebep[-1]])

            return None, spo

    def e_match_e():
        # 当不同的实体有共同的predicate时使用该方法
        result = []

        for e in entity:
            try:
                e_back_ep = get_triple_word(e)
            except:
                continue
            for ep in e_back_ep:
                if predicate in ep:
                    result.append([e, predicate, ep[-1]])

        return result

    entity_back = get_entity_es(e, content)
    print(entity_back)
    # 获取每个entity的备选项
    predicate, entity = e_match_p()
    print(entity)
    if predicate is not None:
        entity_2 = e_match_e()
    else:
        entity_2 = entity

    return entity_2

print(process_ep_e_es(e=['许嵩', '史铁生'], p=['作品'], content='歌手许嵩和史铁生的主要作品是什么？'))
# print(process_so_p_es(s=['许嵩', '史铁生'], o=['务虚笔记', '病隙碎笔', '不如吃茶去'],
#                 content='史铁生与务虚笔记和病隙碎笔以及歌手许嵩与歌曲不如吃茶去是什么关系？'))
# print(process_so_p('柳如是', '湖上草', content='柳如是的湖上草是她的什么？'))
# print(process_e_p(entity='湖上草', p='诗', content='《湖上草》是谁的诗？'))
# process_e_ep_es(e=['许嵩', '史铁生'], content='许嵩和史铁生的主要作品是什么？')
# 测试两者有共同的predicate时的情况
# process_e_ep_es(e=['柳如是', '史铁生'], content='女词人柳如是和史铁生的主要作品是什么？')
# 测试两者不存在共同predicate时的情况