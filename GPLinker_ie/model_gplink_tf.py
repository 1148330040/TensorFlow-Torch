# tf版本GPlink
# 源代码相关来自苏神:
# 三元组抽取任务，基于GlobalPointer的仿TPLinker设计
# 文章介绍：https://kexue.fm/archives/8888


import os
import json
import warnings

import numpy as np
import pandas as pd

print(np.__version__)

import tensorflow as tf

from bert4keras.snippets import sequence_padding
from bert4keras.backend import K

from tensorflow.keras import initializers

from transformers import BertTokenizer, TFBertModel

os.environ["CUDA_VISIBLE_DEVICES"]="0"

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")

train_ds_path = 'duie/duie_train.json/duie_train.json'
valid_ds_path = 'duie/duie_dev.json/duie_dev.json'
test_ds_path = 'duie/duie_test2.json/duie_test2.json'
schema_ds_path = 'duie/duie_schema/duie_schema.json'

tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-12_H-768")

max_len = 128
batch_size = 24

def get_dataset(path):
    """将数据处理成:
    {'text': text,  'spo_list': [(s, p, o)]}
    s-subject-实体, p-predicate-关系, o-object-客实体
    """
    dataset = open(path)
    spo_ds = []
    for ds in dataset.readlines():
        ds = json.loads(ds)
        text = ds['text']
        text = text.replace('\t', '').replace(' ', '')
        if len(text) > max_len:
            continue

        spo_list = ds['spo_list']

        spo_ds.append({'text': text,
                       'spo_list': spo_list})

    spo_ds = pd.DataFrame(spo_ds)

    return spo_ds

get_dataset(path=train_ds_path)


def predicate2seq(dataset):
    spo_predicates = []
    for spo_ls in dataset['spo_list']:
        for spo in spo_ls:
            spo_predicates.append(spo['predicate'])
    spo_predicates = list(set(spo_predicates))

    predicate_vocabs = {word: num for num, word in enumerate(spo_predicates)}

    vocabs = open('duie/predicate_vocabs.json', 'w')
    vocabs.write(json.dumps(predicate_vocabs))

    return predicate_vocabs


def get_predict_vocab(ds):
    if os.path.exists('duie/predicate_vocabs.json'):
        vocabs = json.loads(open('duie/predicate_vocabs.json').read())
    else:
        vocabs = predicate2seq(ds)

    p2id = vocabs
    id2p = {value: key for key, value in vocabs.items()}

    return p2id, id2p

predicate2id, id2predicate = get_predict_vocab(None)


def data_generator(dataset):
    """数据生成器
    """

    def get_seq_pos(seq):
        """不要通过词在text中的位置信息找，因为有可能部分符号或者特殊字符编码后的长度与text长度不一致
        """
        seq_len = len(seq)
        for i in range(len(input_id)):
            if input_id[i:i + seq_len] == seq:
                return i
        return -1

    input_ids, input_masks, input_tokens = [], [], []
    batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []

    for _, ds in dataset.iterrows():

        text = ''.join(ds['text'])
        spo_lists = ds['spo_list']
        inputs = tokenizer.encode_plus(text)

        input_id = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        spoes = set()

        for spo_list in spo_lists:

            s = spo_list['subject']
            o = spo_list['object']['@value']

            predicate = spo_list['predicate']

            p = predicate2id[predicate]

            s = tokenizer.encode(s)[1:-1]
            o = tokenizer.encode(o)[1:-1]
            sh = get_seq_pos(s)
            oh = get_seq_pos(o)
            if sh != -1 and oh != -1:
                spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))

        # 构建标签
        entity_labels = [[] for _ in range(2)]
        head_labels = [[] for _ in range(len(predicate2id))]
        tail_labels = [[] for _ in range(len(predicate2id))]
        for sh, st, p, oh, ot in spoes:
            entity_labels[0].append((sh, st))
            entity_labels[1].append((oh, ot))
            head_labels[p].append((sh, oh))
            tail_labels[p].append((st, ot))

        for label in entity_labels + head_labels + tail_labels:
            if not label:  # 至少要有一个标签
                label.append((0, 0))  # 如果没有则用0填充

        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        # 构建batch
        input_ids.append(input_id)
        input_masks.append(attention_mask)
        input_tokens.append(token_type_ids)
        batch_entity_labels.append(entity_labels)
        batch_head_labels.append(head_labels)
        batch_tail_labels.append(tail_labels)

        if len(input_ids) == batch_size:

            input_ids = tf.constant(sequence_padding(input_ids, max_len), dtype=tf.int32)
            input_masks = tf.constant(sequence_padding(input_masks, max_len), dtype=tf.int32)
            input_tokens = tf.constant(sequence_padding(input_tokens, max_len), dtype=tf.int32)

            batch_entity_labels = sequence_padding(
                batch_entity_labels, seq_dims=2,
            )
            batch_head_labels = sequence_padding(
                batch_head_labels, seq_dims=2,
            )
            batch_tail_labels = sequence_padding(
                batch_tail_labels, seq_dims=2,
            )

            yield [input_ids, input_masks, input_tokens], [
                batch_entity_labels, batch_head_labels, batch_tail_labels
            ]
            # reduce_sum(input_mask) 是为了用于计算loss时截断predict使用(获取了每个句子转id后的长度)
            input_ids, input_masks, input_tokens = [], [], []
            batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []


def align(tensor, axes, ndim=None):
    """重新对齐tensor（批量版expand_dims）
    axes：原来的第i维对齐新tensor的第axes[i]维；
    ndim：新tensor的维度。
    """
    assert len(axes) == tf.keras.backend.ndim(tensor)
    indices = [None] * (ndim or max(axes))
    for i in axes:
        indices[i] = slice(None)
    return tensor[indices]


def sequence_masking(x, mask, value=0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    else:
        x_dtype = tf.keras.backend.dtype(x)
        if x_dtype == 'bool':
            x = tf.cast(x, 'int32')
        if tf.keras.backend.dtype(mask) != tf.keras.backend.dtype(x):
            mask = tf.cast(mask, tf.keras.backend.dtype(x))
        if value == '-inf':
            value = -K.infinity()
        elif value == 'inf':
            value = K.infinity()
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = tf.keras.backend.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        mask = align(mask, [0, axis], tf.keras.backend.ndim(x))
        value = tf.cast(value, tf.keras.backend.dtype(x))
        x = x * mask + value * (1 - mask)
        if x_dtype == 'bool':
            x = K.cast(x, 'bool')
        return x


class SinusoidalPositionEmbedding(tf.keras.layers.Layer):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self,
        output_dim,
        merge_mode='add',
        custom_position_ids=False,
        **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            seq_len = tf.shape(inputs)[1]
            inputs, position_ids = inputs
            if 'float' not in tf.keras.backend.dtype(position_ids):
                position_ids = tf.cast(position_ids, tf.keras.backend.floatx())
        else:
            input_shape = tf.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = tf.keras.backend.arange(0, seq_len, dtype=tf.keras.backend.floatx())[None]

        indices = tf.keras.backend.arange(0, self.output_dim // 2, dtype=tf.keras.backend.floatx())
        indices = tf.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = tf.einsum('bn,d->bnd', position_ids, indices)
        embeddings = tf.stack([tf.sin(embeddings), tf.cos(embeddings)], axis=-1)
        embeddings = tf.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = tf.tile(embeddings, [batch_size, 1, 1])
            return tf.concat([inputs, embeddings])


class GlobalPointer(tf.keras.layers.Layer):

    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """
    def __init__(
        self,
        heads,
        head_size,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
        kernel_initializer='lecun_normal',
        **kwargs
    ):
        super(GlobalPointer, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.use_bias = use_bias
        self.tril_mask = tril_mask
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(GlobalPointer, self).build(input_shape)
        self.dense = tf.keras.layers.Dense(
            units=self.head_size * self.heads * 2,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @tf.function
    def call(self, inputs, mask=None):
        # 输入变换
        inputs = self.dense(inputs)
        inputs = tf.split(inputs, self.heads, axis=-1)
        inputs = tf.stack(inputs, axis=-2)
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = tf.keras.backend.repeat_elements(pos[..., None, 1::2], 2, -1)
            sin_pos = tf.keras.backend.repeat_elements(pos[..., None, ::2], 2, -1)
            qw2 = tf.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = tf.reshape(qw2, tf.shape(qw))
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = tf.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = tf.reshape(kw2, tf.shape(kw))
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = tf.einsum('bmhd,bnhd->bhmn', qw, kw)
        # 排除padding
        logits = sequence_masking(logits, mask, '-inf', 2)
        logits = sequence_masking(logits, mask, '-inf', 3)
        # 排除下三角
        if self.tril_mask:
            mask = tf.linalg.band_part(K.ones_like(logits), 0, -1)
            logits = logits - (1 - mask) * K.infinity()
        # scale返回
        return logits / self.head_size**0.5


class EfficientGlobalPointer(GlobalPointer):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    """
    def build(self, input_shape):
        self.dense_1 = tf.keras.layers.Dense(
            units=self.head_size * 2,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.dense_2 = tf.keras.layers.Dense(
            units=self.heads * 2,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.built = True

    @tf.function
    def call(self, inputs, mask=None):
        # 输入变换
        inputs = self.dense_1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = tf.keras.backend.repeat_elements(pos[..., 1::2], 2, -1)
            sin_pos = tf.keras.backend.repeat_elements(pos[..., ::2], 2, -1)
            qw2 = tf.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = tf.reshape(qw2, K.shape(qw))
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = tf.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = tf.reshape(kw2, tf.shape(kw))
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = tf.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5
        bias = tf.einsum('bnh->bhn', self.dense_2(inputs)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        # 排除padding
        logits = sequence_masking(logits, mask, '-inf', 2)
        logits = sequence_masking(logits, mask, '-inf', 3)
        # 排除下三角
        if self.tril_mask:
            mask = tf.linalg.band_part(tf.ones_like(logits), 0, -1)
            logits = logits - (1 - mask) * K.infinity()
        # 返回最终结果
        return logits


class ModelBert4GpLink(tf.keras.Model):

    def __init__(self, output_dim):
        super(ModelBert4GpLink, self).__init__(output_dim)
        self.output_dim = output_dim
        self.dropout = tf.keras.layers.Dropout(0.01)
        # 必须把egp放入到init中否则将会忽略更新
        self.egp1 = EfficientGlobalPointer(heads=2, head_size=self.output_dim)
        self.egp2 = EfficientGlobalPointer(heads=len(predicate2id), head_size=self.output_dim, RoPE=False, tril_mask=False)
        self.egp3 = EfficientGlobalPointer(heads=len(predicate2id), head_size=self.output_dim, RoPE=False, tril_mask=False)
        self.bert = TFBertModel.from_pretrained("uer/chinese_roberta_L-12_H-768")


    @tf.function(input_signature=[(tf.TensorSpec(shape=(None, max_len), name='input_ids', dtype=tf.int32),
                                  (tf.TensorSpec(shape=(None, max_len), name='input_masks', dtype=tf.int32)),
                                  (tf.TensorSpec(shape=(None, max_len), name='input_tokens', dtype=tf.int32)))])
    def call(self, batch_ds):
        input_ids, input_masks, input_tokens = batch_ds

        hidden = self.bert(input_ids, input_masks, input_tokens)

        hidden = self.dropout(hidden[0], 1)

        entity_output = self.egp1(hidden)

        head_output = self.egp2(hidden)

        tail_output = self.egp3(hidden)

        return entity_output, head_output, tail_output


def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    zeros = tf.zeros_like(y_pred[..., :1])
    y_pred = tf.concat([y_pred, zeros], axis=-1)
    infs = 0
    if mask_zero:
        infs = zeros + tf.keras.utils.get_custom_objects().get('infinity', 1e12)
        y_pred = tf.concat([infs, y_pred[..., 1:]], axis=-1)

    y_pos_2 = tf.gather(y_pred, tf.cast(y_true, dtype=tf.int32),
                        batch_dims=tf.keras.backend.ndim(y_true) - 1)

    y_pos_1 = tf.concat([y_pos_2, zeros], axis=-1)
    if mask_zero:
        y_pred = tf.concat([-infs, y_pred[..., 1:]], axis=-1)
        y_pos_2 = tf.gather(y_pred, tf.cast(y_true, dtype=tf.int32),
                            batch_dims=tf.keras.backend.ndim(y_true) - 1)

    pos_loss = tf.reduce_logsumexp(-y_pos_1, axis=-1)
    all_loss = tf.reduce_logsumexp(y_pred, axis=-1)
    aux_loss = tf.reduce_logsumexp(y_pos_2, axis=-1) - all_loss
    aux_loss = tf.keras.backend.clip(1 - tf.exp(aux_loss), tf.keras.backend.epsilon(), 1)
    neg_loss = all_loss + tf.keras.backend.log(aux_loss)

    return pos_loss + neg_loss


def loss_(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    shape = y_pred.shape
    # 在bert4keras中y_pred由batch中最长的那一组决定max_len
    # 但在tf中max_len是固定的, 需要将y_pred截断至len(text)的长度
    # 比如一个长度为15个word的句子, 那么正确的输出应该是 (1, 48, 15, 15)
    y_true = y_true[..., 0] * tf.cast(shape[2], tf.float32) + y_true[..., 1]
    y_pred = tf.reshape(y_pred, (shape[0], -1, tf.keras.backend.prod(shape[2:])))

    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, True)

    return tf.reduce_sum(loss, axis=1)


def clip_loss(y_true, y_pred, length):
    loss_value = []

    for yt, yp, le in zip(y_true, y_pred, length):
        yp = tf.cast(tf.expand_dims(yp, 0), dtype=tf.float32)
        yt = tf.cast(tf.expand_dims(yt, 0), dtype=tf.float32)
        yt = K.constant(yt)
        yp = K.constant(yp)
        yp = yp[:, :, :le, :le]
        loss_value.append(loss_(y_true=yt, y_pred=yp))

    loss =  tf.reduce_mean(loss_value)

    return loss


def get_text_token(content):
    inputs = tokenizer.encode_plus(content)

    input_id = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_token = inputs['token_type_ids']

    input_id = tf.constant(sequence_padding([input_id], length=max_len), dtype=tf.int32)
    input_mask = tf.constant(sequence_padding([input_mask], length=max_len), dtype=tf.int32)
    input_token = tf.constant(sequence_padding([input_token], length=max_len), dtype=tf.int32)

    return input_id, input_mask, input_token


def predict(model, text, threshold=0):
    """抽取输入text所包含的三元组
    """

    input_id, input_mask, input_token = get_text_token(text)
    length = tf.reduce_sum(input_mask[0])

    outputs = model((input_id, input_mask, input_token))

    outputs = [np.array(o[0])[:, :length, :length] for o in outputs]
    # 抽取subject和object
    subjects, objects = set(), set()

    outputs[0][:, [0, -1]] -= np.inf
    # 去掉列的开始和结尾
    outputs[0][:, :, [0, -1]] -= np.inf
    # 去掉行的开始和结尾
    # 最终这一步的操作相当于是讲一个矩阵数据的最外围设为np.inf
    # [[1.1 1.2 1.3 1.4 1.5]
    #   [2.  2.  2.  2.  2. ]
    #   [3.  3.  3.  3.  3. ]
    #   [4.  4.  4.  4.  4. ]
    #   [5.  5.  5.  5.  5. ]]]
    # [[-inf -inf -inf -inf -inf]
    #   [-inf   2.   2.   2. -inf]
    #   [-inf   3.   3.   3. -inf]
    #   [-inf   4.   4.   4. -inf]
    #   [-inf -inf -inf -inf -inf]]]

    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        # entity_label设置时:
        # entity_labels[0].append((sh, st))
        # entity_labels[1].append((oh, ot))
        # entity_predict的形状为(2, x, x)
        # (0, x, x)给的是subject的起始
        # (1, y, y)给的是object的起始
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes = set()
    # print(subjects)
    # print(objects)
    # print(outputs[1])
    # print(outputs[1][:, 2, 7])
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            # head_predict的形状是(49, len(id), len(id))
            # [: , sh, oh] 其中
            # : 代表的意思是49即所有的predict-id
            # sh: 代表的意思是所有predict-id指引的第sh列
            # oh: 代表的是所有predict-id指引下的sh列的第oh列
            # z = np.array([
            #     [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4],[0, 1, 2, 31, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],],
            #     [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5],[1, 2, 3, 41, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],],
            #     [[2, 3, 4, 5, 6], [2, 3, 4, 5, 6],[2, 3, 4, 51, 6],[2, 3, 4, 5, 6],[2, 3, 4, 5, 6],],
            #     [[3, 4, 5, 6, 7], [3, 4, 5, 6, 7],[3, 4, 5, 61, 7],[3, 4, 5, 6, 7],[3, 4, 5, 6, 7],],
            #     [[4, 5, 6, 7, 8], [4, 5, 6, 7, 8],[4, 5, 6, 71, 8],[4, 5, 6, 7, 8],[4, 5, 6, 7, 8],],
            #
            # ])
            # print(z[:, 2])
            #  [ 1  2  3 41  5]
            #  [ 2  3  4 51  6]
            #  [ 3  4  5 61  7]
            #  [ 4  5  6 71  8]]
            # print(z[:, 2, 3])
            # [31 41 51 61 71]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            # 效果同上只是获取的tai_predict
            ps = set(p1s) & set(p2s)

            for p in ps:
                s = tokenizer.decode(input_id[0][sh:(st+1)]).replace(' ', '')
                o = tokenizer.decode(input_id[0][oh:(ot+1)]).replace(' ', '')
                try:
                    p = id2predicate[p]
                except:
                    p = None
                for mask in ['[SEP]', '[PAD]', '[UNK]', '[UNK]']:
                    while mask in o:
                        o = o.replace(mask, '')
                    while mask in s:
                        s = s.replace(mask, '')

                spoes.add((s, p, o))

    return list(spoes)

class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate():
    """评估函数，计算f1、precision、recall
    """
    model = tf.saved_model.load('model_save/mid_model/')
    len_lab = 1e-10
    len_pre = 1e-10
    len_pre_is_true = 1e-10

    data = get_dataset(valid_ds_path)

    def process(word):
        while ' ' in word:
            word = word.replace(' ', '')
        return word.lower()

    for _, d in data.iterrows():
        predict_value = set([SPO(spo) for spo in predict(model, d['text'])])
        label_value = set([(process(spo['subject']), spo['predicate'], process(word=spo['object']['@value'])) for spo in d['spo_list']])
        label_value = set([SPO(spo) for spo in label_value])

        len_lab += len(label_value)
        len_pre += len(predict_value)
        len_pre_is_true += len(label_value & predict_value)


    f1_value = 2 * len_pre_is_true / (len_lab + len_pre)
    precision = len_pre_is_true / len_pre
    recall = len_pre_is_true / len_lab

    # f1_value = 2 * len_lab_pre / (len_label + len_predict)
    # precision = len_lab_pre / len_predict
    # recall = len_lab_pre / len_label

    return f1_value, precision, recall


def fit():

    model = ModelBert4GpLink(64)
    optimizer_gp = tf.keras.optimizers.Adam(learning_rate=2e-5)
    optimizer_bert = tf.keras.optimizers.Adam(learning_rate=1e-5)
    f1_value = 1e-10
    def step(batch_ds):
        inputs_seq = batch_ds[0]

        labels = batch_ds[-1]
        entity_labels = labels[0]
        head_labels = labels[1]
        tail_labels = labels[2]
        id_length = tf.reduce_sum(inputs_seq[1], axis=1)

        with tf.GradientTape() as tp:

            entity_predict, head_predict, tail_predict = model(inputs_seq)

            entity_loss = clip_loss(entity_labels, entity_predict, id_length)

            head_loss = clip_loss(head_labels, head_predict, id_length)

            tail_loss = clip_loss(tail_labels, tail_predict, id_length)

            loss_v = (entity_loss + head_loss + tail_loss)

            params_bert = []
            params_other = []

            for var in model.trainable_variables:
                model_name = var.name
                none_bert_layer = ['tf_bert_model/bert/pooler/dense/kernel:0',
                                   'tf_bert_model/bert/pooler/dense/bias:0',
                                   'Variable:0']
                if model_name in none_bert_layer:
                    pass
                elif model_name.startswith('tf_bert_model'):
                    params_bert.append(var)
                else:
                    params_other.append(var)

        params_all = tp.gradient(loss_v, [params_bert, params_other])

        gradients_bert = params_all[0]
        gradients_other = params_all[1]

        optimizer_bert.apply_gradients(zip(gradients_bert, params_bert))

        optimizer_gp.apply_gradients(zip(gradients_other, params_other))

        return entity_loss, head_loss, tail_loss


    for _ in range(10):

        train_ds = data_generator(get_dataset(train_ds_path))

        for num, ds in enumerate(train_ds):
            loss1, loss2, loss3 = step(ds)
            if (num+1) % 100 == 0:
                print(f"entity_loss: {loss1}, head_loss: {loss2}, tail_loss: {loss3}")

        model.save('model_save/mid_model/')
        f1, _, _ = evaluate()

        if f1 > f1_value:
            model.save('model_save/best_model')
            f1_value = f1

# fit()

print(evaluate())
