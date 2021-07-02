# *- coding: utf-8 -*

import re
import os
import json
import pymongo
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Layer

tf.compat.v1.disable_v2_behavior()


min_count = 32
max_len = 400
char_size = 64
batch_size = 64
z_dim = 128

# mongo dataset
def get_dataset():
    host = 'xxx'
    port = 'xxx'
    db_name = 'xxx'
    collection = 'xxx'

    client = f"mongodb://{host}:{port}/"
    mongo_sql_client = pymongo.MongoClient(client)
    mongo_sql_db = mongo_sql_client[db_name]
    mongo_sql_collection = mongo_sql_db[collection]

    dataset = mongo_sql_collection.find(
        {'title': {'$nin': [None, ""]}, 'desc': {'$nin': [None, ""]}, 'type': {'$ne': '更新日志'} },
        {'_id': 1, 'title': 1, 'desc': 1}
    ).skip(0)

    dataset = pd.DataFrame(dataset)

    dataset['title'] = dataset['title']

    def desc_pro(desc):
        try:
            return desc.strip()
        except:
            return np.NAN

    dataset['desc'] = dataset['desc'].apply(
        lambda x: desc_pro(x)
    )

    dataset.dropna(subset=['desc'], inplace=True)

    pattern = u'[\u4e00-\u9fa5]+'
    dataset['title'] = dataset['title'].apply(
        lambda x: ''.join(re.findall(pattern=pattern, string=x))
    )

    dataset['desc'] = dataset['desc'].apply(
        lambda x: ''.join(re.findall(pattern=pattern, string=x))
    )

    # return pd.DataFrame(dataset)


def process_word(dataset=None):
    # 首先构建编码表
    # 第一步: 通过统计字在所有文章中出现的数目来去掉一部分出现数目过于少的字
    # 第二步: 通过单字所处的位置为其构建词表(只是为每个字赋予一个编号因此无需考虑其他因素)

    if os.path.exists('seq2seq_config.json'):
        chars_, id2char, char2id = json.load(open('seq2seq_config.json'))
        id2char = {int(i): j for i, j in id2char.items()}
    else:
        # 第一步:
        chars_ = {}
        num = 0
        for t, w in zip(dataset['title'], dataset['content']):
            for i in t:
                chars_[i] = chars_.get(i, 0) + 1
            for i in w:
                chars_[i] = chars_.get(i, 0) + 1

            num = num + 1
            if num % 1000 == 0:
                print(num)

        chars_ = {i: j for i, j in chars_.items() if j >= min_count}
        # 0: mask 1: unk  2: start 3: end
        # 使用四分位数控制min_count会比较好
        # 第二步:
        # 0: mask 1: unk 2: start 3: end
        # 所以说字的位置要从4开始
        id2char = {i + 4: j for i, j in enumerate(chars_)}
        char2id = {j: i for i, j in id2char.items()}
        json.dump([chars_, id2char, char2id], open('seq2seq_config.json', 'w'))

    return chars_, char2id, id2char


# path = '../news_dataset_clean/link_dataset.xlsx'
# path = '../news_dataset_clean/link_dataset.xlsx'
# dataset_ = pd.read_excel(path, usecols=['content', 'title'])
# dataset = dataset_[:640000]
# val_dataset = dataset_[640000:]
chars, char_2id, id_2char = process_word()
# chars 是输入数据所有字以及其出现的次数数目
# char2id 是 字:字编码 的字典
# id2char 是 字编码:字 的字典

print(f"chars: {len(chars) + 4}, char_2id: {len(char_2id)}, id_2char: {len(id_2char)}")

def words2id(words, add_st_end=False):
    # 句子转id(位置编码)
    if add_st_end:
        # 输出即label需要为其填上<start>和<end>标志
        words_ids = [char_2id.get(w, 0) for w in words[:max_len-2]]
        # 如果字表中没有对应单词, 则使用mask(0代表mask)将其替换
        # 这个做法是为了忽略(mask)掉一部分单词, 防止过拟合(加噪音)、提高速度
        # [max_len-2]的原因是为start和end标志提供位置, 当然如果本身长度就小于max_len的话就无所谓了
        # 如果长度大于max_len则放弃掉后面的数据

        words_ids = [2] + words_ids[:max_len-2] + [3]
    else:
        words_ids = [char_2id.get(w, 0) for w in words[:max_len]]

    return words_ids


def ids2words(ids):
    # 最终处理完毕后需要将位置编码转为字
    return ''.join([id_2char.get(i, '') for i in ids])


def padding(ids):
    # 为每一条数据填充至max_len
    # 当然也可以获取到最长数据的长度, 为其他数据填充至改长度
    ids = [i + [0] * (max_len - len(i)) for i in ids]
    return ids


def data_generator(dataset):
    """
    生成的是一个元组, 每个元组的元素是一个batch 包含了[data, label], None
    """
    inputs, labels = [], []
    while True:
        # 为了确保数据可以不断的进入模型可以将其看作是tf.data的.shuffle().repeat()函数组合
        dataset = dataset.sample(frac=1)
        # 添加该语句的目的是确保每轮循环的dataset都是顺序打乱的
        for _, data in dataset.iterrows():
            content = words2id(data['content'])
            tit = words2id(data['title'], add_st_end=True)
            inputs.append(content)
            labels.append(tit)
            if len(inputs) == batch_size:
                inputs = np.array(padding(inputs))
                labels = np.array(padding(labels))
                yield [inputs, labels], None
                inputs, labels = [], []


# data = get_dataset()
# x, y = data_generator(data)
# x_mask = tf.expand_dims(tf.cast(tf.greater(x, 0), np.float32), 2)
# seq_len = tf.cast(tf.reduce_sum(x_mask, axis=1)[:, 0], np.int32)
# print(x_mask)
# print(seq_len)


def to_one_hot(inputs):
    """输出一个词表大小的向量, 来标记该词是否出现在文章
    """
    inputs, inputs_mask = inputs
    inputs = tf.cast(inputs, np.int32)
    inputs = tf.one_hot(inputs, len(chars)+4)
    # one hot编码
    inputs = tf.reduce_sum(inputs_mask*inputs, 1, keepdims=True)
    # inputs_mask功能是: 标识了des编码后每个位置是否存在单词
    # inputs 经过one-hot后标识了每个位置单词的在向量上所处的位置
    # 两者相乘即可得到符合content真实情况的以字为维度的向量展示
    # sum之后即得到了content本身的展示
    inputs = tf.cast(tf.greater(inputs, 0.5), tf.float32)
    return inputs


class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
        self.shift = None
        self.log_scale = None

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        # 继承自Layer的一个方法，可以接受额外字典参数**kwargs

        self.built = True

    def call(self, inputs, **kwargs):
        x_outs = tf.exp(self.log_scale) * inputs + self.shift
        # 为数据进行tf.exp的目的是既不改变数据之间的关系又可以压缩数据尺寸
        # 消除异方差的以及转化计算方法
        return x_outs


class OurLayer(Layer):
    """定义新的layer，增加reuse方法允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        # layer.built 用于控制
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                inputs_shape = [tf.keras.backend.int_shape(x) for x in inputs]
                # int_shape以元祖tuple的形式返回tensor或者变量(tf.keras.backend.variable)的shape
                # 且tf.keras.backend.variable的shape为()
            else:
                inputs_shape = tf.keras.backend.int_shape(inputs)
            layer.build(inputs_shape)
            # 根据传入的inputs_shape 创建一个对应的layer层
        outputs = layer.call(*args, **kwargs)
        # 返回了层本身的inputs

        if not tf.keras.__version__.startswith('2.3.'):
            # 根据keras版本的不同添加其他版本的参数
            for w in layer.trainable_weights:
                if w not in self._trainable_weights:
                    self._trainable_weights.append(w)
            for w in layer.non_trainable_weights:
                if w not in self._non_trainable_weights:
                    self._non_trainable_weights.append(w)
            for u in layer.updates:
                if not hasattr(self, '_updates'):
                    self._updates = []
                if u not in self._updates:
                    self._updates.append(u)

        return outputs


def reverse_sequence(v, mask):
    """mask.shape->[batch_size, seq_len, 1]
    """

    seq_len = tf.round(tf.reduce_sum(mask, axis=1)[:, 0])
    seq_len = tf.cast(seq_len, tf.int32)
    # 此处的seq_len 为每条mask内部为1的值的数目和
    # x_mask = tf.expand_dims(tf.cast(tf.greater(x, 0), np.float32), 2)
    # seq_len = tf.cast(tf.reduce_sum(x_mask, axis=1)[:, 0], np.int32)
    # print(tf.reverse_sequence(x, seq_len, 1))
    # x.row == len(seq_len)
    # 操作是将x的每一行对应的seq_len数据处进行翻转
    # 比如x -> [[1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4], [4, 4, 5, 5], [5, 5, 6, 6]]
    # x.row = 5
    # seq_len=[3, 3, 3, 3, 3]
    # 那么翻转就是x-> [[2, 1, 1, 2], [3, 2, 2, 3], ...]
    return tf.reverse_sequence(v, seq_len, 1)


class OurBidirectional(OurLayer):
    """封装双向RNN, 允许传入mask保证对齐
    """
    def __init__(self, layer, **kwargs):
        super(OurBidirectional, self).__init__(**kwargs)
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())

        self.forward_layer_name = 'forward_' + self.forward_layer.name
        self.backward_layer_name = 'forward_' + self.backward_layer.name

    def call(self, inputs, **kwargs):
        x, mask = inputs

        x_forward = self.reuse(self.forward_layer, x)
        # 继承自类 OurLayer的方法

        x_backward = reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = reverse_sequence(x_backward, mask)

        x = tf.keras.backend.concatenate([x_forward, x_backward], -1)
        if x.shape.rank == 3:
            # 获取x的维度
            return x * mask
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.forward_layer.units * 2, )
        # (2, 2) + (4,) -> (2, 2, 4)


def seq_avg_pool(seq_mask):
    """seq是[None, seq_len, s_size]的格式, mask是[None, seq_len, 1]的格式
    去除mask的部分做avg_pooling
    """
    seq, mask = seq_mask
    seq -= (1 - mask) * 1e10

    return tf.reduce_sum(seq * mask, 1) / (tf.reduce_sum(mask, 1) + 1e-6)


def seq_max_pool(seq_mask):
    """seq是[None, seq_len, s_size]的格式, mask是[None, seq_len, 1]的格式
    去除mask的部分做max_pooling
    """
    seq, mask = seq_mask
    seq -= (1 - mask) * 1e10

    return tf.reduce_max(seq, 1)

class LayerNormalization(Layer):
    # 原始代码的LayerNormalization来自于from keras_layer_normalization import LayerNormalization
    # 但是受限于版本问题(可能要求tf1.+版本)因此重写该类
    # 其实就是将其中的一些函数改为TensorFlow
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = tf.keras.backend.epsilon() * tf.keras.backend.epsilon()
        # 返回数值表达式中使用的模糊因子的值
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer  = tf.keras.initializers.get(beta_initializer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer  = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_constraint  = tf.keras.constraints.get(gamma_constraint)
        self.beta_constraint   = tf.keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer' : tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer' : tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint' : tf.keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint'  : tf.keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):

        mean = tf.keras.backend.mean(inputs, axis=-1, keepdims=True)
        variance = tf.keras.backend.mean(tf.keras.backend.square(inputs - mean), axis=-1, keepdims=True)
        std = tf.keras.backend.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class SelfModulatedLayerNormalization(OurLayer):
    """模仿self-Modulated Batch Normalization
    并将batch normalization 改为 Layer Normalization"""
    # batch-Norm 针对的是每组数据的单一数据相较于组数据维度的缩放
    # tv = [x1, x2, x3....xi]
    # tv_1 = [(i - i/len(tv) )**2 / len(tv) for i in tv]
    # sklearn.preprocessing.Normalizer
    # scale & shift 缩放平移
    # batch-normalization的目的是防止梯度小时和爆炸，加快收敛，可正则化模型
    # 此处是针对层的normalization

    def __init__(self, num_hidden, **kwargs):
        super(SelfModulatedLayerNormalization, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        super(SelfModulatedLayerNormalization, self).build(input_shape)
        output_dim = input_shape[0][-1]
        self.layer_norm = LayerNormalization(center=False, scale=False)
        self.beta_dense_1 = tf.keras.layers.Dense(self.num_hidden, activation='relu')
        self.beta_dense_2 = tf.keras.layers.Dense(output_dim)
        self.gamma_dense_1 = tf.keras.layers.Dense(self.num_hidden, activation='relu')
        self.gamma_dense_2 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, **kwargs):
        inputs, cond = inputs
        inputs = self.reuse(self.layer_norm, inputs)
        beta = self.reuse(self.beta_dense_1, cond)
        beta = self.reuse(self.beta_dense_2, beta)
        gamma = self.reuse(self.gamma_dense_1, cond)
        gamma = self.reuse(self.gamma_dense_2, gamma)

        for _ in range(inputs.shape.rank - cond.shape.rank):
            beta = tf.expand_dims(beta, 1)
            gamma = tf.expand_dims(gamma, 1)

        return inputs * (gamma + 1) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Attention(OurLayer):
    """多头注意力机制
    所谓”多头“（Multi-Head），就是只多做几次同样的事情（参数不共享），然后把结果拼接
    """
    def  __init__(self, heads, size_pre_head, key_size=None, mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_pre_head
        self.out_dim = heads * size_pre_head
        self.key_size = key_size if key_size else size_pre_head
        self.mask_right = mask_right

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = tf.keras.layers.Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = tf.keras.layers.Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = tf.keras.layers.Dense(self.out_dim, use_bias=False)

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(x.shape.rank - mask.shape.rank):
                mask = tf.expand_dims(mask, mask.shape.rank)
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
                # 1-mask 存在两种结果:
                # 1: 0 说明该位置存在值, 相当于x对应位置的值减去0即不变
                # 2: 1 说明该位置对应的x处不存在值, 相当于x对应位置的值减去-1e10即负无穷
                # 此举的目的是计算attention score的时候对padding做mask操作

    # 获取attention score
    # 公式是: https://pic1.zhimg.com/v2-e698e0083f4cc8d0fae45c501fb9aef8_r.jpg
    # 其中q指下方的q_value进行的一系列计算, k指下方的k_value的一系列计算, dk指的是self.key_size
    def call(self, inputs, **kwargs):
        q_value, k_value, v_value = inputs[:3]
        # inputs -> [y, x, x, x_mask]
        # Attention在seq2seq中的思路是重复利用输入值与输出值作为下一个单字的输出
        # 因此这里有两个x

        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变化
        q_linear = self.reuse(self.q_dense, q_value)
        k_linear = self.reuse(self.k_dense, k_value)
        v_linear = self.reuse(self.v_dense, v_value)
        # 形状变化
        # 确保 reshape内部shape的乘积等于q_linear.shape内部的乘积即可
        qw = tf.reshape(q_linear, shape=(-1, tf.shape(q_linear)[1], self.heads, self.key_size))
        # 将维度空间从(64, 128, 128)->(64, 128, 8, 16)的目的是:
        # 转化为多个低维空间最后进行拼接，形成同样维度的输出，借此丰富特性信息，降低了计算量
        # print(f"qw shape: {qw.shape}")
        kw = tf.reshape(k_linear, (-1, tf.shape(k_linear)[1], self.heads, self.key_size))
        vw = tf.reshape(v_linear, (-1, tf.shape(v_linear)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = tf.transpose(qw, (0, 2, 1, 3))
        # print(f"qw transpose shape: {qw.shape}")
        # 首先需要明确这里的0, 2, 1, 3并不是让qw的维度变为0, 2, 1, 3(因为也没有0这个维度)
        # 这里的0, 2, 1, 3可以认为是对维度位置对应维度的调换
        # 比如qw的维度是(64, 128, 8, 16)本身维度的位置是(0, 1, 2, 3)此时transpose希望将它的维度转变
        # 就通过它的位置信息传递, 即将其维度变为(64, 8, 128, 16)此时对应的位置就是(0, 2, 1, 3)
        kw = tf.transpose(kw, (0, 2, 1, 3))
        vw = tf.transpose(vw, (0, 2, 1, 3))
        # Attention
        attention = tf.einsum('ijkl, ijml->ijkm', qw, kw) / self.key_size**0.5
        attention = tf.transpose(attention, (0, 3, 2, 1))
        attention = self.mask(attention, v_mask, 'add')
        attention = tf.transpose(attention, (0, 3, 2, 1))
        if self.mask_right:
            ones = np.ones_like(attention[:1, :1])
            mask = (ones - tf.linalg.band_part(ones, num_lower=-1, num_upper=0))
            attention = attention - mask
            # 以矩阵对角线(从(0, 0)处到右下角结束) num_low代表下三角, num_upper代表上三角
            # 若取负值则视作不处理, 若取正数则当做向上或者向下从起点和结束点平移对应位置画出一个三角
            # 这个三角内部的值全部填充为0
            # 此处相当于将ones的以对角线分割下半区为1上半区为0
            # 假设ones原本为:       现在则变为了:
            # [[1, 1, 1, 1]]      [[1, 0, 0, 0]]
            # [[1, 1, 1, 1]]      [[1, 1, 0, 0]]
            # [[1, 1, 1, 1]]      [[1, 1, 1, 0]]
            # [[1, 1, 1, 1]]      [[1, 1, 1, 1]]

        attention_value = tf.math.softmax(attention)
        attention_value = tf.einsum('ijkl, ijlm->ijkm', attention_value, vw)
        attention_value = tf.transpose(attention_value, (0, 2, 1, 3))
        attention_value = tf.reshape(attention_value, (-1, tf.shape(attention_value)[1], self.out_dim))
        attention_value = self.mask(attention_value, q_mask, 'mul')

        return attention_value

    def compute_output_shape(self, input_shape):
        return [input_shape[0][0], input_shape[0][1], self.out_dim]


x_in = tf.keras.Input(shape=(None, ))
y_in = tf.keras.Input(shape=(None, ))
x_, y_ = x_in, y_in


# 初始值shape: (batch_size, max_len)
x_mask = tf.keras.layers.Lambda(lambda x: tf.cast(tf.greater(tf.expand_dims(x, 2), 0), tf.float32))(x_)
# 将其拆解后可知进行了如下操作
# 1: 增加维度 在shape=2处增加一个维度将原来(10, 40)->(10, 40, 1)
# inputs_mask = k.expand_dims(inputs, 2)
# 2: 将为0的值置为false大于0的则为True
# inputs_mask = k.greater(inputs_mask, 0)
# 3: 转换类型
# inputs_mask = k.cast(inputs_mask, np.float32)
y_mask = tf.keras.layers.Lambda(lambda x: tf.cast(tf.greater(tf.expand_dims(x, 2), 0), tf.float32))(y_)
# print(f"x_mask: {x_mask}")
# print(f"y_mask: {y_mask}")
x_one_hot = tf.keras.layers.Lambda(to_one_hot)([x_, x_mask])
# one_hot: (batch_size, 1, len(char)+4) len(char)+4 指的是字典长度+4是因为mask编码
x_prior = ScaleShift()(x_one_hot)

# scale: (batch_size, 1, len(char)+4) 对内部数据做了处理
# ScaleShift 仅仅是对x_one_hot的数据进行了缩放平移处理, shape保持不变, 因此在embedding时input_dim就是len(chars)+4


# embedding 就是特征转向量, output_dim决定了输出向量的维度 input_dim则限定了输入特征的数目在此处要求input_dim >= len(char)+4
embedding = tf.keras.layers.Embedding(input_dim=len(chars) + 4, output_dim=char_size)
x = embedding(x_)
# embedding: (batch_size, max_len, char_size) 对batch_size组对话内容的每个字进行了向量映射

# encoder 双层双向LSTM
x = LayerNormalization()(x)
# 针对于层的归一化操作, 常用于RNN层, 此处的CuDNN LSTM层继承了部分RNN层的类, 而且LSTM本身就是RNN层的变种之一
# 其他相似的如BatchNor, GroupNorm, InstanceNorm, SwitchableNorm等基本上是针对不同情境下的归一化操作

x = OurBidirectional(tf.keras.layers.LSTM(z_dim // 2, return_sequences=True))([x, x_mask])
# 在tensorflow 2中cudnn lstm并入到了lstm中
# 如果使用CuDNNLSTM需要考虑版本问题, tf.compat.v1.keras.layers.CuDNNLSTM
x = LayerNormalization()(x)
x = OurBidirectional(tf.keras.layers.LSTM(z_dim // 2, return_sequences=True))([x, x_mask])
x_mask_seq = tf.keras.layers.Lambda(seq_max_pool)([x, x_mask])
# seq_max_pool shape: (batch_size, max_le) 此处通过x_mask 将x里面不为空的值进行了处理

y = embedding(y_)
# print(f"y_embedding: {y}, shape: {y.shape}")
# decoder 双层单向LSTM
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_mask_seq])
y = tf.keras.layers.LSTM(z_dim, return_sequences=True)(y)
# 通常，在需要将各个隐层的结果作为下一层的输入时，选择设置 return_sequences=True
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_mask_seq])
y = tf.keras.layers.LSTM(z_dim, return_sequences=True)(y)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_mask_seq])

# 从embedding到此处中间设计层处理后的shape均为: (batch_size, max_len, char_size) 只是对内部数据进行了处理

# Attention交互
xy = Attention(8, 16)([y, x, x, x_mask])

xy = tf.keras.layers.Concatenate()([y, xy])

# 输出分类
xy = tf.keras.layers.Dense(char_size)(xy)
# Relu激活函数的目的是将目标张量内部小于0的值置为0, leaky Relu函数则是将于小于0的值除以固定参数此处是0.2
xy = tf.keras.layers.LeakyReLU(0.2)(xy)
xy = tf.keras.layers.Dense(len(chars)+4)(xy)
xy = tf.keras.layers.Lambda(lambda x: (x[0]+x[1])/2)([xy, x_prior])
xy = tf.keras.layers.Activation('softmax')(xy)


cross_entropy_func = lambda x: tf.keras.losses.sparse_categorical_crossentropy(x[0][:, 1:], x[1][:, :-1])

cross_entropy = tf.keras.layers.Lambda(cross_entropy_func)([y_in, xy])
# cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
cross_entropy = tf.reduce_sum(cross_entropy * y_mask[:, 1:, 0]) / tf.reduce_sum(y_mask[:, 1:, 0])

model = tf.keras.Model([x_in, y_in], xy)
model.add_loss(cross_entropy)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
)


def gen_sent(sentence, top_k=3, max_length=64):
    """
    beam search 解码
    每次只保留top_k个最优候选结果, 如果top_k=1则为贪心搜索
    """
    xid = np.array([words2id(sentence)] * top_k)
    yid = np.array([[2]] * top_k)
    # 选择[2]的原因是<start> 的编码是[2]而解码均是以2为开头的
    scores = [0] * top_k    # 候选答案的分数
    for i in range(max_length):
        pre = model.predict([xid, yid])[:, i, 3:]
        log_pre = np.log(pre + 1e-6)    # 取对数方便计算
        arg_top_k = log_pre.argsort(axis=1)[:, -top_k:]     # 选出每一页的top_k
        _yid = []
        _scores = []
        # 暂存候选目标序列和得分
        if i == 0:
            for j in range(top_k):
                _yid.append(list(yid[j]) + [arg_top_k[0][j] + 3])
                # yid[i] 其实就是<start>[2]
                _scores.append(scores[j] + log_pre[0][arg_top_k[0][j]])
        else:
            for j in range(top_k):
                for t in range(top_k):
                    _yid.append(list(yid[j]) + [arg_top_k[j][t] + 3])
                    _scores.append(scores[j] + log_pre[j][arg_top_k[j][t]])
            _arg_top_k = np.argsort(_scores)[-top_k:] # 获取新的top_k

            _yid = [_yid[t] for t in _arg_top_k]
            _scores = [_scores[t] for t in _arg_top_k]
        yid = np.array(_yid)
        scores = np.array(_scores)
        best_one = np.argmax(scores)
        if yid[best_one][-1] == 3:
            return ids2words(yid[best_one])
        # 如果最后一个字不是<end> 直接返回
    return ids2words(yid[np.argmax(scores)])



# gen_sent(s1)
s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医 。'
s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'


class Evaluate(tf.keras.callbacks.Callback):

    def __init__(self):
        super(Evaluate, self).__init__()
        self.loss_min = 1e10
    # 包括acc和loss的日志，并且可选地包括val_loss和val_acc
    def on_epoch_end(self, epoch, logs=None):
        print(f"\ns1: {gen_sent(s1)} \n")
        print(f"s2: {gen_sent(s2)} \n")
        if logs['loss'] <= self.loss_min:
            self.loss_min = logs['loss']
            model.save_weights('weights/best_model.weights')

evaluator = Evaluate()


def fit_(dataset, val_dataset):
    model.fit(
        data_generator(dataset),
        steps_per_epoch=1000,
        # 正常来说steps_per_epoch和epochs以及len(dataset)的关系是 steps_per_epoch = len(dataset) / batch_size / epoch得到的
        # 但是在数据迭代生成哪里加了个 while True: 之后数据就可以确保循环产生相当于近似的len(dataset)是无限大
        # 在TensorFlow中可以使用.shuffle().repeat()替代这种无限循环
        epochs=50,
        validation_data=data_generator(val_dataset),
        validation_steps=1,
        callbacks=[evaluator]
    )

def eva_():
    model.load_weights(filepath='weights/best_model.weights')
    t1 = '“互联网+”战略的提出，为工业互联网的发展指明了方向。中国互联网与工业融合创新联盟专家委员会专家高新民认为，互联网可以“+”各行各业，但是“互联网+工业”具有更加重要的战略意义。简单而言，“互联网+”应该是全面推进，突出重点的模式，产业互联网化尤其是工业互联网化和制造业的互联网化是重点，同时也是难点。产业互联网不仅是产业形态与互联网技术的融合，也是互联网思维和商业模式对传统产业的渗透，值得一提的是，正是因为工业互联网是一种“跨界”，因此需要搭建沟通和交流的平台。2014年7月，在工信部的指导下，中国互联网与工业融合创新联盟成立。联盟由中国信息通信研究院、电子科学技术情报研究所、中国电子学会、中国互联网协会等单位共同发起。曹淑敏在大会上对联盟的工作进行了介绍和总结，联盟在会上发布了《互联网与工业融合创新蓝皮书》。对于联盟未来的工作，怀进鹏提出了希望和要求，认为联盟要致力于做到三个有利于：有利于企业的发展，促进企业间交流信息、分享经验；有利于消费者更便捷地获得高质量产品，满足消费者需求；有利于政府结合企业和社会需求，出台和制定更好的政策，营造更好的环境。'
    t2 = '中新社北京2月2日电 (记者 刘育英)2018年是中国全面实施工业互联网建设的开局之年，工信部将统筹推进“323”行动，并实施工业互联网三年行动计划。2018工业互联网峰会1日至2日在北京国家会议中心举行。工信部部长苗圩表示，工信部将统筹推进工业互联网发展的“323”行动。据介绍，“323”行动是指打造网络、平台、安全三大体系；推进两类应用，一是大型企业集成创新，二是中小企业应用普及；构建产业、生态和国际化三大支撑。工信部还将组织开展工业互联网项目试点示范，推进工业互联网试点城市和示范基地建设，推动工业互联网应用由过去家电、服装、机械产业向飞机、石化、钢铁、橡胶、物流等更广泛领域普及。'
    s3 = '新浪体育讯 北京时间10月16日,NBA中国赛广州站如约开打,火箭再次胜出,以95-85击败篮网。姚明渐入佳境,打了18分39秒,8投5中,拿下10分5个篮板,他还盖帽1次。火箭以两战皆胜的战绩圆满结束中国行。'
    s4 = '羽白网是提供工业智造信息服务的网站, 包含羽白服务、培训、文库三个功能，羽白服务有产业资讯，生态治理，共享智造三大功能，' \
         '帮助用户了解工业领域产业的基本信息介绍和发展经验，普及绿色工业概念，介绍现代化工业下产生的新技术新发展，羽白培训为用户提' \
         '供教育培训和专家咨询的服务，羽白文库包含新闻资讯，行业信息，时政要闻，热点政策等文章内容。为用户提供工业信息化的服务。'
    s5 = "近年来，上海临港地区积极推动人工智能产业发展，致力于成为以人工智能为特色的科创中心主体承载区，并成为具有全球影响力的" \
         "人工智能技术创新策源地和产业化基地。上海林港人工智能科技有限公司（以下简称“林港人工智能”）是临港引进的首批人工智能企" \
         "业之一，是临港地区2+3+4产业体系中的两大先导产业之一予以重点扶持单位。林港人工智能立足全球视野，秉承“以新一代数字智" \
         "能技术促产业升级”的宗旨，坚持“产学研用”深度融合的创新发展之路，培育了基于AI人工智能的实体经济多场景应用。经过长时" \
         "间探索与实践，公司已经投资孵化形成了以AI智能、5G、大数据、区块链等技术为支撑的产业咨询、生态治理和绿色智造三大业务" \
         "生态。目前，已成功为纺织、铸造、机械装备、水泥、家具等10个传统制造行业提供转型升级服务，累计帮扶4000余家企业。" \
         "当前林港人工智能汇聚了众多行业资深专家和国际管理咨询师，建立了规范运作，科学高效的投融资建设体系，持续投资特种机器人、" \
         "生态治理技术、工业信息软件、智能共享制造与工业互联网等新兴业务"
    print(gen_sent(s4))
    print(gen_sent(s5))

eva_()
