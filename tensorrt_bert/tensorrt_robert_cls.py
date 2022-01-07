import os
import re

import warnings

import numpy as np
import pandas as pd

import tensorflow as tf
from datetime import datetime

import tensorrt as trt
import uff # 目前已被NVIDIA弃用建议使用onnx
import pycuda

import onnx_graphsurgeon
import onnxruntime as ort
import onnx
from bert4keras.snippets import sequence_padding
from transformers import BertTokenizer
from sklearn.metrics import f1_score


from transformers.convert_graph_to_onnx import convert_tensorflow
import tf2onnx.convert

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-2_H-768")
# 进入linux系统-mnt/tang_nlp/tensorrt_bert文件/运行下列命令转换onnx模型
# python -m tf2onnx.convert  --saved-model best_model_2/  --output test.onnx  --opset 11

dataset = pd.read_excel('dataset/dataset_sample.xlsx')
dataset.dropna(inplace=True)
test_dataset = dataset[34000:44000]

def get_seq_ds(content):
    max_len = 68
    ids, masks, tokens = [], [], []

    inputs = tokenizer.encode_plus(content)

    input_id = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_token = inputs['token_type_ids']

    ids.append(tf.constant(input_id, dtype=tf.int32))
    masks.append(tf.constant(input_mask, dtype=tf.int32))
    tokens.append(tf.constant(input_token, dtype=tf.int32))

    ids = sequence_padding(ids, max_len)
    masks = sequence_padding(masks, max_len)
    tokens = sequence_padding(tokens, max_len)

    return  ids, masks, tokens


def test_process_onnx(batch_size):

    onnx_robert = onnx.load_model('robert_2_layer.onnx')

    for input in onnx_robert.graph.input:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = batch_size
    onnx.save_model(onnx_robert, 'robert_2_layer.onnx')


    robert_onnx = ort.InferenceSession("robert_2_layer.onnx")
    robert_onnx.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])

    predicts = []
    ids_ = []
    masks_ = []
    tokens_ = []
    time1 = datetime.now()
    for num, ds in test_dataset.iterrows():
        text = ds['review']
        labels = np.zeros(20)
        labels[ds['label']] = 1

        ids, mask, tokens = get_seq_ds(text)
        ids_.append(ids[0])
        masks_.append(mask[0])
        tokens_.append(tokens[0])
        if len(ids_) < batch_size:
            continue
        onnx_inputs = {'input_ids': ids_, 'masks': masks_, 'tokens': tokens_}
        predict = robert_onnx.run(None, onnx_inputs)[0]
        predict = np.nanargmax(np.array(predict), axis=1)
        predicts = predicts + list(predict)
        ids_ = []
        masks_ = []
        tokens_ = []

    labels = np.array(test_dataset['label'])
    f1 = f1_score(y_true=labels, y_pred=np.array(predicts), average='weighted')
    print(datetime.now() - time1)
    return f1


def test_process_pb(batch_size):
    model = tf.saved_model.load('best_model_2/')
    predicts = []
    ids_ = []
    masks_ = []
    tokens_ = []
    time1 = datetime.now()
    for num, ds in test_dataset.iterrows():
        text = ds['review']
        labels = np.zeros(20)
        labels[ds['label']] = 1
        ids, mask, tokens = get_seq_ds(text)
        ids_.append(ids[0])
        masks_.append(mask[0])
        tokens_.append(tokens[0])
        if len(ids_) < batch_size:
            continue
        predict = model((ids_, masks_, tokens_))
        predict = np.nanargmax(np.array(predict), axis=1)
        predicts = predicts + list(predict)
        ids_ = []
        masks_ = []
        tokens_ = []
    labels = np.array(test_dataset['label'])
    f1 = f1_score(y_true=labels, y_pred=np.array(predicts), average='weighted')
    print(datetime.now() - time1)

    return f1


test_process_onnx(1)
test_process_pb(1)

# batch_size: 1
# 0:00:24.758920    onnx---gpu
# 0:00:56.001462    pb----gpu

# batch_size: 5
# 0:00:17.935403    onnx---gpu
# 0:00:18.646852    pb----gpu

# batch_size: 10
# 0:00:17.418962    onnx---gpu
# 0:00:16.452406    pb----gpu

# batch_size: 20
# 0:00:16.086152    onnx---gpu
# 0:00:15.104701    pb----gpu

# batch_size: 40
# 0:00:15.266314    onnx---gpu
# 0:00:14.499982    pb----gpu