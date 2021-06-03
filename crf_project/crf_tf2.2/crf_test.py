# *- coding: utf-8 -*

import os
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.text.crf import crf_log_likelihood
from bert4keras.models import build_transformer_model

s = pd.DataFrame({
    'a': [1, 2, 3, 4, 5],
    'b': [1, 2, 3, 4, 5],
    'c': [1, 2, 3, 4, 5]
})


s['d'] = s[['a', 'b', 'c']].apply(
    lambda x: list(x), axis=1
)
print(s)