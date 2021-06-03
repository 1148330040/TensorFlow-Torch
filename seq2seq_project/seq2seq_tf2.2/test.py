# *- coding: utf-8 -*


import tensorflow as tf
import torch
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'a': np.arange(100),
    'b': np.arange(100)
})

def test(ds):
    s = 0
    for num, (_, d) in enumerate(ds.iterrows()):
        s = s+d['a'] + d['b']
        if (num+1) % 11 == 0:
            yield s
            s = 0


ds = test(data)

for i in range(5):
    print(i)
    for num, d in enumerate(ds):
        print(f"i: {i}, num: {num}, d: {d}")