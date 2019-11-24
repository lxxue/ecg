from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm
import pandas as pd

STEP = 256

def my_load_dataset(idx, x_all, y_all):
    x_list = []
    for i in idx:
        x_list.append(x_all[i])
    y = y_all[idx]
    # x = pd.read_csv(x_fname)
    # x = x.drop(['id'], axis=1)
    # x = x.fillna(0)
    # x = x.values
    # x = x[idx]
    # x_list = []
    # for i in range(len(x)):
    #     x_i_end = x[i].nonzero()[0][-1]
    #     x_i_end = STEP * int((x_i_end+1) / STEP) 
    #     x_list.append(x[i][:x_i_end+1])
    # y = pd.read_csv(y_fname)
    # y = y.drop(['id'], axis=1)
    # y = y.values.squeeze()
    # y = y[idx]
    return x_list, y

def pad(x, val=0, dtype=np.float32):
    max_len = max([len(x_i) for x_i in x])
    # print(max_len)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for i, x_i in enumerate(x):
        padded[i, :len(x_i)] = x_i
    return padded

class MyPreprocess:
    def __init__(self, x_list, y):
        x_hstack = np.hstack(x_list)
        self.mean = np.mean(x_hstack).astype(np.float32)
        self.std = np.std(x_hstack).astype(np.float32)
        self.step = STEP
        self.classes = [0, 1, 2, 3]

    def process(self, x_b, y_b):
        # assert(len(x_b) == len(y_b))
        x = pad(x_b)
        x = (x - self.mean) / self.std
        x = x[:, :, None]

        y = pad([[y_b[i]] * int(len(x_b[i]) / self.step) for i in range(len(x_b))], val=3, dtype=np.int32)
        y = keras.utils.np_utils.to_categorical(y, num_classes=4)
        return x, y

    def process_x(self, x_b):
        x = pad(x_b)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x


def my_data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    # if this is the case, we won't shuffle our data enough
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)



# 
# class Preproc:
# 
#     def __init__(self, ecg, labels):
#         self.mean, self.std = compute_mean_std(ecg)
#         self.classes = sorted(set(l for label in labels for l in label))
#         self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
#         self.class_to_int = {c : i for i, c in self.int_to_class.items()}
# 
#     def process(self, x, y):
#         return self.process_x(x), self.process_y(y)
# 
#     def process_x(self, x):
#         x = pad(x)
#         x = (x - self.mean) / self.std
#         x = x[:, :, None]
#         return x
# 
#     def process_y(self, y):
#         # TODO, awni, fix hack pad with noise for cinc
#         y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32) 
#         y = keras.utils.np_utils.to_categorical(
#                 y, num_classes=len(self.classes))
#         return y
# 
# def pad(x, val=0, dtype=np.float32):
#     max_len = max(len(i) for i in x)
#     padded = np.full((len(x), max_len), val, dtype=dtype)
#     for e, i in enumerate(x):
#         padded[e, :len(i)] = i
#     return padded
# 
# def compute_mean_std(x):
#     x = np.hstack(x)
#     return (np.mean(x).astype(np.float32),
#            np.std(x).astype(np.float32))
# 
# def load_dataset(data_json):
#     with open(data_json, 'r') as fid:
#         data = [json.loads(l) for l in fid]
#     labels = []; ecgs = []
#     for d in tqdm.tqdm(data):
#         labels.append(d['labels'])
#         ecgs.append(load_ecg(d['ecg']))
#     return ecgs, labels
# 
# def load_ecg(record):
#     if os.path.splitext(record)[1] == ".npy":
#         ecg = np.load(record)
#     elif os.path.splitext(record)[1] == ".mat":
#         ecg = sio.loadmat(record)['val'].squeeze()
#     else: # Assumes binary 16 bit integers
#         with open(record, 'r') as fid:
#             ecg = np.fromfile(fid, dtype=np.int16)
# 
#     trunc_samp = STEP * int(len(ecg) / STEP)
#     return ecg[:trunc_samp]

if __name__ == "__main__":
    x_fname = "../examples/cinc17/data/X_train.csv"
    y_fname = "../examples/cinc17/data/y_train.csv"
    x_list, y = my_load_dataset(np.arange(1000), x_fname, y_fname)
    preprocessor = MyPreprocess(x_list, y)
    print(preprocessor.mean)
    print(preprocessor.std)
    gen = my_data_generator(2, preprocessor, x_list, y)
    for x, y in gen:
        print(x)
        print(y)
        print(x.shape, y.shape)
        break


    # data_json = "examples/cinc17/train.json"
    # train = load_dataset(data_json)
    # preproc = Preproc(*train)
    # gen = data_generator(32, preproc, *train)
    # for x, y in gen:
    #     print(x.shape, y.shape)
    #     break
