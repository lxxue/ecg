import json
import keras
import numpy as np
import scipy.io as sio
import scipy.stats as sst

import sys
sys.path.append("../../ecg")

import myload as load
import network
import util

import pandas as pd
import numpy as np

import argparse

from collections import Counter

STEP = 256

def mypredict(args):
    # ecg = load.load_ecg(record +".mat")
    if args.data_fname[-3:] == "npy":
        x_test = np.load(args.data_fname)
    elif args.data_fname[-3:] == "csv":
        x_test = pd.read_csv(args.data_fname)
        x_test.drop(['id'], axis=1)
        x_test = x_test.fillna(0)
        x_test = x_test.values
        np.save(args.data_fname[:-4], x_test)

    print("data loading done")
    x_test_list = []
    for i in range(len(x_test)):
        x_i_end = x_test[i].nonzero()[0][-1] + 1 
        x_i_end = STEP * int((x_i_end) / STEP) 
        # print(x_i_end)
        x_test_list.append(x_test[i][:x_i_end])



    preproc = util.load(args.prep_fname)
    x_test = preproc.process_x(x_test_list)

    params = json.load(open(args.config_fname))
    params.update({
        "compile" : False,
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    model = network.build_network(**params)
    model.load_weights(args.model_fname)

    # print(x_test.shape)
    probs = model.predict(x_test)
    preds = np.argmax(probs, axis=2)
    results = -np.ones((len(x_test),), dtype=np.int64)
    for i in range(len(x_test)):
        # print(len(x_test_list[i])//STEP)
        # print(preds[i, :len(x_test_list[i])//STEP])
        # print(sst.mode(preds[i, :len(x_test_list[i])//STEP])[0][0])

        results[i] = sst.mode(preds[i, :len(x_test_list[i])//STEP])[0][0]
    # print(probs.shape)
    # prediction = sst.mode(np.argmax(probs, axis=2).squeeze())[0][0]
    print(Counter(results))
    
    sample = pd.read_csv(args.sample_fname)
    sample['y'] = results
    sample.to_csv(args.submit_fname, index=False)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_fname")
    parser.add_argument("prep_fname")
    parser.add_argument("model_fname")
    parser.add_argument("config_fname")
    parser.add_argument("sample_fname")
    parser.add_argument("submit_fname")
    args = parser.parse_args()
    print(args)
    mypredict(args)
