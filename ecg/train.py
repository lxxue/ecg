from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import json
import keras
import numpy as np
import os
import os.path as osp
import random
import time
import pandas as pd

import network
# import load
import myload as load
import util

MAX_EPOCHS = 100

def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5")

def train(args, params):

    x_fname = "examples/cinc17/data/X_train.csv"
    y_fname = "examples/cinc17/data/y_train.csv"
    x_npy_fname = "examples/cinc17/data/x_train.npy"
    y_npy_fname = "examples/cinc17/data/y_train.npy"

    STEP = 256
    print("Loading dataset...")
    if osp.isfile(x_npy_fname):
        x = np.load(x_npy_fname)
    else:
        x = pd.read_csv(x_fname)
        x = x.drop(['id'], axis=1)
        x = x.fillna(0)
        x = x.values
        np.save(x_npy_fname, x)
    
    x_list = []
    for i in range(len(x)):
        x_i_end = x[i].nonzero()[0][-1] + 1
        x_i_end = STEP * int((x_i_end) / STEP) 
        # print(x_i_end)
        x_list.append(x[i][:x_i_end])


    if osp.isfile(y_npy_fname):
        y = np.load(y_npy_fname)
    else:
        y = pd.read_csv(y_fname)
        y = y.drop(['id'], axis=1)
        y = y.values.squeeze()
        np.save(y_npy_fname, y)
    print("done")

    train_idx = np.load(params['train_idx_fname'])
    dev_idx = np.load(params['dev_idx_fname'])
    
    train = load.my_load_dataset(train_idx, x_list, y)
    dev = load.my_load_dataset(dev_idx, x_list, y)
    print("Building preprocessor...")
    preproc = load.MyPreprocess(*train)
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")


    save_dir = make_save_dir(params['save_dir'], args.experiment)

    util.save(preproc, save_dir)

    params.update({
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    model = network.build_network(**params)

    stopping = keras.callbacks.EarlyStopping(patience=8)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params["learning_rate"] * 0.001)

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False)

    batch_size = params.get("batch_size", 32)

    if params.get("generator", False):
        train_gen = load.my_data_generator(batch_size, preproc, *train)
        dev_gen = load.my_data_generator(batch_size, preproc, *dev)
        model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(train[0]) / batch_size),
            epochs=MAX_EPOCHS,
            validation_data=dev_gen,
            validation_steps=int(len(dev[0]) / batch_size),
            callbacks=[checkpointer, reduce_lr, stopping])
    else:
        train_x, train_y = preproc.process(*train)
        dev_x, dev_y = preproc.process(*dev)
        model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=[checkpointer, reduce_lr, stopping])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)
