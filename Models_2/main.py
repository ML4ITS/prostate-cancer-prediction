from datetime import time

import torch
from caffe2.contrib import gloo

from LSTM import training_test_LSTM
from MLP import training_test_MLP
from CNN1D import training_test_CNN1D
from config_file.config_parser import Data
from utils import str2bool, get_std_mean_accuracy, flush_file, multiple_boxplot
from preprocessing import *
import sys
import time
import torch.nn as nn




if __name__ == "__main__":
    start = time.time()
    #parameters from command line
    _, path1, path2, i = sys.argv
    # dummies, delta = str2bool(dummies), str2bool(delta)
    i = int(i)
    p = Data() #get parameters
    p.extractData()

    data1, data2 = upload_db(path1, path2, len=4)
    # data1, data2 = remove_outliers(data1), remove_outliers(data2)

    #case 0: no dummies no delta
    #case 1: no dummies y delta
    #case 2: y dummies no delta
    #case 3: y dummies y delta

    if i == 2 or i == 3:
        data1, data2 = get_dummies_data(data1), get_dummies_data(data2)
    if i == 1 or i == 3:
        data1, data2 = delta_features(data1), delta_features(data2)
    if i == 2 or i == 3:
        data1, data2 = delete_columns(data1), delete_columns(data2)

    data2 = balance_db(data1, data2, p.balanced)

    data1, data2 = assign_target(data1, data2)

    data = concat_data(data1, data2)

    data.to_csv("db/data.csv", index = False)

#models
    torch.set_num_threads(5)
    torch.set_num_interop_threads(5)
    accuracy = []
    case = "case" + str(i+1)
    #CNN1D
    flush_file("cnn1d", case)
    acc = []
    for i in range(p.repetition):
        results = training_test_CNN1D(p.epochs, p.trials, case)
        acc.append(list(results[0].values())[1])
    accuracy.append(acc)
    get_std_mean_accuracy("cnn1d", acc, case)
    # type MLP
    flush_file("mlp", case)
    acc = []
    for i in range(p.repetition):
        results = training_test_MLP(p.epochs, p.trials, case)
        acc.append(list(results[0].values())[1])
    accuracy.append(acc)
    get_std_mean_accuracy("mlp", acc, case)
    #type LSTM
    flush_file("lstm", case)
    acc = []
    for i in range(p.repetition):
        results = training_test_LSTM(p.epochs, p.trials, case)
        acc.append(list(results[0].values())[1])
    accuracy.append(acc)
    get_std_mean_accuracy("lstm", acc, case)
    #box plot for the 3 models
    multiple_boxplot(accuracy, case)
    print("Time taken: ", time.time() - start)




