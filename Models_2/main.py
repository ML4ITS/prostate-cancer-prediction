from LSTM import training_test_LSTM
from MLP import training_test_MLP
from CNN1D import training_test_CNN1D
from config_file.config_parser import Data
from utils import str2bool, get_std_mean_accuracy, flush_file
from preprocessing import *
import sys




if __name__ == "__main__":
    #parameter from command line
    _, path1, path2  = sys.argv
    # dummies, delta = str2bool(dummies), str2bool(delta)

    p = Data() #get parameters
    p.extractData()

    # dummies = False
    # delta = False
    # type = "CNN1D"
    data1, data2 = upload_db(path1, path2, len=4)
    # data1, data2 = remove_outliers(data1), remove_outliers(data2)
    #case 0: no dummies no delta
    #case 1: no dummies y delta
    #case 2: y dummies no delta
    #case 3: y dummies y delta
    for i in range(3,4): #4 combinations of input
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

        case = "case" + str(i+1)
        #CNN1D
        flush_file("cnn1d", case)
        accuracy = []
        for i in range(p.repetition):
            results = training_test_CNN1D(p.epochs, p.trials, case)
            accuracy.append(list(results[0].values())[1])
        get_std_mean_accuracy("cnn1d", accuracy, case)
        #-----------#
        # type MLP
        flush_file("mlp", case)
        accuracy = []
        for i in range(p.repetition):
            results = training_test_MLP(p.epochs, p.trials, case)
            accuracy.append(list(results[0].values())[1])
        get_std_mean_accuracy("mlp", accuracy, case)
        #type LSTM
        flush_file("lstm", case)
        accuracy = []
        for i in range(p.repetition):
            results = training_test_LSTM(p.epochs, p.trials, case)
            accuracy.append(list(results[0].values())[1])
        get_std_mean_accuracy("lstm", accuracy, case)

        #-----------#


