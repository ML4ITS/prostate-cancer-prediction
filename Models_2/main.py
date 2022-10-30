import sys
sys.path.append('../')
import torch
from config_file.config_parser import Data
from manage_dataset import *
from methods.CNN1D import training_test_CNN1D
from methods.LSTM import training_test_LSTM
from methods.MLP import training_test_MLP
from methods.rocket import rocket_algorithm
from methods.inception_time import *
from utils import *
from preprocessing import *
from settings import p
import time


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU AVAILABLE")
    else:
        print("USING CPU")
    #parameters from command line
    _, path1, path2 = sys.argv
    m = Data() #get parameters
    m.extractData()
    p.regularization = p.indicator = p.interpolation = False
    if p.regularization or p.indicator or p.interpolation:
        exit("wrong parameters")
    #loop over each type of input
    # case 0: no dummies no delta
    # case 1: no dummies y delta
    # case 2: y dummies no delta
    # case 3: y dummies y delta
    for i in range(4):
        data1, data2 = upload_db(path1, path2, "days", len=4, model2 = True)
        data1, data2 = remove_outliers(data1), remove_outliers(data2)
        df = extract_info_wrapp(data1, data2)
        if i == 2 or i == 3:
            data1, data2 = get_dummies_data(data1), get_dummies_data(data2)
        if i == 1 or i == 3:
            data1, data2 = delta_features(data1), delta_features(data2)
        if i == 2 or i == 3:
            data1, data2 = delete_columns(data1), delete_columns(data2)
        data1, data2 = assign_target(data1, data2)
        data = concat_data(data1, data2)
        data.to_csv("../complete_dataset.csv", index=False)
        data, test = extract_unbalanced_test(data, data1, model2 = True)
        save_info(df, test)
        train = extract_balanced_train(data, m.balanced)
        del data["days"]
        data.reset_index(inplace=True, drop=True)
        data.to_csv("../data.csv", index=False)
        print(test.tail(20))
        print("Shape train: " + str(len(train["ss_number_id"].unique())))
        print("Shape test: " + str(len(test["ss_number_id"].unique())))
        train.to_csv("../train.csv", index=False)
        test.to_csv("../test.csv", index=False)
        case = "case" + str(i+1)
        #type ROCKET
        rocket_algorithm(case,  m.repetition)
        accuracy = []
        # type CNN1D
        flush_file("cnn1d", case)
        acc = training_test_CNN1D(m.epochs, m.trials, case, m.repetition)
        accuracy.append(acc)
        get_std_mean_accuracy("cnn1d", acc, case)
        # type MLP
        flush_file("mlp", case)
        acc = training_test_MLP(m.epochs, m.trials, case, m.repetition)
        accuracy.append(acc)
        get_std_mean_accuracy("mlp", acc, case)
        #type LSTM
        flush_file("lstm", case)
        acc = training_test_LSTM(m.epochs, m.trials, case, m.repetition, model2 = True)
        accuracy.append(acc)
        get_std_mean_accuracy("lstm", acc, case)
        #type convolutional heads
        p.multiple_kernels = True
        flush_file("cnn1d_heads", case)
        acc = training_test_CNN1D(m.epochs, m.trials, case, m.repetition, m_kernels = p.multiple_kernels)
        accuracy.append(acc)
        get_std_mean_accuracy("cnn1d_heads", acc, case)
        #box plot for the 3 models
        multiple_boxplot_models(accuracy, case)
    print("Time taken: ", time.time() - start)




