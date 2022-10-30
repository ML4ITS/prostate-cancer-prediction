import sys
sys.path.append('../')
from utils import *
from methods.LSTM import training_test_LSTM
from preprocessing import *
from config_file.config_parser import Data
from manage_dataset import *
from settings import p
import torch


if __name__ == "__main__":
    #parameters from command line
    _, path1, path2 = sys.argv
    m = Data() #get parameters
    m.extractData()
    p.regularization = True #set regularization parameter
    # 4 case: interpolation, no interpolation and the advanced cases
    if torch.cuda.is_available():
        print("GPU AVAILABLE")
    else:
        print("USING CPU")
    #frequency windows
    windows = ["6m", "1y", "2y", "8y"]
    for k in range(4):
        accuracy = []
        window = windows[k]
        for i in range(4):
            if p.regularization is False:
                exit("wrong parameters!!!!")
            p.interpolation, p.indicator = combinations(i)
            data1, data2 = upload_db(path1, path2, "days", len=4)
            data1, data2 = remove_outliers(data1), remove_outliers(data2)
            data1, data2 = change_format(data1, data2)
            min, max = define_global_min_max(data1, data2)
            data1, data2 = set_min_max(data1, data2, min, max)
            data1, data2 = prepare_df_wrapp(data1, data2)
            data1, data2 = interpolation_data(data1, data2, p.indicator, p.interpolation, frequency = window)
            data1, data2 = assign_target(data1, data2)
            data = concat_data(data1, data2)
            data.reset_index(inplace=True, drop=True)
            data.to_csv("../complete_dataset.csv", index=False)
            print(data.head())
            data, test = extract_unbalanced_test(data, data1)
            train = extract_balanced_train(data, m.balanced, id = 1)
            print("Shape train: " + str(train.shape))
            print("Shape test: " + str(test.shape))
            train.to_csv("../train.csv", index=False)
            test.to_csv("../test.csv", index=False)
            #type LSTM
            case = "windows" + str(k+1) + "/case" + str(i+1)
            flush_file("lstm", case)
            acc = training_test_LSTM(m.epochs, m.trials, case, m.repetition)
            accuracy.append(acc)
        multiple_boxplot_inputs(accuracy, path = "lstm/window_size_" + str(k+1))




