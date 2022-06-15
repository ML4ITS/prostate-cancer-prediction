import sys
sys.path.append('../')
from utils import flush_file, get_std_mean_accuracy, combinations
from methods.LSTM import training_test_LSTM
import time
from preprocessing import *
from config_file.config_parser import Data
from manage_dataset import *
from settings import p

if __name__ == "__main__":
    start = time.time()
    #parameters from command line
    # _, path1, path2 = sys.argv
    # dummies, delta = str2bool(dummies), str2bool(delta)

    m = Data() #get parameters
    m.extractData()
    p.regularization = True #set regularization parameter
    # 4 case: interpolation, no interpolation and the advanced cases
    for i in range(4):
        p.interpolation, p.indicators = combinations(i)
        data1, data2 = upload_db("../dataset/cancer.csv", "../dataset/nocancer.csv", "days", len=4)
        data2 = balance_db(data1, data2, m.balanced)
        data1, data2 = change_format(data1, data2)
        min, max = define_global_min_max(data1, data2)
        data1, data2 = set_min_max(data1, data2, min, max)
        data1, data2 = prepare_df_wrapp(data1, data2)
        data1, data2 = interpolation_data(data1, data2, p.interpolation, frequency = "2y")
        data1, data2 = assign_target(data1, data2)
        data =concat_data(data1, data2)
        data.to_csv("../data_model.csv", index=False, header = False)
        #type LSTM
        case = "case" + str(i+1)
        flush_file("lstm", case)
        acc = []

        for i in range(m.repetition):
            results = training_test_LSTM(m.epochs, m.trials, case)
            acc.append(list(results[0].values())[1])





