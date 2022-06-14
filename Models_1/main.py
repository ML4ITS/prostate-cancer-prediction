import sys
sys.path.append('../')
from utils import flush_file, get_std_mean_accuracy
from methods.LSTM import training_test_LSTM
import time
from preprocessing import *
from config_file.config_parser import Data
from manage_dataset import *

if __name__ == "__main__":
    start = time.time()
    #parameters from command line
    # _, path1, path2 = sys.argv
    # dummies, delta = str2bool(dummies), str2bool(delta)

    p = Data() #get parameters
    p.extractData()
    if p.regularization is False:
        exit("wrong parameters")
#capire come gestire days
    data1, data2 = upload_db("../dataset/cancer.csv", "../dataset/nocancer.csv", "days", len=4)
    data1, data2 = change_format(data1, data2)
    min, max = define_global_min_max(data1, data2)
    data1, data2 = set_min_max(data1, data2, min, max)
    data1, data2 = prepare_df_wrapp(data1, data2)
    data1, data2 = interpolation_data(data1, data2, p.interpolation, frequency = "48h")
    data1, data2 = assign_target(data1, data2)
    data2 = balance_db(data1, data2, p.balanced)
    data =concat_data(data1, data2)
    data.to_csv("../data_model.csv", index=False, header = False)
    #type LSTM
    flush_file("lstm", "case0")
    acc = []
    #4 case: interpolation, no interpolation and the advanced cases

    for i in range(p.repetition):
        results = training_test_LSTM(p.epochs, p.trials, "case0")
        acc.append(list(results[0].values())[1])





