import sys
sys.path.append('../')
from utils import *
from methods.LSTM import training_test_LSTM
import time
from preprocessing import *
from config_file.config_parser import Data
from manage_dataset import *
from settings import p
from methods.sklearn_classifiers import *

if __name__ == "__main__":
    start = time.time()
    #parameters from command line
    _, path1, path2 = sys.argv
    m = Data() #get parameters
    m.extractData()
    p.regularization = False #set regularization parameter
    #baseline 1 val1 = 2 val2 = 1
    #baseline 3 val1 = 3 val2 = 2
    for baseline in range(1, 3):
        data1, data2 = upload_db(path1, path2, "days", len=4)
        # data1, data2 = remove_outliers(data1), remove_outliers(data2)
        data2 = balance_db(data1, data2, m.balanced)
        if baseline == 1:
            data1, data2 = extract_last_value_wrapp(data1, data2, val1=2, val2=1)
        if baseline == 3:
            data1, data2 = extract_last_value_wrapp(data1, data2, val1 = 3, val2 = 2) #first / third
        if baseline == 2:
            data1, data2 = extract_delta_wrapp(data1, data2) #second
            data1, data2 = extract_statistics_wrapp(data1, data2) #second
        if baseline == 3:
            data1, data2 = extract_velocity_wrapp(data1, data2) #third
        if baseline != 2:
            data1, data2 = clean_df_wrapp(data1, data2, baseline)
        data1, data2 = assign_target(data1, data2)
        case = "results/case" + str(baseline)
        plot_isomap(data1, data2, case)
        data = concat_data(data1, data2)
        print(data.head())
        heatmap(data, case)
        classifiers(data, case, m.repetition)

