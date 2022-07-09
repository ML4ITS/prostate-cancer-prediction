import sys
sys.path.append('../')
from utils import *
from preprocessing import *
from config_file.config_parser import Data
from manage_dataset import *
from settings import p
from methods.sklearn_classifiers import *

if __name__ == "__main__":
    #parameters from command line
    _, path1, path2 = sys.argv
    m = Data() #get parameters
    m.extractData()
    p.regularization = False #set regularization parameter
    #baseline 1 val1 = 2 val2 = 1
    #baseline 3 val1 = 3 val2 = 2
    for baseline in range(1, 4):
        data1, data2 = upload_db(path1, path2, "days", len=4)
        data1, data2 = remove_outliers(data1), remove_outliers(data2)
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
        heatmap(data, case)
        data, test = extract_unbalanced_test(data, data1)
        train = extract_balanced_train(data, m.balanced, id = 1)
        train, test = remove_id(train, test)
        print("Shape train: " + str(train.shape))
        print("Shape test: " + str(test.shape))
        # same db every time
        classifiers(train, test, case, m.repetition)
        print("---------Next baseline-----------")

