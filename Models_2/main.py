from LSTM import training_test_LSTM
from MLP import training_test_MLP
from CNN1D import training_test_CNN1D
from preprocessing import *
import sys




if __name__ == "__main__":
    # _, path1, path2, dummies, delta, type = sys.argv
    dummies = False
    delta = False
    type = "CNN1D"
    data1, data2 = upload_db("db/cancer.csv", "db/nocancer.csv", len=4)
    # data1, data2 = remove_outliers(data1), remove_outliers(data2)

    if dummies is True:
        data1, data2 = get_dummies_data(data1), get_dummies_data(data2)
    if delta is True:
        data1, data2 = delta_features(data1), delta_features(data2)
    if dummies is True:
        data1, data2 = delete_columns(data1), delete_columns(data2)

    # data2 = balance_db(data1, data2)

    data1, data2 = assign_target(data1, data2)

    data = concat_data(data1, data2)

    data.to_csv("db/data.csv", index = False)

    #models
    if type == "LSTM":
        training_test_LSTM()
    elif type == "MLP":
        training_test_MLP()
    else: #CNN1D
        training_test_CNN1D()