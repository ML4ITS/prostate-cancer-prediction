from model import training_test_part
from preprocessing import *
import sys









if __name__ == "__main__":
    _, path1, path2, DUMMIES, DELTA_FEATURES = sys.argv
    data1, data2 = upload_db(path1, path2, len=4)
    data1, data2 = remove_outliers(data1), remove_outliers(data2)
    if DUMMIES:
        data1, data2 = dummies(data1), dummies(data2)
    if DELTA_FEATURES:
        data1, data2 = delta_features(data1), delta_features(data2)
    if DUMMIES:
        data1, data2 = delete_columns(data1), delete_columns(data2)
    data2 = balance_db(data1, data2)
    data1, data2 = assign_target(data1, data2)
    data = concat_data(data1, data2)
    TIMESTEPS = extract_timesteps(data, DELTA_FEATURES)
    data.to_csv("db/data", index = False)
    training_test_part()