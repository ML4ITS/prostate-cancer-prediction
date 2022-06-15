import numpy as np
import pandas as pd

def from_days_to_months(data1, data2):
    data1["months"] = round(data1["days"] / 30)
    data2["months"] = round(data2["days"] / 30)
    del data1["days"]
    del data2["days"]
    return data1, data2

def add_features(data1, data2):
    data1["age"] = round(data1["days"] / (30 * 12))
    data2["age"] = round(data2["days"] / (30 * 12))
    return data1, data2

def upload_db(path1, path2, column, len = 4, model2 = False):
    """read the two dbs and select patients with a minimum number of values"""
    data1 = pd.read_csv(path1).sort_values(by = [column])
    data2 = pd.read_csv(path2).sort_values(by=[column])
    data1, data2 = add_features(data1, data2)
    if model2:
        data1, data2 = from_days_to_months(data1, data2)
    v = data1.ss_number_id.value_counts()
    data1 = data1[data1.ss_number_id.isin(v.index[v.gt(len)])]
    v = data2.ss_number_id.value_counts()
    data2 = data2[data2.ss_number_id.isin(v.index[v.gt(len)])]
    print("Number of patients with cancer: "+ str(data1["ss_number_id"].nunique()))
    print("Number of patients without cancer: "+ str(data2["ss_number_id"].nunique()))
    return data1, data2

def assign_target(data1, data2):
    data1["risk"] = 1 #cancer
    data2["risk"] = 0 #no cancer
    return data1, data2

def remove_outliers(data, regularization = False):
    # """remove patients with age>=100"""
    # if regularization:
    #     return data.loc[data['days'] < (100 * 30 * 12)]
    # else:
    d1 = data.shape[0]
    data = data.loc[data['age'] < 100]
    d2 = data.shape[0]
    print("Outliers removed: " + str(d1 - d2))
    return data

def balance_db(data1, data2, balanced):
    if balanced is not True:
        return data2
    n = data2["ss_number_id"].unique()
    id = np.random.choice(n, size = len(data1["ss_number_id"].unique()), replace = False)
    data2 = data2.loc[data2["ss_number_id"].isin((id))]
    return data2