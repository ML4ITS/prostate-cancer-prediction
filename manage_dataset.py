import numpy as np
import pandas as pd

def from_days_to_months(data1, data2):
    data1["months"] = round(data1["days"] / 30)
    data2["months"] = round(data2["days"] / 30)
    return data1, data2

def upload_db(path1, path2, column, len = 4, model2 = False):
    """read the two dbs and select patients with a minimum number of values"""
    data1 = pd.read_csv(path1).sort_values(by = [column])
    data2 = pd.read_csv(path2).sort_values(by=[column])
    print(data1["ss_number_id"].nunique() == data2["ss_number_id"].nunique())
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

def balance_db(data1, data2, balanced):
    if balanced is not True:
        return data2
    n = data2["ss_number_id"].unique()
    id = np.random.choice(n, size = len(data1["ss_number_id"].unique()), replace = False)
    data2 = data2.loc[data2["ss_number_id"].isin((id))]
    return data2