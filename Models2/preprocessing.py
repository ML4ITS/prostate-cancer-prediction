import pandas as pd
import numpy as np


def upload_db(path1, path2, len = 4):
    print(path1)
    data1 = pd.read_csv(path1).sort_values(by = ["age"])
    data2 = pd.read_csv(path2).sort_values(by=["age"])
    v = data1.ss_number_id.value_counts()
    data1 = data1[data1.ss_number_id.isin(v.index[v.gt(len)])]
    v = data2.ss_number_id.value_counts()
    data2 = data2[data2.ss_number_id.isin(v.index[v.gt(len)])]
    print("Number of patients with cancer: "+ str(data1["ss_number_id"].nunique()))
    print("Number of patients without cancer: "+ str(data2["ss_number_id"].nunique()))
    return data1, data2

def dummies(data):
    bins1 = [0, 30, 40, 50, 60, 70, 80, 120]
    bins2 = [0, 4, 10, 2000]
    labels = [0, 1 , 2, 3, 4, 5, 6]
    data["AgeGroup"] = pd.cut(data["age"], bins = bins1, labels = labels, right = False)
    data["PsaGroup"] = pd.cut(data["psa"], bins=bins2, labels=[0, 1, 2], right=False)
    data = pd.get_dummies(data, columns = ["AgeGroup","PsaGroup"])
    return data

def delta_features(data):
    data["delta_time"] = data.sort_values(by = ["months"]).groupby(["ss_number_id"])["months"].apply(lambda x: x - x.shift()).fillna(np.nan)
    data["delta_psa"] = data.sort_values(by=["months"]).groupby(["ss_number_id"])["psa"].apply(lambda x: x - x.shift()).fillna(np.nan)
    data.dropna(inplace = True)
    return data

def delete_columns(data):
    del data["age"]
    del data["psa"]
    return data

def balance_db(data1, data2):
    n = data2["ss_number_id"].unique()
    id = np.random.choice(n, size = len(data1["ss_number_id"].unique()), replace = False)
    data2 = data2.loc[data2["ss_number_id"].isin((id))]
    return data2

def assign_target(data1, data2):
    data1["risk"] = 1
    data2["risk"] = 0
    return data1, data2

def remove_outliers(data):
    """remove patients with age>=100"""
    return data.loc[data['age'] < 100]

def concat_data(data1, data2):
    data = pd.concat([data1, data2])
    data = data.sort_values(by = ["months"])
    del data["ambiguous_date"]
    del data["date_of_birth_15"]
    del data["months"]
    print("Number of patients: " + str(data["ss_number_id"].nunique()))
    return data

def extract_timesteps(data, DELTA_FEATURES, DUMMIES):
    if DELTA_FEATURES is True and DUMMIES is True :
        feature = "delta_time"
    elif DUMMIES is True and DELTA_FEATURES is False:
        feature = "delta_time"
    else:
        feature = "age"
    return data.groupby(["ss_number_id"])[feature].count().max()