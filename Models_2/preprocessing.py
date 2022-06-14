import pandas as pd
import numpy as np



def get_dummies_data(data):
    """dummies for features age and psa"""
    bins1 = [0, 30, 40, 50, 60, 70, 80, 120] #for age
    bins2 = [0, 4, 10, 2000] #for psa
    labels = [0, 1 , 2, 3, 4, 5, 6]
    data["AgeGroup"] = pd.cut(data["age"], bins = bins1, labels = labels, right = False)
    data["PsaGroup"] = pd.cut(data["psa"], bins=bins2, labels=[0, 1, 2], right=False)
    data = pd.get_dummies(data, columns = ["AgeGroup","PsaGroup"])
    return data

def delta_features(data):
    """delta time and delta psa """
    data["delta_time"] = data.sort_values(by = ["months"]).groupby(["ss_number_id"])["months"].apply(lambda x: x - x.shift()).fillna(np.nan)
    data["delta_psa"] = data.sort_values(by=["months"]).groupby(["ss_number_id"])["psa"].apply(lambda x: x - x.shift()).fillna(np.nan)
    data.dropna(inplace = True)
    return data

def delete_columns(data):
    """useless columns are removed"""
    del data["age"]
    del data["psa"]
    return data

def remove_outliers(data):
    """remove patients with age>=100"""
    return data.loc[data['age'] < 100]

def concat_data(data1, data2):
    """union of the two dbs"""
    data = pd.concat([data1, data2])
    data = data.sort_values(by = ["months"])
    del data["ambiguous_date"]
    del data["date_of_birth_15"]
    del data["months"]
    print("Number of patients: " + str(data["ss_number_id"].nunique()))
    return data
