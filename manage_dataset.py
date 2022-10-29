import numpy as np
import pandas as pd

def drop_duplicates(data):
    size = data.shape[0]
    data["delta_days"] = data.sort_values(by=["days"]).groupby(["ss_number_id"])["days"].apply(lambda x: x - x.shift()).fillna(100)
    #the visits too closed have been removed (too closed means 8 days)
    data = data.loc[data["delta_days"] > 8]
    print("Number of values too closed removed: " + str(size - data.shape[0]))
    del data["delta_days"]
    return data

def from_days_to_months(data1, data2):
    data1["months"] = round(data1["days"] / 30)
    data2["months"] = round(data2["days"] / 30)
    return data1, data2

def add_feature_age(data1, data2):
    data1["age"] = round(data1["days"] / (30 * 12))
    data2["age"] = round(data2["days"] / (30 * 12))
    return data1, data2

def from_categoric_to_numeric(data):
    data['category'].replace(['Low risk', 'High risk','Intermediate risk', 'Regional', 'Metastatic', 'Missing'],
                            [0, 1, 2, 3, 4, 5], inplace=True)
    return data

def upload_db(path1, path2, column, len = 4, model2 = False):
    """read the two dbs and select patients with a minimum number of values"""
    data1 = pd.read_csv(path1).sort_values(by = [column])
    data2 = pd.read_csv(path2).sort_values(by=[column])
    data1, data2 = drop_duplicates(data1), drop_duplicates(data2)
    data1, data2 = add_feature_age(data1, data2)
    if model2:
        data1, data2 = from_days_to_months(data1, data2)
        data1 = from_categoric_to_numeric(data1)
    else:
        del data1["category"]
    v = data1.ss_number_id.value_counts()
    data1 = data1[data1.ss_number_id.isin(v.index[v.gt(len)])]
    v1 = data2.ss_number_id.value_counts()
    data2 = data2[data2.ss_number_id.isin(v1.index[v1.gt(len)])]
    print("Number of patients with cancer: "+ str(data1["ss_number_id"].nunique()))
    print("Number of patients without cancer: "+ str(data2["ss_number_id"].nunique()))
    return data1, data2

def assign_target(data1, data2):
    data1["risk"] = 1 #cancer
    data2["risk"] = 0 #no cancer
    return data1, data2

def remove_outliers(data):
    #outliers means patients with age < 30 and age >=100
    d1 = data.shape[0]
    data = data.loc[(data['age'] >=30) & (data['age'] < 100)]
    d2 = data.shape[0]
    print("Outliers removed: " + str(d1 - d2))
    return data

def balance_db(data1, data2, balanced, id = None):
    if balanced is not True:
        return data2
    if id != None:
        return data2.sample(n=data1.shape[0], random_state = 42)
    n = data2["ss_number_id"].unique()
    np.random.seed(0)
    id = np.random.choice(n, size = len(data1["ss_number_id"].unique()), replace = False)
    data2 = data2.loc[data2["ss_number_id"].isin((id))]
    return data2

def extract_unbalanced_test(data, data1, model2 = None):
    #the test set is unbalnced so the samples have been randomly extracted from the dataset
    if model2 is not None:
        n = data["ss_number_id"].unique()
        dim_test = round(len(data1["ss_number_id"].unique()) * 2 * 0.2)
        np.random.seed(2020)
        id = np.random.choice(n, size= dim_test, replace=False)
        test = data.loc[data["ss_number_id"].isin((id))]
        data = data.loc[~data["ss_number_id"].isin((id))]
        print("Shape data: " + str(len(data["ss_number_id"].unique())))
        print("Shape test: " + str(len(test["ss_number_id"].unique())))
    else:
        # extract x_test that is 0.2 * data1.shape*2
        test = data.sample(n=round(data1.shape[0] * 2 * 0.2), random_state=42)
        # delete from data, samples that are in test
        data = data[~data.apply(tuple, 1).isin(test.apply(tuple, 1))]
    return data, test

def extract_balanced_train(data, balanced, id = None):
    #the training set is balanced so patients with cancer and no cancer have been extracted with a proportion 50%-50%
    data1 = data[data.iloc[:, -1] == 1]
    data2 = data[data.iloc[:, -1] == 0]
    data2 = balance_db(data1, data2, balanced, id)
    if id == None:
        df = pd.concat([data1, data2]).sort_values(by = "days")
        del df["days"]
    else:
        df = pd.concat([data1, data2])
    return df