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

def concat_data(data1, data2):
    """union of the two dbs"""
    data = pd.concat([data1, data2])
    data = data.sort_values(by = ["months"])
    del data["ambiguous_date"]
    del data["date_of_birth_15"]
    del data["months"]
    print("Number of patients: " + str(data["ss_number_id"].nunique()))
    return data

def extract_info_wrapp(data1, data2):
    def extract_info(data, cancer = True):
        if cancer == False:
            data["category"] = -1
        df = data[['ss_number_id', 'category']].drop_duplicates()
        visit = data.groupby("ss_number_id")["psa"].count().to_frame().rename(columns={'psa': 'visits'})
        f = ["mean", "min", "max"]
        mod_age = data.groupby("ss_number_id")["age"].agg(f).rename(columns={"mean": "mean_age", "min": "min_age", "max": "max_age"})
        df = pd.merge(df, visit, how='inner', on='ss_number_id')
        df = pd.merge(df, mod_age, how='inner', on='ss_number_id')
        return df
    df1 = extract_info(data1)
    df2 = extract_info(data2, cancer = False)
    # remove category
    del data1["category"]
    del data2["category"]
    return pd.concat([df1, df2])

def save_info(df, test):
    df = df[df['ss_number_id'].isin(test['ss_number_id'])].reset_index(drop=True)
    df.sort_values(by = "ss_number_id", inplace = True)
    test.sort_values(by=["ss_number_id", "days"], inplace=True)
    del test["days"]
    df.to_csv("df.csv", index=False)
