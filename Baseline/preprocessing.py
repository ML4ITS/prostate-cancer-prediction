import numpy as np
import pandas as pd
from manage_dataset import  *


def extract_last_value_wrapp(data1, data2, val1, val2):
    #first baseline
    #third baseline
    def extract_last_value(data, val1, val2):
        data = data.sort_values(by=["days"]).groupby("ss_number_id").tail(val1)
        return data.groupby("ss_number_id").head(val2)
    data1, data2 = extract_last_value(data1, val1, val2), extract_last_value(data2, val1, val2)
    return data1, data2

def extract_delta_wrapp(data1, data2):
    #second baseline
    data1, data2 = from_days_to_months(data1, data2)
    def extract_delta_value(data):
        data["delta"] = data.sort_values(by=["months"]).groupby(["ss_number_id"])["months"].apply(lambda x: x - x.shift()).fillna(np.nan)
        data.dropna(inplace=True)
        data.reset_index(drop = True, inplace = True)
        return data
    data1, data2 = extract_delta_value(data1), extract_delta_value(data2)
    return data1, data2

def extract_statistics_wrapp(data1, data2):
    #second baseline
    def extract_statistics(data):
        def q1(x): return x.quantile(0.25)
        def q3(x): return x.quantile(0.75)
        f = ["mean", "median", "std", q1, q3]
        data = data.groupby(["ss_number_id"])["delta"].agg(f)
        data["ss_number_id"] = data.index
        data.reset_index(drop = True, inplace = True)
        return data
    data1, data2 = extract_statistics(data1), extract_statistics(data2)
    return data1, data2

def extract_velocity_wrapp(data1, data2):
    #third baseline
    def extract_velocity(data):
        data["delta_x"] = data.sort_values(by=["days"]).groupby(["ss_number_id"])["psa"].apply(
            lambda x: x - x.shift()).fillna(np.nan)
        data["delta_y"] = data.sort_values(by=["days"]).groupby(["ss_number_id"])["days"].apply(
            lambda x: x - x.shift()).fillna(np.nan)
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        data["velocity"] = data["delta_x"] * 30 / data["delta_y"]
        return data
    data1, data2 = extract_velocity(data1), extract_velocity(data2)
    return data1, data2

def clean_df_wrapp(data1, data2, baseline):
    def clean_df(data, baseline):
        if baseline == 3:
            del data["delta_x"]
            del data["delta_y"]
        del data["days"]
        del data["ambiguous_date"]
        del data["date_of_birth_15"]
        return data
    data1 = clean_df(data1, baseline)
    data2 = clean_df(data2, baseline)
    return data1, data2

def concat_data(data1, data2):
    """union of the two dbs"""
    data = pd.concat([data1, data2])
    data.replace([np.inf, -np.inf], np.nan, inplace = True)
    data.dropna(inplace = True)
    data.reset_index(drop = True, inplace = True)
    print("Number of patients: " + str(data["ss_number_id"].nunique()))
    print(data.head())
    print("Features: " + str(data.columns))
    data.reset_index(drop = True, inplace = True)
    return data

def remove_id(data1, data2):
    del data1["ss_number_id"]
    del data2["ss_number_id"]
    return data1, data2