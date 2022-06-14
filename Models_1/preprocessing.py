import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

def change_format(data1, data2):
    data1["date"] = pd.to_datetime(data1["days"], unit = "d")
    data2["date"] = pd.to_datetime(data2["days"], unit = "d")
    return data1, data2

def define_global_min_max(data1, data2):
    def add_months(start_date, delta_period):
        return start_date + relativedelta(months = delta_period)
    max1 = add_months(data1["date"].max(),1).to_pydatetime()
    min1 = add_months(data1["date"].min(),-1).to_pydatetime()
    max2 = add_months(data2["date"].max(),1).to_pydatetime()
    min2 = add_months(data2["date"].min(),-1).to_pydatetime()
    max = max1 if max1 > max2 else max2
    min = min1 if min1 < min2 else min2
    return min, max

def set_min_max(data1, data2, min, max):
    def min_max(data, min, max):
        data = data.sort_values(["ss_number_id", "date"])
        df = data.drop_duplicates("ss_number_id", keep = "last").copy()
        df1 = data.drop_duplicates("ss_number_id", keep = "last").copy()
        df["date"] = min
        df["psa"] = np.nan
        df1["date"] = max
        df1["psa"] = np.nan
        d = pd.concat([data, df, df1]).sort_index(kind = "merge_sort")
        return d
    data1 = min_max(data1, min, max)
    data2 = min_max(data2, min, max)
    return data1, data2

def prepare_df_wrapp(data1, data2):
    def prepare_df(data):
        data.sort_values(by = ["date"], inplace = True)
        data.set_index("date", inplace = True)
        del data["days"]
        del data["age"]
        del data["ambiguous_date"]
        del data["date_of_birth_15"]
        return data
    data1 = prepare_df(data1)
    data2 = prepare_df(data2)
    return data1, data2

def get_interp(data, frequency):
    ris = data["psa"].resample(frequency).mean().fillna(-1)
    nan = (np.array(ris) != -1).astype(int)
    ris[ris == -1] = np.nan
    inter = ris.interpolate(method = "linear", limit_direction = "both")
    nan[nan == 0] = -1
    return inter * nan

def interpolation_data(data1, data2, interp = False, frequency = "6M"):
    def interpolation(data, interp):
        if interp:
            return data.groupby("ss_number_id").apply(lambda x: get_interp(x, frequency))
        else:
            return data.groupby("ss_number_id")["psa"].resample(frequency).mean().fillna(-1).unstack("date")
    return interpolation(data1, interp), interpolation(data2, interp)

def concat_data(data1, data2):
    """union of the two dbs"""
    data = pd.concat([data1, data2])
    data["ss_number_id"] = data.index
    print("Number of patients: " + str(data["ss_number_id"].nunique()))
    print("Number of patients: " + str(data["ss_number_id"].count()))
    print("data shape: " + str(data.shape[0]))
    del data["ss_number_id"]
    # column_to_move = data.pop("ss_number_id")
    # data.insert(0, "ss_number_id", column_to_move)
    data.reset_index(drop = True, inplace = True)

    return data

