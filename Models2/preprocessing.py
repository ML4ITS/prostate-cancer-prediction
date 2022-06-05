import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import plotly.express as px
import matplotlib.plyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap

def upload_db(path1, path2, len = 2):
    data1 = pd.read_csv(path1).sort_values(by = ["age"])
    data2 = pd.read_csv(path2).sort_values(by=["age"])
    v = data1.ss_number_id.value_counts()
    data1 = data1[data1.ss_number_id.isin(v.index[v.gt(len)])]
    v = data2.ss_number_id.value_counts()
    data2 = data2[data2.ss_number_id.isin(v.index[v.gt(len)])]
    print("Number of patients with cancer: "+ str(len(data1["ss_number_id"].unique())))
    print("Number of patients without cancer: "+ str(len(data2["ss_number_id"].unique())))
    data1.head()
    return data1, data2

def dummies(data):
    bins1 = [0, 30, 40, 50, 60, 70, 80, 120]
    bins2 = [0, 4, 10, 2000]
    labels = [0, 1 , 2, 3, 4, 5, 6]
    data["AgeGroup"] = pd.cut(data1["age"], bins = bins1, labels = labels, right = False)