import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from packaging import version
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers
from torchmetrics.functional import auc
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from torchmetrics import Accuracy
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torchmetrics import F1Score
from torchmetrics.functional import precision_recall
from torchmetrics.functional import auc
from torchmetrics import AUROC
from torch.nn import functional as F
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import seaborn as sn
from config_file.config_parser import Data
from utils import get_binary_indicators
from settings import p

def extract_timesteps():
    """extract the maximum timesteps"""
    data = pd.read_csv("../data_model.csv")
    return data.shape[1] - 1 if p.regularization else data.groupby(["ss_number_id"]).count().max()[0]

def extract_n_features():
    """extract the total number of features from data
    data.shape[1] - 2 because I dont consider the target and the id"""
    data = pd.read_csv("../data_model.csv")
    if p.regularization:
        size = 2 if p.indicator else 1
    else:
        size = data.shape[1] - 2
    return size

def manage_db(df):
    if p.regularization is not True:
        sequences = []
        label = []
        FEATURES_COLUMNS = df.columns
        for ss_number_id, group in df.groupby("ss_number_id"):
            sequence_features = group[FEATURES_COLUMNS[1:-1]]
            sequences.append((np.array(sequence_features.values)))
            label.append(group[FEATURES_COLUMNS[-1]].values[0])
        return np.array(sequences, dtype = "object"), np.array(label)
    else:
        return np.array(df.iloc[:, :-1]), np.array(df.iloc[:, -1])

def pad_collate(data):
    """data is a list of tuples with (example, label, length)
        where example is a tensor of arbitrary shape
        and label/length are scalars"""
    _, labels, lengths = zip(*data)
    max_len = extract_timesteps()
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])
    return features.float(), labels.long(), lengths.long()


class psaDataset(Dataset):
    """Custom Dataset subclass.

    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape"""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):

        self.X = X
        y = torch.LongTensor(y)
        self.y = y
        self.seq_len = seq_len
        self.indicator = p.indicator
        self.regularization = p.regularization

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        """"
        return x,y with dimension x: n_timesteps, n_features
                                y: scalar value
        """
        x = np.array(self.X[index:index+self.seq_len])
        x = torch.FloatTensor(x) if self.regularization else torch.FloatTensor(x[0])
        if self.indicator is True:
            x = get_binary_indicators(x)
        if self.indicator is False and self.regularization is True:
            x = x.reshape(-1,1)
        y = self.y[index+self.seq_len-1]
        return x, y, x.shape[0]

class psaDataModule(pl.LightningDataModule):
    def __init__(self, seq_len=1, batch_size=128):

        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = 0
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.y_test = None
        self.preprocessing = StandardScaler()
        self.regularization = p.regularization
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' and self.X_train is not None:
            return
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return
        data = pd.read_csv("../data_model.csv")
        x,y = manage_db(data)
        #every time I call train_test_split, test and train are the same
        x, test, y, test_y = train_test_split(x, y, train_size=0.8, shuffle=True, random_state = 42)
        print("SHAPE TRAIN: "+ str(x.shape))
        print("SHAPE TEST: " + str(test.shape))
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=True, random_state = None)


        if stage == 'fit' or stage is None:
            self.X_train = X_train if self.regularization else np.array(X_train, dtype = object)
            self.y_train = np.array(y_train)  # labels
            self.X_val = X_val if self.regularization else np.array(X_val, dtype = object)
            self.y_val = np.array(y_val)  # labels

        if stage == 'test' or stage is None:
            self.X_test = test if self.regularization else np.array(test, dtype = object)
            self.y_test = np.array(test_y)  # labels

    def train_dataloader(self):
        train_dataset = psaDataset(self.X_train, self.y_train, seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=pad_collate, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_dataset = psaDataset(self.X_val, self.y_val, seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=pad_collate, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_dataset = psaDataset(self.X_test, self.y_test, seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, collate_fn=pad_collate, num_workers=self.num_workers)
        return test_loader
