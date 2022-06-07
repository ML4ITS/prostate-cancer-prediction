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

def extract_n_features():
    data = pd.read_csv("db/data")
    return data.shape[1] - 2

def manage_db(df):
    sequences = []
    label = []
    FEATURES_COLUMNS = df.columns
    for ss_number_id, group in df.groupby("ss_number_id"):
        sequence_features = group[FEATURES_COLUMNS[1:-1]]
        sequences.append((np.array(sequence_features.values)))
        label.append(group[FEATURES_COLUMNS[-1]].values[0])
    return np.array(sequences, dtype = "object"), np.array(label)

def pad_collate(data):
    """data is a list of tuples with (example, label, length)
        where example is a tensor of arbitrary shape
        and label/length are scalars"""
    _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])
    return features.float(), labels.long(), lengths.long()

class psaDataset(Dataset):
    '''
Custom Dataset subclass.
Serves as input to DataLoader to transform X
  into sequence data using rolling window.
DataLoader using this dataset will output batches
  of `(batch_size, seq_len, n_features)` shape.
'''
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):

        self.X = X
        y = torch.LongTensor(y)
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        """"
        return x,y with dimension x: n_timesteps, n_features
                                y: scalar value
        """
        x = self.X[index:index+self.seq_len]
        x = torch.FloatTensor(x[0])
        y = self.y[index+self.seq_len-1]
        return (x,y, x.shape[0])

class psaDataModule(pl.LightningDataModule):
    def __init__(self, seq_len=1, batch_size=128, num_workers=0):

        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.preprocessing = StandardScaler()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' and self.X_train is not None:
            return
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return
        data = pd.read_csv("db/data")
        x,y = manage_db(data)
        x, test, y, test_y = train_test_split(x, y, train_size=0.95, shuffle=True)
        print("SHAPE TRAIN: "+ str(x.shape))
        print("SHAPE TEST: " + str(test.shape))
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=True)


        if stage == 'fit' or stage is None:
            self.X_train = np.array(X_train, dtype = object)
            self.y_train = np.array(y_train)  # labels
            self.X_val = np.array(X_val, dtype = object)
            self.y_val = np.array(y_val)  # labels

        if stage == 'test' or stage is None:
            self.X_test = np.array(test, dtype = object)
            self.y_test = np.array(test_y)  # labels

    def train_dataloader(self):
        train_dataset = psaDataset(self.X_train, self.y_train, seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=pad_collate)
        return train_loader

    def val_dataloader(self):
        val_dataset = psaDataset(self.X_val, self.y_val, seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=pad_collate)
        return val_loader

    def test_dataloader(self):
        test_dataset = psaDataset(self.X_test, self.y_test, seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=pad_collate)
        return test_loader
