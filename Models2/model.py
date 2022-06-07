import numpy as np
import optuna as optuna
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Specificity, F1Score
from torchmetrics.functional import accuracy
from pytorch_lightning.loggers.csv_logs import CSVLogger
from torchmetrics.functional import precision_recall
from torch.nn import functional as F
from sklearn import metrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from Models2.dataset import *


class LSTMClassification(pl.LightningModule):

    def __init__(self, N_FEATURES, hidden_size, learning_rate,dropout,
                              num_layers, rnn_type, bidirectional):
        super(LSTMClassification, self).__init__()
        hidden_size = hidden_size
        num_layers = num_layers
        dropout = dropout
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size= N_FEATURES,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout if num_layers>1 else 0,
                                batch_first=True,
                                bidirectional = bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size= N_FEATURES,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers>1 else 0,
                    batch_first=True,
                    bidirectional = bidirectional)
        else:
            self.rnn = nn.RNN(input_size= N_FEATURES,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers>1 else 0,
                batch_first=True,
                bidirectional = bidirectional)

        self.linear = nn.Linear(hidden_size*2, 1) if bidirectional else nn.Linear(hidden_size, 1)
        self.test_F1score = F1Score()
        self.specificity = Specificity()
        self.target = []
        self.preds = []
        self.prob = []

    def forward(self, x):
        x, _= self.rnn(x)
        x = self.linear(x[:,-1])
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)


    def training_step(self,batch,batch_idx):
        """"input size:
        x.shape = (batch_size, timesteps, number of features
        y.shape = (batch_size)"""
        x, y, _ = batch
        preds = self.forward(x)
        y = y.reshape(-1,1)
        loss = self.criterion(preds, y.type(torch.FloatTensor))
        acc = accuracy(preds, y)
        logs = {"train_loss" : loss, "train_acc" : acc}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger = True)
        return loss

    def validation_step(self, batch, batch_nb):
        """"input size:
        x.shape = (batch_size, timesteps, number of features
        y.shape = (batch_size)"""
        x, y, _ = batch
        preds = self.forward(x)
        y = y.reshape(-1,1)
        loss = self.criterion(preds, y.type(torch.FloatTensor))
        acc = accuracy(preds, y)
        logs = {"valid_loss" : loss, "valid_acc" : acc}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger = True)
        return loss


    def test_step(self, batch, batch_nb):
        """"input size:
        x.shape = (batch_size, timesteps, number of features
        y.shape = (batch_size)"""
        x, y, _ = batch
        preds = self.forward(x)
        y = y.reshape(-1,1)
        loss = self.criterion(preds, y.type(torch.FloatTensor))
        self.prob.extend(preds.numpy())
        preds = (preds>0.5).float()
        self.target.extend(y.numpy())
        self.preds.extend((preds.numpy()))
        acc = accuracy(preds, y)
        self.test_F1score.update(preds,y)
        self.specificity.update(preds,y)
        precision, recall = precision_recall(preds,y, average = "micro")
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('f1 score', self.test_F1score)
        self.log('precision', precision)
        self.log('recall', recall)
        self.log('specificity', self.specificity)
        return loss


    def test_epoch_end(self, outputs):
        #confusion matrix
        fig = plt.figure(figsize = (7,6))
        cm = confusion_matrix(self.target, self.preds)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()

        #create ROC curve
        fig = plt.figure(figsize=(7, 6))
        fpr, tpr, _ = metrics.roc_curve(np.array(self.target), np.array(self.prob))
        plt.plot(fpr, tpr)
        plt.ylabel("true positive rate")
        plt.xlabel("false positive rate")
        plt.show()


def objective(trial: optuna.trial.Trial) -> float:
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    learning_rate = trial.suggest_uniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    rnn_type = trial.suggest_categorical("rnn_type", ["lstm", "gru", "rnn"])
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    N_FEATURES = extract_n_features()
    LSTMmodel = LSTMClassification(N_FEATURES, hidden_size, learning_rate, batch_size, dropout, rnn_type, bidirectional)

    dm = psaDataModule(batch_size=batch_size)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=False,
        mode='min'
    )

    EPOCHS = 100


    trainer = pl.Trainer(max_epochs= EPOCHS,
                         callbacks=[early_stop_callback]
                         )

    hyperparameters_LSTM = dict(hidden_size=hidden_size,
                                dropout=dropout, learning_rate=learning_rate, batch_size=batch_size,
                                rnn_type=rnn_type, bidirectional=bidirectional)

    trainer.logger.log_hyperparams(hyperparameters_LSTM)
    trainer.fit(LSTMmodel, datamodule=dm)

    return trainer.callback_metrics["valid_loss"].item()

def hyperparameter_tuning():
    N_TRIALS = 100
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials= N_TRIALS)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("Best Params : {}".format(study.best_params))
    print("\nBest loss : {}".format(study.best_value))
    return study

def training_test_part():
    study = hyperparameter_tuning()
    trial = study.best_trial
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    learning_rate = trial.suggest_uniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    rnn_type = trial.suggest_categorical("rnn_type", ["lstm", "gru", "rnn"])
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    N_FEATURES = extract_n_features()

    model = LSTMClassification(N_FEATURES, hidden_size, learning_rate, batch_size, dropout, rnn_type, bidirectional)
    EPOCHS = 100

    dm = psaDataModule(batch_size = batch_size)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                        save_top_k=1,
                        save_last=True,
                        save_weights_only=True,
                        filename='Models2/checkpoint/{epoch:02d}-{val_loss:.4f}',
                        verbose=False,
                        mode='min')

    early_stop_callback = EarlyStopping(
       monitor='valid_loss',
       patience=30,
       verbose=False,
       mode='min'
    )
    logger = CSVLogger(save_dir="Models2/logs/")

    trainer = pl.Trainer(
                        accelerator="auto",
                        devices = 1 if torch.cuda.is_available() else None,
                        max_epochs=EPOCHS,
                        logger=logger,
                        callbacks=[early_stop_callback,checkpoint_callback, lr_monitor]
                        )
    trainer.fit(model, datamodule = dm)
    trainer.test(model, datamodule=dm)








