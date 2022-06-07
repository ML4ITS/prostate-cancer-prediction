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
from dataset import *


class MLPClassification(pl.LightningModule):

    def __init__(self, n_features, timesteps, learning_rate, dropout1, dropout2, dropout3, dropout4, activation, out1, out2, out3, out4):
        super(MLPClassification, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.flatten = nn.Flatten()
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(timesteps * n_features, out1),
            self.activation,
            nn.Dropout(dropout1),
            nn.Linear(out1, out2),
            self.activation,
            nn.Dropout(dropout2),
            nn.Linear(out2, out3),
            self.activation,
            nn.Dropout(dropout3),
            nn.Linear(out3, out4),
            self.activation,
            nn.Dropout(dropout4),
            nn.Linear(out4, 1),
        )


        self.test_F1score = F1Score()
        self.specificity = Specificity()
        self.target = []
        self.preds = []
        self.prob = []

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return torch.sigmoid(x)

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
    out1 = trial.suggest_int("out1", 512, 1024, step=32)
    out2 = trial.suggest_int("out2", 256, 512, step=32)
    out3 = trial.suggest_int("out3", 128, 256, step=32)
    out4 = trial.suggest_int("out4", 64, 128, step=32)
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    dropout1 = trial.suggest_uniform("dropout1", 0.1, 0.8)
    dropout2 = trial.suggest_uniform("dropout2", 0.1, 0.8)
    dropout3 = trial.suggest_uniform("dropout3", 0.1, 0.8)
    dropout4 = trial.suggest_uniform("dropout4", 0.1, 0.8)
    activation = trial.suggest_categorical("activation", ["tanh", "relu"])
    timesteps, n_features = extract_timesteps(), extract_n_features()
    MLPmodel = MLPClassification(n_features, timesteps, learning_rate, dropout1, dropout2, dropout3, dropout4, activation, out1, out2, out3, out4)

    dm = psaDataModule(batch_size=batch_size)

    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        patience=30,
        verbose=False,
        mode='min'
    )

    EPOCHS = 100


    trainer = pl.Trainer(max_epochs= EPOCHS,
                         callbacks=[early_stop_callback])

    trainer.fit(MLPmodel, datamodule=dm)

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

def training_test_MLP():
    study = hyperparameter_tuning()
    trial = study.best_trial
    out1 = trial.suggest_int("out1", 512, 1024, step=32)
    out2 = trial.suggest_int("out2", 256, 512, step=32)
    out3 = trial.suggest_int("out3", 128, 256, step=32)
    out4 = trial.suggest_int("out4", 64, 128, step=32)
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    dropout1 = trial.suggest_uniform("dropout1", 0.1, 0.8)
    dropout2 = trial.suggest_uniform("dropout2", 0.1, 0.8)
    dropout3 = trial.suggest_uniform("dropout3", 0.1, 0.8)
    dropout4 = trial.suggest_uniform("dropout4", 0.1, 0.8)
    activation = trial.suggest_categorical("activation", ["tanh", "relu"])
    timesteps, n_features = extract_timesteps(), extract_n_features()
    MLPmodel = MLPClassification(n_features, timesteps, learning_rate, dropout1, dropout2, dropout3, dropout4, activation, out1, out2, out3, out4)
    EPOCHS = 200

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
    trainer.fit(MLPmodel, datamodule = dm)
    trainer.test(MLPmodel, datamodule=dm)








