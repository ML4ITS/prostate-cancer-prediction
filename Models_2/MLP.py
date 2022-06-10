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
import matplotlib.pyplot as plt
from utils import get_random_numbers, save_evaluation_metric, plot_accuracy_loss, dispatcher
from dataset import *
import random

def getMultiLayerPerceptron(InputNetworkSize, layers,
                                   hidden_dimension_size, activationFunction, dropout):
    model = torch.nn.Sequential(
        torch.nn.LazyLinear(hidden_dimension_size[0]) if InputNetworkSize == None
                    else torch.nn.Linear(InputNetworkSize, hidden_dimension_size[0]),
        activationFunction,
        nn.Dropout(dropout[0]),)
    for i in range(layers):
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(hidden_dimension_size[i], hidden_dimension_size[i+1]),
            activationFunction,
            nn.Dropout(dropout[i+1]),)

    model = torch.nn.Sequential(
        model, torch.nn.Linear(hidden_dimension_size[-1], 1))
    return model

class MLPClassification(pl.LightningModule):

    def __init__(self, n_features, timesteps, learning_rate, layers, dropout, hidden_dimension_size, activation, case = None):
        super(MLPClassification, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.activation = dispatcher[activation]
        self.flatten = nn.Flatten()
        self.linear_act_stack = getMultiLayerPerceptron(n_features * timesteps, layers,
                                   hidden_dimension_size, self.activation, dropout)

        self.test_F1score = F1Score()
        self.specificity = Specificity()
        self.target = []
        self.preds = []
        self.prob = []
        self.case = case

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_act_stack(x)
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
        self.acc = accuracy(preds, y)
        self.test_F1score.update(preds,y)
        self.specificity.update(preds,y)
        self.precision, self.recall = precision_recall(preds,y, average = "micro")
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.acc, prog_bar=True)
        self.log('f1 score', self.test_F1score)
        self.log('precision', self.precision)
        self.log('recall', self.recall)
        self.log('specificity', self.specificity)
        return loss


    def test_epoch_end(self, outputs):
        save_evaluation_metric("mlp", self.acc, self.test_F1score.compute(), self.precision, self.recall, self.specificity.compute(), self.case)
        #confusion matrix
        cm = confusion_matrix(self.target, self.preds)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()
        disp.figure_.savefig("mlp/" + self.case + "/conf_mat.png",dpi=300)
        plt.clf()
        #create ROC curve
        fpr, tpr, _ = metrics.roc_curve(np.array(self.target), np.array(self.prob))
        plt.plot(fpr, tpr)
        plt.ylabel("true positive rate")
        plt.xlabel("false positive rate")
        plt.savefig("mlp/" + self.case + "/roc_curve.png")



def objective(trial: optuna.trial.Trial) -> float:
    layers = trial.suggest_int("layers", 1, 15, step=1)
    dropout = get_random_numbers(layers, trial, 0.0, 0.9, "dropout", int = False, desc = False)
    hidden_dimension_size = get_random_numbers(layers, trial, 32, 1024, "hidden_dim",step=64)
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    activation = trial.suggest_categorical("activation", ["nn.Tanh()", "nn.ReLU()", "nn.ELU()", "nn.LeakyReLU()","nn.Sigmoid()"])
    timesteps, n_features = extract_timesteps(), extract_n_features()

    MLPmodel = MLPClassification(n_features, timesteps, learning_rate, layers, dropout, hidden_dimension_size, activation)

    dm = psaDataModule(batch_size=batch_size)

    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        patience=30,
        verbose=False,
        mode='min'
    )

    EPOCHS = 1


    trainer = pl.Trainer(max_epochs= EPOCHS,
                         callbacks=[early_stop_callback])

    trainer.fit(MLPmodel, datamodule=dm)

    return trainer.callback_metrics["valid_loss"].item()

def hyperparameter_tuning(trials):

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials= trials)

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

def training_test_MLP(epochs, trials, case):
    study = hyperparameter_tuning(trials)
    trial = study.best_trial
    layers = trial.suggest_int("layers", 1, 15, step=1)
    dropout = get_random_numbers(layers, trial, 0.0, 0.9, "dropout", int = False, desc = False)
    hidden_dimension_size = get_random_numbers(layers, trial, 32, 1024, "hidden_dim",step=64)
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    activation = trial.suggest_categorical("activation", ["nn.Tanh()", "nn.ReLU()", "nn.ELU()", "nn.LeakyReLU()","nn.Sigmoid()"])
    timesteps, n_features = extract_timesteps(), extract_n_features()

    MLPmodel = MLPClassification(n_features, timesteps, learning_rate, layers, dropout, hidden_dimension_size, activation, case)

    dm = psaDataModule(batch_size = batch_size)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                        save_top_k=1,
                        save_last=True,
                        save_weights_only=True,
                        filename='checkpoint/{epoch:02d}-{val_loss:.4f}',
                        verbose=False,
                        mode='min')

    early_stop_callback = EarlyStopping(
       monitor='valid_loss',
       patience=30,
       verbose=False,
       mode='min'
    )
    logger = CSVLogger(save_dir="/logs/")

    trainer = pl.Trainer(
                        accelerator="auto",
                        devices = 1 if torch.cuda.is_available() else None,
                        max_epochs=epochs,
                        logger=logger,
                        callbacks=[early_stop_callback,checkpoint_callback, lr_monitor]
                        )
    trainer.fit(MLPmodel, datamodule = dm)
    torch.save(MLPmodel, "mlp/" + case + "/model")
    model = torch.load( "mlp/" + case + "/model")
    print("------------STARTING TEST PART------------")
    results = trainer.test(model, datamodule=dm)
    print("------------FINISH TEST PART------------")
    plot_accuracy_loss("mlp", trainer, case)
    return results









