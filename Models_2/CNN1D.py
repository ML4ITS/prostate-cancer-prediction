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
from MLP import getMultiLayerPerceptron
from utils import get_random_numbers
from dataset import *
from IPython.display import display

def getConv1d(n_features, layers, hidden_dimension_size, activationFunction, dropout, kernel_size):
    model = torch.nn.Sequential(
        nn.Conv1d(n_features, hidden_dimension_size[0], kernel_size=kernel_size[0]),
        nn.BatchNorm1d(hidden_dimension_size[0]),
        activationFunction,
        nn.Dropout(dropout[0]),)
    for i in range(layers):
        model = torch.nn.Sequential(
            model,
            nn.Conv1d(hidden_dimension_size[i], hidden_dimension_size[i+1], kernel_size=kernel_size[i+1]),
            nn.BatchNorm1d(hidden_dimension_size[i+1]),
            activationFunction,
            nn.Dropout(dropout[i+1]),)
    return model

class CNN1DClassification(pl.LightningModule):

    def __init__(self, n_features, timesteps, learning_rate, layers_c, hidden_dimension_size_c, activation, dropout_c, kernel_size, layers_m, hidden_dimension_size_m, dropout_m):
        super(CNN1DClassification, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.test_F1score = F1Score()
        self.specificity = Specificity()
        self.target = []
        self.preds = []
        self.prob = []

        self.cnn1d = getConv1d(n_features, layers_c, hidden_dimension_size_c, activation, dropout_c, kernel_size)
        n_channels = self.cnn1d(torch.empty(1,n_features,timesteps)).size(-1)
        self.flatten = nn.Flatten()
        self.linear = getMultiLayerPerceptron(n_channels, layers_m, hidden_dimension_size_m, activation, dropout_m)


    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.cnn1d(x)
        x = self.flatten(x)
        x = self.linear(x)
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
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    layers_c = trial.suggest_int("layers", 1, 15, step=1)
    dropout_c = get_random_numbers(layers_c, trial, 0.1, 0.9, "dropout_c", int = False, desc = False)
    hidden_dimension_size_c = get_random_numbers(layers_c, trial, 32, 1024, "hidden_dim_c", desc = False)
    kernel_size = get_random_numbers(layers_c, trial, 1, 7, "kernel")
    #--------------#
    layers_m = trial.suggest_int("layers_m", 1, 15, step=1)
    dropout_m = get_random_numbers(layers_m, trial, 0.1, 0.9, "dropout_m", int = False, desc = False)
    hidden_dimension_size_m = get_random_numbers(layers_m, trial, 32, 1024, "hidden_dim_m")
    activation = trial.suggest_categorical("activation", ["tanh", "relu"])
    timesteps, n_features = extract_timesteps(), extract_n_features()

    model = CNN1DClassification(n_features,timesteps, learning_rate, layers_c, hidden_dimension_size_c, activation, dropout_c, kernel_size, layers_m, hidden_dimension_size_m, dropout_m)

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

    trainer.fit(model, datamodule=dm)

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

def training_test_CNN1D():
    study = hyperparameter_tuning()
    trial = study.best_trial
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    layers_c = trial.suggest_int("layers", 1, 15, step=1)
    dropout_c = get_random_numbers(layers_c, trial, 0.1, 0.9, "dropout_c", int = False, desc = False)
    hidden_dimension_size_c = get_random_numbers(layers_c, trial, 32, 1024, "hidden_dim_c", desc = False)
    kernel_size = get_random_numbers(layers_c, trial, 1, 7, "kernel")
    #--------------#
    layers_m = trial.suggest_int("layers_m", 1, 15, step=1)
    dropout_m = get_random_numbers(layers_m, trial, 0.1, 0.9, "dropout_m", int = False, desc = False)
    hidden_dimension_size_m = get_random_numbers(layers_m, trial, 32, 1024, "hidden_dim_m")
    activation = trial.suggest_categorical("activation", ["tanh", "relu"])
    timesteps, n_features = extract_timesteps(), extract_n_features()

    model = CNN1DClassification(n_features,timesteps, learning_rate, layers_c, hidden_dimension_size_c, activation, dropout_c, kernel_size, layers_m, hidden_dimension_size_m, dropout_m)
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
    trainer.fit(model, datamodule = dm)
    torch.save(model, "cnn1d/model")
    cnn1d_model = torch.load("cnn1d/model")
    trainer.test(cnn1d_model, datamodule=dm)
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace = True)
    display(metrics.dropna(axis =1, how = "all").head())
    g = sn.relplot(data=metrics, kind = "line")
    plt.gcf().set_size_inches(12,4)
    plt.grid()
    plt.savefig("cnn1d/table")








