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
from utils import get_random_numbers, save_evaluation_metric, plot_accuracy_loss, dispatcher
from dataset import *
from IPython.display import display

def get_Kernel_Stride(input_size, output_size,padding):
    # Stride = (input_size // output_size)
    # Kernel = input_size - (output_size - 1) * Stride
    Kernel = 2 * padding + input_size + 1 - output_size
    return Kernel



def getConv1d(n_features, layers, hidden_dimension_size, activationFunction, dropout, padding):
    Kernel= get_Kernel_Stride(n_features, hidden_dimension_size[0], padding[0])
    model = torch.nn.Sequential(
        nn.Conv1d(in_channels = n_features, out_channels=hidden_dimension_size[0], kernel_size=Kernel, padding=padding[0]),
        nn.BatchNorm1d(hidden_dimension_size[0]),
        activationFunction,
        nn.AvgPool1d(hidden_dimension_size[0]),
        nn.Dropout(dropout[0]),)
    for i in range(layers):
        Kernel= get_Kernel_Stride(hidden_dimension_size[i], hidden_dimension_size[i+1], padding[i+1])
        if(Kernel<0):
            print("Kernel:  " + str(Kernel))
            print(c)
        model = torch.nn.Sequential(
            model,
            nn.Conv1d(in_channels = hidden_dimension_size[i], out_channels=hidden_dimension_size[i+1], kernel_size = Kernel,  padding=padding[0]),
            nn.BatchNorm1d(hidden_dimension_size[i+1]),
            activationFunction,
            nn.AvgPool1d(hidden_dimension_size[i+1]),
            nn.Dropout(dropout[i+1]),)
    model = torch.nn.Sequential(model, torch.nn.Flatten())
    return model

class CNN1DClassification(pl.LightningModule):

    def __init__(self, n_features, timesteps, learning_rate, layers_c, hidden_dimension_size_c, activation, dropout_c, layers_m, hidden_dimension_size_m, dropout_m, padding):
        super(CNN1DClassification, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        self.activation = dispatcher[activation]
        self.test_F1score = F1Score()
        self.specificity = Specificity()
        self.target = []
        self.preds = []
        self.prob = []

        self.cnn1d = getConv1d(n_features, layers_c, hidden_dimension_size_c, self.activation, dropout_c, padding)

        self.linear = getMultiLayerPerceptron(None, layers_m, hidden_dimension_size_m, self.activation, dropout_m)


    def forward(self, x):
        x = x.permute(0,2,1)
        print(x.shape)
        x = self.cnn1d(x)
        x = self.linear(x)
        return x


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
        preds = torch.sigmoid(preds)
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
        save_evaluation_metric("cnn1d", self.acc, self.test_F1score.compute(), self.precision, self.recall, self.specificity.compute())
        #confusion matrix
        cm = confusion_matrix(self.target, self.preds)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()
        disp.figure_.savefig('cnn1d/conf_mat.png',dpi=300)
        #create ROC curve
        fpr, tpr, _ = metrics.roc_curve(np.array(self.target), np.array(self.prob))
        plt.plot(fpr, tpr)
        plt.ylabel("true positive rate")
        plt.xlabel("false positive rate")
        plt.savefig("cnn1d/roc_curve.png")

def objective(trial: optuna.trial.Trial) -> float:
    timesteps, n_features = extract_timesteps(), extract_n_features()
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    layers_c = trial.suggest_int("layers", 1, 15, step=1)
    activation = trial.suggest_categorical("activation", ["nn.Tanh()", "nn.ReLU()", "nn.ELU()", "nn.LeakyReLU()","nn.Sigmoid()"])
    dropout_c = get_random_numbers(layers_c, trial, 0.1, 0.9, "dropout_c", int = False, desc = False)
    n_filters = get_random_numbers(layers_c, trial, 1, n_features-1, "n_filters", desc = True)
    padding = get_random_numbers(layers_c, trial, 0, 2, "padding")
    # parameters for linear layers
    layers_m = trial.suggest_int("layers_m", 1, 7, step=1)
    dropout_m = get_random_numbers(layers_m, trial, 0.1, 0.9, "dropout_m", int = False, desc = False)
    hidden_dimension_size_m = get_random_numbers(layers_m, trial, 128, 512, "hidden_dim_m",step=64)



    model = CNN1DClassification(n_features,timesteps, learning_rate, layers_c, n_filters, activation, dropout_c, layers_m, hidden_dimension_size_m, dropout_m, padding)

    dm = psaDataModule(batch_size=batch_size)

    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        patience=30,
        verbose=False,
        mode='min'
    )

    EPOCHS = 3


    trainer = pl.Trainer(max_epochs= EPOCHS,
                         callbacks=[early_stop_callback])

    trainer.fit(model, datamodule=dm)

    return trainer.callback_metrics["valid_loss"].item()

def hyperparameter_tuning():
    N_TRIALS = 200
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
    timesteps, n_features = extract_timesteps(), extract_n_features()
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    layers_c = trial.suggest_int("layers", 1, 15, step=1)
    dropout_c = get_random_numbers(layers_c, trial, 0.1, 0.9, "dropout_c", int = False, desc = False)
    n_filters = get_random_numbers(layers_c, trial, 1, n_features-1, "n_filters", desc = True)
    padding = get_random_numbers(layers_c, trial, 0, 2, "padding")
    #--------------#
    layers_m = trial.suggest_int("layers_m", 1, 7, step=1)
    dropout_m = get_random_numbers(layers_m, trial, 0.1, 0.9, "dropout_m", int = False, desc = False)
    hidden_dimension_size_m = get_random_numbers(layers_m, trial, 128, 512, "hidden_dim_m",step=64)
    activation = trial.suggest_categorical("activation", ["nn.Tanh()", "nn.ReLU()", "nn.ELU()", "nn.LeakyReLU()","nn.Sigmoid()"])


    model = CNN1DClassification(n_features,timesteps, learning_rate, layers_c, n_filters, activation, dropout_c, layers_m, hidden_dimension_size_m, dropout_m, padding)

    EPOCHS = 20

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
    logger = CSVLogger(save_dir="logs/")

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
    print("------------STARTING TEST PART------------")
    trainer.test(cnn1d_model, datamodule=dm)
    plot_accuracy_loss("cnn1d", trainer)
    print("------------FINISH TEST PART------------")








