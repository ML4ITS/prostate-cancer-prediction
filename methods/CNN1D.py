import sys
sys.path.append('../')
from methods.MLP import getMultiLayerPerceptron
import numpy as np
import optuna as optuna
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import torch
torch.manual_seed(0)
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Specificity, F1Score, Accuracy
from pytorch_lightning.loggers.csv_logs import CSVLogger
from sklearn import metrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from settings import p
from utils import *
from dataset import *


def getConv1d(n_features, layers, hidden_dimension_size, activationFunction, dropout, padding):
    kernel= get_kernel_size(n_features, hidden_dimension_size[0], padding[0])
    model = torch.nn.Sequential(
        nn.Conv1d(in_channels = n_features, out_channels=hidden_dimension_size[0], kernel_size=kernel, padding=padding[0]),
        nn.BatchNorm1d(hidden_dimension_size[0]),
        activationFunction,
        nn.Dropout(dropout[0]),)

    for i in range(layers):
        kernel= get_kernel_size(hidden_dimension_size[i], hidden_dimension_size[i+1], padding[i+1])
        if(kernel<0):
            print("Kernel size:  " + str(kernel))
        model = torch.nn.Sequential(
                        model,
                        nn.Conv1d(in_channels = hidden_dimension_size[i], out_channels=hidden_dimension_size[i+1], kernel_size = kernel,  padding=padding[0]),
                        nn.BatchNorm1d(hidden_dimension_size[i+1]),
                        activationFunction,
                        nn.Dropout(dropout[i+1]),)
    model = torch.nn.Sequential(model, nn.AvgPool1d(hidden_dimension_size[-1]), torch.nn.Flatten())
    return model

class CNN1DClassification(pl.LightningModule):

    def __init__(self, n_features, learning_rate, layers_c, n_filt_1, n_filt_2, n_filt_3, activation, dropout_c, layers_m, hidden_dimension_size_m, dropout_m, padding, case = None, m_kernels = False):
        super(CNN1DClassification, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        self.activation = dispatcher[activation]
        self.test_F1score = F1Score()
        self.specificity = Specificity()
        self.accuracy_train = Accuracy()
        self.accuracy_val = Accuracy()
        self.accuracy_test = Accuracy()
        self.target = []
        self.preds = []
        self.prob = []
        self.case = case
        self.m_kernels = m_kernels

        self.cnn1d_1 = getConv1d(n_features, layers_c, n_filt_1, self.activation, dropout_c, padding)
        self.cnn1d_2 = getConv1d(n_features, layers_c, n_filt_2, self.activation, dropout_c, padding)
        self.cnn1d_3 = getConv1d(n_features, layers_c, n_filt_3, self.activation, dropout_c, padding)

        self.linear = getMultiLayerPerceptron(None, layers_m, hidden_dimension_size_m, self.activation, dropout_m)

        self.flatten = torch.nn.Flatten()
    def forward(self, x):
        x = x.permute(0,2,1)
        if self.m_kernels:
            x1 = self.cnn1d_1(x)
            x2 = self.cnn1d_2(x)
            x3 = self.cnn1d_3(x)
            x = torch.cat((x1,x2,x3), axis = -1)
            x = self.flatten(x)
        else:
            x = self.cnn1d_1(x)
        x = self.linear(x)
        return x


    def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)


    def training_step(self,batch,batch_idx):
        """"input size:
        x.shape = (batch_size, timesteps, number of features
        y.shape = (batch_size)"""
        x, y, _ = batch
        preds = self.forward(x).type_as(x)
        y = y.reshape(-1,1)
        loss = self.criterion(preds, y.type(torch.FloatTensor).type_as(preds))
        preds = torch.sigmoid(preds)
        self.accuracy_train.update(preds,y)
        logs = {"train_loss" : loss, "train_acc" : self.accuracy_train}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger = True)
        return loss

    def validation_step(self, batch, batch_nb):
        """"input size:
        x.shape = (batch_size, timesteps, number of features
        y.shape = (batch_size)"""
        x, y, _ = batch
        preds = self.forward(x).type_as(x)
        y = y.reshape(-1,1)
        loss = self.criterion(preds, y.type(torch.FloatTensor).type_as(preds))
        preds = torch.sigmoid(preds)
        self.accuracy_val.update(preds,y)
        logs = {"valid_loss" : loss, "valid_acc" : self.accuracy_val}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger = True)
        return loss


    def test_step(self, batch, batch_nb):
        """"input size:
        x.shape = (batch_size, timesteps, number of features
        y.shape = (batch_size)"""
        x, y, _ = batch
        preds = self.forward(x).type_as(x)
        y = y.reshape(-1,1)
        loss = self.criterion(preds, y.type(torch.FloatTensor).type_as(preds))
        preds = torch.sigmoid(preds)
        self.prob.extend(preds.cpu().detach().numpy())
        preds = (preds>0.5).float()
        self.target.extend(y.cpu().detach().numpy())
        self.preds.extend(preds.cpu().detach().numpy())
        self.accuracy_test.update(preds,y)
        self.test_F1score.update(preds,y)
        self.specificity.update(preds,y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.accuracy_test, prog_bar=True)
        self.log('f1 score', self.test_F1score)
        self.log('specificity', self.specificity)
        return loss


    def test_epoch_end(self, outputs):
        path =  "cnn1d_heads/" if self.m_kernels else "cnn1d/"
        save_evaluation_metric(path, self.accuracy_test.compute(), self.test_F1score.compute(), self.specificity.compute(), self.case)
        #save results
        save_result_df(path, self.target, self.preds, self.case)
        #confusion matrix
        cm = confusion_matrix(self.target, self.preds)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()
        disp.figure_.savefig(path + self.case + "/conf_mat.png",dpi=300)
        #create ROC curve
        plt.clf()
        fpr, tpr, _ = metrics.roc_curve(np.array(self.target), np.array(self.prob))
        plt.plot(fpr, tpr)
        plt.ylabel("true positive rate")
        plt.xlabel("false positive rate")
        plt.savefig(path + self.case + "/roc_curve.png")

def objective(trial: optuna.trial.Trial) -> float:
    n_features = extract_n_features()
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    layers_c = trial.suggest_int("layers", 1, 15, step=1)
    activation = trial.suggest_categorical("activation", ["nn.Tanh()", "nn.ReLU()", "nn.ELU()", "nn.LeakyReLU()","nn.Sigmoid()"])
    dropout_c = get_random_numbers(layers_c, trial, 0.0, 0.9, "dropout_c", int = False, desc = False)
    n_filt_1 = get_random_numbers(layers_c, trial, 1, n_features-1, "n_filt_1", desc = True)
    n_filt_2 = get_random_numbers(layers_c, trial, 1, n_features-1, "n_filt_2", desc = True)
    n_filt_3 = get_random_numbers(layers_c, trial, 1, n_features-1, "n_filt_3", desc = True)
    padding = get_random_numbers(layers_c, trial, 0, 2, "padding")
    # parameters for linear layers
    layers_m = trial.suggest_int("layers_m", 1, 7, step=1)
    dropout_m = get_random_numbers(layers_m, trial, 0.1, 0.9, "dropout_m", int = False, desc = False)
    hidden_dimension_size_m = get_random_numbers(layers_m, trial, 128, 512, "hidden_dim_m",step=64)
    m_kernels = p.multiple_kernels


    model = CNN1DClassification(n_features, learning_rate, layers_c, n_filt_1, n_filt_2, n_filt_3, activation, dropout_c, layers_m, hidden_dimension_size_m, dropout_m, padding, m_kernels = m_kernels)

    dm = psaDataModule(batch_size=batch_size)

    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        patience=10,
        verbose=False,
        mode='min'
    )

    EPOCHS = 1


    trainer = pl.Trainer(max_epochs= EPOCHS,
                         accelerator="auto",
                         devices=-1 if torch.cuda.is_available() else None,
                         callbacks=[early_stop_callback])

    trainer.fit(model, datamodule=dm)

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

def training_test_CNN1D(epochs, trials, case, iterations, m_kernels = False):
    study = hyperparameter_tuning(trials)
    trial = study.best_trial
    path = "cnn1d_heads/" if m_kernels else "cnn1d/"
    save_hyperparameter(path, study, case)
    n_features = extract_n_features()
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    layers_c = trial.suggest_int("layers", 1, 15, step=1)
    activation = trial.suggest_categorical("activation", ["nn.Tanh()", "nn.ReLU()", "nn.ELU()", "nn.LeakyReLU()","nn.Sigmoid()"])
    dropout_c = get_random_numbers(layers_c, trial, 0.0, 0.9, "dropout_c", int = False, desc = False)
    n_filt_1 = get_random_numbers(layers_c, trial, 1, n_features-1, "n_filt_1", desc = True)
    n_filt_2 = get_random_numbers(layers_c, trial, 1, n_features-1, "n_filt_2", desc = True)
    n_filt_3 = get_random_numbers(layers_c, trial, 1, n_features-1, "n_filt_3", desc = True)
    padding = get_random_numbers(layers_c, trial, 0, 2, "padding")
    # parameters for linear layers
    layers_m = trial.suggest_int("layers_m", 1, 7, step=1)
    dropout_m = get_random_numbers(layers_m, trial, 0.1, 0.9, "dropout_m", int = False, desc = False)
    hidden_dimension_size_m = get_random_numbers(layers_m, trial, 128, 512, "hidden_dim_m",step=64)

    accuracy = []
    for i in range(iterations):
        model = CNN1DClassification(n_features, learning_rate, layers_c, n_filt_1, n_filt_2, n_filt_3, activation, dropout_c, layers_m, hidden_dimension_size_m, dropout_m, padding, case, m_kernels)

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
        logger = CSVLogger(save_dir="../Models_2/logs/")

        trainer = pl.Trainer(
                            accelerator="auto",
                            devices=-1 if torch.cuda.is_available() else None,
                            max_epochs=epochs,
                            logger=logger,
                            callbacks=[early_stop_callback,checkpoint_callback, lr_monitor]
                            )
        trainer.fit(model, datamodule = dm)
        torch.save(model, path + case + "/model")
        cnn1d_model = torch.load(path + case + "/model")
        print("------------STARTING TEST PART------------")
        results = trainer.test(cnn1d_model, datamodule=dm)
        accuracy.append(list(results[0].values())[1])
        plot_accuracy_loss(path, trainer, case)
        print("------------FINISH TEST PART------------")
    return accuracy








