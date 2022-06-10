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
from utils import save_evaluation_metric, plot_accuracy_loss, dispatcher, get_random_numbers
from dataset import *
from IPython.display import display



class LSTMClassification(pl.LightningModule):

    def __init__(self, N_FEATURES, hidden_size, learning_rate,dropout,
                              num_layers, rnn_type, bidirectional, activation,layers_m,dropout_m,hidden_dimension_size_m, case = None):
        super(LSTMClassification, self).__init__()
        hidden_size = hidden_size
        num_layers = num_layers
        activation = dispatcher[activation]
        dropout = dropout
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        self.case = case
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

        input_size = hidden_size*2 if bidirectional else hidden_size
        self.linear = getMultiLayerPerceptron(input_size, layers_m, hidden_dimension_size_m, activation, dropout_m)
        self.test_F1score = F1Score()
        self.specificity = Specificity()
        self.target = []
        self.preds = []
        self.prob = []

    def forward(self, x):
        x, _= self.rnn(x)
        x = self.linear(x[:,-1])
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
        preds = torch.sigmoid(preds)
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
        preds = torch.sigmoid(preds)
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
        save_evaluation_metric("lstm", self.acc, self.test_F1score.compute(), self.precision, self.recall, self.specificity.compute(), self.case)
        # #confusion matrix
        cm = confusion_matrix(self.target, self.preds)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()
        disp.figure_.savefig("lstm/" + self.case +"/conf_mat.png",dpi=300)
        plt.clf()
        #create ROC curve
        fpr, tpr, _ = metrics.roc_curve(np.array(self.target), np.array(self.prob))
        plt.plot(fpr, tpr)
        plt.ylabel("true positive rate")
        plt.xlabel("false positive rate")
        plt.savefig("lstm/" + self.case +"/roc_curve.png")





def objective(trial: optuna.trial.Trial) -> float:
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3, step=1)
    learning_rate = trial.suggest_uniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    dropout = trial.suggest_uniform("dropout", 0.0, 0.8)
    rnn_type = trial.suggest_categorical("rnn_type", ["lstm", "gru", "rnn"])
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    activation = trial.suggest_categorical("activation", ["nn.Tanh()", "nn.ReLU()", "nn.ELU()", "nn.LeakyReLU()","nn.Sigmoid()"])
    #MLP
    layers_m = trial.suggest_int("layers_m", 1, 7, step=1)
    dropout_m = get_random_numbers(layers_m, trial, 0.0, 0.9, "dropout_m", int = False, desc = False)
    hidden_dimension_size_m = get_random_numbers(layers_m, trial, 128, 1024, "hidden_dim_m", step = 64)
    N_FEATURES = extract_n_features()
    LSTMmodel = LSTMClassification(N_FEATURES, hidden_size, learning_rate, dropout, num_layers, rnn_type, bidirectional, activation, layers_m,dropout_m,hidden_dimension_size_m)

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

    trainer.fit(LSTMmodel, datamodule=dm)

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

def training_test_LSTM(epochs, trials, case):
    study = hyperparameter_tuning(trials)
    trial = study.best_trial
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3, step=1)
    learning_rate = trial.suggest_uniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    dropout = trial.suggest_uniform("dropout", 0.0, 0.8)
    rnn_type = trial.suggest_categorical("rnn_type", ["lstm", "gru", "rnn"])
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    activation = trial.suggest_categorical("activation", ["nn.Tanh()", "nn.ReLU()", "nn.ELU()", "nn.LeakyReLU()","nn.Sigmoid()"])
    #MLP
    layers_m = trial.suggest_int("layers_m", 1, 7, step=1)
    dropout_m = get_random_numbers(layers_m, trial, 0.0, 0.9, "dropout_m", int = False, desc = False)
    hidden_dimension_size_m = get_random_numbers(layers_m, trial, 128, 1024, "hidden_dim_m", step = 64)
    N_FEATURES = extract_n_features()

    model = LSTMClassification(N_FEATURES, hidden_size, learning_rate, dropout, num_layers, rnn_type, bidirectional, activation, layers_m,dropout_m,hidden_dimension_size_m, case)

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
                        max_epochs=epochs,
                        logger=logger,
                        callbacks=[early_stop_callback,checkpoint_callback, lr_monitor]
                        )
    trainer.fit(model, datamodule = dm)
    torch.save(model, "lstm/" + case + "/model")
    lstm_model = torch.load( "lstm/" + case + "/model")
    print("------------STARTING TEST PART------------")
    results = trainer.test(lstm_model, datamodule=dm)
    plot_accuracy_loss("lstm", trainer, case)
    print("------------FINISH TEST PART------------")
    return results







