import matplotlib.pyplot as plt
import numpy as np
import optuna as optuna
import pytorch_lightning as pl
import torch
torch.manual_seed(0)
import torch.nn as nn
from dataset import *
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.csv_logs import CSVLogger
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from torchmetrics import Specificity, F1Score, Accuracy
from torchmetrics.functional import precision_recall
from utils import *


def getMultiLayerPerceptron(InputNetworkSize, layers,
                                   hidden_dimension_size, activationFunction, dropout):
    #the number of layers have been defined dinamically
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
        self.accuracy_train = Accuracy()
        self.accuracy_val = Accuracy()
        self.accuracy_test = Accuracy()
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
        preds = self.forward(x).type_as(x)
        y = y.reshape(-1,1)
        loss = self.criterion(preds, y.type(torch.FloatTensor).type_as(preds))
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
        self.prob.extend(preds.cpu().detach().numpy())
        preds = (preds>0.5).float()
        self.target.extend(y.cpu().detach().numpy())
        self.preds.extend(preds.cpu().detach().numpy())
        self.accuracy_test.update(preds, y)
        self.test_F1score.update(preds,y)
        self.specificity.update(preds,y)
        self.precision, self.recall = precision_recall(preds,y, average = "micro")
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.accuracy_test, prog_bar=True)
        self.log('f1 score', self.test_F1score)
        self.log('specificity', self.specificity)
        return loss


    def test_epoch_end(self, outputs):
        save_evaluation_metric("mlp", self.accuracy_test.compute(), self.test_F1score.compute(), self.specificity.compute(), self.case)
        #save results
        save_result_df("mlp", self.target, self.preds, self.case)
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
        patience=10,
        verbose=False,
        mode='min'
    )

    EPOCHS = 15


    trainer = pl.Trainer(max_epochs= EPOCHS,
                         accelerator="auto",
                         devices=-1 if torch.cuda.is_available() else None,
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

def training_test_MLP(epochs, trials, case, iterations):
    #create model and make prediction
    study = hyperparameter_tuning(trials)
    trial = study.best_trial
    save_hyperparameter("mlp/", study, case)
    layers = trial.suggest_int("layers", 1, 15, step=1)
    dropout = get_random_numbers(layers, trial, 0.0, 0.9, "dropout", int = False, desc = False)
    hidden_dimension_size = get_random_numbers(layers, trial, 32, 1024, "hidden_dim",step=64)
    learning_rate = trial.suggest_uniform("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    activation = trial.suggest_categorical("activation", ["nn.Tanh()", "nn.ReLU()", "nn.ELU()", "nn.LeakyReLU()","nn.Sigmoid()"])
    timesteps, n_features = extract_timesteps(), extract_n_features()

    accuracy = []

    for i in range(iterations):
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
        logger = CSVLogger(save_dir="../Models_2/logs/")

        trainer = pl.Trainer(
                            accelerator="auto",
                            devices =-1 if torch.cuda.is_available() else None,
                            max_epochs=epochs,
                            logger=logger,
                            callbacks=[early_stop_callback,checkpoint_callback, lr_monitor]
                            )
        trainer.fit(MLPmodel, datamodule = dm)
        torch.save(MLPmodel, "mlp/" + case + "/model")
        model = torch.load( "mlp/" + case + "/model")
        print("------------STARTING TEST PART------------")
        results = trainer.test(model, datamodule=dm)
        accuracy.append(list(results[0].values())[1])
        print("------------FINISH TEST PART------------")
        plot_accuracy_loss("mlp", trainer, case)
    return accuracy









