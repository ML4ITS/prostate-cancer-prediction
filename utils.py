import numpy as np
import pandas as pd
import torch
from IPython.display import display
import seaborn as sn
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
import functools
from sklearn.manifold import Isomap

dispatcher= {'nn.Tanh()': nn.Tanh(), 'nn.ReLU()': nn.ReLU(), 'nn.ELU()': nn.ELU(), 'nn.LeakyReLU()': nn.LeakyReLU(), 'nn.Sigmoid()': nn.Sigmoid()}

def get_random_numbers(layers, trial, min, max, element, int = True, desc = True, step = 1):
    random_float_list = []
    for i in range(layers+1):
        el = element + str(i)
        if int is True:
            x = trial.suggest_int(el, min, max, step)
        else:
            x= trial.suggest_uniform(el, min, max)
        random_float_list.append(x)
    return np.sort(np.array(random_float_list))[::-1] if desc else np.sort(np.array(random_float_list))

def get_kernel_size(input_size, output_size,padding):
    return 2 * padding + input_size + 1 - output_size

def save_evaluation_metric(model, accuracy, f1score, precision, recall, specificity, case):
    file = open(model+ "/" + case +"/results.txt", "a+")
    accuracy = "accuracy: " + str(accuracy.item()) + "\n"
    f1score = "f1score: " + str(f1score.item()) + "\n"
    precision = "precision: " + str(precision.item()) + "\n"
    recall = "recall: " + str(recall.item()) + "\n"
    specificity = "specificity: " + str(specificity.item()) + "\n"
    next = ".... next ....\n"
    content = accuracy + f1score + precision + recall + specificity + next
    file.write(content)
    file.close()

def flush_file(model, case):
    fo = open(model+ "/" + case +"/results.txt", "wb")
    fo.flush()
    fo.close()

def get_std_mean_accuracy(model, accuracy, case):
    file = open(model+ "/" + case +"/results.txt", "a+")
    file.seek(0)
    mean = "mean accuracy: " + str(np.mean(accuracy)) + "\n"
    std = "std accuracy: " + str(np.std(accuracy)) + "\n"
    content = mean + std
    file.write(content)
    file.close()

def plot_accuracy_loss(model,trainer,case):
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace = True)
    display(metrics.dropna(axis =1, how = "all").head())
    g = sn.relplot(data=metrics, kind = "line")
    plt.gcf().set_size_inches(12,4)
    plt.savefig(model+"/" +case + "/table.png")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def multiple_boxplot_models(accuracy, case):
    # Pandas dataframe
    data = pd.DataFrame({"CNN1D": accuracy[0], "MLP": accuracy[1], "LSTM": accuracy[2] })
    # Plot the dataframe
    ax = data[['CNN1D', 'MLP', 'LSTM']].plot(kind='box', title='boxplot')
    # Display the plot
    plt.savefig("boxplot/" + case + "/table.png")

def multiple_boxplot_inputs(accuracy):
    # Pandas dataframe
    data = pd.DataFrame({"NO_INTERP": accuracy[2], "NO_INTERP+INDIC": accuracy[3], "INTERP": accuracy[1], "INTERP+INDIC": accuracy[0] })
    # Plot the dataframe
    ax = data[["NO_INTERP", "NO_INTERP+INDIC", "INTERP", "INTERP+INDIC"]].plot(kind='box', title='boxplot')
    # Display the plot
    plt.savefig("boxplot.png")

def get_binary_indicators(x):
    preds = (x < 0).float()
    x = np.where(x<0, -x, x)
    x = np.hstack((x.reshape(-1,1), preds.reshape(-1,1))).reshape(x.shape[1], 2)
    return torch.FloatTensor(x)

def combinations(i):
    switcher = {
        0: "True,True",

        1: "True,False",

        2: "False,False",

        3: "False,True"
    }
    x, y = switcher.get(i).split(",")
    return str2bool(x), str2bool(y)

def plot_isomap(data1, data2):
    embedding = Isomap(n_components = 2)
    dat = pd.concat([data1.iloc[:50, :-1], data2.iloc[:50, :-1]])
    color = pd.concat([data1.iloc[:50, :-1], data2.iloc[:50, :-1]])
    X_iso = embedding.fit_transform(dat)
    plt.figure(figsize = (10, 6))
    plt.scatter(X_iso[:,0], X_iso[:,1], c = color, cmap = plt.cm.rainbow)
    plt.title("Isomap")
    plt.show()

def heatmap(data, baseline):
    plt.figure(figsize = (9,9))
    sn.heatmap(data = data.corr().round(2), cmap = "coolwarm", linewidth = .5, annot = True, annot_kws = {"size":12})
    plt.savefig(baseline + "/heatmap")

