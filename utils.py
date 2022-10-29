import numpy as np
import pandas as pd
import torch
from IPython.display import display
import seaborn as sn
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
from sklearn.metrics import *
import functools
from sklearn.manifold import Isomap
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from settings import p

dispatcher= {'nn.Tanh()': nn.Tanh(), 'nn.ReLU()': nn.ReLU(), 'nn.ELU()': nn.ELU(), 'nn.LeakyReLU()': nn.LeakyReLU(), 'nn.Sigmoid()': nn.Sigmoid()}

def get_random_numbers(layers, trial, min, max, element, int = True, desc = True, step = 1):
    #hyperparameter tuning needs to generate random numbers for the number of layers, hidden neurons, dropout etc.
    #return the list of the generated values for a given hyperparameter
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
    #return the kernel size from the given parameters: input size, output size and padding
    return 2 * padding + input_size + 1 - output_size

def save_evaluation_metric(model, accuracy, f1score, specificity, case):
    #accuracy, f1score and specificity are saved in a file for a given model
    file = open(model+ "/" + case +"/results.txt", "a+")
    accuracy = "accuracy: " + str(accuracy.item()) + "\n"
    f1score = "f1score: " + str(f1score.item()) + "\n"
    specificity = "specificity: " + str(specificity.item()) + "\n"
    next = ".... next ....\n"
    content = accuracy + f1score + specificity + next
    file.write(content)
    file.close()

def save_hyperparameter(model, study, case):
    #the selected hyperparameters are saved in a file
    file = open(model + case + "/results.txt", "a+")
    content = study.best_params
    file.write(str(content) + "\n")
    file.close()

def flush_file(model, case):
    #clear the internal buffer of the file
    fo = open(model+ "/" + case +"/results.txt", "wb")
    fo.flush()
    fo.close()

def get_std_mean_accuracy(model, accuracy, case):
    #the mean and the standard deviation for the accuracy have been saved in the file
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
    #the function converts the given input into a boolean value
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def multiple_boxplot_models(accuracy, case):
    #bloxplot using accuracy paramaters for the irregular time series approach
    # Pandas dataframe
    data = pd.DataFrame({"CASE1": accuracy[0], "CASE2": accuracy[1], "CASE3": accuracy[2], "CASE4": accuracy[3]})
    # Plot the dataframe
    ax = data[["CASE1", "CASE2", "CASE3", "CASE4"]].plot(kind='box', title='boxplot')
    plt.show()
    # Display the plot
    # plt.savefig("boxplot/" + case + "/table.png")

def multiple_boxplot_inputs(accuracy, path):
    #bloxplot using accuracy paramaters for the time series regularization approach
    # Pandas dataframe
    data = pd.DataFrame({"NO_INTERP": accuracy[2], "NO_INTERP+INDIC": accuracy[3], "INTERP": accuracy[1], "INTERP+INDIC": accuracy[0] })
    # Plot the dataframe
    ax = data[["NO_INTERP", "NO_INTERP+INDIC", "INTERP", "INTERP+INDIC"]].plot(kind='box', title='boxplot')
    # Display the plot
    plt.title(path)
    plt.savefig(path + "_boxplot.png")

def get_binary_indicators(x):
    #given the input, the negative values indicate the not real values while the positive values the real values.
    #then the binary missing vector is extracted
    preds = (x < 0).float()
    if p.interpolation:
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

def plot_isomap(data1, data2, baseline):
    #the function plot the isomap to understand the distribution of the data
    embedding = Isomap(n_components = 2)
    dat = pd.concat([data1.iloc[:50, :-1], data2.iloc[:50, :-1]])
    color = pd.concat([data1.iloc[:50, -1], data2.iloc[:50, -1]])
    X_iso = embedding.fit_transform(dat)
    plt.figure(figsize = (10, 6))
    plt.scatter(X_iso[:,0], X_iso[:,1], c = color, cmap = plt.cm.rainbow)
    plt.title("Isomap")
    plt.savefig(baseline + "/isomap")

def heatmap(data, baseline):
    #the function plots the heatmap
    plt.figure(figsize = (9,9))
    sn.heatmap(data = data.corr().round(2), cmap = "coolwarm", linewidth = .5, annot = True, annot_kws = {"size":12})
    plt.savefig(baseline + "/heatmap")

def feature_importance(X_test, y_test, model, baseline, name):
    #the function calls the permutation importance algorithm to understand which features are the most important for the prediction of the test set
    feature_names = [f"feature{i}" for i in range(X_test.shape[1])]
    result = permutation_importance(model, X_test, y_test, n_repeats = 10, random_state = 42, n_jobs = 2)
    importances = pd.Series(result.importances_mean, index = feature_names)
    fig, ax = plt.subplots()
    importances.plot.bar(yerr = result.importances_std, ax = ax)
    ax.set_title("feature importances using permutation on full model")
    ax.set_ylabel("mean accuracy decrease")
    fig.tight_layout()
    plt.savefig(baseline + "/feature_importance_"+ name)

def conf_matrix_categ(file, df, model, feature1, feature2, inf, sup, case, feature3 = None, feature4 = None):
    #the results (f1score, confusion mantrix, accuracy) are saved in a file for each type categories
    mean, median, mode = None, None, None
    if sup == -1:
        c = df.loc[(df[feature1] == inf)]
    else:
        if feature3 is not None and feature4 is not None:
            c = df.loc[(df[feature1] >= inf) & (df[feature1] < sup) & (df[feature2] < feature4) & (df[feature2] >= feature3)]
        else:
            c = df.loc[(df[feature1] >= inf) & (df[feature2] < sup)]
    if c.empty is not True:
        cm = confusion_matrix(c["target"], c["pred"])
        f1 = f1_score(c["target"], c["pred"], average='macro')
        accuracy = accuracy_score(c["target"], c["pred"])
        if feature3 is None:
            file.write(str(inf) + "<=" + feature1 + "<" + str(sup) +"\taccuracy: "+ str(accuracy) + "\tf1score: "+ str(f1) + "\n")
        else:
            mean, median, mode = c["visits"].mean(), c["visits"].median(), c["visits"].mode()[0]
            file.write(str(inf) + "<=" + feature1 + "<" + str(sup)  + "  "+ str(feature3) + "<=" + feature2 + "<" + str(feature4) +"\taccuracy: " + str(accuracy) + "\tf1score: " + str(f1) + "\n")
            file.write("visits: \t mean: " + str(mean) + "\tmedian: "+ str(median) + "\tmode: "+ str(mode) + "\n")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        if feature3 is None:
            disp.figure_.savefig(model + "/" + case + "/categories/conf_mat" + str(inf) + "_" + feature1 + "_" + str(sup)+ ".png", dpi=300)
        else:
            disp.figure_.savefig(model + "/" + case + "/categories/conf_mat_age_" + str(inf) +  "_" + str(feature4) + ".png",dpi=300)
        plt.clf()
        plt.cla()
    else:
        file.write("No results for this category\n")

def save_result_df(model, target, preds, case):
    #the performance is extracted using the different categories: visits, age and cancer risk
    file = open(model + "/" + case + "/categories/results.txt", "w")
    df = pd.read_csv("df.csv")
    df["target"] = np.array(target).astype(int)
    df["pred"] = np.array(preds).astype(int)

    #visit < 10 10 - 20 > 20
    conf_matrix_categ(file, df, model, "visits", "visits", 0, 10, case)
    conf_matrix_categ(file, df, model, "visits", "visits", 10, 20, case)
    conf_matrix_categ(file, df, model, "visits", "visits", 20, 30, case)
    conf_matrix_categ(file, df, model, "visits", "visits", 30, 40, case)
    conf_matrix_categ(file, df, model, "visits", "visits", 40, 500, case)
    #age 30-50 50-70 70-100
    conf_matrix_categ(file, df, model, "min_age", "max_age", 30, 45, case, feature3=55,feature4=65)
    conf_matrix_categ(file, df, model, "min_age", "max_age", 30, 45, case, feature3=65, feature4=75)
    conf_matrix_categ(file, df, model, "min_age", "max_age", 30, 45, case, feature3=75, feature4=85)
    conf_matrix_categ(file, df, model, "min_age", "max_age", 45, 55, case, feature3=55, feature4=65)
    conf_matrix_categ(file, df, model, "min_age", "max_age", 45, 55, case, feature3=65, feature4=75)
    conf_matrix_categ(file, df, model, "min_age", "max_age", 45, 55, case, feature3=75, feature4=85)
    conf_matrix_categ(file, df, model, "min_age", "max_age", 55, 65, case, feature3=65, feature4=75)
    conf_matrix_categ(file, df, model, "min_age", "max_age", 55, 65, case, feature3=75, feature4=85)
    conf_matrix_categ(file, df, model, "min_age", "max_age", 65, 75, case, feature3=75, feature4=85)
    #risk category
    df = df.loc[(df["target"] == 1)]
    for i in range(6):
        conf_matrix_categ(file, df, model, "category", "category", i, -1, case)
    file.close()




