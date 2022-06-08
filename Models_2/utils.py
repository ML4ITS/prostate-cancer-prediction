import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sn
import matplotlib.pyplot as plt

def get_random_numbers(layers, trial, min, max, element, int = True, desc = True):
    random_float_list = []
    for i in range(layers+1):
        el = element + str(i)
        #todo add step
        if int is True:
            x = trial.suggest_int(el, min, max)
        else:
            x= trial.suggest_uniform(el, min, max)
        random_float_list.append(x)
    return np.sort(np.array(random_float_list))[::-1] if desc else np.sort(np.array(random_float_list))

def save_evaluation_metric(model, accuracy, f1score, precision, recall, specificity):

    with  open(model+"/results.txt", "w") as file:
        accuracy = "accuracy: " + str(accuracy.item()) + "\n"
        f1score = "f1score: " + str(f1score.item()) + "\n"
        precision = "precision: " + str(precision.item()) + "\n"
        recall = "recall: " + str(recall.item()) + "\n"
        specificity = "specificity: " + str(specificity.item()) + "\n"
        content = accuracy + f1score + precision + recall + specificity
        file.write(content)
        file.close()


def plot_accuracy_loss(model,trainer):
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace = True)
    display(metrics.dropna(axis =1, how = "all").head())
    g = sn.relplot(data=metrics, kind = "line")
    plt.gcf().set_size_inches(12,4)
    plt.savefig(model+"/table.png")
