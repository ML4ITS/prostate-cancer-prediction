import numpy as np
import torch
from dataset import *
from sktime.transformations.panel.rocket import Rocket, MiniRocketMultivariate, MiniRocket
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from utils import *


def flush_file(case):
    fo = open(case + "/results.txt", "wb")
    fo.flush()
    fo.close()

def padding(x):
    #the time series are irregular in the number of points so padding is necessary
    max_len = extract_timesteps()
    n_ftrs = x[0].shape[1]
    features = np.zeros((len(x), max_len, n_ftrs))

    for i in range(len(x)):
        j, k = x[i].shape[0], x[i].shape[1]
        features[i] = np.vstack((x[i], np.full((max_len - j, k), -1)))
    return features

def get_models():
    models = [('LR', LogisticRegression()),
              ('RIDGE', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
              ('SVM', SVC()),
              ('RF',  RandomForestClassifier())]
    return models

def save_evaluation_metric(name, mean, std, case):
    file = open("rocket/" + case + "/results.txt", "a+")
    msg = "%s: mean %f (std %f) \n\n" % (name, mean, std)
    file.write(msg)
    file.close()

def boxplot(results, names, case):
    #boxplot to compare the different ML models
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('How to compare rocket performances')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig("rocket/" + case + "/boxplot")

def get_parameters():
    #hyperparamters have been set for each model
    params = []
    params_lr = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
    params_ridge = {'normalize' : [True, False]}
    params_svm = {
        'C': [0.1, 1, 100, 1000],
        'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
        'degree': [1, 2, 3, 4, 5, 6]
    }
    params_rf = {
        'bootstrap': [True, False],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 15],
        'min_samples_leaf': [3, 4, 5, 6],
        'min_samples_split': [2, 6, 10],
        'n_estimators': [5,20]
    }
    params.append([params_lr, params_ridge, params_svm, params_rf])
    return params

def classifiers(x_train, x_test, y_train, y_test, case, iterations):
    #given the training and test set and the different ML models, the prediction have been made
    flush_file("rocket/" + case)
    models = get_models()
    params = get_parameters()[0]
    results = []
    names = []
    index = 0
    for name, model in models:
        grid_search = GridSearchCV(estimator= model,
                                   param_grid=params[index],
                                   cv=5, n_jobs=-1, verbose=1, scoring="accuracy")
        grid_search.fit(x_train, y_train)
        model = grid_search.best_estimator_
        accuracy = []
        for i in range(iterations):
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            accuracy.append(accuracy_score(preds, y_test))
            save_result_df("rocket", y_test, preds, case)
        results.append(accuracy)
        names.append(name)
        save_evaluation_metric(name, np.mean(accuracy), np.std(accuracy), case)
        index += 1
    boxplot(results, names, case)

def rocket_algorithm(case, iterations):
    train = pd.read_csv("../train.csv", header=0, index_col=False)
    test = pd.read_csv("../test.csv", header=0, index_col=False)
    x_train, y_train = manage_db(train)
    x_test, y_test = manage_db(test)
    x_train = padding(x_train)
    x_test = padding(x_test)
    rocket = Rocket()
    rocket.fit(x_train)
    X_train_transform = rocket.transform(x_train)
    X_test_transform = rocket.transform(x_test)
    classifiers(X_train_transform, X_test_transform, y_train, y_test, case, iterations)
