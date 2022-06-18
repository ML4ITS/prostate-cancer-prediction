import sys
sys.path.append('../')
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.utils.fixes import loguniform
from sklearn import tree
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from utils import *


def boxplot(results, names, baseline):
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('How to compare sklearn classification algorithms')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig(baseline + "/classification_algoriths")

def save_evaluation_metric(name, mean, std, baseline):
    file = open(baseline + "/results.txt", "a+")
    msg = "%s: mean %f (std %f) \n\n" % (name, mean, std)
    file.write(msg)
    file.close()

def flush_file(baseline):
    fo = open(baseline + "/results.txt", "wb")
    fo.flush()
    fo.close()

def split_train_test(data):
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
    print("X train shape: " + str(X_train.shape))
    print("X test shape: " + str(X_test.shape))
    return X_train, X_test, y_train, y_test

def get_models():
    models = [('KNN', KNeighborsClassifier()),
              ('DT', DecisionTreeClassifier(random_state=42)),
              ('NB', GaussianNB()),
              ('SVM', SVC(random_state=42)),
              ('ADABOOST', AdaBoostClassifier(random_state=42)),
              ('RF',  RandomForestClassifier(random_state=42))]
    return models

def get_parameters():
    params = []

    params_knn = {
        'leaf_size': list(range(1, 50)),
        'n_neighbors':  list(range(1, 30)),
        'p': [1, 2],
    }

    params_dt = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100],
        'criterion': ["gini", "entropy"]
    }

    params_nb = {
        'var_smoothing': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
    }

    params_svm = {
        'C': [0.1, 1, 100, 1000],
        'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
        'degree': [1, 2, 3, 4, 5, 6]
    }

    params_ada = {
        'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30]
    }

    params_rf = {
        'bootstrap': [True, False],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 15],
        'min_samples_leaf': [3, 4, 5, 6],
        'min_samples_split': [2, 6, 10],
        'n_estimators': [5,20]
    }
    params.append([params_knn, params_dt, params_nb, params_svm, params_ada, params_rf])
    return params

def classifiers(data, baseline):
    X_train, X_test, y_train, y_test = split_train_test(data)
    flush_file(baseline)
    models = get_models()
    params = get_parameters()[0]
    seed = 50
    results = []
    names = []
    scoring = 'accuracy'
    index = 0
    for name, model in models:
        grid_search = GridSearchCV(estimator= model,
                                   param_grid=params[index],
                                   cv=5, n_jobs=-1, verbose=1, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle = True)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        save_evaluation_metric(name, cv_results.mean(), cv_results.std(), baseline)
        feature_importance(X_test, y_test, model, baseline, name)
        index += 1
    boxplot(results, names, baseline)