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
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('How to compare rocket performances')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig("rocket/" + case + "/boxplot")

def get_parameters():
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
            accuracy.append(accuracy_score(model.predict(x_test), y_test))
        results.append(accuracy)
        names.append(name)
        save_evaluation_metric(name, np.mean(accuracy), np.std(accuracy), case)
        index += 1
    boxplot(results, names, case)

def rocket_algorithm(data, case, iterations):
    x, y = manage_db(data)
    x = padding(x)
    print(x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, shuffle=True)
    rocket = Rocket()
    rocket.fit(x_train)
    X_train_transform = rocket.transform(x_train)
    X_test_transform = rocket.transform(x_test)
    classifiers(X_train_transform, X_test_transform, y_train, y_test, case, iterations)













#
# #----------------------------#
# data = pd.read_csv("data_model.csv")
# x = np.array(data.iloc[:, :-1])
# y = np.array(data.iloc[:, -1])
#
# # every time I call train_test_split, test and train should be different
# x, test, y, test_y = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=None)
# # print(x.shape)
# # print(y.shape)
# # rocket = Rocket(num_kernels=112)
# # rocket.fit(x)
# # x = rocket.transform(x)
# # print(x.shape)
# # exit()
# # classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
# # classifier.fit(x, y)
# # test = rocket.transform(test)
# # predictions = classifier.predict(test)
#
# kernels = generate_kernels(x.shape[-1], 10_000)
#
# # transform training set and train classifier
# X_training_transform = apply_kernels(x, kernels)
# classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
# classifier.fit(X_training_transform, y)
#
# # transform test set and predict
# X_test_transform = apply_kernels(test, kernels)
# predictions = classifier.predict(X_test_transform)
# print(accuracy_score(predictions, test_y))
