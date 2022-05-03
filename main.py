import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time

from utils import two_dim_visualization
from svm import LinearSVM, LinearSVMParallel

if __name__ == '__main__':

    # load data

    # toydata = pd.read_csv('data/toydata_tiny.csv')
    # X, y = np.array(toydata[['x1', 'x2']]), np.array(toydata['y'])

    toydata = pd.read_csv('data/toydata_large.csv')
    X, y = np.array(toydata[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]), np.array(toydata['y'])

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # set parameters
    learning_rate = 1e-1
    regularization = 1e-2
    n = 2  # number of machines
    print(f'Learning rate: {learning_rate} \nRegularization: {regularization} \nNumber of machines: {n}')

    # fit and predict linear non-parallel
    print('\n--- Linear Sequential SVM ---')
    start = time.time()
    linear_sgd = LinearSVM(learning_rate, regularization)
    y_predicted = linear_sgd.fit_predict(X, y)
    end = time.time()
    print("Elapsed time fit/predict:", end - start)
    print("Accuracy: {}".format(accuracy_score(y, y_predicted)))

    # fit and predict linear parallel
    print('\n--- Linear Parallel SVM ---')
    start = time.time()
    linear_sgd = LinearSVMParallel(learning_rate, regularization)
    y_predicted = linear_sgd.fit_predict(X, y, n)
    end = time.time()
    print("Elapsed time fit/predict:", end - start)
    print("Accuracy: {}".format(accuracy_score(y, y_predicted)))

    # fit and predict sklearn SVC
    print('\n--- Sklearn SVC ---')
    start = time.time()
    sklearn_svc = SVC()
    sklearn_svc.fit(X_train, y_train)
    y_predicted = sklearn_svc.predict(X_test)
    end = time.time()
    print("Elapsed time fit/predict:", end - start)
    print("Accuracy: {}".format(accuracy_score(y_test, y_predicted)))

    # Plot true labels
    # two_dim_visualization(X_PCA, y, 'True Labels', 'true')
    # Plot predicted labels
    # two_dim_visualization(X_PCA, y_predicted, 'Predicted Labels', 'predicted')
