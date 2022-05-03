import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time

from utils import two_dim_visualization
from svm import LinearSequentialSVM, LinearParallelSVM, RFFSequentialSVM

if __name__ == '__main__':

    # toydata = pd.read_csv('data/toydata_tiny.csv')
    # X, y = np.array(toydata[['x1', 'x2']]), np.array(toydata['y'])

    toydata = pd.read_csv('data/toydata_large.csv')
    X, y = np.array(toydata[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]), np.array(toydata['y'])

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    """
    # load data
    data = np.load("mnist.npz")
    X, y = np.array(data["train"]), np.array(data["train_labels"])
    X_test, y_test = np.array(data["test"]), np.array(data["test_labels"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=10000, train_size=50000, random_state=42)
    """

    # set parameters
    learning_rate = 1e-1
    regularization = 1e-2
    num_threads = 10
    print(f'Learning rate: {learning_rate} \nRegularization: {regularization} \nNumber of threads: {num_threads}')

    # fit and predict linear sequential
    print('\n--- Linear Sequential SVM ---')
    start = time.time()
    linear_sequential_svm = LinearSequentialSVM(learning_rate, regularization)
    linear_sequential_svm.fit(X_train, y_train)
    y_predicted_linear_sequential = linear_sequential_svm.predict(X_test)
    end = time.time()
    print("Runtime fit and predict:", end - start)
    print("Accuracy: {}".format(accuracy_score(y_test, y_predicted_linear_sequential)))

    # fit and predict linear parallel
    print('\n--- Linear Parallel SVM ---')
    start = time.time()
    linear_parallel_svm = LinearParallelSVM(learning_rate, regularization, num_threads)
    linear_parallel_svm.fit(X_train, y_train)
    y_predicted_linear_parallel = linear_parallel_svm.predict(X_test)
    end = time.time()
    print("Runtime fit and predict::", end - start)
    print("Accuracy: {}".format(accuracy_score(y_test, y_predicted_linear_parallel)))
    """
    # fit and predict rff sequential
    print('\n--- RFF Sequential SVM ---')
    start = time.time()
    rff_sequential_svm = RFFSequentialSVM(learning_rate, regularization)
    rff_sequential_svm.fit(X_train, y_train)
    y_predicted_rff_sequential = rff_sequential_svm.predict(X_test)
    end = time.time()
    print("Runtime fit and predict:", end - start)
    print("Accuracy: {}".format(accuracy_score(y_test, y_predicted_rff_sequential)))
    """
    # fit and predict sklearn SVC
    print('\n--- Sklearn SVC ---')
    start = time.time()
    sklearn_svc = SVC()
    sklearn_svc.fit(X_train, y_train)
    y_predicted_sklearn_svc = sklearn_svc.predict(X_test)
    end = time.time()
    print("Elapsed time fit/predict:", end - start)
    print("Accuracy: {}".format(accuracy_score(y_test, y_predicted_sklearn_svc)))

    # Plot true labels
    # two_dim_visualization(X, y, 'True Labels', 'true')
    # Plot predicted labels
    # two_dim_visualization(X, y_predicted, 'Predicted Labels', 'predicted')
