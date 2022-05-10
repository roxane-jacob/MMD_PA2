import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time

from svm import SequentialSVM, ParallelSVM, NonLinearFeatures
from utils import two_dim_visualization


def load_data(path):
    with np.load(path) as f:
        X_train, y_train = f['train'], f['train_labels']
        X_test, y_test = f['test'], f['test_labels']
        X = np.concatenate((X_train.T, X_test.T))
        y = np.concatenate((y_train.flatten().astype(int), y_test.flatten().astype(int)))
        return X, y

if __name__ == '__main__':

    # toydata = pd.read_csv('data/toydata_tiny.csv')
    # X, y = np.array(toydata[['x1', 'x2']]), np.array(toydata['y'])

    # toydata = pd.read_csv('data/toydata_large.csv')
    # X, y = np.array(toydata[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]), np.array(toydata['y'])

    X, y = load_data('data/mnist.npz')

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    #train_subset = np.linspace(0, 59999, 1000).astype(int)
    #test_subset = np.linspace(0, 9999, 200).astype(int)
    #X_train = X_train[train_subset]
    #y_train = y_train[train_subset]
    #X_test = X_test[test_subset]
    #y_test = y_test[test_subset]
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
    num_threads = 8
    print(f'Learning rate: {learning_rate} '
          f'\nRegularization: {regularization} '
          f'\nNumber of threads for parallel implementation: {num_threads}')

    # fit and predict sklearn SVC
    #print('\n--- Sklearn SVC ---')
    #start = time.time()
    #sklearn_svc = SVC()
    #sklearn_svc.fit(X_train, y_train)
    #y_predicted_sklearn_svc = sklearn_svc.predict(X_test)
    #end = time.time()
    #print("Elapsed time fit/predict:", end - start)
    #print("Accuracy: {}".format(accuracy_score(y_test, y_predicted_sklearn_svc)))

    # fit and predict linear sequential
    print('\n--- Linear Sequential SVM ---')
    start = time.time()
    linear_sequential_svm = SequentialSVM(learning_rate=learning_rate,
                                          regularization_parameter=regularization)
    linear_sequential_svm.fit(X_train, y_train)
    y_predicted_linear_sequential = linear_sequential_svm.predict(X_test)
    end = time.time()
    print("Runtime fit and predict:", end - start)
    print("Accuracy: {}".format(accuracy_score(y_test, y_predicted_linear_sequential)))

    # fit and predict linear parallel
    print('\n--- Linear Parallel SVM ---')
    start = time.time()
    linear_parallel_svm = ParallelSVM(learning_rate=learning_rate,
                                      regularization_parameter=regularization,
                                      num_threads=num_threads)
    linear_parallel_svm.fit(X_train, y_train)
    y_predicted_linear_parallel = linear_parallel_svm.predict(X_test)
    end = time.time()
    print("Runtime fit and predict::", end - start)
    print("Accuracy: {}".format(accuracy_score(y_test, y_predicted_linear_parallel)))

    # Create non-linear features
    print('\n--- Compute non-linear features ---')
    start = time.time()
    nlf = NonLinearFeatures(m=20, sigma=2.0)
    X = nlf.fit_transform(X)
    end = time.time()
    print(f'Runtime transformation to non-linear features: {end-start}')

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # fit and predict rff sequential
    print('\n--- RFF Sequential SVM ---')
    start = time.time()
    rff_sequential_svm = SequentialSVM(learning_rate, regularization)
    rff_sequential_svm.fit(X_train, y_train)
    y_predicted_rff_sequential = rff_sequential_svm.predict(X_test)
    end = time.time()
    print("Runtime fit and predict:", end - start)
    print("Accuracy: {}".format(accuracy_score(y_test, y_predicted_rff_sequential)))

    # fit and predict rff parallel
    print('\n--- RFF Parallel SVM ---')
    start = time.time()
    rff_parallel_svm = ParallelSVM(learning_rate, regularization)
    rff_parallel_svm.fit(X_train, y_train)
    y_predicted_rff_parallel = rff_parallel_svm.predict(X_test)
    end = time.time()
    print("Runtime fit and predict:", end - start)
    print("Accuracy: {}".format(accuracy_score(y_test, y_predicted_rff_parallel)))

    # Plot true labels
    # two_dim_visualization(X, y, 'True Labels', 'true')
    # Plot predicted labels
    # two_dim_visualization(X, y_predicted, 'Predicted Labels', 'predicted')
