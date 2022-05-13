import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from svm import NonLinearFeatures


def load_csv(path):
    data = pd.read_csv(path)
    X, y = np.array(data[['x1', 'x2']]), np.array(data['y'])
    return X, y


def load_mnist(path):
    with np.load(path) as f:
        X_train, y_train = f['train'], f['train_labels']
        X_test, y_test = f['test'], f['test_labels']
        X_train, y_train = shuffle(X_train.T, y_train.flatten().astype(int), random_state=0)
        X_test, y_test = shuffle(X_test.T, y_test.flatten().astype(int), random_state=0)

        return X_train, y_train, X_test, y_test


def gridsearch(method, X_train, X_test, y_train, y_test, lr_params, reg_params):
    params = []
    accuracies = []
    for lr in lr_params:
        for reg in reg_params:
            y_pred, runtime, accuracy = method(X_train, X_test, y_train, y_test,
                                               learning_rate=lr, regularization=reg)
            params.append((lr, reg))
            accuracies.append(accuracy)
    max_accuracy_index = accuracies.index(max(accuracies))

    return params[max_accuracy_index]


def gridsearch_rff(method, X_train, X_test, y_train, y_test, lr, reg, m_params, sigma_params):
    params = []
    accuracies = []
    for m in m_params:
        for sigma in sigma_params:
            nlf = NonLinearFeatures(m=m, sigma=sigma)
            X_rff_train = nlf.fit_transform(X_train)
            X_rff_test = nlf.transform(X_test)
            y_pred, runtime, accuracy = method(X_rff_train, X_rff_test, y_train, y_test,
                                               learning_rate=lr, regularization=reg)
            params.append((m, sigma))
            accuracies.append(accuracy)
    max_accuracy_index = accuracies.index(max(accuracies))

    return params[max_accuracy_index]


def five_fold_cross_validation(method, X_train, X_test, y_train, y_test, lr, reg, num_threads=None):
    y_pred = None
    runtimes = []
    accuracies = []

    for _ in range(5):
        if num_threads:
            y_pred, runtime, accuracy = method(X_train, X_test, y_train, y_test,
                                               learning_rate=lr, regularization=reg,
                                               num_threads=num_threads)
        else:
            y_pred, runtime, accuracy = method(X_train, X_test, y_train, y_test,
                                               learning_rate=lr, regularization=reg)
        accuracies.append(accuracy)
        runtimes.append(runtime)

    runtime = sum(runtimes) / len(runtimes)
    accuracy = sum(accuracies) / len(accuracies)

    return y_pred, runtime, accuracy


def sgd_progress(method, X_train, X_test, y_train, y_test, lr, reg):
    _, _, _, progress = method(X_train, X_test, y_train, y_test,
                               learning_rate=lr, regularization=reg, store_sgd_progress=True)

    return progress


def plot_sgd_convergence(convergence_data_linear, convergence_data_rff, path):
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(1, len(convergence_data_linear)+1), convergence_data_linear, label='linear')
    plt.plot(np.arange(1, len(convergence_data_rff) + 1), convergence_data_rff, label='rff')
    plt.xlabel('number of SGD epochs')
    plt.ylabel('SGD training error')
    plt.legend()
    plt.savefig(path, dpi=150)


def two_dim_scatterplot(data, labels, path):
    """
    Create a 2D scatter plot from the given input data with coloring corresponding to the labelling.

    Parameters
    ----------
    data : ndarray of shape (n samples, 2 features)
        Input data.
    labels : list (int)
        List of labels, corresponding to the respective labeling of the data points.
    path : str
        String referring to the path, where the plot shall be stored.
    """

    x1, x2 = data[:, 0], data[:, 1]
    plt.figure(figsize=(10, 10))
    plt.scatter(x1, x2, c=labels)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14)
    plt.savefig(path, dpi=150, format='png')


def plot_parallel_runtimes(number_of_machines_tiny, number_of_machines_large, number_of_machines_mnist,
                           parallel_runtimes_tiny, parallel_runtimes_large, parallel_runtimes_mnist, path):
    plt.figure(figsize=(10, 10))
    plt.plot(number_of_machines_tiny, parallel_runtimes_tiny)
    plt.plot(number_of_machines_large, parallel_runtimes_large)
    plt.plot(number_of_machines_mnist, parallel_runtimes_mnist)
    plt.xlabel('Number of machines')
    plt.ylabel('Runtime in seconds')
    plt.legend(['tiny toydata', 'large toydata', 'MNIST'])
    plt.savefig(path, dpi=150)


def plot_parallel_accuracies(number_of_machines_tiny, number_of_machines_large, number_of_machines_mnist,
                             parallel_accuracies_tiny, parallel_accuracies_large, parallel_accuracies_mnist, path):
    plt.figure(figsize=(10, 10))
    plt.plot(number_of_machines_tiny, parallel_accuracies_tiny)
    plt.plot(number_of_machines_large, parallel_accuracies_large)
    plt.plot(number_of_machines_mnist, parallel_accuracies_mnist)
    plt.xlabel('Number of machines')
    plt.ylabel('Accuracy')
    plt.legend(['tiny toydata', 'large toydata', 'MNIST'])
    plt.savefig(path, dpi=150)
