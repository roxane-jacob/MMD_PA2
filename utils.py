import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def load_csv(path):
    data = pd.read_csv(path)
    X, y = np.array(data[['x1', 'x2']]), np.array(data['y'])
    return X, y


def load_mnist(path):
    with np.load(path) as f:
        X_train, y_train = f['train'], f['train_labels']
        X_test, y_test = f['test'], f['test_labels']
        #X = np.concatenate((X_train.T, X_test.T))
        #y = np.concatenate((y_train.flatten().astype(int), y_test.flatten().astype(int)))
        X_train, y_train = shuffle(X_train.T, y_train.flatten().astype(int), random_state=0)
        X_test, y_test = shuffle(X_test.T, y_test.flatten().astype(int), random_state=0)
        #rnd = np.random.RandomState()
        #random_positions = rnd.permutation(X.shape[0])
        #subset_indices = random_positions[:3000]
        #X = X[subset_indices]
        #y = y[subset_indices]

        return X_train, y_train, X_test, y_test


def gridsearch(method, X_train, X_test, y_train, y_test, lr_params, reg_params):
    params = []
    #y_preds = []
    #runtimes = []
    accuracies = []
    for lr in lr_params:
        for reg in reg_params:
            y_pred, runtime, accuracy = method(X_train, X_test, y_train, y_test,
                                               learning_rate=lr, regularization=reg)
            params.append((lr, reg))
            #y_preds.append(y_pred)
            #runtimes.append(runtime)
            accuracies.append(accuracy)
    max_accuracy_index = accuracies.index(max(accuracies))
    #results = {}
    #results['lr'] = params[max_accuracy_index][0]
    #results['reg'] = params[max_accuracy_index][1]
    #results['runtime'] = runtimes[max_accuracy_index]
    #results['accuracy'] = accuracies[max_accuracy_index]
    #results['y_pred'] = y_preds[max_accuracy_index]

    return params[max_accuracy_index]


def five_fold_cross_validation(method, X_train, X_test, y_train, y_test, lr, reg):
    y_pred = None
    runtimes = []
    accuracies = []

    for _ in range(5):
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


def two_dim_visualization(data, labels, path):
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
    plt.figure(figsize=(8, 8))
    plt.scatter(x1, x2, c=labels)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14)
    plt.savefig(path, dpi=150, format='png')
