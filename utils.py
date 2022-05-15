import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from svm import NonLinearFeatures


def load_csv(path):
    """
    Load data samples X and labels y from the MNIST.npz file.

        Parameters
        ----------
        path : str
            String referring to the path, where the plot shall be stored

        Returns
        -------
        X, y : ndarray, ndarray
            data samples and the corresponding labels
    """
    data = pd.read_csv(path)
    X, y = np.array(data[['x1', 'x2']]), np.array(data['y'])
    return X, y


def load_mnist(path):
    """
    Load training and test data from the MNIST.npz file.

        Parameters
        ----------
        path: str
            String referring to the path, where the plot shall be stored.

        Returns
        -------
        X_train, y_train, X_test, y_test : ndarray, ndarray, ndarray, ndarray
            Training data, test data, and the corresponding training labels and test labels
    """
    with np.load(path) as f:
        X_train, y_train = f['train'], f['train_labels']
        X_test, y_test = f['test'], f['test_labels']
        X_train, y_train = shuffle(X_train.T, y_train.flatten().astype(int), random_state=0)
        X_test, y_test = shuffle(X_test.T, y_test.flatten().astype(int), random_state=0)

        return X_train, y_train, X_test, y_test


def gridsearch(method, X_train, X_test, y_train, y_test, lr_params, reg_params):
    """
    Perform a grid search on a given model within a given range of learning rates and regularization parameters.

        Parameters
        ----------
        method : function
            The svm method that should be used. Will be either the sklearn_svc, sequential_svm, or parallel_svm method
            from runner_svm_models.py
        X_train : ndarray of shape (n samples, m features)
            Training data
        X_test : ndarray of shape (n samples, m features)
            Test data
        y_train : ndarray of shape (n samples,)
            Training labels
        y_test : ndarray of shape (n samples,)
            Test labels
        lr_params : list (float)
            List of learning rates for the SGD algorithm
        reg_params : list (float)
            List of regularization parameters for the SGD algorithm

        Returns
        -------
        params[max_accuracy_index] : (float, float)
            Tuple containing the (learning rate, regularization parameter) that achieved the highest accuracy in the
            grid search run
    """
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
    """
    Perform a grid search on a given model within a given range of learning rates and regularization parameters.

        Parameters
        ----------
        method : function
            The svm method that should be used. Will be either the sklearn_svc, sequential_svm, or parallel_svm method
            from runner_svm_models.py
        X_train : ndarray of shape (n samples, m features)
            Training data
        X_test : ndarray of shape (n samples, m features)
            Test data
        y_train : ndarray of shape (n samples,)
            Training labels
        y_test : ndarray of shape (n samples,)
            Test labels
        lr : float
            Learning rate of the SGD algorithm
        reg : float
            Regularization parameter of the SGD algorithm
        m_params : list (float)
            List of sample dimensions m for the transformation to RFF
        sigma_params : list (float)
            List of sigma parameters for the transformation to RFF

        Returns
        -------
        params[max_accuracy_index] : (float, float)
            Tuple containing the (m, sigma) that achieved the highest accuracy in the
            grid search run
    """
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
    """
    Perform five runs of the given method and return the predicted labels, the mean of the runtime and the mean of the
    calculated accuracies.

        Parameters
        ----------
        method : function
            The svm method that should be used. Will be either the sklearn_svc, sequential_svm, or parallel_svm method
            from runner_svm_models.py
        X_train : ndarray of shape (n samples, m features)
            Training data
        X_test : ndarray of shape (n samples, m features)
            Test data
        y_train : ndarray of shape (n samples,)
            Training labels
        y_test : ndarray of shape (n samples,)
            Test labels
        lr : float
            Learning rate of the SGD algorithm
        reg : float
            Regularization parameter of the SGD algorithm
        num_threads : int, optional
            Number of splits that the parallel SGD algorithm will work on. If empty, the non-parallel SVC will be used.

        Returns
        -------
        y_pred, runtime, accuracy : ndarray, list(float), list(float)
            Predicted labels, runtime, and accuracy of the predicted labels on the test set.
    """
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
    """
    Run an SVC model on the given data and return the progress of the SGD solver for the given set of parameters.

        Parameters
        ----------
        method : function
            The svm method that should be used. Will be the sklearn_svc method from runner_svm_models.py
        X_train : ndarray of shape (n samples, m features)
            Training data
        X_test : ndarray of shape (n samples, m features)
            Test data
        y_train : ndarray of shape (n samples,)
            Training labels
        y_test : ndarray of shape (n samples,)
            Test labels
        lr : float
            Learning rate of the SGD algorithm
        reg : float
            Regularization parameter of the SGD algorithm

        Returns
        -------
        progress : list(flaot)
            List containing the progress of the SGD solver towards the weights vector of its last iteration.
    """
    _, _, _, progress = method(X_train, X_test, y_train, y_test,
                               learning_rate=lr, regularization=reg, store_sgd_progress=True)

    return progress


def plot_sgd_convergence(convergence_data_linear, convergence_data_rff, path):
    """
    Create a 2D plot from the given input data depicting the convergence of the SGD solver versus the number of
    iterations.

        Parameters
        ----------
        convergence_data_linear : list(float)
        convergence_data_rff : list(flaot)
        path : str
            String referring to the path, where the plot shall be stored.

        Returns
        -------
        None
    """
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(1, len(convergence_data_linear)+1), convergence_data_linear, label='linear')
    plt.plot(np.arange(1, len(convergence_data_rff) + 1), convergence_data_rff, label='rff')
    plt.xlabel('number of SGD epochs', fontsize=16)
    plt.ylabel('SGD training error', fontsize=16)
    plt.legend(fontsize=16)
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

        Returns
        -------
        None
    """

    x1, x2 = data[:, 0], data[:, 1]
    plt.figure(figsize=(10, 10))
    plt.scatter(x1, x2, c=labels, cmap='coolwarm')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')
    plt.xlabel(r"$x_1$", fontsize=16)
    plt.ylabel(r"$x_2$", fontsize=16)
    plt.savefig(path, dpi=150, format='png')


def plot_parallel_runtimes(number_of_machines_tiny, number_of_machines_large, number_of_machines_mnist,
                           parallel_runtimes_tiny, parallel_runtimes_large, parallel_runtimes_mnist, path):
    """
    Create a 2D plot from the given input data of the runtimes of the parallelized SVC versus the number of machines.

        Parameters
        ----------
        number_of_machines_tiny : list(int)
        number_of_machines_large : list(int)
        number_of_machines_mnist : list(int)
        parallel_runtimes_tiny : list(float)
        parallel_runtimes_large : list(float)
        parallel_runtimes_mnist : list(float)
        path : str
            String referring to the path, where the plot shall be stored.

        Returns
        -------
        None
    """
    plt.figure(figsize=(10, 10))
    plt.plot(number_of_machines_tiny, parallel_runtimes_tiny)
    plt.plot(number_of_machines_large, parallel_runtimes_large)
    plt.plot(number_of_machines_mnist, parallel_runtimes_mnist)
    plt.xlabel('Number of machines', fontsize=16)
    plt.xticks(number_of_machines_tiny)
    plt.ylabel('Runtime in seconds', fontsize=16)
    plt.legend(['tiny toydata', 'large toydata', 'MNIST'], fontsize=16)
    plt.savefig(path, dpi=150)


def plot_parallel_accuracies(number_of_machines_tiny, number_of_machines_large, number_of_machines_mnist,
                             parallel_accuracies_tiny, parallel_accuracies_large, parallel_accuracies_mnist, path):
    """
    Create a 2D plot from the given input data of the accuracies of the parallelized SVC versus the number of machines.

        Parameters
        ----------
        number_of_machines_tiny : list(int)
        number_of_machines_large : list(int)
        number_of_machines_mnist : list(int)
        parallel_accuracies_tiny : list(float)
        parallel_accuracies_large : list(float)
        parallel_accuracies_mnist : list(float)
        path : str
            String referring to the path, where the plot shall be stored.

        Returns
        -------
        None
    """
    plt.figure(figsize=(10, 10))
    plt.plot(number_of_machines_tiny, parallel_accuracies_tiny)
    plt.plot(number_of_machines_large, parallel_accuracies_large)
    plt.plot(number_of_machines_mnist, parallel_accuracies_mnist)
    plt.xlabel('Number of machines', fontsize=16)
    plt.xticks(number_of_machines_tiny)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(['tiny toydata', 'large toydata', 'MNIST'], fontsize=16)
    plt.savefig(path, dpi=150)
