from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
import ray

from svm import SequentialSVM, ParallelSVM


def sklearn_svc(X_train, X_test, y_train, y_test):
    """
    Train the SVC of the sklearn implementation, predict labels, and calculate the accuracy on given test set.

        Parameters
        ----------
        X_train : ndarray of shape (n samples, m features)
            Training data
        X_test : ndarray of shape (n samples, m features)
            Test data
        y_train : ndarray of shape (n samples,)
            Training labels
        y_test : ndarray of shape (n samples,)
            Test labels

        Returns
        -------
        y_predicted, runtime, accuracy : (ndarray of shape (n samples,), list(float), list(float))
            Predicted labels, runtime, and accuracy of the predicted labels on the test set.
    """
    start = time.time()
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    accuracy = accuracy_score(y_test, y_predicted)

    return y_predicted, runtime, accuracy


def sequential_svm(X_train, X_test, y_train, y_test, learning_rate, regularization, store_sgd_progress=False):
    """
    Train the own implementation of an SVC, predict labels, and calculate the accuracy on given test set.

        Parameters
        ----------
        X_train : ndarray of shape (n samples, m features)
            Training data
        X_test : ndarray of shape (n samples, m features)
            Test data
        y_train : ndarray of shape (n samples,)
            Training labels
        y_test : ndarray of shape (n samples,)
            Test labels
        learning_rate : float
            Learning rate of the SGD algorithm
        regularization : float
            Regularization parameter of the SGD algorithm
        store_sgd_progress : bool, optional
            If true, a list reporting the progress of the SGD solver is returned

        Returns
        -------
        y_predicted, runtime, accuracy : (ndarray of shape (n samples,), list(float), list(float))
            Predicted labels, runtime, and accuracy of the predicted labels on the test set.
        sgd_progress : list(float), optional
             A list reporting the progress of the SGD solver
    """
    start = time.time()
    clf = SequentialSVM(learning_rate, regularization, store_sgd_progress=store_sgd_progress)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    accuracy = accuracy_score(y_test, y_predicted)

    if store_sgd_progress:
        sgd_progress = clf.get_sgd_progress()
        return y_predicted, runtime, accuracy, sgd_progress

    else:
        return y_predicted, runtime, accuracy


def parallel_svm(X_train, X_test, y_train, y_test, learning_rate, regularization, num_threads=4):
    """
    Train the own parallel implementation of an SVC, predict labels, and calculate the accuracy on given test set.

        Parameters
        ----------
        X_train : ndarray of shape (n samples, m features)
            Training data
        X_test : ndarray of shape (n samples, m features)
            Test data
        y_train : ndarray of shape (n samples,)
            Training labels
        y_test : ndarray of shape (n samples,)
            Test labels
        learning_rate : float
            Learning rate of the SGD algorithm
        regularization : float
            Regularization parameter of the SGD algorithm
        num_threads : int
            Number of splits that the parallel SGD algorithm will work on

        Returns
        -------
        y_predicted, runtime, accuracy : ndarray, list(float), list(float)
            Predicted labels, runtime, and accuracy of the predicted labels on the test set.
    """
    start = time.time()
    clf = ParallelSVM(learning_rate, regularization, num_threads)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    accuracy = accuracy_score(y_test, y_predicted)

    return y_predicted, runtime, accuracy
