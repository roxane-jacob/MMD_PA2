from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time

from svm import SequentialSVM, ParallelSVM


def sklearn_svc(X_train, X_test, y_train, y_test):

    start = time.time()
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    accuracy = accuracy_score(y_test, y_predicted)

    return y_predicted, runtime, accuracy


def sequential_linear_svm(X_train, X_test, y_train, y_test, learning_rate, regularization):

    start = time.time()
    clf = SequentialSVM(learning_rate, regularization)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    accuracy = accuracy_score(y_test, y_predicted)

    return y_predicted, runtime, accuracy


def sequential_rff_svm(X_train, X_test, y_train, y_test, learning_rate, regularization):

    start = time.time()
    clf = SequentialSVM(learning_rate, regularization)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    accuracy = accuracy_score(y_test, y_predicted)

    return y_predicted, runtime, accuracy


def parallel_linear_svm(X_train, X_test, y_train, y_test, learning_rate, regularization, num_threads=8):

    start = time.time()
    clf = ParallelSVM(learning_rate, regularization, num_threads)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    accuracy = accuracy_score(y_test, y_predicted)

    return y_predicted, runtime, accuracy


def parallel_rff_svm(X_train, X_test, y_train, y_test, learning_rate, regularization, num_threads=8):

    start = time.time()
    clf = ParallelSVM(learning_rate, regularization, num_threads)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    accuracy = accuracy_score(y_test, y_predicted)

    return y_predicted, runtime, accuracy
