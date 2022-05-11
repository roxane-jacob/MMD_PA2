from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time

from svm import SequentialSVM, ParallelSVM


def sklearn_svc(X_train, X_test, y_train, y_test):

    print('\n--- Sklearn SVC ---')
    start = time.time()
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    print(f"Elapsed time fit/predict: {runtime}")
    accuracy = accuracy_score(y_test, y_predicted)
    print(f"Accuracy: {accuracy}")

    return y_predicted, runtime, accuracy


def sequential_linear_svm(X_train, X_test, y_train, y_test, learning_rate, regularization):

    print('\n--- Sequential Linear SVM ---')
    print(f'Learning rate: {learning_rate} 'f'\nRegularization: {regularization}')

    # fit and predict linear sequential
    start = time.time()
    clf = SequentialSVM(learning_rate, regularization)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    print(f"Elapsed time fit/predict: {runtime}")
    accuracy = accuracy_score(y_test, y_predicted)
    print(f"Accuracy: {accuracy}")

    return y_predicted, runtime, accuracy


def sequential_rff_svm(X_train, X_test, y_train, y_test, learning_rate, regularization):

    print('\n--- Sequential RFF SVM ---')
    print(f'Learning rate: {learning_rate} 'f'\nRegularization: {regularization}')

    # fit and predict rff sequential
    start = time.time()
    clf = SequentialSVM(learning_rate, regularization)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    print(f"Elapsed time fit/predict: {runtime}")
    accuracy = accuracy_score(y_test, y_predicted)
    print(f"Accuracy: {accuracy}")

    return y_predicted, runtime, accuracy


def parallel_linear_svm(X_train, X_test, y_train, y_test, learning_rate, regularization, num_threads):

    print('\n--- Parallel Linear SVM ---')
    print(f'Learning rate: {learning_rate} '
          f'\nRegularization: {regularization} '
          f'\nNumber of threads: {num_threads}')

    # fit and predict linear parallel
    start = time.time()
    clf = ParallelSVM(learning_rate, regularization, num_threads)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    print(f"Elapsed time fit/predict: {runtime}")
    accuracy = accuracy_score(y_test, y_predicted)
    print(f"Accuracy: {accuracy}")

    return y_predicted, runtime, accuracy


def parallel_rff_svm(X_train, X_test, y_train, y_test, learning_rate, regularization, num_threads):

    print('\n--- Parallel RFF SVM ---')
    print(f'Learning rate: {learning_rate} '
          f'\nRegularization: {regularization} '
          f'\nNumber of threads: {num_threads}')

    # fit and predict rff parallel
    start = time.time()
    clf = ParallelSVM(learning_rate, regularization, num_threads)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    end = time.time()
    runtime = end - start
    print(f"Elapsed time fit/predict: {runtime}")
    accuracy = accuracy_score(y_test, y_predicted)
    print(f"Accuracy: {accuracy}")

    return y_predicted, runtime, accuracy
