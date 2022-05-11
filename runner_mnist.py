import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from svm import NonLinearFeatures
from runner_svm_models import sklearn_svc, sequential_linear_svm, sequential_rff_svm, parallel_linear_svm, parallel_rff_svm
from utils import load_mnist


def runner_mnist(path):

    print(f'\n---------- Running Procedure on {path} ----------')

    X, y = load_mnist(path)

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # compute baseline accuracy (dummy classifier)
    dummy_clf = DummyClassifier(strategy='stratified')
    dummy_clf.fit(X_train, y_train)
    print("\nBaseline Accuracy: {}".format(dummy_clf.score(X_test, y_test)))

    # run sequential linear svm
    _ = sequential_linear_svm(X_train, X_test, y_train, y_test, learning_rate=1e-1, regularization=1e-2)

    # ---------- Compute RFF Features ----------

    # Create RFF features
    print('\n--- Compute RFF features ---')
    start = time.time()
    nlf = NonLinearFeatures(m=1000, sigma=2.0)
    X_rff = nlf.fit_transform(X)
    end = time.time()
    print(f'Runtime transformation to RFF features: {end - start}')

    # train/test split of RFF features
    X_rff_train, X_rff_test, y_train, y_test = train_test_split(X_rff, y, random_state=42)

    # run sequential RFF svm
    _ = sequential_rff_svm(X_rff_train, X_rff_test, y_train, y_test, learning_rate=1e-1, regularization=1e-2)

    # run parallel linear svm
    _ = parallel_linear_svm(X_train, X_test, y_train, y_test, learning_rate=1e-1, regularization=1e-2, num_threads=8)

    # run parallel RFF svm
    _ = parallel_rff_svm(X_rff_train, X_rff_test, y_train, y_test, learning_rate=1e-1, regularization=1e-2, num_threads=8)

    # make plots for runtime/performance comparison when training on 1000, 2000, 3000 training samples
    # and using own implementation vs. sklearn's svm.svc

    training_size = [1000, 2000, 3000]
    runtimes_sequential_rff = []
    accuracies_sequential_rff = []
    runtimes_sklearn = []
    accuracies_sklearn = []

    for size in training_size:
        _, runtime, accuracy = sequential_rff_svm(X_rff_train[:size], X_rff_test, y_train[:size], y_test,
                                                  learning_rate=1e-1, regularization=1e-2)
        runtimes_sequential_rff.append(runtime)
        accuracies_sequential_rff.append(accuracy)

        _, runtime, accuracy = sklearn_svc(X_rff_train[:size], X_rff_test, y_train[:size], y_test)
        runtimes_sklearn.append(runtime)
        accuracies_sklearn.append(accuracy)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(training_size, runtimes_sequential_rff, label='sequential RFF')
    axs[0].plot(training_size, runtimes_sklearn, label='sklearn')
    axs[0].set_xticks(training_size)
    axs[0].set_ylabel('runtime in seconds')
    axs[0].legend()
    axs[1].plot(training_size, accuracies_sequential_rff, label='sequential RFF')
    axs[1].plot(training_size, accuracies_sklearn, label='sklearn')
    axs[1].set_xticks(training_size)
    axs[1].set_xlabel('training set size')
    axs[1].set_ylabel('accuracy')
    axs[1].legend()
    plt.savefig('output/mnist_sequential_rff_vs_sklearn.png', dpi=150)
