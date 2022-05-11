import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from svm import NonLinearFeatures
from runner_svm_models import sklearn_svc, sequential_svm, sequential_svm, parallel_svm, parallel_svm
from utils import load_mnist, gridsearch


def runner_mnist(path):

    print(f'\n---------- Running Procedure on {path} ----------')

    X_train, y_train, X_test, y_test = load_mnist(path)

    # scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # compute baseline accuracy (dummy classifier)
    dummy_clf = DummyClassifier(strategy='stratified')
    dummy_clf.fit(X_train, y_train)
    print("\nBaseline Accuracy: {}".format(dummy_clf.score(X_test, y_test)))

    # define learning rate and regularization parameter range for gridsearch:
    lr_params = [1, 1e-1, 1e-2]
    reg_params = [1, 1e-1, 1e-2]

    # run sequential linear svm
    print('\n--- Sequential Linear SVM ---')
    lr, reg = gridsearch(sequential_svm, X_train, X_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))
    _, runtime, accuracy = sequential_svm(X_train, X_test, y_train, y_test, lr, reg)
    print('Runtime with best parameters: {}'.format(runtime))
    print('Accuracy with best parameters: {}'.format(accuracy))

    # ---------- Compute RFF Features ----------

    # Create RFF features
    print('\n--- Compute RFF features ---')
    start = time.time()
    nlf = NonLinearFeatures(m=2000, sigma=1.0)
    X_rff_train = nlf.fit_transform(X_train[:3000])
    X_rff_test = nlf.fit_transform(X_test[:500])
    end = time.time()
    print(f'Runtime transformation to RFF features: {end - start}')

    # run sequential RFF svm
    print('\n--- Sequential RFF SVM ---')
    lr, reg = gridsearch(sequential_svm, X_rff_train, X_rff_test, y_train[:3000], y_test[:500], lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))
    _, runtime, accuracy = sequential_svm(X_rff_train, X_rff_test, y_train[:3000], y_test[:500], lr, reg)

    print('Runtime with best parameters: {}'.format(runtime))
    print('Accuracy with best parameters: {}'.format(accuracy))

    # run parallel linear svm
    print('\n--- Parallel Linear SVM ---')
    lr, reg = gridsearch(parallel_svm, X_train, X_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))
    _, runtime, accuracy = parallel_svm(X_train, X_test, y_train, y_test, lr, reg)

    print('Runtime with best parameters: {}'.format(runtime))
    print('Accuracy with best parameters: {}'.format(accuracy))

    # run parallel RFF svm
    print('\n--- Parallel RFF SVM ---')
    lr, reg = gridsearch(parallel_svm, X_rff_train, X_rff_test, y_train[:3000], y_test[:500], lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))
    _, runtime, accuracy = parallel_svm(X_rff_train, X_rff_test, y_train[:3000], y_test[:500], lr, reg)

    print('Runtime with best parameters: {}'.format(runtime))
    print('Accuracy with best parameters: {}'.format(accuracy))

    # make plots for runtime/performance comparison when training on 1000, 2000, 3000 training samples
    # and using own implementation vs. sklearn's svm.svc

    training_size = [1000, 2000, 3000]
    runtimes_sequential_rff = []
    accuracies_sequential_rff = []
    runtimes_sklearn = []
    accuracies_sklearn = []

    for size in training_size:
        _, runtime, accuracy = sequential_svm(X_rff_train[:size], X_rff_test, y_train[:size], y_test[:500],
                                              learning_rate=1e-1, regularization=1e-2)
        runtimes_sequential_rff.append(runtime)
        accuracies_sequential_rff.append(accuracy)

        _, runtime, accuracy = sklearn_svc(X_rff_train[:size], X_rff_test, y_train[:size], y_test[:500])
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
