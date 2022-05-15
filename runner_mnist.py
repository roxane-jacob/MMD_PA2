import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import time

from svm import NonLinearFeatures
from runner_svm_models import sklearn_svc, sequential_svm, parallel_svm
from utils import load_mnist, gridsearch, gridsearch_rff


def runner_mnist(path):
    """
    Run experiments on the MNIST dataset. The MNIST dataset consist of 70000 images of handwritten digits of
    dimension (28 x 28) with class labels from 0 to 9. The images are flattened into arrays of dimension (784,).
    The training/test split of the data is 60000/10000.

        Parameters
        ----------
        path : str
            Path where the data is stored

        Returns
        -------
        number_of_machines, parallel_runtimes, parallel_accuracies : list(int), list(float), list(float)
            The runtimes and accuracies versus the number of machines are returned for further comparison of the
            parallelization on the three different datasets
    """

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

    # define learning rate and regularization parameter range for gridsearch
    lr_params = [1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4]
    reg_params = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    # run gridsearch on sequential linear svm
    print('\n--- Sequential Linear SVM ---')
    lr, reg = gridsearch(sequential_svm, X_train, X_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))

    # run sequential linear svm with optimized learning rate and regularisation parameter
    _, runtime, accuracy = sequential_svm(X_train, X_test, y_train, y_test, lr, reg)
    print('Runtime with best parameters: {}'.format(round(runtime, 4)))
    print('Accuracy with best parameters: {}'.format(round(accuracy, 4)))

    # run gridsearch on parallel linear svm with fixed number of 8 machines
    print('\n--- Parallel Linear SVM ---')
    lr, reg = gridsearch(parallel_svm, X_train, X_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))

    # run parallel linear svm with optimized learning rate and regularisation parameter with fixed number of 8 machines
    _, runtime, accuracy = parallel_svm(X_train, X_test, y_train, y_test, lr, reg)
    print('Runtime with best parameters: {}'.format(round(runtime, 4)))
    print('Accuracy with best parameters: {}'.format(round(accuracy, 4)))

    # run parallel linear svm with increasing number of machines
    # this run takes into account the learning rate and the regularisation parameter of the previous gridsearch
    print('Running parallel linear svm with increasing number of machines...')
    number_of_machines = [1, 2, 3, 4]
    parallel_runtimes = []
    parallel_accuracies = []
    for num_threads in number_of_machines:
        _, runtime, accuracy = parallel_svm(X_train, X_test,
                                            y_train, y_test, lr, reg, num_threads)
        parallel_runtimes.append(runtime)
        parallel_accuracies.append(accuracy)

    # use a subset of the original sample space to reduce runtime of SVMs with rff features
    X_train = X_train[:3000]
    X_test = X_test[:500]
    y_train = y_train[:3000]
    y_test = y_test[:500]

    # create RFF features for the upcoming gridsearch procedure
    print('\n--- Compute RFF features ---')
    start = time.time()
    nlf = NonLinearFeatures(m=1000, sigma=200)
    X_rff_train = nlf.fit_transform(X_train)
    X_rff_test = nlf.transform(X_test)
    end = time.time()
    print(f'Runtime transformation to RFF features: {end - start}')

    # run sequential RFF svm gridsearch to find optimal learning rate and regularisation parameter
    lr_seq, reg_seq = gridsearch(sequential_svm, X_rff_train, X_rff_test, y_train, y_test, lr_params, reg_params)

    # run gridsearch on RFF feature hyperparameters (m, sigma) with optimized learning rate and regularisation parameter
    print('\n--- RFF Gridsearch ---')
    m_params = [1000, 2000, 3000]
    sigma_params = [1, 100, 200, 300, 1000]
    m_seq, sigma_seq = gridsearch_rff(sequential_svm, X_train, X_test, y_train, y_test, lr_seq, reg_seq,  m_params,
                                      sigma_params)
    print('Best feature dimension m: {}'.format(m_seq))
    print('Best hyperparameter sigma: {}'.format(sigma_seq))

    # Recalculate RFF features with optimized hyperparameters for the upcoming calculations
    print('\n--- Recompute RFF features ---')
    start = time.time()
    nlf = NonLinearFeatures(m=m_seq, sigma=sigma_seq)
    X_rff_train = nlf.fit_transform(X_train)
    X_rff_test = nlf.transform(X_test)
    end = time.time()
    print(f'Runtime transformation to RFF features: {end - start}')

    # run sequential RFF svm with optimized learning rate, regularisation parameter, feature dimension m, and sigma
    print('\n--- Sequential RFF SVM ---')
    print('Best learning rate: {}'.format(lr_seq))
    print('Best regularization parameter: {}'.format(reg_seq))
    _, runtime, accuracy = sequential_svm(X_rff_train, X_rff_test, y_train, y_test, lr_seq, reg_seq)
    print('Runtime with best parameters: {}'.format(round(runtime, 4)))
    print('Accuracy with best parameters: {}'.format(round(accuracy, 4)))

    # run gridsearch on parallel RFF svm
    print('\n--- Parallel RFF SVM ---')
    lr, reg = gridsearch(parallel_svm, X_rff_train, X_rff_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))

    # run parallel RFF svm with optimized learning rate, regularisation parameter, feature dimension m, and sigma
    _, runtime, accuracy = parallel_svm(X_rff_train, X_rff_test, y_train, y_test, lr, reg)
    print('Runtime with best parameters: {}'.format(round(runtime, 4)))
    print('Accuracy with best parameters: {}'.format(round(accuracy, 4)))

    # make plots for runtime/performance comparison when training on 1000, 2000, 3000 training samples
    # and using own implementation vs. sklearn's svm.svc
    training_size = [1000, 2000, 3000]
    runtimes_sequential_rff = []
    accuracies_sequential_rff = []
    runtimes_sklearn = []
    accuracies_sklearn = []

    for size in training_size:
        _, runtime, accuracy = sequential_svm(X_rff_train[:size], X_rff_test, y_train[:size], y_test, lr_seq, reg_seq)
        runtimes_sequential_rff.append(runtime)
        accuracies_sequential_rff.append(accuracy)

        _, runtime, accuracy = sklearn_svc(X_rff_train[:size], X_rff_test, y_train[:size], y_test)
        runtimes_sklearn.append(runtime)
        accuracies_sklearn.append(accuracy)

    # plot runtime and accuracy versus the size of the training set
    fig, axs = plt.subplots(2, 1)
    axs[0].semilogy(training_size, runtimes_sequential_rff, label='sequential RFF')
    axs[0].semilogy(training_size, runtimes_sklearn, label='sklearn')
    axs[0].set_xticks(training_size)
    axs[0].set_ylabel('runtime in seconds', fontsize=16)
    axs[0].legend(fontsize=16, bbox_to_anchor=(1.02, 1))
    axs[1].plot(training_size, accuracies_sequential_rff, label='sequential RFF')
    axs[1].plot(training_size, accuracies_sklearn, label='sklearn')
    axs[1].set_xticks(training_size)
    axs[1].set_xlabel('training set size', fontsize=16)
    axs[1].set_ylabel('accuracy', fontsize=16)
    #axs[1].legend(fontsize=16, bbox_to_anchor=(1.04,1))
    plt.savefig('output/mnist_sequential_rff_vs_sklearn.png', dpi=150, bbox_inches="tight")

    return number_of_machines, parallel_runtimes, parallel_accuracies
