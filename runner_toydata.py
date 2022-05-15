from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from svm import NonLinearFeatures
from runner_svm_models import sklearn_svc, sequential_svm, parallel_svm
from utils import load_csv, gridsearch, five_fold_cross_validation, \
    sgd_progress, plot_sgd_convergence, two_dim_scatterplot


def runner_toydata(path, tiny):
    """
    Run experiments on the given toy datasets. There are two datasets, a tiny and a large one. The class labels are
    either +1 or -1. The tiny toy dataset is of dimension (200, 2), the large one is of dimension (200000, 8).

        Parameters
        ----------
        path : str
            Path where the data is stored
        tiny : bool
            Specify, whether the tiny or large dataset is given. If true, some additional plots will be generated

        Returns
        -------
        number_of_machines, parallel_runtimes, parallel_accuracies : list(int), list(float), list(float)
            The runtimes and accuracies versus the number of machines are returned for further comparison of the
            parallelization on the three different datasets
    """
    print(f'\n---------- Running Procedure on {path} ----------')

    X, y = load_csv(path)

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # compute baseline accuracy (dummy classifier)
    dummy_clf = DummyClassifier(strategy='stratified')
    dummy_clf.fit(X_train, y_train)
    print("\nBaseline Accuracy: {}".format(dummy_clf.score(X_test, y_test)))

    # run sklearn svc
    print('\n--- Sklearn SVC ---')
    y_pred_sklearn_svc, runtime_sklearn_svc, accuracy_sklearn_svc = sklearn_svc(X_train, X_test, y_train, y_test)
    print(f"Elapsed time fit/predict: {runtime_sklearn_svc}")
    print(f"Accuracy: {accuracy_sklearn_svc}")

    # define learning rate and regularization parameter range for gridsearch:
    lr_params = [1e2, 1e1, 1, 1e-1, 1e-2, 1e-3]
    reg_params = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    # run sequential linear svm
    print('\n--- Sequential Linear SVM ---')
    lr, reg = gridsearch(sequential_svm, X_train, X_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))
    y_pred_seq_linear, runtime, accuracy = five_fold_cross_validation(sequential_svm, X_train, X_test, y_train,
                                                                      y_test, lr, reg)
    print('Runtime with best parameters: {}'.format(runtime))
    print('Accuracy with best parameters: {}'.format(accuracy))
    sgd_progress_linear = sgd_progress(sequential_svm, X_train, X_test, y_train, y_test, lr, reg)

    # run parallel linear svm
    print('\n--- Parallel Linear SVM ---')
    lr, reg = gridsearch(parallel_svm, X_train, X_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))
    y_pred_par_linear, runtime, accuracy = five_fold_cross_validation(parallel_svm, X_train, X_test,
                                                                      y_train, y_test, lr, reg)
    print('Runtime with best parameters: {}'.format(runtime))
    print('Accuracy with best parameters: {}'.format(accuracy))

    # run parallel linear svm with increasing number of machines
    print('Running parallel linear svm with increasing number of machines...')
    number_of_machines = [1, 2, 3, 4, 5, 6, 7, 8]
    parallel_runtimes = []
    parallel_accuracies = []
    for num_threads in number_of_machines:
        _, runtime, accuracy = five_fold_cross_validation(parallel_svm, X_train, X_test,
                                                          y_train, y_test, lr, reg, num_threads)
        parallel_runtimes.append(runtime)
        parallel_accuracies.append(accuracy)

    # Create RFF features
    print('\n--- Compute RFF features ---')
    start = time.time()
    nlf = NonLinearFeatures(m=100, sigma=1.0)
    X_rff = nlf.fit_transform(X)
    end = time.time()
    print(f'Runtime transformation to RFF features: {end - start}')

    # train/test split of RFF features
    X_rff_train, X_rff_test, y_train, y_test = train_test_split(X_rff, y, random_state=42)

    # run sequential RFF svm
    print('\n--- Sequential RFF SVM ---')
    lr, reg = gridsearch(sequential_svm, X_rff_train, X_rff_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))
    y_pred_seq_rff, runtime, accuracy = five_fold_cross_validation(sequential_svm, X_rff_train, X_rff_test,
                                                                   y_train, y_test, lr, reg)
    print('Runtime with best parameters: {}'.format(runtime))
    print('Accuracy with best parameters: {}'.format(accuracy))
    sgd_progress_rff = sgd_progress(sequential_svm, X_rff_train, X_rff_test, y_train, y_test, lr, reg)

    # run parallel RFF svm
    print('\n--- Parallel RFF SVM ---')
    lr, reg = gridsearch(parallel_svm, X_rff_train, X_rff_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(lr))
    print('Best regularization parameter: {}'.format(reg))
    y_pred_par_rff, runtime, accuracy = five_fold_cross_validation(parallel_svm, X_rff_train, X_rff_test,
                                                                   y_train, y_test, lr, reg)

    print('Runtime with best parameters: {}'.format(runtime))
    print('Accuracy with best parameters: {}'.format(accuracy))

    # ---------- Plot results ----------

    if tiny:
        # Plot true labels
        two_dim_scatterplot(X_test, y_test, 'output/true.png')
        # Plot predicted labels
        two_dim_scatterplot(X_test, y_pred_sklearn_svc, 'output/predicted_sklearn_svc.png')
        two_dim_scatterplot(X_test, y_pred_seq_linear, 'output/predicted_linear_sequential.png')
        two_dim_scatterplot(X_test, y_pred_par_linear, 'output/predicted_linear_parallel.png')
        two_dim_scatterplot(X_test, y_pred_seq_rff, 'output/predicted_rff_sequential.png')
        two_dim_scatterplot(X_test, y_pred_par_rff, 'output/predicted_rff_parallel.png')

        plot_sgd_convergence(sgd_progress_linear, sgd_progress_rff, 'output/sgd_progress_tiny.png')

    if not tiny:
        plot_sgd_convergence(sgd_progress_linear, sgd_progress_rff, 'output/sgd_progress_large.png')

    return number_of_machines, parallel_runtimes, parallel_accuracies
