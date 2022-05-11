from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from svm import NonLinearFeatures
from runner_svm_models import sklearn_svc, sequential_linear_svm, sequential_rff_svm, parallel_linear_svm, parallel_rff_svm
from utils import load_csv, two_dim_visualization, gridsearch


def runner_toydata_tiny(path):

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
    lr_params = [1, 1e-1, 1e-2]
    reg_params = [1, 1e-1, 1e-2]

    # run sequential linear svm
    print('\n--- Sequential Linear SVM ---')
    results_seq_linear = gridsearch(sequential_linear_svm, X_train, X_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(results_seq_linear['lr']))
    print('Best regularization parameter: {}'.format(results_seq_linear['reg']))
    print('Runtime with best parameters: {}'.format(results_seq_linear['runtime']))
    print('Accuracy with best parameters: {}'.format(results_seq_linear['accuracy']))
    y_pred_seq_linear = results_seq_linear['y_pred']

    # ---------- Compute RFF Features ----------

    # Create RFF features
    print('\n--- Compute RFF features ---')
    start = time.time()
    nlf = NonLinearFeatures(m=20, sigma=2.0)
    X_rff = nlf.fit_transform(X)
    end = time.time()
    print(f'Runtime transformation to RFF features: {end - start}')

    # train/test split of RFF features
    X_rff_train, X_rff_test, y_train, y_test = train_test_split(X_rff, y, random_state=42)

    # run sequential RFF svm
    print('\n--- Sequential RFF SVM ---')
    results_seq_rff = gridsearch(sequential_rff_svm, X_rff_train, X_rff_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(results_seq_rff['lr']))
    print('Best regularization parameter: {}'.format(results_seq_rff['reg']))
    print('Runtime with best parameters: {}'.format(results_seq_rff['runtime']))
    print('Accuracy with best parameters: {}'.format(results_seq_rff['accuracy']))
    y_pred_seq_rff = results_seq_rff['y_pred']

    # run parallel linear svm
    print('\n--- Parallel Linear SVM ---')
    results_par_linear = gridsearch(parallel_linear_svm, X_train, X_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(results_par_linear['lr']))
    print('Best regularization parameter: {}'.format(results_par_linear['reg']))
    print('Runtime with best parameters: {}'.format(results_par_linear['runtime']))
    print('Accuracy with best parameters: {}'.format(results_par_linear['accuracy']))
    y_pred_par_linear = results_par_linear['y_pred']

    # run parallel RFF svm
    print('\n--- Parallel RFF SVM ---')
    results_par_rff = gridsearch(parallel_rff_svm, X_rff_train, X_rff_test, y_train, y_test, lr_params, reg_params)
    print('Best learning rate: {}'.format(results_par_rff['lr']))
    print('Best regularization parameter: {}'.format(results_par_rff['reg']))
    print('Runtime with best parameters: {}'.format(results_par_rff['runtime']))
    print('Accuracy with best parameters: {}'.format(results_par_rff['accuracy']))
    y_pred_par_rff = results_par_rff['y_pred']

    # ---------- Plot results ----------

    # Plot true labels
    two_dim_visualization(X_test, y_test, 'output/true.png')
    # Plot predicted labels
    two_dim_visualization(X_test, y_pred_sklearn_svc, 'output/predicted_sklearn_svc.png')
    two_dim_visualization(X_test, y_pred_seq_linear, 'output/predicted_linear_sequential.png')
    two_dim_visualization(X_test, y_pred_par_linear, 'output/predicted_linear_parallel.png')
    two_dim_visualization(X_test, y_pred_seq_rff, 'output/predicted_rff_sequential.png')
    two_dim_visualization(X_test, y_pred_par_rff, 'output/predicted_rff_parallel.png')
