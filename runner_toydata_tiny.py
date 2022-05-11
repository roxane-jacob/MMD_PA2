from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from svm import NonLinearFeatures
from runner_svm_models import sklearn_svc, sequential_linear_svm, sequential_rff_svm, parallel_linear_svm, parallel_rff_svm
from utils import load_csv, two_dim_visualization


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

    # run sequential linear svm
    y_pred_seq_linear, runtime_seq_linear, accuracy_seq_linear = sequential_linear_svm(X_train, X_test, y_train, y_test,
                                                                                       learning_rate=1e-1, regularization=1e-2)

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
    y_pred_seq_rff, runtime_seq_rff, accuracy_seq_rff = sequential_rff_svm(X_rff_train, X_rff_test, y_train, y_test,
                                                                           learning_rate=1e-1, regularization=1e-2)

    # run parallel linear svm
    y_pred_par_linear, runtime_par_linear, accuracy_par_linear = parallel_linear_svm(X_train, X_test, y_train, y_test,
                                                                                     learning_rate=1e-1, regularization=1e-2, num_threads=8)

    # run parallel RFF svm
    y_pred_par_rff, runtime_par_rff, accuracy_par_rff = parallel_rff_svm(X_rff_train, X_rff_test, y_train, y_test,
                                                                         learning_rate=1e-1, regularization=1e-2, num_threads=8)

    # ---------- Plot results ----------

    # Plot true labels
    two_dim_visualization(X_test, y_test, 'output/true.png')
    # Plot predicted labels
    two_dim_visualization(X_test, y_pred_sklearn_svc, 'output/predicted_sklearn_svc.png')
    two_dim_visualization(X_test, y_pred_seq_linear, 'output/predicted_linear_sequential.png')
    two_dim_visualization(X_test, y_pred_par_linear, 'output/predicted_linear_parallel.png')
    two_dim_visualization(X_test, y_pred_seq_rff, 'output/predicted_rff_sequential.png')
    two_dim_visualization(X_test, y_pred_par_rff, 'output/predicted_rff_parallel.png')
