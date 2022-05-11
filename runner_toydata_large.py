from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from svm import NonLinearFeatures
from runner_svm_models import sklearn_svc, sequential_linear_svm, sequential_rff_svm, parallel_linear_svm, parallel_rff_svm
from utils import load_csv


def runner_toydata_large(path):

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
    _ = sklearn_svc(X_train, X_test, y_train, y_test)

    # run sequential linear svm
    _ = sequential_linear_svm(X_train, X_test, y_train, y_test, learning_rate=1e-1, regularization=1e-2)

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
    _ = sequential_rff_svm(X_rff_train, X_rff_test, y_train, y_test, learning_rate=1e-1, regularization=1e-2)

    # run parallel linear svm
    _ = parallel_linear_svm(X_train, X_test, y_train, y_test, learning_rate=1e-1, regularization=1e-2, num_threads=8)

    # run parallel RFF svm
    _ = parallel_rff_svm(X_rff_train, X_rff_test, y_train, y_test, learning_rate=1e-1, regularization=1e-2, num_threads=8)
