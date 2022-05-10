import numpy as np
from joblib import Parallel, delayed


class SequentialSVM:
    def __init__(self, learning_rate=1e-3, regularization_parameter=1e-2, tolerance=1e-6, max_num_iterations=1000):
        self.lr = learning_rate
        self.reg = regularization_parameter
        self.tol = tolerance
        self.n_max = max_num_iterations
        self.w = None
        self.wi = []
        self.multiclass = False
        self.n_classes = None

    def _init_weights(self, X):
        if self.multiclass:
            _, n_features = X.shape
            w = np.zeros((n_features, self.n_classes))
        else:
            _, n_features = X.shape
            w = np.zeros(n_features)
        return w

    def _get_weights(self):
        return np.array(self.w)

    def _update_weights(self, x, y, w):
        if self.multiclass:
            val = 0
            for i in range(self.n_classes):
                if i is not y:
                    val = max(np.dot(w[:, i], x), val)
            if np.dot(x, w[:, y]) - val >= 1:
                pass
            else:
                dw = -x
                # update w
                w[:, y] -= self.lr * dw
                # project w
                w *= min(1, 1 / (np.linalg.norm(w) * np.sqrt(self.reg)))
            return w

        else:
            # compute gradient
            if y * np.dot(x, w) >= 1:
                pass
            else:
                dw = -y * x
                # update w
                w -= self.lr * dw
                # project w
                w *= min(1, 1/(np.linalg.norm(w)*np.sqrt(self.reg)))
            return w

    def fit(self, X, y):
        self.n_classes = max(y) + 1
        if self.n_classes > 2:
            self.multiclass = True
        else:
            self.multiclass = False

        n_samples, n_features = X.shape

        # incorporate bias term b into features X
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)
        self.runner(X, y)
        self.w = self.wi[0]

    def runner(self, X, y):
        w = self._init_weights(X)

        for _ in range(self.n_max):
            old_w = w
            for idx, x in enumerate(X):
                w = self._update_weights(x, y[idx], w)
            diff = old_w - w
            if np.linalg.norm(diff, 1) < self.tol:
                break

        self.wi.append(w)

    def predict(self, X):
        n_samples, n_features = X.shape

        # incorporate bias term b into features X
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        if self.multiclass:
            prediction = np.argmax(self.w.T @ X.T, axis=0)
        else:
            prediction = np.sign(np.dot(X, self.w)).astype(int)
        return prediction


class ParallelSVM(SequentialSVM):
    def __init__(self, learning_rate=1e-3, regularization_parameter=1e-2, tolerance=1e-6, max_num_iterations=1000,
                 num_threads=1):
        SequentialSVM.__init__(self, learning_rate, regularization_parameter, tolerance, max_num_iterations)
        self.n_threads = num_threads

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.n_classes = max(y) + 1
        if self.n_classes > 2:
            self.multiclass = True
        else:
            self.multiclass = False

        # incorporate bias term b into features X
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        sample_to_thread = np.random.randint(0, self.n_threads, n_samples)
        Xs = [X[sample_to_thread == i] for i in range(self.n_threads)]
        ys = [y[sample_to_thread == i] for i in range(self.n_threads)]

        Parallel(n_jobs=self.n_threads, backend='threading')(delayed(self.runner)(Xi, yi) for Xi, yi in zip(Xs, ys))

        self.w = sum(self.wi) / self.n_threads  # compute w by taking the average of sub_ws


class NonLinearFeatures:
    def __init__(self, m=None, sigma=None):
        self.m = m
        self.sigma = sigma

    def _update_random_variables(self, d):
        self.omega = 1 / self.sigma * np.random.standard_cauchy((d, self.m))
        self.b = 2 * np.pi * np.random.rand(self.m)

    def _non_linear_features(self, X):
        n_samples, _ = X.shape
        X_new = np.zeros((n_samples, self.m))
        for i, x in enumerate(X):
            X_new[i, :] = np.sqrt(2 / self.m) * \
                          np.array([np.cos(np.dot(self.omega[:, i], x) + self.b[i]) for i in range(self.m)])
        return X_new

    def fit_transform(self, X):
        _, n_features = X.shape
        self._update_random_variables(n_features)
        X = self._non_linear_features(X)
        return X

    def fit(self, X):
        _, n_features = X.shape
        self._update_random_variables(n_features)

    def transform(self, X):
        X = self._non_linear_features(X)
        return X
