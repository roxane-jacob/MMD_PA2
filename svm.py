import numpy as np
from joblib import Parallel, delayed


class SequentialSVM:

    def __init__(self, learning_rate=1e-3, regularization_parameter=1e-2, tolerance=1e-6, max_num_iterations=1000):
        self.lr = learning_rate
        self.reg = regularization_parameter
        self.tol = tolerance
        self.n_max = max_num_iterations
        self.w = None
        self.omega = None
        self.b = None
        self.multiclass = False

    def _init_weights(self, X):
        _, n_features = X.shape
        self.w = np.zeros(n_features)

    def _init_weights_multiclass(self, X, n_classes):
        _, n_features = X.shape
        self.w = np.zeros((n_features, n_classes))

    def _get_weights(self):
        return np.array(self.w)

    def _update_weights(self, x, y):
        # compute gradient
        if y * np.dot(x, self.w) >= 1:
            pass
        else:
            dw = -y * x
            # update w
            self.w -= self.lr * dw
            # project w
            self.w *= min(1, 1/(np.linalg.norm(self.w)*np.sqrt(self.reg)))

    def _update_weights_multiclass(self, x, y, n_classes):
        val = 0
        for i in range(n_classes):
            if i is not y:
                val = max(np.dot(self.w[:, i], x), val)
        if np.dot(x, self.w[:, y]) - val >= 1:
            pass
        else:
            dw = -x
            # update w
            self.w[:, y] -= self.lr * dw
            # project w
            self.w *= min(1, 1/(np.linalg.norm(self.w)*np.sqrt(self.reg)))

    def fit(self, X, y):
        n_classes = max(y) + 1
        if n_classes > 2:
            self.multiclass = True
        else:
            self.multiclass = False

        n_samples, n_features = X.shape

        # incorporate bias term b into features X
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        # multiclass case
        if self.multiclass:
            self._init_weights_multiclass(X, n_classes)
            for _ in range(self.n_max):
                old_w = self._get_weights()
                for idx, x in enumerate(X):
                    self._update_weights_multiclass(x, y[idx], n_classes)
                diff = old_w - self.w
                if np.linalg.norm(diff, 1) < self.tol:
                    break
        # binary case
        else:
            self._init_weights(X)
            for _ in range(self.n_max):
                old_w = self._get_weights()
                for idx, x in enumerate(X):
                    self._update_weights(x, y[idx])
                diff = old_w - self.w
                if np.linalg.norm(diff, 1) < self.tol:
                    break

    def predict(self, X):
        n_samples, n_features = X.shape

        # incorporate bias term b into features X
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        prediction = np.sign(np.dot(X, self.w)).astype(int)
        return prediction


class ParallelSVM:

    def __init__(self, learning_rate=1e-3, regularization_parameter=1e-2, num_threads=1, tolerance=1e-6,
                 max_num_iterations=1000):
        self.lr = learning_rate
        self.reg = regularization_parameter
        self.n_threads = num_threads
        self.tol = tolerance
        self.n_max = max_num_iterations
        self.w = None
        self.sub_w = None
        self.sub_ws = []

    def _init_sub_w(self, X):
        _, n_features = X.shape
        self.sub_w = np.zeros(n_features)

    def _get_sub_w(self):
        return self.sub_w

    def _update_sub_w(self, x, y):
        # compute gradient
        if y * np.dot(x, self.sub_w) >= 1:
            pass
        else:
            dw = -y * x
            # update w
            self.sub_w -= self.lr * dw
            # project w
            self.sub_w *= min(1, 1 / (np.linalg.norm(self.sub_w) * np.sqrt(self.reg)))

    def subfit(self, Xi, yi):

        self._init_sub_w(Xi)

        for n in range(self.n_max):
            old_sub_w = self._get_sub_w()
            for idx, x in enumerate(Xi):
                self._update_sub_w(x, yi[idx])
            if n > 0:
                diff = old_sub_w - self.sub_w
                if np.linalg.norm(diff, 1) < self.tol:
                    break

        self.sub_ws.append(self.sub_w)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # incorporate bias term b into features X
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        sample_to_thread = np.random.randint(0, self.n_threads, n_samples)
        Xs = [X[sample_to_thread == i] for i in range(self.n_threads)]
        ys = [y[sample_to_thread == i] for i in range(self.n_threads)]

        Parallel(n_jobs=self.n_threads, backend='threading')(delayed(self.subfit)(Xi, yi) for Xi, yi in zip(Xs, ys))

        self.w = sum(self.sub_ws) / self.n_threads  # compute w by taking the average of sub_ws

    def predict(self, X):
        n_samples, n_features = X.shape

        # incorporate bias term b into features X
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        prediction = np.sign(np.dot(X, self.w)).astype(int)
        return prediction


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
