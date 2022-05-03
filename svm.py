import numpy as np
import threading


class LinearSequentialSVM:

    def __init__(self, learning_rate=1e-3, regularization_parameter=1e-2, tolerance=1e-6, max_num_iterations=1000):
        self.lr = learning_rate
        self.reg = regularization_parameter
        self.tol = tolerance
        self.n_max = max_num_iterations
        self.w = None

    def _init_weights(self, X):
        _, n_features = X.shape
        self.w = np.zeros(n_features)

    def _get_weights(self):
        return self.w

    def _update_weights(self, x, y):
        # compute gradient
        if y * np.dot(x, self.w) >= 1:
            dw = self.reg * self.w
        else:
            dw = self.reg * self.w - np.dot(y, x)
        # update w
        self.w -= self.lr * dw

    def fit(self, X, y):

        # incorporate bias term b into features X
        n_samples, n_features = X.shape
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        self._init_weights(X)

        for _ in range(self.n_max):
            old_w = self._get_weights()
            for idx, x in enumerate(X):
                self._update_weights(x, y[idx])
            diff = old_w - self.w
            if np.linalg.norm(diff, 1) < self.tol:
                break

    def predict(self, X):

        # incorporate bias term b into features X
        n_samples, n_features = X.shape
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        prediction = np.sign(np.dot(X, self.w)).astype(int)
        return prediction


class LinearParallelSVM:

    def __init__(self, learning_rate=1e-3, regularization_parameter=1e-2, num_threads=1, tolerance=1e-6, max_num_iterations=1000):
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
            dw = self.reg * self.sub_w
        else:
            dw = self.reg * self.sub_w - np.dot(y, x)
        # update w
        self.sub_w -= self.lr * dw

    def fit(self, X, y):

        # incorporate bias term b into features X
        n_samples, n_features = X.shape
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        for i in range(self.n_threads):

            # take a subset of points
            positions = np.random.randint(0, int(len(X)), int(len(X)/self.n_threads))
            Xi = []
            yi = []
            for j in positions:
                Xi.append(X[j])
                yi.append(y[j])
            Xi = np.array(Xi)
            yi = np.array(yi)

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

        self.w = sum(self.sub_ws) / self.n_threads  # compute w by taking the average of sub_ws

    def predict(self, X):

        # incorporate bias term b into features X
        n_samples, n_features = X.shape
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        prediction = np.sign(np.dot(X, self.w)).astype(int)
        return prediction


class Laplacian:
    def __init__(self, sigma):
        self.sigma = sigma

    def get_sigma(self):
        return self.sigma

    def sample_rffs(self, d):
        omega = 1 / self.get_sigma() * np.random.standard_cauchy(d)
        b = 2 * np.pi * np.random.rand(d)
        return omega, b


class RFF_SVM:

    def __init__(self, learning_rate, regularization_parameter):
        self.lr = learning_rate
        self.reg = regularization_parameter
        self.w = None

    def _init_weights(self, X):
        _, n_features = X.shape
        self.w = np.zeros(n_features)

    def _classify(self, x):
        label = np.sign(np.dot(x, self.w))
        if label == 0:
            label = 1
        return label

    def _update_weights(self, x, y, t):
        if y * np.dot(x, self.w) < 1:
            # w = self.w - ((1/np.sqrt(t+1)) * -np.dot(y, x))
            w = self.w - (self.lr * -np.dot(y, x))
            self.w = w * min(1, (1 / np.linalg.norm(w)) * (1 / np.sqrt(self.reg)))

    def fit_predict(self, X, y):
        # incorporate bias term b into features X
        n_samples, n_features = X.shape
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        self._init_weights(X)  # initialize w0
        predictions = np.zeros_like(y)  # initialize array for predicted labels
        k = Laplacian(sigma=1)  # initialize kernel

        # generate kernel approximation
        d = 200
        omega, b = k.sample_rffs(d)
        k_approx = np.zeros_like(X)
        for i, Y in zip(range(len(k_approx)), X):
            x = 0
            zx = np.sqrt(2) / np.sqrt(d) * np.cos(omega * x + b)
            zy = np.sqrt(2) / np.sqrt(d) * np.cos(np.dot(omega, Y) + b)
            k_approx[i] = np.dot(zx, zy)

        for t, x in enumerate(X):
            # classify x according to sign(wTx)
            predictions[t] = self._classify(x)
            # update w based on (x,y)
            self._update_weights(x, y[t], t)

        return predictions
