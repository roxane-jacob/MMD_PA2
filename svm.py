import numpy as np
import threading


class LinearSVM:

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

        for t, x in enumerate(X):
            # classify x according to sign(wTx)
            predictions[t] = self._classify(x)
            # update w based on (x,y)
            self._update_weights(x, y[t], t)

        return predictions


class Laplacian:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, delta):
        return np.exp(-np.abs(delta) / self.sigma)

    def get_sigma(self):
        return self.sigma

    def sample_rffs(self, d):
        w = 1 / self.get_sigma() * np.random.standard_cauchy(d)
        b = 2 * np.pi * np.random.rand(d)
        return w, b


class MyThread (threading.Thread):
    def __init__(self, thread_id):
        threading.Thread.__init__(self)
        self.threadID = thread_id

    def run(self):
        print("Starting Thread" + self.thread_id)
        print_time(self.name, self.counter, 5)
        print("Exiting Thread" + self.thread_id)


class LinearSVMParallel:

    def __init__(self, learning_rate, regularization):
        self.lr = learning_rate
        self.reg = regularization
        self.w = None
        self.sub_w = None
        self.sub_ws = []

    def _init_sub_w(self, X):
        _, n_features = X.shape
        self.sub_w = np.zeros(n_features)

    def _classify(self, x):
        label = np.sign(np.dot(x, self.w))
        if label == 0:
            label = 1
        return label

    def _update_weights(self, x, y, t):
        if y * np.dot(x, self.sub_w) < 1:
            # w = self.w - ((1/np.sqrt(t+1)) * -np.dot(y, x))
            sub_w = self.sub_w - (self.lr * -np.dot(y, x))
            self.sub_w = sub_w * min(1, (1 / np.linalg.norm(sub_w)) * (1 / np.sqrt(self.reg)))

    def fit_predict(self, X, y, n):

        # incorporate bias term b into features X
        n_samples, n_features = X.shape
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        for i in range(n):

            # take a subset of points
            positions = np.random.randint(0, int(len(X)), int(len(X)/n))
            Xi = []
            yi = []
            for j in positions:
                Xi.append(X[j])
                yi.append(y[j])
            Xi = np.array(Xi)
            yi = np.array(yi)

            self._init_sub_w(Xi)

            for t, x in enumerate(Xi):
                self._update_weights(x, yi[t], t)

            self.sub_ws.append(self.sub_w)

        self.w = sum(self.sub_ws) / n  # compute w by taking the average of sub_ws

        predictions = np.zeros_like(y)  # initialize array for predicted labels
        for t, x in enumerate(X):
            predictions[t] = self._classify(x)  # classify x according to sign(wTx)

        return predictions
