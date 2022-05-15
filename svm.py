import numpy as np


class SequentialSVM:
    def __init__(self, learning_rate=1e-3, regularization_parameter=1e-2, store_sgd_progress=False):
        """
        Implementation of a Support Vector Machine (SVM) classifier. The classifier can work with binary and
        multiclass labelled data. The classifier optimizes the hinge loss of the training data via Stochastic
        Gradient Descent (SGD).

            Parameters
            ----------
            learning_rate : float
                Learning rate of the SGD algorithm
            regularization_parameter : float
                Regularization parameter of the SDG algorithm.
            store_sgd_progress : bool, optional
                If selected, the algorithm will return a learning curve of the SGD algorithm. The learning curve is
                defined as ||w-w*||_1 / ||w*||, where w is the weight vector of the decision boundary of the current
                iteration and w* is the optimal weight vector, which is returned after the final iteration.
        """
        self.lr = learning_rate
        self.reg = regularization_parameter
        self.w = None
        self.multiclass = False
        self.n_classes = None
        self.store_sgd_progress = store_sgd_progress
        self.stored_weights = []
        self.sgd_progress = []

    def _init_weights(self, X):
        """
        Initialize the weight vector w. The vector w defines the decision boundary of the SVM classifier. If the given
        data is binary, the weight vector has the dimension m, where m is the number of the features per sample.
        In the case of multiclass labelled data, one weight vector per class c is generated. This results in a matrix
        of shape (m, c), where m is the number of features per sample and c is the number of classes in the label space.

            Parameters
            ----------
            X : ndarray of shape (n samples, m features)
                Training data

            Returns
            -------
            w : ndarray of shape m (binary classification), or (m, c) (multiclass classification)
                An initial array of the specific dimension containing zeros.
        """
        if self.multiclass:
            _, n_features = X.shape
            w = np.zeros((n_features, self.n_classes))
        else:
            _, n_features = X.shape
            w = np.zeros(n_features)
        return w

    def _get_weights(self):
        """getter for the weight vector w"""
        return np.array(self.w)

    def _update_weights(self, x, y, w):
        """
        Update the weight vector w with a given sample x and label y. The SGD algorithm optimizes the hinge loss of the
        given data points. There are two separate update rules for the case of binary and multiclass classification.

            Parameters
            ----------
            x : ndarray of shape (m features,)
                training sample
            y : float
                label of the training sample
            w : ndarray of shape (m features,) or (m features, c classes)
                current weight vector

            Returns
            -------
            w : ndarray of shape (m features,) or (m features, c classes)
                the updated weight vector
        """
        if self.multiclass:
            val = 0
            for i in range(self.n_classes):
                if i is not y:
                    val = max(np.dot(w[:, i], x), val)
            if np.dot(x, w[:, y]) - val >= 1:
                pass
            else:
                dw = -x  # compute gradient
                w[:, y] -= self.lr * dw  # update w
                w *= min(1, 1 / (np.linalg.norm(w) * np.sqrt(self.reg)))  # project w
            return w

        else:
            if y * np.dot(x, w) >= 1:
                pass
            else:
                dw = -y * x  # compute gradient
                w -= self.lr * dw  # update w
                w *= min(1, 1/(np.linalg.norm(w)*np.sqrt(self.reg)))  # project w
            return w

    def fit(self, X, y):
        """
        Fit the SVM classifier to the given training data. The weight vector of the resulting decision boundary is
        stored within the class for further label prediction of given test data.

            Parameters
            ----------
            X : ndarray of shape (n samples, m features)
                Training data
            y : ndarray of shape (n samples,)
                Training data labels

            Returns
            -------
            None
        """
        self.n_classes = max(y) + 1
        if self.n_classes > 2:
            self.multiclass = True
        else:
            self.multiclass = False

        n_samples, n_features = X.shape

        # incorporate bias term b into features X
        b = np.ones((n_samples, 1))
        X = np.concatenate((X, b), axis=1)

        self.w = self.runner(X, y)

        if self.store_sgd_progress:
            for stored_w in self.stored_weights:
                self.sgd_progress.append(np.linalg.norm(self.w - stored_w, ord=1) / np.linalg.norm(self.w))

    def runner(self, X, y):
        """
        Run the SGD optimizer on the given training data and return a weight vector w, which determines the decision
        boundary.

            Parameters
            ----------
            X : ndarray of shape (n samples, m features)
                Training data
            y : ndarray of shape (n samples,)
                Training data labels

            Returns
            -------
            w : ndarray of shape (m features,) or (m features, c classes)
                the weight vector of the decision boundary
        """
        w = self._init_weights(X)

        if self.store_sgd_progress:
            for idx, x in enumerate(X):
                self.stored_weights.append(np.array(w))
                w = self._update_weights(x, y[idx], w)
        else:
            for idx, x in enumerate(X):
                w = self._update_weights(x, y[idx], w)
        return w

    def get_sgd_progress(self):
        """getter for the sgd learning curve"""
        return self.sgd_progress

    def predict(self, X):
        """
        Predict the class labels for a given dataset X.

            Parameters
            ----------
            X : ndarray of shape (n samples, m features)
                Test or validation data

            Returns
            -------
            prediction : ndarray of shape (n samples,)
                Predicted class labels for the given test or validation data.
        """
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
    def __init__(self, learning_rate=1e-3, regularization_parameter=1e-2, num_threads=8):
        """
        Parallel version of the SVM classifier class. The class takes one additional argument, which is the number of
        threads for the parallel SGD optimizer.

            Parameters
            ----------
            learning_rate : float
                Learning rate of the SGD algorithm
            regularization_parameter : float
                Regularization parameter of the SGD algorithm
            num_threads : int
                Number of threads for the parallelization
        """
        SequentialSVM.__init__(self, learning_rate, regularization_parameter)
        self.n_threads = num_threads

    def fit(self, X, y):
        """
        Fit the SVM classifier to the given training data. The parallelized SGD algorithm splits the given data into
        approximately equally sized subsets. The number of subsets equals the number of threads. The runner is called
        on each of the subsets. The final weight vector is the average of multiple runner jobs. It is stored within the
        class for further label prediction of given test data.

            Parameters
            ----------
            X : ndarray of shape (n samples, m features)
                Training data
            y : ndarray of shape (n samples,)
                Training data labels

            Returns
            -------
            None
        """
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

        ws = []
        for Xi, yi in zip(Xs, ys):
            ws.append(self.runner(Xi, yi))

        #ws = Parallel(n_jobs=self.n_threads, backend='threading')(delayed(self.runner)(Xi, yi) for Xi, yi in zip(Xs, ys))

        self.w = sum(ws) / self.n_threads  # compute w by taking the average of sub_ws


class NonLinearFeatures:
    def __init__(self, m=None, sigma=None):
        """
        The NonLinearFeatures class implements an explicit kernel function and projects given input data of dimension
        n x d onto a higher dimensional features space of dimension n x m, where d < m. The projection uses Random
        Fourier Features (RFF) to approximate a Laplacian Kernel.

            Parameters
            ----------
            m : int
                Feature dimension of the higher dimensional space of the projection
            sigma : float
                Hyper parameter of the RFF representation
        """
        self.m = m
        self.sigma = sigma

    def _update_random_variables(self, d):
        """
        The RFF method draws m random samples omega_1, ..., omega_m and b_1, ..., b_m. omega_i are vectors of
        dimension d, drawn from a standard cauchy distribution and b_i are random numbers, drawn from
        a continuous distribution from the interval [0, 2*pi]. Both omega (m, d) and b (m,) are stored within the class
        for further usage.

            Parameters
            ----------
            d : int
                Feature dimension of the given original data

            Returns
            -------
            None
        """
        self.omega = 1 / self.sigma * np.random.standard_cauchy((self.m, d))
        self.b = 2 * np.pi * np.random.rand(self.m)

    def _non_linear_features(self, X):
        """
        Projection of the given input data X through the explicit representation of the kernel function through the RFF
        approximation.

            Parameters
            ----------
            X : ndarray of shape (n samples, d features)
                Training data of original feature dimension

            Returns
            -------
            Z : ndarray of shape (n samples, m features)
                Training data of projected feature dimension
        """
        n_samples, _ = X.shape
        Z = np.sqrt(2 / self.m) * np.cos(self.omega @ X.T + np.outer(self.b, np.ones(n_samples))).T
        return Z

    def fit_transform(self, X):
        """
        Draw random samples for the RFF approximation and transform the data.

            Parameters
            ----------
            X : ndarray of shape (n samples, d features)
                Training data of original feature dimension

            Returns
            -------
            X : ndarray of shape (n samples, m features)
                Training data of projected feature dimension
        """
        _, n_features = X.shape
        self._update_random_variables(n_features)
        X = self._non_linear_features(X)
        return X

    def fit(self, X):
        """
        Draw random samples for the RFF approximation.

            Parameters
            ----------
            X : ndarray of shape (n samples, d features)
                Training data of original feature dimension

            Returns
            -------
            None
        """
        _, n_features = X.shape
        self._update_random_variables(n_features)

    def transform(self, X):
        """
        Transform the data.

            Parameters
            ----------
            X : ndarray of shape (n samples, d features)
                Training data of original feature dimension

            Returns
            -------
            X : ndarray of shape (n samples, m features)
                Training data of projected feature dimension
        """
        X = self._non_linear_features(X)
        return X
