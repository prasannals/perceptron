import numpy as np


class Perceptron(object):
    """
    Implements a Perceptron
    """
    def __init__(self, learning_rate, max_iters):
        """

        :param learning_rate: float  The learning rate for the perceptron algorithm.
        :param max_iters: int  The maximum allowed iterations of the learning algorithm.
        """
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.weights = []
        self.errors_ = []


    def fit(self, data_frame, y):
        """

        :param data_frame: A matrix of m x n where m is the number of samples and n is the number of features.
        :param y: An array like object with the true labels for the m samples
        :return: self
        """
        self.weights = np.zeros(1 + data_frame.shape[1])
        self.errors_ = []

        for _ in range(self.max_iters):
            errors = 0
            for xi, target in zip(data_frame, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)

            self.errors_.append(errors)
        return self

    def score(self, sample, weights):
        """

        :param sample: A data point with n features
        :param weights: An array of weights of size n + 1
        :return: The "score" of the current data point for the provided weights
        """
        return np.dot(sample, weights[1:]) + weights[0]

    def predict(self, sample):
        """

        :param sample: A data frame with m data points and n features
        :return: The predicted class of the data point
        """
        if self.score(sample, self.weights) >= 0:
            return 1
        else:
            return -1