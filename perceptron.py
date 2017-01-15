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

    def fit(self, data_frame, y):
        """

        :param data_frame: A matrix of m x n where m is the number of samples and n is the number of features.
        :param y: An array like object with the true labels for the m samples
        :return: self
        """
        self.weights = np.zeros(len(y) + 1)   # initialize weights to 0



    def score(self, sample, weights):
        """

        :param sample: A data point with n features
        :param weights: An array of weights of size n + 1
        :return: The "score" of the current data point for the provided weights
        """
        return np.dot(sample, weights[1:])

    def predict(self, sample):
        """

        :param sample: A data frame with m data points and n features
        :return: The predicted class of the data point
        """
        pred = []
        for entry in sample:
            if self.score(entry, self.weights) >= 0:
                pred.append(1)
            else:
                pred.append(-1)
        return np.array(pred)