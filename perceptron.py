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

    def fit(self, data_frame, y):
        """

        :param data_frame: A matrix of m x n where m is the number of samples and n is the number of features.
        :param y: An array like object with the true labels for the m samples
        :return: self
        """
        pass

    def score(self, sample, weights):
        """

        :param sample: A data point with n features
        :param weights: An array of weights of size n + 1
        :return: The "score" of the current data point for the provided weights
        """
        pass

    def predict(self, sample):
        """

        :param sample: A data point with n features
        :return: The predicted class of the data point
        """
        pass