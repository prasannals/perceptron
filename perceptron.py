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
            preds = self.predict(data_frame)
            indicator = (y - preds)
            row_update = indicator * self.learning_rate

            self.weights[0] = self.weights[0] + np.sum(row_update)
            self.weights[1:] = self.weights[1:] + np.sum( ((data_frame[:,1:].transpose() * row_update).transpose()), axis=0 )

            self.errors_.append((preds != y).sum())

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
        pred = []

        for row in sample:
            if self.score(row, self.weights) >= 0:
                pred.append(1)
            else:
                pred.append(-1)

        return np.array(pred)