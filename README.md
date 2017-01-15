# perceptron

A class which implements a perceptron and provides functionality to fit a perceptron to a data frame. Provides binary(1 or -1) prediction using the predict method.

## Usage

import perceptron

per = perceptron.Perceptron(learning_rate, max_iterations)

per.fit(data_frame, labels) # lables should contain 1 or -1

per.predict(data) # prediction list (consists of 1 or -1)

## Dependencies

NumPy