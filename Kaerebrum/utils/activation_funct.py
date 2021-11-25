from scipy.special import expit
import numpy as np


def activate_sigmoid(z):
    return expit(z)


def derive_sigmoid(z):
    return z * (1. - z)


def activate_relu(z):
    return np.maximum(0, z)


def derive_relu(z):
    return z > 0


def activate_softmax(z):
    return np.exp(z)/np.sum(np.exp(z))


def sigmoid_func():
    activation = []
    activation.append(activate_sigmoid)
    activation.append(derive_sigmoid)
    return activation


sigmoid = sigmoid_func()


def relu_func():
    activation = []
    activation.append(activate_relu)
    activation.append(derive_relu)
    return activation


relu = relu_func()
