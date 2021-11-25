import numpy as np


def logloss(X, y, activated_layer):
    cost = (1/(2*X.shape[0]))*np.sum(np.sum((-y * np.log(activated_layer))-(1-y)*np.log(1-activated_layer)))
    #cost = np.sum(-np.log(activated_layer[y])/(1/2*X.shape[0]))
    return cost


def mean_square_error(X, y, activated_layer):
    cost = (1/(2*X.shape[0]))*sum((activated_layer-y)**2)
    return cost
