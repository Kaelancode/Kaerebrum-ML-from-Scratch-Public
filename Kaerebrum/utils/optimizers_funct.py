import numpy as np
from scipy import sqrt as sqrt


def adam(timer, velocityParam, param_grad,  stepParam, alpha=0.001, momentum=0.9, gamma=0.999, eps=1e-7):
    gradient_adapt = []

    for i in range(len(velocityParam)):
        stepParam[i] = gamma * stepParam[i] + (1.-gamma)*np.power(param_grad[i], 2)
        adam_stepParam = stepParam[i]/(1.-np.power(gamma, timer))
        alpha_adaptParam = alpha/((sqrt(adam_stepParam)+eps))

        velocityParam[i] = momentum*velocityParam[i] + (1.-momentum)*param_grad[i]
        adam_velocityParam = velocityParam[i]/(1.-np.power(momentum, timer))
        gradient_adapt.append(alpha_adaptParam * adam_velocityParam)

    return gradient_adapt, velocityParam, stepParam


def momentum(timer, velocityParam, param_grad, stepParam, alpha=0.001, momentum=0.9):
    for i in range(len(velocityParam)):
        velocityParam[i] = momentum * velocityParam[i] + alpha * param_grad[i]
    return velocityParam, velocityParam, stepParam


def nesterov(timer, velocityParam, param_grad, stepParam, alpha=0.001, momentum=0.9):
    advance_weights = weights - momentum * velocityParam
    self.decay*self.velocity - self.alpha*gradient
    return advance_weights
