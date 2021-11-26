import numpy as np
import pandas as pd
import keyboard
import warnings
from scipy import sqrt as sqrt
import joblib

from .utils.activation_funct import *
from .utils.loss_funct import *
from .utils.optimizers_funct import *
from .utils.Utils import check_y, check_x, multinomial_check
from .utils import *


class Axon:
    def __init__(self, hidden_layers=[], output_layer=None, params_init='N', params_modifier=None):
        self.__hidden = hidden_layers
        self.__output = output_layer

        self._params_init = params_init
        self._params_modifier = params_modifier
        self.params = []
        self.biase = []

    def _normal_initialize(self, input, output):
        params = np.random.randn(input, output)
        return params

    def _uniform_initialize(self, input, output):
        params = np.random.uniform(low=-1.0, high=1.0, size=(input, output))
        return params

    def _He_method(self, params_init, inputs, outputs):
        if params_init == 'N':
            modifier = sqrt(2./inputs)
            return modifier
        elif params_init == 'U':
            modifier = sqrt(6./inputs)
            return modifier

    def _Xavier_method(self, params_init, inputs, outputs):
        if params_init == 'N':
            modifier = sqrt(2./(inputs+outputs))
            return modifier
        elif params_init == 'U':
            modifier = sqrt(6./(inputs+outputs))
            return modifier

    def _initialize_params(self, layers, params_init=0, params_modifier=None):
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        Type of distributions to initialize params be
        1) normal: 'N'         -----> initialize weights in normal distribution
        2) uniform" 'U'        -----> initialize weights in uniform distribution
        3) 0: '0'              -----> initialize weights  as zeros
        4) None or otherwise   -----> initialize weights as rand - 0.5

        Type of intialization Methods
        1) He Kai Ming method: 'He'  -> uniform: sqrt(6/inputs) normal: sqrt(1/inputs)  relu: sqrt(2/inputs)
        2) Xavier method: 'X'        -> uniform: sqrt(6/inputs + outputs) normal: sqrt(2/inputs + outputs)
        3) None                      -> *1
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        params_list = []
        if params_init == 'N':
            initizer = self._normal_initialize
        elif params_init == 'U':
            initizer = self._uniform_initialize
        elif params_init == 0:
            for i in range(len(layers)-1):
                params = np.zeros((layers[i], layers[i+1]))
                params_list.append(params)
            return params_list
        else:
            for i in range(len(layers)-1):
                params = np.random.rand(layers[i], layers[i+1])-0.5
                params_list.append(params)
            return params_list

        if params_modifier == 'He':
            modifier = self._He_method
        elif params_modifier == 'X':
            modifier = self._Xavier_method
        else:
            modifier = lambda x, y, z: 1
        for i in range(len(layers)-1):
            # implement modifier specific to activations in future
            params = initizer(layers[i], layers[i+1]) * modifier(params_init, layers[i], layers[i+1])
            params_list.append(params)

        return params_list

    def _initialize_biase(self, layers):
        biase_list = []
        for i in range(len(layers)-1):
            if self._activations[i] == relu:
                biase = np.ones((1, layers[i+1]))*0.1
                biase_list.append(biase)
            else:
                # implement modifier specific to activations in future
                biase = np.zeros((1, layers[i+1]))
                biase_list.append(biase)
        return biase_list

    def gen_weights(self, layers, params_init, params_modifier=None):
        params = self._initialize_params(layers, params_init, params_modifier)
        biase = self._initialize_biase(layers)
        return (params, biase)

    def build(self, activations=[sigmoid, sigmoid], optimizer=adam, loss=logloss):
        # number of activations must be equal to params or total_years - 1
        self._activations = []
        self._activations = self._activations + activations

        self._optimizer_func = optimizer
        self._loss_func = logloss

    def kill_button(self):
        if keyboard.is_pressed("q"):
            return True
        else:
            return False

    def check_batch(self, batch):
        if batch == 0:
            batch = self._m
            print('Full Batch will be Trained')
        else:
            assert batch <= self._m, 'Batch size is bigger than number of samples.'
        return batch

    def print_info(self, iters, alpha, opt, loss, reg, rand):
        print('No of iterations/ n_iters:', iters)
        print('Learning rate/ alpha:', alpha)
        print('Optimizer', opt)
        print('Loss Function:', loss)
        print('Regularization:', reg)
        print('Shuffle data on each iteration:', rand)

    def train(self, X, y, alpha=0.001, epoch=10, batch=0, params=None, biases=None):
        X, self._n, self._m = check_x(X)
        y = check_y(y)  # Ensure y is in column vector
        labels, n_labels = multinomial_check(y)
        Y_hotcoded = (np.arange(n_labels) == np.atleast_2d(y))*1
        self.__input = self._n
        self.total_layers = [self.__input] + self.__hidden + [self.__output]
        print(self.total_layers)
        self._activated_layers = list(range(len(self.total_layers))) # should be in train
        self._activated_layers[0] = X
        self.gradients = list(range(len(self.total_layers)-1))
        self.biases_grad = list(range(len(self.total_layers)-1))
        self.cost = []

        batch = self.check_batch(batch)
        extra = (1 if X.shape[0] % batch else 0)
        iters = X.shape[0]//batch + extra
        self.params, self.biases = self.gen_weights(self.total_layers, self._params_init, self._params_modifier)
        '''
        if params is not None:
            self.params = params
        if biases is not None:
            self.biases = biases
        '''
        velocityParam = [i * 0 for i in self.params]
        stepParam = [i * 0 for i in self.params]

        self.print_info(epoch, alpha, self._optimizer_func.__str__(), self._loss_func.__str__(), None, False)
        timer = 1
        self.m = self._m

        for e in range(epoch):
            if batch == self._m:
                cost, gradient_adapt, velocityParam, stepParam = self.trainer(X, Y_hotcoded, alpha, timer, velocityParam, stepParam)

                if self.kill_button():
                    print('\n(q)Killed Training')
                    break

                for j in range(len(self.params)):
                    #print('Change', len(gradient_adapt))
                    #print(gradient_adapt)
                    self.params[j] -= gradient_adapt[j]
                    #self.biases[j] -= biase_adapt[j]
                    self.biases[j] -= self.biase_grad[j]/self.m
                timer += 1

            else:
                for i in range(iters):
                    start = i * batch
                    end = start + batch
                    end = self._m if end >= self._m else end
                    batch_range = list(range(start, end))
                    X_batch = X[batch_range, :]
                    y_batch = y[batch_range]
                    self.m = X_batch.shape[0]
                    self._activated_layers[0] = X_batch
                    Y_hotcoded = (np.arange(n_labels) == np.atleast_2d(y_batch))*1

                    cost, gradient_adapt, velocityParam, stepParam = self.trainer(X_batch, Y_hotcoded, alpha, timer, velocityParam, stepParam)
                    self.gd=gradient_adapt


                    if self.kill_button():
                        print('\n(q)Killed Training')
                        break

                    for j in range(len(self.params)):
                        #print('Change', len(gradient_adapt))
                        #print(gradient_adapt)
                        #print('l', gradient_adapt[j], 'L2', len(gradient_adapt))
                        #print('P',self.params, 'P2', len(self.params))
                        self.params[j] -= gradient_adapt[j]
                        #self.biases[j] -= biase_adapt[j]
                        self.biases[j] -= self.biase_grad[j]/self.m
                        if self.kill_button():
                            print('\n(q)Killed Training')
                            break

                    timer += 1

            self._activated_layers[0] = X
            output = self._forward_propagate(X, self.params, self.biases, self._activated_layers, self._activations)
            current_output = self.get_predictions(output)
            print('Current accuracy: {} for epoch {}:'.format(self.accuracy(current_output, y), e+1))
            if self.kill_button():
                print('\n(q)Killed Training')
                break



            print('Cost:', cost)
            self.cost.append(cost)
            #self.cost.append[cost]
        return output

    def trainer(self, X_batch, Y_hotcoded, alpha, timer, velocityParam, stepParam):
        output = self._forward_propagate(X_batch, self.params, self.biases, self._activated_layers, self._activations)
        cost = logloss(X_batch, Y_hotcoded, self._activated_layers[-1])
        error = output - Y_hotcoded
        self.gradients, self.biase_grad = self._backward_propagate(error, self.params, self.biases, self._activated_layers, self.gradients, self.biases_grad, self._activations)

        gradient_adapt, velocityParam, stepParam = self._optimizer_func(timer, velocityParam, self.gradients, stepParam, alpha)

        return cost, gradient_adapt, velocityParam, stepParam

    def _forward_propagate(self, X, params, biases, activated_layers, activations):
        '''
        for i, param, biase, activated_layer, activation in enumerate(params, biases, activated_layers, activations):
            hypothesis = np.dot(X, param) + biase
            activated = activation(hypothesis)
            activated_layers[i] = activated
        '''

        for i, param in enumerate(params):
            hypothesis = np.dot(activated_layers[i], param) + biases[i]
            #activate = activations[i]  # get the activation function
            activated = activations[i][0](hypothesis)
            self._activated_layers[i+1] = activated
        return activated

    def _backward_propagate(self, error, params, biases, activated_layers, gradients, biase_grad, activations):
        #error /= self.m
        #biase_grad = []
        #biase_grad.append(np.sum(error, 0, keepdims=True))

        biase_grad[-1] = np.sum(error, 0, keepdims=True)
        gradients[-1] = activated_layers[-2].T@error

        for i in reversed(range(len(gradients)-1)):
            # i start from last number
            #activate = activations[i]
            delta = error@params[i+1].T * activations[i][1](activated_layers[i+1])
            gradients[i] = 1/self.m * (activated_layers[i].T@delta)
            #gradients[i] = (activated_layers[i].T@delta)
            error = delta
            #biase_grad.insert(0, np.sum(error, 0, keepdims=True))
            biase_grad[i] = np.sum(error, 0, keepdims=True)

        return gradients, biase_grad

    def get_predictions(self, output):
        return np.argmax(output, axis=1)

    def accuracy(self, predictions, y):
        return np.sum(predictions == y.T)/y.shape[0]

    def predict(self, X, y=None, params=None, biases=None):
        X, _, _ = check_x(X)
        if params is not None and biases is not None:
            _params = params
            _biases = biases
        elif (params is not None and biases is None) or (params is  None and biases is not None):
            print('params and biases must be both filled, or both None')
        else:
            _params = self.params
            _biases = self.biases

        #y = (np.arange(self.n_labels) == np.atleast_2d(y).T)*1
        self._activated_layers[0] = X
        output = self._forward_propagate(X, _params, _biases, self._activated_layers, self._activations)
        predictions = self.get_predictions(output)
        return predictions, output

    def save(self, model, name):
        joblib.dump(model, name)
