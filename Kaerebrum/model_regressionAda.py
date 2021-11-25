import numpy as np
import pandas as pd
import math
import keyboard
import warnings

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.optimize as op
from mlcode.utilsGit import*
import plotly.express as px


class Regression:
    def __init__(self, alpha=0.01, n_iters=1000, ld=0):
        self.alpha = alpha
        self.n_iters = n_iters
        self.ld = ld  # ld is lambda for regularization
        self.binomial = False

        self.m = 0  # no of n_samples
        self.n = 0  # no of features
        self.weights = None
        self.cost = []
        self.loss = None
        self.reg = None
        self.y = None
        self.loss_function = None
        self.reg_function = None
        self.learning_function = None
        self.momentum_function = None
        self.weights_function = None
        self.animate = None

    def L2_loss(self, hypothesis, X, y, reg, reg_d, bounce=False):
        #print('L2 loss', bounce)
        cost = (1/(2*self.m))*sum((hypothesis-y)**2)+reg
        if bounce:
            return cost
        gradient = (1/self.m)*((hypothesis-y)@X) + reg_d
        weights = self.weights-self.alpha*gradient  # diff weights so that scipy will not update self.weights
        return weights, cost, gradient

    def L1_loss(self, hypothesis, X, y, reg, reg_d, bounce=False):
        #print('L1 loss', bounce)
        cost = (1/(2*self.m))*sum(np.abs(hypothesis-y))+reg
        if bounce:
            return cost
        gradient = (1/self.m)*(np.sign(hypothesis-y)@X)+reg_d
        weights = self.weights-self.alpha*gradient  # diff weights so that scipy will not update self.weights
        return weights, cost, gradient

    def L2_reg(self, weights):
        # print('weight[1:]',weights[1:])
        reg = self.ld/(2*self.m)*(weights[1:]@weights[1:].T)
        # print('reg', reg)  # square itself and sum it up
        reg_d = np.insert(self.ld/self.m*weights[1:], 0, 0)
        # print('reg_d', reg_d)
        return reg, reg_d

    def L1_reg(self, weights):
        np.seterr(divide='ignore', invalid='ignore')
        reg = self.ld/(2*self.m)*(sum(np.abs(weights[1:])))  # sum to 1 number to add to cost
        reg_d = np.insert(self.ld/self.m*np.sign(weights[1:]), 0, 0)
        return reg, reg_d

    def opt_none(self, alpha, adapt_rate, gamma, gradient, timer=None, gradient_adapt=None):
        return None, alpha

    def opt_weights(self, velocity, momentum, weights):
        return weights

    def opt_Nesterov(self, velocity, momentum, weights):
        advance_weights = weights - momentum*velocity
        return advance_weights

    def opt_adagrad(self, alpha, adapt_rate, gamma, gradient, timer=None, gradient_adapt=None):
        adapt_rate += gradient**2
        alpha_adapt = alpha/(np.sqrt(adapt_rate) + 1e-8)
        return adapt_rate, alpha_adapt

    def opt_adadelta(self, alpha, adapt_rate, gamma, gradient, timer=None, gradient_adapt=None):
        adapt_rate = gamma*adapt_rate + (1-gamma)*gradient**2
        self.gradient_rate = gamma*self.gradient_rate + (1-gamma)*gradient_adapt**2
        rms_grad = np.sqrt(adapt_rate + 1e-6)
        rms_theta = np.sqrt(self.gradient_rate + 1e-6)
        alpha_adapt = rms_theta/rms_grad
        return adapt_rate, alpha_adapt

    def opt_rmsprop(self, alpha, adapt_rate, gamma, gradient, timer=None, gradient_adapt=None):
        adapt_rate = gamma * adapt_rate + (1-gamma)*gradient**2
        alpha_adapt = alpha/(np.sqrt(adapt_rate)+1e-6)
        return adapt_rate, alpha_adapt

    def opt_adam(self, alpha, adapt_rate, gamma, gradient, timer, gradient_adapt=None):
        adapt_rate = gamma * adapt_rate + (1-gamma)*gradient**2
        adam_correcteded_rate = adapt_rate/(1-gamma**timer)
        alpha_adapt = alpha/((np.sqrt(adam_correcteded_rate)+1e-8))
        return adapt_rate, alpha_adapt

    def opt_adam_momentum(self, velocity, momentum, alpha_adapt, gradient, timer):
        velocity = momentum*velocity + (1-momentum)*gradient
        adam_corrected_velocity = velocity/(1-momentum**timer)
        gradient_adapt = alpha_adapt * adam_corrected_velocity
        return velocity, gradient_adapt

    def opt_nadam_momentum(self, velocity, momentum, alpha_adapt, gradient, timer):
        velocity = momentum*velocity + (1-momentum)*gradient
        adam_corrected_velocity = velocity/(1-momentum**timer)
        advance_weights = (momentum*adam_corrected_velocity)+((1-momentum)*gradient)/(1-momentum**timer)
        gradient_adapt = alpha_adapt * advance_weights
        return velocity, gradient_adapt

    def opt_momentum(self, velocity, momentum, alpha_adapt, gradient, timer=None):
        velocity = momentum*velocity + alpha_adapt*gradient
        return velocity, velocity

    def check_y(self, y):
        if isinstance(y, (list)):
            y = np.asarray(y)
        elif isinstance(y, (pd.core.series.Series, pd.DataFrame)):
            y = y.to_numpy()

        if isinstance(y, (pd.core.series.Series, np.ndarray)):
            y = y.ravel()
        else:
            pass
        return y

    def check_x(self, X):
        if isinstance(X, (list)):
            X = np.asarray(X)
        elif isinstance(X, (pd.core.series.Series, pd.DataFrame)):
            X = X.to_numpy()
        m = X.shape[0]
        try:
            n = X.shape[1]
            X = np.insert(X, 0, np.ones(m), axis=1)
        except IndexError:
            n = 1
            X = np.stack((np.ones(m), X), 1)
        return X, n, m

    def check_multinomial(self):
        # Determine loss type and reg type
        if not self.multinomial:
            if self.loss == 'L2':
                self.loss_function = self.L2_loss
            if self.loss == 'L1':
                self.loss_function = self.L1_loss
            if self.reg == 'L2':
                self.reg_function = self.L2_reg
            if self.reg == 'L1':
                self.reg_function = self.L1_reg
        else:
            pass
        return None

    def check_optimizer(self, optimizer, momentum, alpha, gamma):
        if optimizer == 'adam':
            if gamma != 0.999 or momentum != 0.9 or alpha != 0.001:
                warnings.warn('Recommended setting for ADAM: alpha = 0.001, gamma = 0.999, momentum = 0.9')
                warnings.warn('Press q to kill training and reset')
            self.weights_function = self.opt_weights
            self.learning_function = self.opt_adam
            self.momentum_function = self.opt_adam_momentum
            return None
        elif optimizer == 'nadam':
            if gamma != 0.999 or momentum != 0.9 or alpha != 0.001:
                warnings.warn('Recommended setting for NADAM: alpha = 0.001, gamma = 0.999, momentum = 0.9')
                warnings.warn('Press q to kill training and reset')
            self.weights_function = self.opt_weights
            self.learning_function = self.opt_adam
            self.momentum_function = self.opt_nadam_momentum
        elif optimizer == 'nesterov':
            if momentum == 0 or alpha != 0.001 or gamma > 0:
                warnings.warn('momentum needs to > 0 for impact, recommend lower alpha to 0.001 for small batches <64, gamma to be 0')
                warnings.warn('Press q to kill training and reset')
            self.learning_function = self.opt_none
            self.weights_function = self.opt_Nesterov
            self.momentum_function = self.opt_momentum
            return None
        elif optimizer == 'adadelta':
            if alpha > 0 or gamma == 0:
                warnings.warn('Adadelta no tuning for alpha, gamma > 0 is required . Optimal setting at 0.9. Lower according to batch sizes (Try 0.7 at batch size 64)')
                warnings.warn('Press q to kill training and reset')
            self.learning_function = self.opt_adadelta
        elif optimizer == 'adagrad':
            if gamma > 0 or alpha != 0.01:
                warnings.warn('Recommended setting for ADGRAD: alpha = 0.1, start high as gradient get smaller fast, lower alpha if batch sizes are big ( lower for larger batches or sparse data) , gamma = zero , No tuning needed for gamma')
                warnings.warn('Press q to kill training and reset')
            self.learning_function = self.opt_adagrad
        elif optimizer == 'rmsprop':
            if momentum != 0 or alpha != 0.001:
                warnings.warn('momentum needs to be 0 to work as intended, Recommended setting for RMSPROP: alpha = 0.001, gamma = 0.9')
                warnings.warn('Press q to kill training and reset')
            self.learning_function = self.opt_rmsprop
        else:
            print('WARNING :Optimizer not recognized. No optimizer will be utilized')
            self.learning_function = self.opt_none
        self.weights_function = self.opt_weights
        self.momentum_function = self.opt_momentum
        return None

    def check_weights(self, weights):
        checked_weights = weights
        if checked_weights is None:
            checked_weights = np.zeros(self.n+1)
        return checked_weights

    def check_batch(self, batch):
        if batch == 0:
            batch = self.m
            print('Full Batch will be Trained')
        else:
            assert batch < self.m, 'Batch size is bigger than number of samples.'
        return batch

    def check_best_fit(self, best_fit, cost_list, weights):
        if best_fit['lowest_cost'] >= cost_list[-1]:
            best_fit['weights'] = weights
            best_fit['lowest_cost'] = cost_list[-1]
        return best_fit

    def check_convergence(self, previous_cost, cost_list, threshold, iter):
        if (abs(previous_cost-cost_list[-1]) < threshold and previous_cost > cost_list[-1]) or cost_list[-1] <= 0:
            print('\nConverged at threshold:{}, epoch:{}'.format(threshold, iter))
            return True, previous_cost
        else:
            previous_cost_ = cost_list[-1]
            return False, previous_cost_

    def check_rand(self, check_best_fit, check_convergence, best_fit, cost_list, weights, previous_cost, threshold, iter):
        best = check_best_fit(best_fit, cost_list, weights)
        if self.batch < self.m:
            check, previous_cost_ = check_convergence(previous_cost, cost_list, threshold, iter)
        return best, check, previous_cost_

    def print_info(self, iters, alpha, ld, loss, reg, rand):
        print('No of iterations/ n_iters:', iters)
        print('Learning rate/ alpha:', alpha)
        print('Lamda/ ld:', ld)
        print('Loss Function:', loss)
        print('Regularization:', reg)
        print('Shuffle data on each iteration:', rand)

    def print_results(self, cost_history):
        print('\nMin Cost:', min(cost_history))
        print('Cost:', cost_history[-1])

    def kill_button(self):
        if keyboard.is_pressed("q"):
            return True
        else:
            return False

    def fit(self, X, y, weights=None, loss='L2', reg='L2', threshold=1e-5, rand=False):
        y = self.check_y(y)
        self.y = y  # self.y , self.hypothesis only utilize in predictions

        # m to get number of rows in data X
        # self.m = X.shape[0]
        # n to get number of columns in data X and shape X for ones in 1st col
        X, self.n, self.m = self.check_x(X)
        # check weights
        self.weights = self.check_weights(weights)

        self.cost, self.loss, self.reg, self.rand = [], loss, reg, rand  # reset the cost
        self.batch, self.previous = self.m, np.inf
        self.best_fit = {'lowest_cost': np.inf, 'weights': self.weights}
        # Determine loss type and reg type
        self.check_multinomial()

        # Print out Info
        self.print_info(self.n_iters, self.alpha, self.ld, self.loss, self.reg, self.rand)
        print('Optimizing Function: Gradient Descent')

        # perform gradient descent
        for i in range(1, self.n_iters+1):
            if (not i % 100) and (i > 1):
                print('Iteration {} running, Current cost: {}'.format(i, self.cost[i-2]), end='\r')
            if self.kill_button():
                print('\n(q)Killed Training')
                break

            if rand:
                shuffle = np.random.permutation(self.m)
                X = X[shuffle, :]
                y = y[shuffle]

            hypothesis = self.implement_regression(self.weights, X)
            reg, reg_d = self.reg_function(self.weights)
            self.weights, cost, _ = self.loss_function(hypothesis, X, y, reg, reg_d)
            self.cost.append(cost)
            self.best_fit = self.check_best_fit(self.best_fit, self.cost, self.weights)
            if self.batch < self.m:
                converge, self.previous = self.check_convergence(self.previous, self.cost, threshold, i)
                if converge:
                    break
            #self.previous = self.cost[i-1]
        self.print_results(self.cost)

    def sgd_fit(self, X, y, weights=None, batch=1, loss='L2', reg='L2', momentum=0.8, threshold=1e-5, rand=True):
        y = self.check_y(y)
        self.y = y  # self.y , self.hypothesis only utilize in predictions

        # m to get number of rows in data X
        # self.m = X.shape[0]
        # n to get number of columns in data X and shape X for ones in 1st col
        X, self.n, self.m = self.check_x(X)
        batch = self.check_batch(batch)

        # check weights
        self.weights = self.check_weights(weights)
        self.cost, self.loss, self.reg, self.rand = [], loss, reg, rand  # reset the cost
        self.batch, self.batch_cost, self.previous = batch, [], np.inf
        self.velocity, self.momentum = self.weights*0, momentum
        self.best_fit = {'lowest_cost': np.inf, 'weights': self.weights}
        # Determine loss type and reg type
        self.check_multinomial()

        # Print out Info
        self.print_info(self.n_iters, self.alpha, self.ld, self.loss, self.reg, self.rand)
        print('Batch size:', self.batch)
        print('Momentum:', self.momentum)
        print('Optimizing Function: (Mini-Batch)Stocastic Descent with Momentum')

        fig, ax = self.set_plot()
        # check for remainders, addition batch needed
        extra = (1 if self.m % self.batch else 0)

        # perform SGD Descent
        for i in range(1, self.n_iters+1):
            if not i % 100:
                print('Epoch {} running, Current cost: {}'.format(i, self.cost[i-2]), end='\r')
            if self.kill_button():
                print('\n(q)Killed Training')
                break
            if rand:
                shuffle = np.random.permutation(self.m)
                X = X[shuffle, :]
                y = y[shuffle]
            if keyboard.is_pressed("w"):
                self.cost_update(fig, ax)

            for j in range(X.shape[0]//self.batch + extra):
                start = j * self.batch
                end = start + self.batch
                end = X.shape[0] if end >= X.shape[0] else end
                batch = list(range(start, end))
                X_batch = X[batch, :]
                y_batch = y[batch]

                self.m = X_batch.shape[0]
                hypothesis = self.implement_regression(self.weights, X_batch)
                reg, reg_d = self.reg_function(self.weights)
                _, batch_cost, gradient = self.loss_function(hypothesis, X_batch, y_batch, reg, reg_d)
                self.velocity = self.momentum*self.velocity + self.alpha*gradient
                if self.multinomial:
                    self.weights -= self.velocity
                else:
                    self.weights -= self.velocity.ravel()
                self.batch_cost.append(batch_cost)

            self.m = X.shape[0]
            hypothesis = self.implement_regression(self.weights, X)
            cost = self.loss_function(hypothesis, X, y, reg, reg_d, bounce=True)
            self.cost.append(cost)
            self.best_fit, converged, self.previous = self.check_rand(self.check_best_fit, self.check_convergence, self.best_fit, self.cost, self.weights, self.previous, threshold, i)
            if converged:
                break

        self.print_results(self.cost)

    def nag_fit(self, X, y, weights=None, batch=32, loss='L2', reg='L2', decay=0.8, threshold=1e-5, rand=True):
        y = self.check_y(y)
        self.y = y  # self.y , self.hypothesis only utilize in predictions

        # m to get number of rows in data X
        # self.m = X.shape[0]
        # n to get number of columns in data X and shape X for ones in 1st col
        X, self.n, self.m = self.check_x(X)
        batch = self.check_batch(batch)
        # check weights
        self.weights = self.check_weights(weights)

        self.cost, self.loss, self.reg, self.rand = [], loss, reg, rand  # reset the cost
        self.batch, self.batch_cost, self.previous = batch, [], np.inf
        self.velocity, self.decay = self.weights*0, decay
        self.best_fit = {'lowest_cost': np.inf, 'weights': self.weights}
        # Determine loss type and reg type
        self.check_multinomial()

        # Print out Info
        self.print_info(self.n_iters, self.alpha, self.ld, self.loss, self.reg, self.rand)
        print('Batch size:', self.batch)
        print('Decay:', self.decay)
        print('Optimizing Function: Nestero Acelerated Gradient Descent')

        # check for remainders, addition batch needed
        extra = (1 if self.m % self.batch else 0)
        # perform NAG Descent
        for i in range(1, self.n_iters + 1):
            if not i % 100 and i > 1:
                print('Epoch {} running, Current cost: {}'.format(i, self.cost[i-2]), end='\r')
            if self.kill_button():
                print('\n(q)Killed Training')
                break
            if rand:
                shuffle = np.random.permutation(self.m)
                X = X[shuffle, :]
                y = y[shuffle]

            for j in range(X.shape[0]//self.batch + extra):
                start = j * self.batch
                end = start + self.batch
                end = X.shape[0] if end >= X.shape[0] else end
                batch = list(range(start, end))
                X_batch = X[batch, :]
                y_batch = y[batch]

                self.m = X_batch.shape[0]
                advance_weights = self.weights + self.decay * self.velocity
                hypothesis = self.implement_regression(advance_weights, X_batch)
                reg, reg_d = self.reg_function(advance_weights)
                _, batch_cost, gradient = self.loss_function(hypothesis, X_batch, y_batch, reg, reg_d)
                self.velocity = self.decay*self.velocity - self.alpha*gradient
                if self.multinomial:
                    self.weights += self.velocity
                else:
                    self.weights += self.velocity.ravel()
                self.batch_cost.append(batch_cost)

            self.m = X.shape[0]
            hypothesis = self.implement_regression(self.weights, X)
            cost = self.loss_function(hypothesis, X, y, reg, reg_d, bounce=True)
            self.cost.append(cost)
            self.best_fit, converged, self.previous = self.check_rand(self.check_best_fit, self.check_convergence, self.best_fit, self.cost, self.weights, self.previous, threshold, i)
            if converged:
                break

        self.print_results(self.cost)

    def adaptive_fit(self, X, y, weights=None, batch=32, loss='L2', reg='L2', momentum=0, gamma=0.9, threshold=1e-5, opt='rmsprop', rand=True):
        ''' Optimal learning rate at 0.001, yearns the best result under most circumstances
            Base optimizer is stochastic gradient descent if opt is set to None or the optimizer is not recognized.
            Momentum will be activated as long as > 0
            Default optimizer is 'rmsprop'
            List of optimizers
            1) 'rmsprop'    momentum should be set to 0 for intended effect. gamma at 0.9
            2) 'adam'       recommend to set momentum at 0.9 and gamma at 0.999
            3  'nesterov'   recommend to set momentum at 0.9 and gamma at 0.999
            4) 'adadelta' increasing gamma will increase the denominator, resulting in a small alpha
            5) 'adagrad'
            if batch is set to 0 , then the FULL batch will be trained
        '''
        y = self.check_y(y)
        self.y = y  # self.y , self.hypothesis only utilize in predictions
        X, self.n, self.m = self.check_x(X)  # n to get number of columns in data X and shape X for ones in 1st col
        batch = self.check_batch(batch)
        self.weights = self.check_weights(weights)  # check weights
        self.check_optimizer(opt, momentum, self.alpha, gamma)
        # initialize all necesarry variables
        self.cost, self.loss, self.reg, self.rand = [], loss, reg, rand
        self.batch, self.batch_cost, self.previous = batch, [], np.inf
        self.velocity, self.adapt_rate, self.momentum, self.gamma = self.weights*0, self.weights*0, momentum, gamma
        self.gradient_adapt, self.gradient_rate = self.weights*0, self.weights*0
        self.best_fit = {'lowest_cost': np.inf, 'weights': self.weights}
        self.optimizer = opt

        # Determine loss type and reg type
        self.check_multinomial()
        # Print out Info
        self.print_info(self.n_iters, self.alpha, self.ld, self.loss, self.reg, self.rand)
        print('Batch size:', self.batch)
        print('Momentum:', self.momentum)
        print('Gamma:', self.gamma)
        print('Optimizing Function:', self.optimizer)
        # check for remainders, addition batch needed
        extra = (1 if self.m % self.batch else 0)
        timer = 1
        # perform SGD Descent
        for i in range(1, self.n_iters+1):
            if not i % 100:
                print('Epoch {} running, Current cost: {}'.format(i, self.cost[i-2]), end='\r')
            if self.kill_button():
                print('\n(q)Killed Training')
                break
            if rand:
                shuffle = np.random.permutation(self.m)
                X = X[shuffle, :]
                y = y[shuffle]

            for j in range(X.shape[0]//self.batch + extra):
                start = j * self.batch
                end = start + self.batch
                end = X.shape[0] if end >= X.shape[0] else end
                batch = list(range(start, end))
                X_batch = X[batch, :]
                y_batch = y[batch]
                if self.kill_button():
                    break
                self.m = X_batch.shape[0]
                checked_weights = self.weights_function(self.velocity, self.momentum, self.weights)
                hypothesis = self.implement_regression(checked_weights, X_batch)
                reg, reg_d = self.reg_function(checked_weights)
                _, batch_cost, gradient = self.loss_function(hypothesis, X_batch, y_batch, reg, reg_d)
                #  self.adapt_rate = self.gamma * self.adapt_rate + (1-self.gamma)*gradient**2
                #  alpha_adapt = self.alpha/((np.sqrt(self.adapt_rate)+1e-6))
                self.adapt_rate, alpha_adapt = self.learning_function(self.alpha, self.adapt_rate, self.gamma, gradient, timer, self.gradient_adapt)
                #  self.velocity = self.momentum*self.velocity + alpha_adapt*gradient
                self.velocity, self.gradient_adapt = self.momentum_function(self.velocity, self.momentum, alpha_adapt, gradient, timer)

                if self.multinomial:
                    self.weights -= self.gradient_adapt
                else:
                    self.weights -= self.gradient_adapt.ravel()
                self.batch_cost.append(batch_cost)

                timer += 1

            self.m = X.shape[0]
            hypothesis = self.implement_regression(self.weights, X)
            cost = self.loss_function(hypothesis, X, y, reg, reg_d, bounce=True)
            self.cost.append(cost)
            self.best_fit, converged, self.previous = self.check_rand(self.check_best_fit, self.check_convergence, self.best_fit, self.cost, self.weights, self.previous, threshold, i)
            if converged:
                break

        self.print_results(self.cost)

    def adam_fit(self, X, y, weights=None, batch=32, loss='L2', reg='L2', momentum=0.9, gamma=0.999, threshold=1e-5, opt='adam', rand=True):
        ''' default learning rate at 0.001, learns the best result under most circumstances
        '''
        y = self.check_y(y)
        self.y = y  # self.y , self.hypothesis only utilize in predictions
        X, self.n, self.m = self.check_x(X)  # n to get number of columns in data X and shape X for ones in 1st col
        batch = self.check_batch(batch)
        self.weights = self.check_weights(weights)  # check weights
        # initialize all necesarry variables
        self.cost, self.loss, self.reg, self.rand = [], loss, reg, rand
        self.batch, self.batch_cost, self.previous = batch, [], np.inf
        self.velocity, self.adapt_rate, self.momentum, self.gamma = self.weights*0, self.weights*0, momentum, gamma
        self.best_fit = {'lowest_cost': np.inf, 'weights': self.weights}
        self.optimizer = opt
        assert self.optimizer in ['adam', 'nadam', 'nesdam'], 'Optimizer must be adam, nadam or nesdam'

        # Determine loss type and reg type
        self.check_multinomial()
        # Print out Info
        self.print_info(self.n_iters, self.alpha, self.ld, self.loss, self.reg, self.rand)
        print('Batch size:', self.batch)
        print('Momentum:', self.momentum)
        print('Gamma:', self.gamma)
        print('Optimizing Function:', self.optimizer)
        # check for remainders, addition batch needed
        extra = (1 if self.m % self.batch else 0)
        timer = 1
        # perform SGD Descent
        for i in range(1, self.n_iters+1):
            if not i % 100:
                print('Epoch {} running, Current cost: {}'.format(i, self.cost[i-2]), end='\r')
            if self.kill_button():
                print('\n(q)Killed Training')
                break
            if rand:
                shuffle = np.random.permutation(self.m)
                X = X[shuffle, :]
                y = y[shuffle]

            for j in range(X.shape[0]//self.batch + extra):
                start = j * self.batch
                end = start + self.batch
                end = X.shape[0] if end >= X.shape[0] else end
                batch = list(range(start, end))
                X_batch = X[batch, :]
                y_batch = y[batch]

                self.m = X_batch.shape[0]
                if opt == 'nesdam':
                    advance_weights = self.weights - self.momentum * self.velocity
                    hypothesis = self.implement_regression(advance_weights, X_batch)
                else:
                    hypothesis = self.implement_regression(self.weights, X_batch)
                reg, reg_d = self.reg_function(self.weights)
                _, batch_cost, gradient = self.loss_function(hypothesis, X_batch, y_batch, reg, reg_d)

                self.adapt_rate = self.gamma * self.adapt_rate + (1-self.gamma)*gradient**2
                if opt == 'nesdam':
                    #self.velocity = self.momentum*self.velocity + (1-self.momentum)*gradient
                    self.velocity = self.momentum*self.velocity
                else:
                    self.velocity = self.momentum*self.velocity + (1-self.momentum)*gradient
                adam_alpha = self.adapt_rate/(1-self.gamma**timer)
                adam_velocity = self.velocity/(1-self.momentum**timer)
                alpha_adapt = self.alpha/((np.sqrt(adam_alpha)+1e-8))
                if opt == 'nadam':
                    advance_weights = (self.momentum*adam_velocity)+((1-self.momentum)*gradient)/(1-self.momentum**timer)
                    gradient_adapt = alpha_adapt * advance_weights
                elif opt == 'nesdam':
                    #gradient_adapt = alpha_adapt*self.velocity
                    gradient_adapt = self.velocity + alpha_adapt*gradient
                else:
                    gradient_adapt = alpha_adapt * adam_velocity
                #self.velocity = self.momentum*self.velocity + alpha_adapt*(1-self.momentum)*gradient
                if self.multinomial:
                    self.weights -= gradient_adapt
                else:
                    self.weights -= gradient_adapt.ravel()
                timer += 1
                self.batch_cost.append(batch_cost)

            self.m = X.shape[0]
            hypothesis = self.implement_regression(self.weights, X)
            cost = self.loss_function(hypothesis, X, y, reg, reg_d, bounce=True)
            self.cost.append(cost)
            self.best_fit, converged, self.previous = self.check_rand(self.check_best_fit, self.check_convergence, self.best_fit, self.cost, self.weights, self.previous, threshold, i)
            if converged:
                break

        self.print_results(self.cost)

    def scipy_fit(self, X, y, weights=None, loss='L2', reg='L2', opt='TNC'):
        y = self.check_y(y)
        self.y = y

        # m to get number of rows in data X
        # self.m = X.shape[0]
        # n to get number of columns in data X and shape X for ones in 1st col
        X, self.n, self.m = self.check_x(X)
        # check weights
        self.weights = self.check_weights(weights)

        self.cost = []  # reset the cost
        self.loss = loss
        self.reg = reg
        # Determine loss type and reg type
        self.check_multinomial()

        # Print out Info
        self.print_info(self.n_iters, self.alpha, self.ld, self.loss, self.reg, False)
        print('Scipy optimise:', opt)

        def scipy_cost_func(weights, X, y):
            hypothesis = self.implement_regression(weights, X)
            reg, reg_d = self.reg_function(weights)
            _, cost, gradient = self.loss_function(hypothesis, X, y, reg, reg_d)
            # gradient = (1/self.m)*((hypothesis-y)@X) + reg_d
            self.cost.append(cost)
            return cost, gradient

        if self.multinomial:
            opt_weights = []
            for i in range(self.weights.shape[0]):
                options = {'maxiter': self.n_iters}
                Result = op.minimize(scipy_cost_func, x0=self.weights[i, :], args=(X, y), jac=True, method=opt, options=options)
                opt_weights.append(Result.x)
            self.weights = opt_weights
            self.weights = np.asarray(self.weights)
            self.cost = np.asarray(self.cost, dtype=float)/self.weights.shape[0]
        else:
            options = {'maxiter': self.n_iters}
            Result = op.minimize(scipy_cost_func, x0=self.weights, args=(X, y), jac=True, method=opt, options=options)
            self.weights = Result.x

        self.print_results(self.cost)

    def implement_regression(self, weights, X):
        raise NotImplementedError

    def cost_matplot(self):
        plt.figure(figsize=(10, 8))
        plt.plot(self.cost, color='red', linewidth=2, label='Cost')
        plt.legend('Aha')
        plt.show()

    def cost_px(self):
        fig = px.line(self.cost)
        fig.show()

    def cost_update(self, figure, chart):
        chart.plot(self.cost, color='red', linewidth=1)
        chart.set_xlabel("Epochs / Iterations")
        chart.set_ylabel('Cost')
        plt.title('Real Time Loss Chart')
        # drawing updated values
        figure.canvas.draw()
        figure.canvas.flush_events()

    def set_plot(self):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1,1,1)

        return fig, ax


class LinearRegression(Regression):
    def __init__(self, alpha=0.01, n_iters=1000, ld=0):
        # Regression.__init__(self, alpha=0.01, n_iters=1000, ld=0)
        super().__init__(alpha, n_iters, ld)
        self.mse = None
        self.rmse = None
        self.r2 = None
        self.r2_adj = None
        self.hypothesis = None
        self.multinomial = False

    def implement_regression(self, weights, X):
        hypothesis = X@weights
        self.hypothesis = hypothesis
        return hypothesis

    def predict(self, X, y=None):
        self.mse = None
        if isinstance(X, (list)):
            X.insert(0, 1)
        else:
            X, self.n, self.m = self.check_x(X)

        predictions = X@self.weights

        if y is not None:
            y = self.check_y(y)
            self.y = y
            self.hypothesis = predictions
            self.mse = np.mean((y - predictions)**2)
            self.rmse = math.sqrt(self.mse)
            sse = sum((y - predictions)**2)
            sst = sum((y-np.mean(self.y))**2)
            self.r2 = sse/sst
            self.r2_adj = 1-((1-self.r2)*(y.shape[0]-1))/(y.shape[0]-self.n-1)
        else:
            pass

        return predictions

    def metrics(self):
        SST = sum((self.y - np.mean(self.y))**2)  # Total sum of squares
        SSR = sum((self.hypothesis - np.mean(self.y))**2)  # sum of squares due to regression
        SSE = sum((self.y - self.hypothesis)**2)  # sum of squares due to error
        samples = self.y.shape[0]

        R2 = 1-(SSE/SST)  # SSR/SSE
        print('R2 =', R2)
        R2_adjusted = 1-((1-R2)*(samples-1))/(samples-self.n-1)
        print('R2_adjusted =', R2_adjusted)
        MAE = np.mean(np.abs(self.y - self.hypothesis))
        print('MAE =', MAE)
        MSE = np.mean((self.y - self.hypothesis)**2)
        print('MSE =', MSE)
        RMSE = math.sqrt(MSE)
        print('RMSE =', RMSE)
        MAPE = np.mean(np.abs((self.y - self.hypothesis)/self.y))*100  # mean absolute percentage error
        print('MAPE =', MAPE)
        Accuracy = 100-MAPE
        print('Accuracy =', Accuracy, '%')
        return R2, R2_adjusted, MAE, MSE, RMSE, MAPE, Accuracy


class MultinomialRegression(Regression):
    def __init__(self, alpha=0.01, n_iters=1000, ld=0):
        # Regression.__init__(self, alpha=0.01, n_iters=1000, ld=0)
        super().__init__(alpha, n_iters, ld)
        self.labels = None
        self.n_labels = None
        self.hypothesis = None
        self.loss_function = self.logloss
        self.reg_function = self.multinomial_L2_reg
        self.multinomial = True

        self.accuracy = None
        self.prob = None
        self.mse = None
        self.r = None
        self.r2 = None

    def logloss(self, hypothesis, X, y, reg, reg_d, bounce=False):
        #print('M logloss', bounce)
        coded = encoder(y)
        cost = np.sum(hypothesis*coded, axis=1)
        cost = np.mean(-np.log(cost))+reg
        if bounce:
            return cost
        hypothesis[np.arange(self.m), y.astype('int')] = hypothesis[np.arange(self.m), y.astype('int')]-1  # all the correct outcomes - 1
        gradient = (1/self.m)*(hypothesis.T.dot(X))+reg_d
        weights = self.weights - gradient*self.alpha

        return weights, cost, gradient

    def multinomial_L2_reg(self, weights):
        # reg = self.ld/(2*self.m)*(weights[:, 1:]@weights[:, 1:].T)
        reg = self.ld/(2*self.m)*np.sum(weights[:, 1:]**2)  # square itself and sum it up
        reg_d = self.ld/self.m*weights
        reg_d[:, 0] = 0
        return reg, reg_d

    def multinomial_L1_reg(self, weights):
        np.seterr(divide='ignore', invalid='ignore')
        reg = self.ld/(2*self.m)*(np.sum(np.abs(weights[:, 1:])))  # sum to 1 number to add to cost
        # reg_d = np.insert(self.ld/self.m*np.sign(weights[:, 1:]), 0, 0)
        reg_d = self.ld/self.m*np.sign(weights)
        reg_d[:, 0] = 0
        return reg, reg_d

    def implement_regression(self, weights, X):
        # to calculate the probabilities of each outcome , softmax
        z = X@weights.T
        exp = np.exp(z)
        sumOfArr = np.sum(exp, axis=1)
        sumOfArr = sumOfArr.reshape(self.m, 1)
        hypothesis = exp/sumOfArr
        self.hypothesis = hypothesis
        return hypothesis

    def multinomial_check(self, y):
        if isinstance(y, (pd.core.series.Series, pd.DataFrame, np.ndarray)):
            # number of classifications
            y = self.check_y(y)

            labels = np.unique(y)
            n_labels = labels.shape[0]
            m = y.shape[0]
        elif isinstance(y, (list, tuple)):
            labels = list(set(y))
            n_labels = len(labels)
            m = len(y)
        else:
            raise Exception('wrong type')
        if n_labels >= 2:
            return labels, n_labels, m
        else:
            return Exception('Only 1 label found')

    def multinomial_build(self, X, y):
        n = X.shape[1]
        y_outcomes = np.ones([self.m, self.n_labels])
        # weights = np.random.rand(self.n_labels, n+1) - generate random weights
        weights = np.zeros([self.n_labels, n+1])
        return y_outcomes, weights

    def fit(self, X, y, weights=None, loss='logloss', reg='L2'):
        self.labels, self.n_labels, self.m = self.multinomial_check(y)
        if weights is None:
            _, self.weights = self.multinomial_build(X, y)
        else:
            self.weights = weights
        if reg == 'L2':
            self.reg_function = self.multinomial_L2_reg
        if reg == 'L1':
            self.reg_function = self.multinomial_L1_reg
        super().fit(X, y, weights=self.weights, loss=loss, reg=reg)

    def sgd_fit(self, X, y, weights=None, batch=1, loss='Logloss', reg='L2', momentum=0.8, threshold=1e-7):
        self.labels, self.n_labels, self.m = self.multinomial_check(y)
        if weights is None:
            _, self.weights = self.multinomial_build(X, y)
        else:
            self.weights = weights
        if reg == 'L2':
            self.reg_function = self.multinomial_L2_reg
        if reg == 'L1':
            self.reg_function = self.multinomial_L1_reg
        super().sgd_fit(X, y, weights=self.weights, batch=batch, loss=loss, reg=reg, momentum=momentum, threshold=threshold)

    def NAG_fit(self, X, y, weights=None, batch=32, loss='Logloss', reg='L2', decay=0.9, threshold=1e-7):
        self.labels, self.n_labels, self.m = self.multinomial_check(y)
        if weights is None:
            _, self.weights = self.multinomial_build(X, y)
        else:
            self.weights = weights
        if reg == 'L2':
            self.reg_function = self.multinomial_L2_reg
        if reg == 'L1':
            self.reg_function = self.multinomial_L1_reg
        super().NAG_fit(X, y, weights=self.weights, batch=batch, loss=loss, reg=reg, decay=decay, threshold=threshold)

    def scipy_fit(self, X, y, weights=None, loss='logloss', reg='L2', opt='TNC'):
        return print('Does not work with scipy_fit')
        self.labels, self.n_labels, self.m = self.multinomial_check(y)
        if weights is None:
            _, self.weights = self.multinomial_build(X, y)
        else:
            self.weights = weights
        if reg == 'L2':
            self.reg_function = self.multinomial_L2_reg
        if reg == 'L1':
            self.reg_function = self.multinomial_L1_reg
        self.loss_function = self.scipy_logloss
        super().scipy_fit(X, y, weights=self.weights, loss='logloss', reg=reg, opt=opt)

    def predict(self, X, y=None):
        self.mse = None
        if isinstance(X, (list)):
            X.insert(0, 1)
        else:
            X, self.n, self.m = self.check_x(X)

        predictions = X@self.weights.T
        exp = np.exp(predictions)
        sumOfArr = np.sum(exp, axis=1)
        sumOfArr = sumOfArr.reshape(self.m, 1)
        self.prob = exp/sumOfArr
        predictions_class = np.argmax(self.prob, axis=1)
        # calculate mse
        if y is not None:
            y = self.check_y(y)
            self.y = y
            self.accuracy = np.mean(predictions_class == y)*100
        return predictions_class


class LogisticRegression(Regression):
    def __init__(self, alpha=0.01, n_iters=1000, ld=0):
        super().__init__(alpha, n_iters, ld)
        self.hypothesis = None
        self.accuracy = None
        self.prob = None
        self.multinomial = False

        self.loss_function = self.logloss

    def logloss(self, hypothesis, X, y, reg, reg_d, bounce=False):
        #print('L logloss', bounce)
        Pos_y = np.log(hypothesis)
        Neg_y = np.log(1-hypothesis)
        cost = (1/self.m)*sum((-y*Pos_y)-((1-y)*Neg_y))+reg
        if bounce:
            return cost
        gradient = (1/self.m)*((hypothesis-y)@X) + reg_d
        weights = self.weights-self.alpha*gradient
        return weights, cost, gradient

    def implement_regression(self, weights, X):
        hypothesis = self.sigmoid(X@weights)
        self.hypothesis = hypothesis
        return hypothesis

    def sigmoid(self, x):
        return 1./(1+np.exp(-x))

    def fit(self, X, y, weights=None, loss='Logloss', reg='L2'):
        super().fit(X, y, weights, loss, reg)

    def sgd_fit(self, X, y, weights=None, batch=1, loss='Logloss', reg='L2', momentum=0.8, threshold=1e-7):
        super().sgd_fit(X, y, weights, batch, loss, reg, momentum, threshold)

    def NAG_fit(self, X, y, weights=None, batch=32, loss='Logloss', reg='L2', decay=0.9, threshold=1e-7):
        super().NAG_fit(X, y, weights, batch, loss, reg, decay, threshold)

    def scipy_fit(self, X, y, weights=None, loss='Logloss', reg='L2', opt='TNC'):
        super().scipy_fit(X, y, weights, loss, reg)

    def predict(self, X, y=None):
        self.mse = None
        if isinstance(X, (list)):
            X.insert(0, 1)
        else:
            X, self.n, self.m = self.check_x(X)

        predictions = X@self.weights
        self.prob = self.sigmoid(predictions)
        predictions_class = (self.prob > 0.5).astype(int)
        # calculate mse
        if y is not None:
            y = self.check_y(y)
            self.y = y
            self.accuracy = np.mean(predictions_class == y)*100
        return predictions_class
