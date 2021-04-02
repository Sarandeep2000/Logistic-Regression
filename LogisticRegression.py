# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:17:29 2021

@author: Jaskirath Singh
"""

import numpy as np
from math import exp
from Functions import accuracy_score
from matplotlib import pyplot as plt

class Logistic_Regression:
    """
    A class for binary Logistic Regression Classifier.
    
    ...
    
    Attributes
    ----------
    coeffs : numpy.ndarray
        An array containing the coefficients of the model
    iterations : int
        Maximum number of iterations for stopping the model before convergence.
    r_type : string, optional
        Type of regularizer to be used. Supported types are "L1" and "L2". The default is "L2"
    alpha : float
        Learning rate (alpha) of the logistic regression gradient descent. The default is 0.01.
    lamb : float
        Regularization parameter (lambda) of the logistic regression regularizer. The default is 1.
    tol : float
        tolerance for stopping.
    
    Methods
    -------
    fit(X, y)
        fits the model according to the input matrix and target vector.
    predict(X)
        predicts the fitted model and returns the predicted numpy array.
    plot_log_loss_vs_iteration()
        plots the logit function against iterations.
    plot_loss_vs_iteration()
        plots the exponential of logit function (with regularizer term) against iterations.
    plot_accuracy_vs_iteration()
        plots the accuracy in predicting the train set against iterations.
    """
    def __init__(self, iterations = 100, r_type = "L2", alpha=0.01, lamb = 1, tol = 10**(-4)):
        self.coeffs = None
        self.iterations = iterations
        self.r_type = r_type
        self.alpha = alpha
        self.lamb = lamb
        self.loss = []
        self.accuracy = []
        self.tol = tol

    def sigmoid(self, z):
        """
        return the sigmoid value for the given z

        Parameters
        ----------
        z : float
            Input to the sigmoid function.

        Returns
        -------
        float
            a number between [0, 1]. The probability for Logistic Regression.

        """
        if(z>25):
            return 1
        if(z<-25):
            return 0
        return 1/(1+exp(-z))

    def regularizer(self, m):
        """
        

        Parameters
        ----------
        m : int
            Total number of samples given.

        Returns
        -------
        float
            the cost of regularizer for the total cost calculation.

        """
        if(self.r_type == "L1" or self.r_type == "l1"):
            arr = np.sum(np.absolute(self.coeffs))
            arr -= abs(self.coeffs[0])
            return (self.lamb/m)*arr
        else:
            arr = np.sum(np.square(self.coeffs))
            arr -= self.coeffs[0]**2
            return (self.lamb/(2*m))*arr

    def total_cost(self, X, y):
        """
        Calculates the total cost (log-loss) of the model.

        Parameters
        ----------
        X : pandas.DataFrame
            input matrix.
        y : pandas.Series
            target vector.

        Returns
        -------
        cost : float
            cost of the current fit.

        """
        m = len(X)
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        y_pred = np.array([self.sigmoid(i) for i in X_np @ self.coeffs])
        cost = (-1/m)*((y_np.T @ np.log(y_pred)) + ((1-y_np).T @ np.log(1-y_pred)))
        cost += self.regularizer(m)
        return cost

    def grad_reg(self, m):           #regularizer update of gradient ascent
        """
        gradient descent of the regularizer term in cost function.

        Parameters
        ----------
        m : int
            total number of input samples.

        Returns
        -------
        numpy.ndarray
            the cost for every coefficient of regularizer.

        """
        if(self.r_type == "L1" or self.r_type == "l1"):
            arr = np.sign(self.coeffs)
            arr[0] = 0
            return (self.lamb/m)*arr
        else:
            arr = np.array(self.coeffs)
            arr[0] = 0
            return (self.lamb/m)*arr

    def update_coeffs(self, X, y):
        """
        Updates the coefficients to fit the model

        Parameters
        ----------
        X : pandas.DataFrame
            input matrix.
        y : pandas.Series
            target vector.

        Returns
        -------
        None.

        """
        m = len(X)
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        y_pred = np.array([self.sigmoid(i) for i in X_np @ self.coeffs])
        self.coeffs = self.coeffs - self.alpha*self.grad_reg(m)                               #regularizer update
        self.coeffs = self.coeffs + (self.alpha/m)*(X_np.T @ (y_np - y_pred))                 #loss function update

    def fit(self, X, y):
        """
        Fits the given input data to the model.

        Parameters
        ----------
        X : pandas.DataFrame
            input matrix.
        y : pandas.Series
            target vector.

        Returns
        -------
        None.

        """
        self.coeffs = np.zeros(len(X.columns))
        X_np = X.to_numpy()
        for i in range(self.iterations):
            self.update_coeffs(X, y)
            self.loss.append(self.total_cost(X, y))
            self.accuracy.append(accuracy_score(y, np.round(np.array([self.sigmoid(i) for i in X_np @ self.coeffs]))))
            if(len(self.loss) == 1):
                pass
            elif(self.loss[-2] - self.loss[-1] < self.tol):
                break

    def plot_log_loss_vs_iteration(self):
        """
        Plots the logit function against the number of iterations.

        Returns
        -------
        None.

        """
        iterations = [i for i in range(len(self.loss))]
        plt.plot(iterations, self.loss)
        plt.xlabel("Iterations")
        plt.ylabel("Log-Loss")
        plt.title("Log-Loss vs. Iteration graph")
        plt.show()

    def plot_loss_vs_iteration(self):
        """
        Plots the exponent of logit function (with regularizer term) against the iterations.

        Returns
        -------
        None.

        """
        iterations = [i for i in range(len(self.loss))]
        plt.plot(iterations, np.exp(self.loss))
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss vs. Iteration graph")
        plt.show()

    def plot_accuracy_vs_iteration(self):
        """
        Plots the accuracy on train set against the iteration

        Returns
        -------
        None.

        """
        iterations = [i for i in range(len(self.accuracy))]
        plt.plot(iterations, self.accuracy)
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Iteration graph")
        plt.show()

    def predict(self, X):
        """
        Predicts the dataset on the fitted model

        Parameters
        ----------
        X : pandas.DataFrame
            input matrix for prediction.

        Returns
        -------
        numpy.ndarray
            prediction vector for the fitted dataset.

        """
        if(self.coeffs is not None):
            X_np = X.to_numpy()
            y_pred = np.array([self.sigmoid(i) for i in X_np @ self.coeffs])
            return np.round(y_pred)
        else:
            raise Exception("Model not fitted.")
