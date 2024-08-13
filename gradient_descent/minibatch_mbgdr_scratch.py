import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error,  r2_score

diabetes = datasets.load_diabetes()

diabetes_X  =  diabetes.data 

diabetes_X_train = diabetes_X[:-50]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-50]
diabetes_y_test = diabetes.target[-20:]

class mini_batch_mbgdr:
    def __init__(self, lr, epochs, batch_size):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.intercept = None
        self.coefficients = None
    
    def mbgdr(self,X, y):
        self.intercept = 0
        self.coefficients = np.ones(X.shape[1])

        for _ in range(self.epochs):
            for j in range(int(X.shape[0]/self.batch_size)):
                index = random.sample(range(X.shape[0]), self.batch_size)
                y_pred = X[index] @ self.coefficients + self.intercept
                intercept_slope = -2 * np.mean(y[index] - y_pred) 
                coefficients_slope = -2 * np.dot(X[index].T, (y[index] - y_pred)) / self.batch_size 
                self.intercept = self.intercept - (self.lr * intercept_slope)
                self.coefficients = self.coefficients - (self.lr * coefficients_slope)
        return self.intercept, self.coefficients
    
    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

gdr = mini_batch_mbgdr(0.37, 100, 35)

intercept, coefficients = gdr.mbgdr(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = gdr.predict(diabetes_X_test)

print("Mean squared error is:", mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Weights:  ",  coefficients)
print("Intercept:  ",  intercept)
print("Score ", r2_score(diabetes_y_test, diabetes_y_pred))

