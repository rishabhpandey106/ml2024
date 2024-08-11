import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_X  =  diabetes.data 

diabetes_X_train = diabetes_X[:-50]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-50]
diabetes_y_test = diabetes.target[-20:]

class batch_gdr:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.intercept = None
        self.coefficients = None
    
    def batch_gdr(self,X, y):
        self.intercept = 0
        self.coefficients = np.ones(X.shape[1])
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.coefficients) + self.intercept
            intercept_slope = -2 * np.mean(y - y_pred)
            coefficients_slope = -2 * np.dot(X.T, y - y_pred) / X.shape[0]
            self.intercept = self.intercept - (self.lr * intercept_slope)
            self.coefficients = self.coefficients - (self.lr * coefficients_slope)
        return self.intercept, self.coefficients
    
    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

gdr = batch_gdr(0.98, 100000)

intercept, coefficients = gdr.batch_gdr(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = gdr.predict(diabetes_X_test)

print("Mean squared error is:", mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Weights:  ",  coefficients)
print("Intercept:  ",  intercept)

