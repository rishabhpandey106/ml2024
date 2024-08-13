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

class stochastic_sgdr_scratch:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.intercept = None
        self.coefficients = None
    
    def sgdr(self,X, y):
        self.intercept = 0
        self.coefficients = np.ones(X.shape[1])

        for _ in range(self.epochs):
            for j in range(X.shape[0]):
                index = np.random.randint(0, X.shape[0])
                y_pred = X[index] @ self.coefficients + self.intercept
                intercept_slope = -2 * (y[index] - y_pred) # n = 1 as we are calculating for one row
                coefficients_slope = -2 * X[index] * (y[index] - y_pred) # n = 1 as we are calculating for one row
                self.intercept = self.intercept - (self.lr * intercept_slope)
                self.coefficients = self.coefficients - (self.lr * coefficients_slope)
        return self.intercept, self.coefficients
    
    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

gdr = stochastic_sgdr_scratch(0.013, 100)

intercept, coefficients = gdr.sgdr(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = gdr.predict(diabetes_X_test)

print("Mean squared error is:", mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Weights:  ",  coefficients)
print("Intercept:  ",  intercept)
print("Score ", r2_score(diabetes_y_test, diabetes_y_pred))

# used when large dataset, non convex function
# faster than batch_gdr but when epochs are same batch_gdr can be faster as for 1 epoch we calculate only 1 time, but in sgdr we calculate 1 * X.shape[0] times
# used alot
# batch_gdr may find local minima instead of global minima in case of large dataset
# batch_gdr uses a lot of ram in case of large dataset, so ram intensive method
# batch_gdr can be used when dataset is small and epochs are large