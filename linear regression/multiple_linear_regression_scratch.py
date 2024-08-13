import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

diabetes_X  =  diabetes.data 

diabetes_X_train = diabetes_X[:-50]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-50]
diabetes_y_test = diabetes.target[-20:]

def linear_regression(X, y):
    X = np.insert(X, 0, 1, axis=1)
    b = np.linalg.inv(X.T @ X) @ X.T @ y
    intercept = b[0]
    cofficients = b[1:]
    return intercept, cofficients

intercept, cofficients = linear_regression(diabetes_X_train, diabetes_y_train)

def predict(X, intercept, cofficients):
    return intercept + np.dot(X, cofficients)

# Predict on the test set
diabetes_y_pred = predict(diabetes_X_test, intercept, cofficients)

# def mean_squared_error(y_true, y_pred):
#     return np.average((y_true - y_pred) ** 2)

# mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("Mean squared error is:", mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Weights:  ",  cofficients)
print("Intercept:  ",  intercept)
print("Score ", r2_score(diabetes_y_test, diabetes_y_pred))

