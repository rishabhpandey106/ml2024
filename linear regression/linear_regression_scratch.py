import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_X  =  diabetes.data[:,np.newaxis,2] # bmi

diabetes_X_train = diabetes_X[:-50]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-50]
diabetes_y_test = diabetes.target[-20:]

def linear_regression(X, y):
    n = len(X)
    num = 0
    den = 0
    for i in range(n):
        num += (X[i] - X.mean()) * (y[i] - y.mean())
        den += (X[i] - X.mean()) ** 2
    m = num / den
    b = y.mean() - m * X.mean()
    return m, b

m, b = linear_regression(diabetes_X_train, diabetes_y_train)

def predict(X, m, b):
    return m * X + b

# Predict on the test set
diabetes_y_pred = predict(diabetes_X_test, m, b)

# def mean_squared_error(y_true, y_pred):
#     return np.average((y_true - y_pred) ** 2)

# mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("Mean squared error is:", mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Weights:  ",  m)
print("Intercept:  ",  b)

plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue')
plt.xlabel('Radius')
plt.ylabel('Target')
plt.show()
