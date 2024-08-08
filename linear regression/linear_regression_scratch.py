import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error

breast_cancer = datasets.load_breast_cancer()

breast_cancer_X  =  breast_cancer.data[:,np.newaxis,0] # radius

breast_cancer_X_train = breast_cancer_X[:-50]
breast_cancer_X_test = breast_cancer_X[-20:]

breast_cancer_y_train = breast_cancer.target[:-50]
breast_cancer_y_test = breast_cancer.target[-20:]

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

m, b = linear_regression(breast_cancer_X_train, breast_cancer_y_train)

def predict(X, m, b):
    return m * X + b

# Predict on the test set
breast_cancer_y_pred = predict(breast_cancer_X_test, m, b)

# def mean_squared_error(y_true, y_pred):
#     return np.average((y_true - y_pred) ** 2)

# mse = mean_squared_error(breast_cancer_y_test, breast_cancer_y_pred)
print("Mean squared error is:", mean_squared_error(breast_cancer_y_test, breast_cancer_y_pred))
print("Weights:  ",  m)
print("Intercept:  ",  b)

plt.scatter(breast_cancer_X_test, breast_cancer_y_test, color='black')
plt.plot(breast_cancer_X_test, breast_cancer_y_pred, color='blue')
plt.xlabel('Radius')
plt.ylabel('Target')
plt.show()
