import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# print(diabetes.DESCR)
diabetes_X  =  diabetes.data[:,np.newaxis,2] # bmi
# print(diabetes_X)

diabetes_X_train = diabetes_X[:-50]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-50]
diabetes_y_test = diabetes.target[-20:]

regression = linear_model.LinearRegression()
regression.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = regression.predict(diabetes_X_test)

print("Mean squared error is:  ", mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Weights:  ",  regression.coef_)
print("Intercept:  ",  regression.intercept_)

plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue')
plt.show()

