import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

breast_cancer = datasets.load_breast_cancer()

# print(breast_cancer.DESCR)
breast_cancer_X  =  breast_cancer.data[:,np.newaxis,0] # radius
# print(breast_cancer_X)

breast_cancer_X_train = breast_cancer_X[:-50]
breast_cancer_X_test = breast_cancer_X[-20:]

breast_cancer_y_train = breast_cancer.target[:-50]
breast_cancer_y_test = breast_cancer.target[-20:]

regression = linear_model.LinearRegression()
regression.fit(breast_cancer_X_train, breast_cancer_y_train)

breast_cancer_y_pred = regression.predict(breast_cancer_X_test)

print("Mean squared error is:  ", mean_squared_error(breast_cancer_y_test, breast_cancer_y_pred))
print("Weights:  ",  regression.coef_)
print("Intercept:  ",  regression.intercept_)

plt.scatter(breast_cancer_X_test, breast_cancer_y_test, color='black')
plt.plot(breast_cancer_X_test, breast_cancer_y_pred, color='blue')
plt.show()

