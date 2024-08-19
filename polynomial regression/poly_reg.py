import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_X  =  diabetes.data # bmi

diabetes_X_train = diabetes_X[:-50]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-50]
diabetes_y_test = diabetes.target[-20:]

regression = PolynomialFeatures(degree=1)
x_train_new = regression.fit_transform(diabetes_X_train)
x_test_new = regression.transform(diabetes_X_test)

lr = LinearRegression()
lr.fit(x_train_new, diabetes_y_train)

diabetes_y_pred = lr.predict(x_test_new)

print("Mean squared error is:  ", mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Weights:  ",  lr.coef_)
print("Intercept:  ",  lr.intercept_)
print("r2 score ", r2_score(diabetes_y_test, diabetes_y_pred))

# better fit to non linear data
# higher-degree polynomials can lead to overly complex models
# low degree polynomials can lead to underfitting
# higher-degree polynomials can lead to overfitting
