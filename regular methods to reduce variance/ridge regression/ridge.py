import numpy as np
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

diabetes_X  =  diabetes.data # bmi

diabetes_X_train = diabetes_X[:-50]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-50]
diabetes_y_test = diabetes.target[-20:]

regression = Ridge(alpha=0.0001)
regression.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = regression.predict(diabetes_X_test)

print("Mean squared error is:  ", mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Weights:  ",  regression.coef_)
print("Intercept:  ",  regression.intercept_)
print("r2 score ", r2_score(diabetes_y_test, diabetes_y_pred))

# alpha lower -> overfitting -> lower bias , higher variance
# alpha higher -> underfitting -> higher bias , lower variance
# alpha increases -> shrinks the coefficients (tends to 0 but never 0)
# higher value are more affected more i.e. hop btw two coefficients are very high as well as lower value have low hops
# bias - variance tradeoff, always wants to minimize bias and maximize variance as much as possible
# alpha increases -> loss function tends to 0
# used to handle multicollinearity, prevent overfitting, and control model complexity(problem of linear regression)
# extension of linear regression
# bias -> follows every training example, variance -> differnce btw training and testing examples(mse)