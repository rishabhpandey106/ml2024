import numpy as np
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1,n_classes=2, random_state=41, n_clusters_per_class=1,hypercube=False, class_sep=10)

def sigmoid_method(x,y):
    x = np.insert(x, 0, 1, axis=1)
    weights = np.ones(x.shape[1])
    lr = 0.1
    epochs = 100
    for _ in range(epochs):
        j = np.random.randint(x.shape[0])
        y_pred = sigmoid(np.dot(x[j], weights))
        weights = weights + lr * (y[j] - y_pred) * x[j]

    return weights[0], weights[1:]

def sigmoid(z):
    return 1 / (1+np.exp(-z))

intercept, coefficients = sigmoid_method(x,y)

print("intercepts: ", intercept, "coefficients: ", coefficients)

m = - coefficients[0] / coefficients[1]
b = - intercept / coefficients[1]

print("slope: ", m, "intercept: ", b)

x_input = np.linspace(-3,3,10)
y_input = m * x_input + b

plt.figure(figsize=(10,6))
plt.scatter(x[:,0],x[:,1],c=y,cmap='winter',edgecolors='b')
plt.plot(x_input,y_input,c='red')
plt.show()