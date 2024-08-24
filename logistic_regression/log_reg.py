import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1,n_classes=2, random_state=41, n_clusters_per_class=1,hypercube=False, class_sep=10)

model = LogisticRegression().fit(x,y)
m = -(model.coef_[0][0] / model.coef_[0][1])
b = -(model.intercept_ / model.coef_[0][1])

print("slope: ", m, "intercept: ", b)

x_input = np.linspace(-3,3,10)
y_input = m * x_input + b

plt.figure(figsize=(10,6))
plt.scatter(x[:,0],x[:,1],c=y,cmap='winter',edgecolors='b')
plt.plot(x_input,y_input,c='red')
plt.show()