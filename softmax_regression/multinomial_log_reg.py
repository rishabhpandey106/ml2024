from seaborn import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_dataset('exercise')
# print(data.head())

encoder = LabelEncoder()
data['kind'] = encoder.fit_transform(data['kind'])
# print(data.head())

# data cleaning
data['time'] = data['time'].str.split(' ').str.get(0)
data['time'] = data['time'].astype(int)
data = data[['pulse', 'time', 'kind']]
# print(data.head())

x = data.iloc[:,0:2]
y = data.iloc[:,-1]
# print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

classifier = LogisticRegression(multi_class='multinomial')
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred))

print(pd.DataFrame(confusion_matrix(y_test, y_pred)))

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.values, y.values, clf=classifier, legend=2)
plt.show()


# used in case of multi-class ,it uses softmax function 