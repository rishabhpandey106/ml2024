import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('cancer_detection\data.csv')
backup = data.copy()

LabelEncoder = LabelEncoder()
data['diagnosis'] = LabelEncoder.fit_transform(data['diagnosis'])

data = data.drop("Unnamed: 32", axis=1)
data = data.drop("id", axis=1)
# print(data.head())

x = data.iloc[:,1:]
y = data.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

new_input = np.array([[16.11,18.05,105.1,813,0.09721,0.1137,0.09447,0.05943,0.1861,0.06248,0.7049,1.332,4.533,74.08,0.00677,0.01938,0.03067,0.01167,0.01875,0.003434,19.92,25.27,129,1233,0.1314,0.2236,0.2802,0.1216,0.2792,0.08158],[11.43,17.31,73.66,398,0.1092,0.09486,0.02031,0.01861,0.1645,0.06562,0.2843,1.908,1.937,21.38,0.006664,0.01735,0.01158,0.00952,0.02282,0.003526,12.78,26.76,82.66,503,0.1413,0.1792,0.07708,0.06402,0.2584,0.08096]])
new_input_scaled = sc.transform(new_input)

output_2 = lr.predict(new_input_scaled)
print('output ',output_2)

from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))