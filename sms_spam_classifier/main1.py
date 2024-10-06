import numpy as np
import pandas as pd

df = pd.read_csv('sms_spam_classifier/new_spam.csv')
df = df.dropna(subset=['transformed_text'])

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# cv = CountVectorizer()
# X = cv.fit_transform(df['transformed_text']).toarray()

tfd = TfidfVectorizer(max_features=3000)
X = tfd.fit_transform(df['transformed_text']).toarray()

y = df['v1'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
mnb = MultinomialNB()
# gnb = GaussianNB()
# bnb = BernoulliNB()
spam_detect_model_mnb = mnb.fit(X_train, y_train)
# spam_detect_model_gnb = gnb.fit(X_train, y_train)
# spam_detect_model_bnb = bnb.fit(X_train, y_train)

y_pred_mnb = spam_detect_model_mnb.predict(X_test)
# y_pred_gnb = spam_detect_model_gnb.predict(X_test)
# y_pred_bnb = spam_detect_model_bnb.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
print(accuracy_score(y_test, y_pred_mnb),precision_score(y_test, y_pred_mnb),confusion_matrix(y_test, y_pred_mnb))
# print(accuracy_score(y_test, y_pred_gnb),precision_score(y_test, y_pred_gnb),confusion_matrix(y_test, y_pred_gnb))
# print(accuracy_score(y_test, y_pred_bnb),precision_score(y_test, y_pred_bnb),confusion_matrix(y_test, y_pred_bnb))

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier

# lr = LogisticRegression(solver='liblinear',penalty='l1')
# rfc = RandomForestClassifier(n_estimators=50, random_state=2)
# svc = SVC(kernel='sigmoid',gamma='1.0')
# knc = KNeighborsClassifier()
# dtc = DecisionTreeClassifier(max_depth=5)

import pickle
pickle.dump(tfd, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))